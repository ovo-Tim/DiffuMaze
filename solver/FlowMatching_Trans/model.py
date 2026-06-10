import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def _get_1d_sincos_pos_embed(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / embed_dim))
    pos = pos.flatten()
    out = pos[:, None] * omega[None, :]
    return torch.cat([out.sin(), out.cos()], dim=-1)


def get_2d_sincos_pos_embed(embed_dim: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    assert embed_dim % 2 == 0
    grid_h = torch.arange(H, device=device, dtype=torch.float32)
    grid_w = torch.arange(W, device=device, dtype=torch.float32)
    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid_h)
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid_w)
    pos_embed = torch.cat([
        emb_h[:, None, :].expand(H, W, -1),
        emb_w[None, :, :].expand(H, W, -1),
    ], dim=-1)
    return pos_embed.reshape(H * W, embed_dim)


class AdaLNModulation(nn.Module):
    def __init__(self, hidden_size: int, num_modulations: int = 6):
        super().__init__()
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, num_modulations * hidden_size),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.proj(c)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        return x


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_ratio * hidden_size),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_size, hidden_size),
        )
        self.adaLN_modulation = AdaLNModulation(hidden_size, num_modulations=6)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, shift2, scale2, gate1, gate2 = self.adaLN_modulation(c).chunk(6, dim=-1)
        x_norm = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        attn_out = self.attn(x_norm)
        x = x + gate1.unsqueeze(1) * attn_out
        x_norm = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        mlp_out = self.mlp(x_norm)
        x = x + gate2.unsqueeze(1) * mlp_out
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_channels * patch_size * patch_size)
        self.adaLN_modulation = AdaLNModulation(hidden_size, num_modulations=2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        patch_size: int = 1,
        time_emb_dim: int = 256,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.use_checkpoint = use_checkpoint

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.patch_embed = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, out_channels, patch_size)

    def unpatchify(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        p = self.patch_size
        B = x.shape[0]
        if p == 1:
            return x.transpose(1, 2).reshape(B, self.out_channels, H, W)
        h, w = H // p, W // p
        x = x.reshape(B, h, w, p, p, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, self.out_channels, H, W)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        t_emb = self.time_embed(t)

        x = self.patch_embed(x)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)

        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, h, w, x.device)
        x = x + pos_embed.unsqueeze(0)

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(
                    partial(block.__class__.forward, block),
                    x, t_emb,
                    use_reentrant=False,
                )
            else:
                x = block(x, t_emb)

        x = self.final_layer(x, t_emb)
        x = self.unpatchify(x, H, W)
        return x


class TransSmall(Transformer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int = 288,
        depth: int = 5,
        num_heads: int = 4,
        mlp_ratio: int = 3,
        patch_size: int = 2,
        time_emb_dim: int = 576,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, hidden_size, depth, num_heads, mlp_ratio, patch_size, time_emb_dim, **kwargs)


class TransXSmall(Transformer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int = 128,
        depth: int = 5,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        patch_size: int = 2,
        time_emb_dim: int = 256,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, hidden_size, depth, num_heads, mlp_ratio, patch_size, time_emb_dim, **kwargs)


class TransXXSmall(Transformer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int = 80,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        patch_size: int = 2,
        time_emb_dim: int = 80,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, hidden_size, depth, num_heads, mlp_ratio, patch_size, time_emb_dim, **kwargs)