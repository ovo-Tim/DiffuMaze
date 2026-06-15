import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X /= X.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-7)
    if X.size(-2) > X.size(-1):
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer for hidden 2D+ weights.

    Use AdamW for biases, norm parameters, embeddings, and output heads. This
    implementation is intentionally local and single-process safe; DDP averages
    gradients before the optimizer step, so no optimizer-internal collectives are
    needed here.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.0, nesterov: bool = True):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.ndim < 2:
                    raise ValueError("Muon expects only 2D or higher-dimensional parameters")
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if nesterov else buf

                original_shape = update.shape
                update = update.reshape(update.shape[0], -1)
                update = zeropower_via_newtonschulz5(update)
                update *= max(1.0, update.size(0) / update.size(1)) ** 0.5
                p.add_(update.reshape(original_shape), alpha=-lr)

        return loss


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
    omega = 1.0 / (10000 ** (omega / (embed_dim // 2)))
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


def precompute_2d_rope(head_dim: int, H: int, W: int, device: torch.device, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 4 == 0, f"head_dim ({head_dim}) must be divisible by 4 for 2D RoPE"
    half_pairs = head_dim // 4
    inv_freq = 1.0 / (base ** (torch.arange(0, half_pairs, device=device, dtype=torch.float32) * 2.0 / (head_dim // 2)))
    h_pos = torch.arange(H, device=device, dtype=torch.float32)
    h_angles = h_pos[:, None] * inv_freq[None, :]
    h_cos, h_sin = h_angles.cos(), h_angles.sin()
    w_pos = torch.arange(W, device=device, dtype=torch.float32)
    w_angles = w_pos[:, None] * inv_freq[None, :]
    w_cos, w_sin = w_angles.cos(), w_angles.sin()
    cos = torch.cat([
        h_cos[:, None, :].expand(H, W, -1),
        w_cos[None, :, :].expand(H, W, -1),
    ], dim=-1).reshape(H * W, head_dim // 2)
    sin = torch.cat([
        h_sin[:, None, :].expand(H, W, -1),
        w_sin[None, :, :].expand(H, W, -1),
    ], dim=-1).reshape(H * W, head_dim // 2)
    return cos, sin


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0).to(x.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(x.dtype)
    return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).reshape(x.shape)


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
    def __init__(self, hidden_size: int, num_heads: int, use_rope: bool = False, qk_norm: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.use_rope = use_rope
        self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, cos=None, sin=None) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.use_rope and cos is not None:
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        return x


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int = 4, use_rope: bool = False, qk_norm: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = Attention(hidden_size, num_heads, use_rope=use_rope, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_ratio * hidden_size),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_size, hidden_size),
        )
        self.adaLN_modulation = AdaLNModulation(hidden_size, num_modulations=6)

    def forward(self, x: torch.Tensor, c: torch.Tensor, cos=None, sin=None) -> torch.Tensor:
        shift1, scale1, shift2, scale2, gate1, gate2 = self.adaLN_modulation(c).chunk(6, dim=-1)
        x_norm = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        attn_out = self.attn(x_norm, cos=cos, sin=sin)
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

        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

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


class TransformerWithRoPE(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_dim = self.hidden_size // self.num_heads
        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_size, self.num_heads, self.mlp_ratio, use_rope=True)
            for _ in range(self.depth)
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        t_emb = self.time_embed(t)
        x = self.patch_embed(x)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)

        cos, sin = precompute_2d_rope(self.head_dim, h, w, x.device)

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(
                    partial(block.__class__.forward, block),
                    x, t_emb, cos, sin,
                    use_reentrant=False,
                )
            else:
                x = block(x, t_emb, cos=cos, sin=sin)

        x = self.final_layer(x, t_emb)
        x = self.unpatchify(x, H, W)
        return x


class TransSmall_rope(TransformerWithRoPE):
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


class TransXSmall_rope(TransformerWithRoPE):
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


class TransXXSmall_rope(TransformerWithRoPE):
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


class VAEEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, hidden_size, 3, 1, 1)
        self.down = nn.Conv2d(hidden_size, hidden_size, 3, 2, 1)
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, 1, 1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, 1, 1)
        self.norm = nn.GroupNorm(32, hidden_size)
        self.silu = nn.SiLU()
        self.mean_conv = nn.Conv2d(hidden_size, hidden_size, 1)
        self.logvar_conv = nn.Conv2d(hidden_size, hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.silu(self.conv_in(x))
        x = self.silu(self.down(x))
        x = self.silu(self.conv1(x))
        x = self.conv2(x)
        x = self.silu(self.norm(x))
        return self.mean_conv(x), self.logvar_conv(x)


class VAEDecoder(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.conv_in = nn.Conv2d(out_channels, hidden_size, 3, 1, 1)
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, 1, 1)
        self.norm = nn.GroupNorm(32, hidden_size)
        self.silu = nn.SiLU()
        self.up = nn.ConvTranspose2d(hidden_size, hidden_size // 2, 4, 2, 1)
        self.conv_out = nn.Conv2d(hidden_size // 2, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.silu(self.conv_in(x))
        x = self.silu(self.conv1(x))
        x = self.silu(self.norm(x))
        x = self.silu(self.up(x))
        x = self.conv_out(x)
        return x


class TransSmall_rope_vae(TransformerWithRoPE):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int = 288,
        depth: int = 5,
        num_heads: int = 4,
        mlp_ratio: int = 3,
        time_emb_dim: int = 576,
        **kwargs,
    ):
        kwargs.pop("patch_size", None)
        super().__init__(in_channels, out_channels, hidden_size, depth, num_heads, mlp_ratio, patch_size=1, time_emb_dim=time_emb_dim, **kwargs)
        self.patch_embed = nn.Conv2d(hidden_size, hidden_size, 1, 1)
        self.vae_encoder = VAEEncoder(in_channels, hidden_size)
        self.vae_decoder = VAEDecoder(hidden_size, out_channels)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        t_emb = self.time_embed(t)
        mean, logvar = self.vae_encoder(x)
        z = self.reparameterize(mean, logvar)
        zH, zW = z.shape[2], z.shape[3]
        x = self.patch_embed(z)
        x = x.flatten(2).transpose(1, 2)
        cos, sin = precompute_2d_rope(self.head_dim, zH, zW, x.device)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(
                    partial(block.__class__.forward, block),
                    x, t_emb, cos, sin,
                    use_reentrant=False,
                )
            else:
                x = block(x, t_emb, cos=cos, sin=sin)
        x = self.final_layer(x, t_emb)
        x = self.unpatchify(x, zH, zW)
        x = self.vae_decoder(x)
        return x, mean, logvar
