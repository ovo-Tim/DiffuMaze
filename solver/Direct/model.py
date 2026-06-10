import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return F.silu(h + self.skip(x))


class DirectUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_ch: int = 64,
        ch_mults: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        ch_list = [base_ch * m for m in ch_mults]

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_ch = base_ch
        for i, ch in enumerate(ch_list):
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                blocks.append(ResBlock(prev_ch if j == 0 else ch, ch))
            self.down_blocks.append(blocks)
            if i < len(ch_mults) - 1:
                self.downsamples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            else:
                self.downsamples.append(None)
            prev_ch = ch

        self.bottleneck = nn.ModuleList([
            ResBlock(ch_list[-1], ch_list[-1]),
            ResBlock(ch_list[-1], ch_list[-1]),
        ])

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in range(len(ch_list) - 2, -1, -1):
            ch = ch_list[i]
            ch_prev = ch_list[i + 1]
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                in_ch = (ch_prev + ch) if j == 0 else ch
                blocks.append(ResBlock(in_ch, ch))
            self.up_blocks.append(blocks)
            self.upsamples.append(nn.Upsample(scale_factor=2, mode="nearest"))

        self.conv_out = nn.Conv2d(base_ch, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)

        skips = []
        for blocks, down in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                x = block(x)
            skips.append(x)
            if down is not None:
                x = down(x)

        for block in self.bottleneck:
            x = block(x)

        for blocks, up, skip in zip(self.up_blocks, self.upsamples, reversed(skips[:-1])):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                x = block(x)

        return self.conv_out(x)


class DirectUNetSmall(DirectUNet):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_ch: int = 64,
        ch_mults: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
    ):
        super().__init__(in_channels, out_channels, base_ch, ch_mults, num_res_blocks)


class DirectUNetXSmall(DirectUNet):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_ch: int = 32,
        ch_mults: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 1,
    ):
        super().__init__(in_channels, out_channels, base_ch, ch_mults, num_res_blocks)


class DirectUNetXXSmall(DirectUNet):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_ch: int = 16,
        ch_mults: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 1,
    ):
        super().__init__(in_channels, out_channels, base_ch, ch_mults, num_res_blocks)