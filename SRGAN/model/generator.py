import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        norm,
        use_act=True,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels) if norm else nn.Identity(),
            nn.PReLU(out_channels) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(dim, dim, 3, 1, 1, True, use_act=False),
            ConvBlock(dim, dim, 3, 1, 1, True, use_act=False),
        )

    def forward(self, x):
        return self.block(x) + x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels * scale_factor ** 2, 3, 1, 1, bias=False
            ),
            nn.PixelShuffle(scale_factor),
            nn.PReLU(in_channels),
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, dim, num_residuals=16):
        super().__init__()
        self.initial = ConvBlock(input_channels, dim, 9, 1, 4, False)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(dim) for _ in range(num_residuals)]
        )
        self.conv_block = ConvBlock(dim, dim, 3, 1, 1, True, False)
        self.upsample_blocks = nn.Sequential(
            UpsampleBlock(dim, 2),
            UpsampleBlock(dim, 2),
        )
        self.final = nn.Sequential(
            nn.Conv2d(dim, output_channels, 9, 1, 4, bias=False), nn.Tanh()
        )

    def forward(self, x):
        initial = self.initial(x)
        x = self.residual_blocks(initial)
        x = self.conv_block(x) + initial
        x = self.upsample_blocks(x)
        return self.final(x)


if __name__ == "__main__":
    input_channels = 3
    output_channels = 3
    batch_size = 4
    x = torch.rand([batch_size, input_channels, 256, 256])
    G = Generator(input_channels, output_channels, 64, num_residuals=12)
    G_ = G(x)
    print(G_.shape)
