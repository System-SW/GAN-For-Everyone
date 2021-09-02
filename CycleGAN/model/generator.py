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
        down=True,
        use_act=True,
        **kwargs,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                padding_mode="reflect",
                bias=False,
                **kwargs,
            )
            if down
            else nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
                **kwargs,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(dim, dim, 3, 1, 1),
            ConvBlock(dim, dim, 3, 1, 1, use_act=False),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, dim, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, dim, 7, 1, 3, padding_mode="reflect", bias=False),
            nn.ReLU(inplace=True),
        )

        self.down_blocks = nn.Sequential(
            ConvBlock(dim, dim * 2, 3, 2, 1),
            ConvBlock(dim * 2, dim * 4, 3, 2, 1),
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(dim * 4) for _ in range(num_residuals)]
        )

        self.up_blocks = nn.Sequential(
            ConvBlock(dim * 4, dim * 2, 3, 2, 1, down=False, output_padding=1),
            ConvBlock(dim * 2, dim * 1, 3, 2, 1, down=False, output_padding=1),
        )

        self.last = nn.Sequential(
            nn.Conv2d(dim * 1, output_channels, 7, 1, 3, padding_mode="reflect"),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down_blocks(x)
        x = self.residual_blocks(x)
        x = self.up_blocks(x)
        return self.last(x)


if __name__ == "__main__":
    input_channels = 3
    output_channels = 3
    batch_size = 4
    x = torch.rand([batch_size, input_channels, 256, 256])
    G = Generator(input_channels, output_channels, 64)
    G_ = G(x)
    print(G_.shape)
