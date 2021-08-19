import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strid, padding, norm):
        super().__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                strid,
                padding,
                bias=True,
                padding_mode="reflect",
            )
        )
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self, input_channels, dim):
        super().__init__()

        self.disc = nn.Sequential(
            Block(input_channels, dim, 4, 2, 1, False),
            Block(dim, dim * 2, 4, 2, 1, True),
            Block(dim * 2, dim * 4, 4, 2, 1, True),
            Block(dim * 4, dim * 8, 4, 1, 1, True),
            nn.Conv2d(dim * 8, 1, 4, 1, 1, bias=False, padding_mode="reflect"),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


if __name__ == "__main__":
    D = Discriminator(3, 64)
    x = torch.rand([4, 3, 256, 256])

    x_ = D(x)
    print(x_.shape)
