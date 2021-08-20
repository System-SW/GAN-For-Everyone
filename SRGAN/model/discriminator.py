import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm):
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
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self, output_channels, dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
            Block(output_channels, dim, 3, 1, 1, False),
            Block(dim * 1, dim * 1, 3, 2, 1, True),
            Block(dim * 1, dim * 2, 3, 1, 1, True),
            Block(dim * 2, dim * 2, 3, 2, 1, True),
            Block(dim * 2, dim * 4, 3, 1, 1, True),
            Block(dim * 4, dim * 4, 3, 2, 1, True),
            Block(dim * 4, dim * 8, 3, 1, 1, True),
            Block(dim * 8, dim * 8, 3, 2, 1, True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(6),
            nn.Flatten(),
            nn.Linear(dim * 8 * 6 * 6, dim * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim * 16, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.classifier(x)


if __name__ == "__main__":
    dim = 64
    output_channels = 3
    x = torch.rand([4, output_channels, 512, 512])
    D = Discriminator(output_channels, dim)
    print(D(x).shape)
