import torch
from torch._C import layout
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_img, dim):
        super(Discriminator, self).__init__()

        self.encoder = nn.Sequential(
            self._block(channels_img, dim * 1, 4, 2, 1, True, False),
            self._block(dim * 1, dim * 2, 4, 2, 1, True, True),
            self._block(dim * 2, dim * 4, 4, 2, 1, True, True),
        )

        self.decoder = nn.Sequential(
            self._block(dim * 4, dim * 2, 4, 2, 1, False, True),
            self._block(dim * 2, dim * 1, 4, 2, 1, False, True),
            nn.ConvTranspose2d(dim * 1, channels_img, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels,
               kernel_size, stride, padding, encode, norm):
        layers = []
        conv = nn.Conv2d if encode else nn.ConvTranspose2d
        layers.append(
            conv(in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 bias=False)
        )
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x):
        hidden = self.encoder(x)
        return self.decoder(hidden), hidden


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(z_dim, dim * 16, 4, 1, 0),  # img: 4x4
            self._block(dim * 16, dim * 8, 4, 2, 1),  # img: 8x8
            self._block(dim * 8, dim * 4, 4, 2, 1),  # img: 16x16
            self._block(dim * 4, dim * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                dim * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),  # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
