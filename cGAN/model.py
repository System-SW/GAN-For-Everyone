import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_img, dim, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img + 1, dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(dim * 1, dim * 2, 4, 2, 1),
            self._block(dim * 2, dim * 4, 4, 2, 1),
            self._block(dim * 4, dim * 8, 4, 2, 1),
            nn.Conv2d(dim * 8, 1, kernel_size=4, stride=2, padding=0),
        )
        self.embed = nn.Embedding(num_classes, img_size * img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(
            labels.shape[0], 1, self.img_size, self.img_size
        )
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, dim, num_classes, img_size, embed_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            self._block(z_dim + embed_size, dim * 16, 4, 1, 0),  # img: 4x4
            self._block(dim * 16, dim * 8, 4, 2, 1),  # img: 8x8
            self._block(dim * 8, dim * 4, 4, 2, 1),  # img: 16x16
            self._block(dim * 4, dim * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                dim * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),  # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )
        self.embed = nn.Embedding(num_classes, embed_size)

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

    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.net(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
