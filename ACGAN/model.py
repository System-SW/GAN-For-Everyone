import torch
import torch.nn as nn
from torch.nn.modules import batchnorm


class Discriminator(nn.Module):
    def __init__(self, input_image_size, output_channels, dim, num_classes=10):
        super(Discriminator, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(output_channels, dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(dim * 1, dim * 2, 4, 2, 1),
            self._block(dim * 2, dim * 4, 4, 2, 1),
            self._block(dim * 4, dim * 8, 4, 2, 1),
        )
        shape = dim * 8 * (input_image_size // (2 ** 4)) ** 2
        self.d_head = nn.Sequential(nn.Flatten(), nn.Linear(shape, 1))
        self.code_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(shape, num_classes),
            nn.Softmax(),
        )

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

    def forward(self, x):
        x = self.conv_blocks(x)
        return self.d_head(x), self.code_head(x)


class Generator(nn.Module):
    def __init__(self, output_image_size,z_dim, output_channels, dim, num_classes):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(z_dim, dim * 16, 4, 1, 0),  # img: 4x4
            self._block(dim * 16, dim * 8, 4, 2, 1),  # img: 8x8
            self._block(dim * 8, dim * 4, 4, 2, 1),  # img: 16x16
            self._block(dim * 4, dim * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                dim * 2, output_channels, kernel_size=4, stride=2, padding=1
            ),  # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )
        self.embed = nn.Embedding(num_classes, z_dim)
        self.z_dim = z_dim

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
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        embed = self.embed(labels)
        embed = embed.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        return self.net(x * embed)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


if __name__ == "__main__":
    batch_size = 4
    input_channels = 3
    output_channels = 3
    image_size = 64

    z = torch.rand([batch_size, input_channels, image_size, image_size])
    D = Discriminator(output_channels, 64, 10)
    val, label = D(z)
    print("val :", val.shape)
    print("label :", label.shape)
