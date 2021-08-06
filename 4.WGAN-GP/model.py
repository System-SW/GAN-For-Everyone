import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, channels_image: int, dim: int):
        super(Critic, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(channels_image, dim,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),

            self._block(dim, dim * 2, 4, 2, 1),
            self._block(dim * 2, dim * 4, 4, 2, 1),
            self._block(dim * 4, dim * 8, 4, 2, 1),

            nn.Conv2d(dim * 8, 1, kernel_size=4, stride=2, padding=0)
        )

    def _block(self, in_channels: int, out_channels: int,
               kernel_size: int, stride: int, padding: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, tensor: torch.Tensor):
        return self.main(tensor)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_image, dim):
        super().__init__()
        self.main = nn.Sequential(
            self._block(z_dim, dim * 16, 4, 1, 0),      # 4x4
            self._block(dim * 16, dim * 8, 4, 2, 1),    # 8x8
            self._block(dim * 8, dim * 4, 4, 2, 1),     # 16x16
            self._block(dim * 4, dim * 2, 4, 2, 1),     # 32x32
            nn.ConvTranspose2d(dim * 2, channels_image, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def _block(self, in_channels: int, out_channels: int,
               kernel_size: int, stride: int, padding: int):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                               stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, tensor: torch.Tensor):
        return self.main(tensor)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Critic(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
