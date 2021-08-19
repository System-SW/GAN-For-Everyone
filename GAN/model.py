import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, z_dim: int, img_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)
