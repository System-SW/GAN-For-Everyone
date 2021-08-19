import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_dim, channels_img, dim):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            nn.Conv2d(
                input_dim + channels_img,
                dim,
                4,
                2,
                1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.LeakyReLU(0.2),
            self._block(dim, dim * 2, 4, 2, 1),
            self._block(dim * 2, dim * 4, 4, 2, 1),
            self._block(dim * 4, dim * 8, 4, 1, 1),
            nn.Conv2d(dim * 8, 1, 4, 1, 1, padding_mode="reflect", bias=False),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, y):
        return self.disc(torch.cat([x, y], dim=1))


class Block(nn.Module):
    def __init__(
        self, in_channels, out_channels, down=True, act="relu", use_dropout=False
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
                padding_mode="reflect",
            )
            if down
            else nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self._use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self._use_dropout else x


class Generator(nn.Module):
    def __init__(self, input_dim, channels_img, dim):
        super(Generator, self).__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        self.down_1 = Block(
            dim, dim * 2, down=True, act="leaky", use_dropout=False
        )  # 64
        self.down_2 = Block(
            dim * 2, dim * 4, down=True, act="leaky", use_dropout=False
        )  # 32
        self.down_3 = Block(
            dim * 4, dim * 8, down=True, act="leaky", use_dropout=False
        )  # 16
        self.down_4 = Block(
            dim * 8, dim * 8, down=True, act="leaky", use_dropout=False
        )  # 8
        self.down_5 = Block(
            dim * 8, dim * 8, down=True, act="leaky", use_dropout=False
        )  # 4
        self.down_6 = Block(
            dim * 8, dim * 8, down=True, act="leaky", use_dropout=False
        )  # 2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(dim * 8, dim * 8, 4, 2, 1, padding_mode="reflect"), nn.ReLU()
        )

        self.up_1 = Block(dim * 8, dim * 8, down=False, act="relu", use_dropout=True)
        self.up_2 = Block(
            dim * 8 * 2, dim * 8, down=False, act="relu", use_dropout=True
        )
        self.up_3 = Block(
            dim * 8 * 2, dim * 8, down=False, act="relu", use_dropout=True
        )
        self.up_4 = Block(
            dim * 8 * 2, dim * 8, down=False, act="relu", use_dropout=False
        )
        self.up_5 = Block(
            dim * 8 * 2, dim * 4, down=False, act="relu", use_dropout=False
        )
        self.up_6 = Block(
            dim * 4 * 2, dim * 2, down=False, act="relu", use_dropout=False
        )
        self.up_7 = Block(dim * 2 * 2, dim, down=False, act="relu", use_dropout=False)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=dim * 2,
                out_channels=channels_img,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down_1(d1)
        d3 = self.down_2(d2)
        d4 = self.down_3(d3)
        d5 = self.down_4(d4)
        d6 = self.down_5(d5)
        d7 = self.down_6(d6)
        bottleneck = self.bottleneck(d7)

        up1 = self.up_1(bottleneck)
        up2 = self.up_2(torch.cat([up1, d7], 1))
        up3 = self.up_3(torch.cat([up2, d6], 1))
        up4 = self.up_4(torch.cat([up3, d5], 1))
        up5 = self.up_5(torch.cat([up4, d4], 1))
        up6 = self.up_6(torch.cat([up5, d3], 1))
        up7 = self.up_7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3, 256, 256)
    D = Discriminator(3, 64)
    G = Generator(3, 64)
    preds = G(x)
    print("G: ", preds.shape)
    preds = D(preds, y)
    print("D: ", preds.shape)
