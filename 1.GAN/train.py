import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

import hyperparameters as hp
from dataset import Dataset
from model import Discriminator, Generator
from opt import Template


class GAN(Template):
    def __init__(self):
        super().__init__(hp, "GAN")
        self.disc = Discriminator(hp.IMAGE_DIM).to(hp.DEVICE)
        self.gen = Generator(hp.Z_DIM, hp.IMAGE_DIM).to(hp.DEVICE)

        self.dataset = Dataset(hp.DATASET_DIR)
        self.fixed_noise = torch.randn((hp.BATCH_SIZE, hp.Z_DIM)).to(hp.DEVICE)

        # optimizer
        self.opt_disc = optim.Adam(self.disc.parameters(), lr=hp.LR)
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=hp.LR)
        self.scaler_dis = torch.cuda.amp.GradScaler()
        self.scaler_gen = torch.cuda.amp.GradScaler()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def train(self):
        loader = self.dataset.create_dataloader(
            hp.BATCH_SIZE)
        self.gen.train()
        self.disc.train()

        # Training
        for epoch in range(hp.NUM_EPOCHS):
            pbar = tqdm(enumerate(loader), total=len(loader))
            for batch_idx, (real, _) in pbar:
                real = real.view(-1, 784).to(hp.DEVICE)
                batch_size = real.shape[0]
                noise = torch.randn(batch_size, hp.Z_DIM).to(hp.DEVICE)

                # Train Discriminator
                with torch.cuda.amp.autocast():
                    fake = self.gen(noise)
                    disc_real = self.disc(real).view(-1)
                    lossD_real = self.bce_loss(
                        disc_real, torch.ones_like(disc_real))
                    disc_fake = self.disc(fake.detach()).view(-1)
                    lossD_fake = self.bce_loss(
                        disc_fake, torch.zeros_like(disc_fake))
                    lossD = (lossD_real + lossD_fake) / 2

                self.disc.zero_grad()
                self.scaler_dis.scale(lossD).backward()
                self.scaler_dis.step(self.opt_disc)
                self.scaler_dis.update()

                # Train Generator
                with torch.cuda.amp.autocast():
                    output = self.disc(fake).view(-1)
                    lossG = self.bce_loss(output, torch.ones_like(output))

                self.gen.zero_grad()
                self.scaler_gen.scale(lossG).backward()
                self.scaler_gen.step(self.opt_gen)
                self.scaler_gen.update()

                with torch.no_grad():
                    if batch_idx % hp.LOG_INTERVAL == 0:
                        self.tb.add_scalar(
                            'Real Prob', self.gan_prob(disc_real).item(),
                            global_step=self.itr)
                        self.tb.add_scalar(
                            'Fake Prob', self.gan_prob(disc_fake).item(),
                            global_step=self.itr)

                    if batch_idx == 0:
                        pbar.set_description_str(
                            f'Epoch[{epoch} / {hp.NUM_EPOCHS}], '
                            f'real:{self.gan_prob(disc_real).item(): .2f}, '
                            f'fake:{self.gan_prob(disc_fake).item(): .2f}, '
                            f'lossD:{lossD.item(): .2f}, '
                            f'lossG:{lossG.item(): .2f} '
                        )
                        self.test(real)
                self.itr += 1

    def test(self, real):
        fake = self.gen(
            self.fixed_noise).reshape(-1, 1, 28, 28)
        data = real.reshape(-1, 1, 28, 28)

        img_grid_fake = torchvision.utils.make_grid(
            fake, normalize=True)
        img_grid_real = torchvision.utils.make_grid(
            data, normalize=True)

        self.tb.add_image(
            "Fake Images", img_grid_fake,
            global_step=self.itr
        )
        self.tb.add_image(
            "Real Images", img_grid_real,
            global_step=self.itr
        )


if __name__ == '__main__':
    trainer = GAN().train()