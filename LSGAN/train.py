import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

import hyperparameters as hp
from dataset import Dataset
from model import Discriminator, Generator, initialize_weights
from opt import Template


class LSGAN(Template):
    def __init__(self):
        super().__init__(
            device=hp.DEVICE, seed=hp.SEED, model_name='LSGAN')

        self.dataset = Dataset(hp.DATASET)
        self.gen = Generator(hp.NOISE_DIM, hp.IMAGE_CHANNELS,
                             hp.GEN_DIM).to(self.device)
        self.disc = Discriminator(hp.IMAGE_CHANNELS,
                                  hp.DISC_DIM).to(self.device)
        initialize_weights(self.gen)
        initialize_weights(self.disc)

        self.FIXED_NOISE = torch.randn(32, hp.NOISE_DIM, 1, 1,
                                       device=self.device)
        # optimizer
        self.opt_gen = optim.Adam(
            self.gen.parameters(), lr=hp.LEARNING_RATE, betas=hp.BETAS)
        self.opt_disc = optim.Adam(
            self.disc.parameters(), lr=hp.LEARNING_RATE, betas=hp.BETAS)

        self.scaler_gen = torch.cuda.amp.GradScaler()
        self.scaler_disc = torch.cuda.amp.GradScaler()
        self.criterion = lambda x, y: (
            (torch.mean(x) - torch.mean(y)) ** 2
        )

    def train(self):
        loader = self.dataset.create_dataloader(hp.BATCH_SIZE)
        self.gen.train()
        self.disc.train()

        # Training
        for epoch in range(hp.NUM_EPOCHS):
            pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
            for batch_idx, (real, _) in pbar:
                real = real.to(self.device)
                batch_size = real.shape[0]
                noise = torch.randn(
                    batch_size, hp.NOISE_DIM, 1, 1, device=self.device)

                # Train Critic
                with torch.cuda.amp.autocast():
                    fake = self.gen(noise)
                    disc_fake = self.disc(fake.detach()).reshape(-1)
                    disc_real = self.disc(real).reshape(-1)
                    loss_real = self.criterion(
                        x=disc_real, y=torch.ones_like(disc_real))
                    loss_fake = self.criterion(
                        x=disc_fake, y=torch.zeros_like(disc_fake))
                    lossD = (loss_fake + loss_real) / 2

                self.disc.zero_grad()
                self.scaler_disc.scale(lossD).backward()
                self.scaler_disc.step(self.opt_disc)
                self.scaler_disc.update()

                # Train Generator
                with torch.cuda.amp.autocast():
                    output = self.disc(fake).reshape(-1)
                    lossG = self.criterion(
                        x=output, y=torch.ones_like(output)) / 2

                self.gen.zero_grad()
                self.scaler_gen.scale(lossG).backward()
                self.scaler_gen.step(self.opt_gen)
                self.scaler_gen.update()

                with torch.no_grad():
                    if batch_idx % hp.LOG_INTERVAL == 0:
                        self.tb.add_scalar(
                            'TRAIN/Discriminator loss', lossD, global_step=self.itr)
                        self.tb.add_scalar(
                            'TRAIN/Generator loss', lossG, global_step=self.itr)

                    if batch_idx % hp.TEST_INTERVAL == 0:
                        pbar.set_description_str(
                            f'Epoch[{epoch+1} / {hp.NUM_EPOCHS}], '
                            f'lossD:{lossD.item(): .2f}, '
                            f'lossG:{lossG.item(): .2f}, ')
                        self.test(real)
                self.itr += 1

    def test(self, real):
        self.gen.eval()
        fake = self.gen(self.FIXED_NOISE)
        img_grid_fake = torchvision.utils.make_grid(
            fake, normalize=True)
        img_grid_real = torchvision.utils.make_grid(
            real[:32], normalize=True)
        self.tb.add_image(
            "Fake Images", img_grid_fake,
            global_step=self.itr
        )
        self.tb.add_image(
            "Real Images", img_grid_real,
            global_step=self.itr
        )
        self.gen.train()


if __name__ == '__main__':
    trainer = LSGAN().train()
    print('Train Done')
