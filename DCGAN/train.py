import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

import hyperparameters as hp
from dataset import Dataset
from model import Discriminator, Generator, initialize_weights
from opt import Template


class DCGAN(Template):
    def __init__(self):
        super().__init__(device=hp.DEVICE, seed=hp.SEED, model_name="DCGAN")

        self.dataset = Dataset(hp.DATASET)
        self.gen = Generator(hp.NOISE_DIM, hp.IMAGE_CHANNELS, hp.GEN_DIM).to(
            self.device
        )
        self.disc = Discriminator(hp.IMAGE_CHANNELS, hp.DISC_DIM).to(self.device)
        initialize_weights(self.gen)
        initialize_weights(self.disc)

        self.FIXED_NOISE = torch.randn(32, hp.NOISE_DIM, 1, 1, device=self.device)
        # optimizer
        self.opt_gen = optim.Adam(
            self.gen.parameters(), lr=hp.LEARNING_RATE, betas=hp.BETAS
        )
        self.opt_disc = optim.Adam(
            self.disc.parameters(), lr=hp.LEARNING_RATE, betas=hp.BETAS
        )

        self.scaler_gen = torch.cuda.amp.GradScaler()
        self.scaler_dis = torch.cuda.amp.GradScaler()
        self.bce_loss = nn.BCEWithLogitsLoss()

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
                noise = torch.randn(batch_size, hp.NOISE_DIM, 1, 1, device=self.device)

                with torch.cuda.amp.autocast():
                    fake = self.gen(noise)
                    disc_real = self.disc(real).reshape(-1)
                    lossD_real = self.bce_loss(disc_real, torch.ones_like(disc_real))
                    disc_fake = self.disc(fake.detach()).reshape(-1)
                    lossD_fake = self.bce_loss(disc_fake, torch.zeros_like(disc_fake))
                    lossD = (lossD_fake + lossD_real) / 2

                self.disc.zero_grad()
                self.scaler_dis.scale(lossD).backward()
                self.scaler_dis.step(self.opt_disc)
                self.scaler_dis.update()

                # Train Generator
                with torch.cuda.amp.autocast():
                    output = self.disc(fake).reshape(-1)
                    lossG = self.bce_loss(output, torch.ones_like(output))

                self.gen.zero_grad()
                self.scaler_gen.scale(lossG).backward()
                self.scaler_gen.step(self.opt_gen)
                self.scaler_gen.update()

                with torch.no_grad():
                    if batch_idx % hp.LOG_INTERVAL == 0:
                        self.tb.add_scalar(
                            "TRAIN/Real Prob",
                            self.gan_prob(disc_real).item(),
                            global_step=self.itr,
                        )
                        self.tb.add_scalar(
                            "TRAIN/Fake Prob",
                            self.gan_prob(disc_fake).item(),
                            global_step=self.itr,
                        )

                    if batch_idx % hp.TEST_INTERVAL == 0:
                        pbar.set_description_str(
                            f"Epoch[{epoch+1} / {hp.NUM_EPOCHS}], "
                            f"real:{self.gan_prob(disc_real).item(): .2f}, "
                            f"fake:{self.gan_prob(disc_fake).item(): .2f}, "
                            f"lossD:{lossD.item(): .2f}, "
                            f"lossG:{lossG.item(): .2f} "
                        )
                        self.test(real)
                self.itr += 1

    def test(self, real):
        self.gen.eval()
        fake = self.gen(self.FIXED_NOISE)
        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
        img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
        self.tb.add_image("Fake Images", img_grid_fake, global_step=self.itr)
        self.tb.add_image("Real Images", img_grid_real, global_step=self.itr)
        self.gen.train()


if __name__ == "__main__":
    trainer = DCGAN().train()
    print("Train Done")
