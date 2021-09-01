import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

import hyperparameters as hp
from dataset import DataLoader
from model import Discriminator, Generator
from opt import Template, Metrics


class GAN(Template):
    def __init__(self):
        super().__init__(
            device=hp.DEVICE,
            seed=hp.SEED,
            model_name=self.__class__.__name__,
        )

        self.disc = Discriminator(hp.IMAGE_SIZE ** 2).to(hp.DEVICE)
        self.gen = Generator(hp.NOISE_DIM, hp.IMAGE_SIZE ** 2).to(hp.DEVICE)

        self.dataloader = DataLoader(hp.DATASET)

        # optimizer
        self.opt_disc = optim.Adam(self.disc.parameters(), lr=hp.LR)
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=hp.LR)
        self.scaler_dis = torch.cuda.amp.GradScaler()
        self.scaler_gen = torch.cuda.amp.GradScaler()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.real_prob = Metrics("TRAIN/Real Prob")
        self.fake_prob = Metrics("TRAIN/Fake Prob")
        self.lossD_real = Metrics("TRAIN/lossD_real")
        self.lossD_fake = Metrics("TRAIN/lossD_fake")
        self.lossD = Metrics("TRAIN/lossD")
        self.lossG = Metrics("TRAIN/lossG")

    def train(self):
        loader, _ = self.dataloader.create_dataloader(hp.BATCH_SIZE, hp.SAMPLE_SIZE)
        FIXED_NOISE = torch.randn([hp.SAMPLE_SIZE, hp.NOISE_DIM], device=self.device)
        self.gen.train()
        self.disc.train()

        # Training
        for epoch in range(hp.NUM_EPOCHS):
            pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
            for batch_idx, (real, _) in pbar:
                self.itr += 1
                batch_size = real.shape[0]
                real = real.view(batch_size, -1).to(hp.DEVICE)
                noise = torch.randn([batch_size, hp.NOISE_DIM], device=self.device)

                # Train Discriminator
                with torch.cuda.amp.autocast():
                    fake = self.gen(noise)
                    disc_real = self.disc(real).view(-1)
                    lossD_real = self.bce_loss(disc_real, torch.ones_like(disc_real))
                    disc_fake = self.disc(fake.detach()).view(-1)
                    lossD_fake = self.bce_loss(disc_fake, torch.zeros_like(disc_fake))
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

                # Logging
                self.real_prob.update_state(self.gan_prob(disc_real))
                self.fake_prob.update_state(self.gan_prob(disc_fake))
                self.lossD.update_state(lossD)
                self.lossG.update_state(lossG)

                if batch_idx % hp.LOG_INTERVAL == 0:
                    real_prob = self.logging_scaler(self.real_prob)
                    fake_prob = self.logging_scaler(self.fake_prob)
                    lossD = self.logging_scaler(self.lossD)
                    lossG = self.logging_scaler(self.lossG)
                    pbar.set_description_str(
                        f"Epoch[{epoch+1} / {hp.NUM_EPOCHS}], "
                        f"real:{real_prob: .2f}, "
                        f"fake:{fake_prob: .2f}, "
                        f"lossD:{lossD: .2f}, "
                        f"lossG:{lossG: .2f} "
                    )

                if batch_idx == 0:
                    test_image = self.test(real[: hp.SAMPLE_SIZE], FIXED_NOISE)

            self.logging_weight_and_gradient("GEN", self.gen, self.itr)
            self.logging_weight_and_gradient("DISC", self.disc, self.itr)
            self.save_image_to_logdir(test_image, epoch + 1)
            self.save_checkpoint(
                self.gen, self.disc, self.opt_gen, self.opt_disc, epoch + 1
            )

    @torch.no_grad()
    def test(self, real, fixed_noise):
        self.gen.eval()
        fake = self.gen(fixed_noise).reshape(-1, 1, hp.IMAGE_SIZE, hp.IMAGE_SIZE)
        real = real.reshape(-1, 1, hp.IMAGE_SIZE, hp.IMAGE_SIZE)
        nrow = int(fixed_noise.shape[0] ** 0.5)
        img_grid_fake = torchvision.utils.make_grid(fake, nrow, normalize=True)
        img_grid_real = torchvision.utils.make_grid(real, nrow, normalize=True)

        self.tb.add_image("Fake Images", img_grid_fake, global_step=self.itr)
        self.tb.add_image("Real Images", img_grid_real, global_step=self.itr)
        self.gen.train()
        return img_grid_fake


if __name__ == "__main__":
    trainer = GAN().train()
    print("Train Done")
