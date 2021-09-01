import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

import hyperparameters as hp
from dataset import DataLoader
from model import Discriminator, Generator, initialize_weights
from opt import Template, Metrics


class DCGAN(Template):
    def __init__(self):
        super().__init__(
            device=hp.DEVICE,
            seed=hp.SEED,
            model_name=self.__class__.__name__,
        )

        self.dataloader = DataLoader(hp.DATASET)
        self.gen = Generator(hp.NOISE_DIM, hp.IMAGE_CHANNELS, hp.GEN_DIM).to(
            self.device
        )
        self.disc = Discriminator(hp.IMAGE_CHANNELS, hp.DISC_DIM).to(self.device)
        initialize_weights(self.gen)
        initialize_weights(self.disc)

        self.FIXED_NOISE = torch.randn(
            hp.SAMPLE_SIZE, hp.NOISE_DIM, 1, 1, device=self.device
        )
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

        self.real_prob = Metrics("TRAIN/Real Prob")
        self.fake_prob = Metrics("TRAIN/Fake Prob")
        self.lossD_real = Metrics("TRAIN/lossD_real")
        self.lossD_fake = Metrics("TRAIN/lossD_fake")
        self.lossD = Metrics("TRAIN/lossD")
        self.lossG = Metrics("TRAIN/lossG")

    def train(self):
        loader, _ = self.dataloader.create_dataloader(hp.BATCH_SIZE, hp.SAMPLE_SIZE)
        self.gen.train()
        self.disc.train()

        # Training
        for epoch in range(hp.NUM_EPOCHS):
            pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
            for batch_idx, (real, _) in pbar:
                self.itr += 1
                real = real.to(self.device)
                batch_size = real.shape[0]
                noise = torch.randn(batch_size, hp.NOISE_DIM, 1, 1, device=self.device)

                # Train Discriminator
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

                # Logging
                self.real_prob.update_state(self.gan_prob(disc_real))
                self.fake_prob.update_state(self.gan_prob(disc_fake))
                self.lossD_real.update_state(lossD_real)
                self.lossD_fake.update_state(lossD_fake)
                self.lossD.update_state(lossD)
                self.lossG.update_state(lossG)

                if batch_idx % hp.LOG_INTERVAL == 0:
                    real_prob = self.logging_scaler(self.real_prob)
                    fake_prob = self.logging_scaler(self.fake_prob)
                    lossD_real = self.logging_scaler(self.lossD_real)
                    lossD_fake = self.logging_scaler(self.lossD_fake)
                    lossD = self.logging_scaler(self.lossD)
                    lossG = self.logging_scaler(self.lossG)
                    pbar.set_description_str(
                        f"Epoch[{epoch+1} / {hp.NUM_EPOCHS}], "
                        f"real:{real_prob: .2f}, "
                        f"fake:{fake_prob: .2f}, "
                        f"lossD:{lossD: .2f}, "
                        f"lossG:{lossG: .2f} "
                    )
                if batch_idx % hp.TEST_INTERVAL == 0:
                    self.test(self.gen, real[: hp.SAMPLE_SIZE], self.FIXED_NOISE)

            self.logging_weight_and_gradient("GEN", self.gen, self.itr)
            self.logging_weight_and_gradient("DISC", self.disc, self.itr)

            test_image = self.test(self.gen, real[: hp.SAMPLE_SIZE], self.FIXED_NOISE)
            self.save_image_to_logdir(test_image, epoch + 1)
            self.save_checkpoint(
                self.gen, self.disc, self.opt_gen, self.opt_disc, epoch + 1
            )


if __name__ == "__main__":
    trainer = DCGAN().train()
    print("Train Done")
