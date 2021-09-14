from math import log2

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

import hyperparameters as hp
from dataset import DataLoader
from model import Discriminator, Generator
from opt import Metrics, Template


class ProGAN(Template):
    def __init__(self):
        super().__init__(
            device=hp.DEVICE,
            seed=hp.SEED,
            model_name=self.__class__.__name__,
        )

        self.gen = Generator(
            hp.NOISE_DIM,
            hp.GEN_DIM,
            hp.IMAGE_CHANNELS,
        ).to(self.device)
        self.disc = Discriminator(
            hp.NOISE_DIM,
            hp.DISC_DIM,
            hp.IMAGE_CHANNELS,
        ).to(self.device)

        # optimizer
        self.opt_gen = optim.Adam(
            self.gen.parameters(), lr=hp.LEARNING_RATE, betas=hp.BETAS
        )
        self.opt_disc = optim.Adam(
            self.disc.parameters(), lr=hp.LEARNING_RATE, betas=hp.BETAS
        )

        self.scaler_gen = torch.cuda.amp.GradScaler()
        self.scaler_dis = torch.cuda.amp.GradScaler()

        self.lossD = Metrics("TRAIN/lossD")
        self.lossG = Metrics("TRAIN/lossG")
        self.lossD_gp = Metrics("TRAIN/lossD_gp")
        self.lossD_wgan = Metrics("TRAIN/lossD_wgan")

    @staticmethod
    def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
        BATCH_SIZE, C, H, W = real.shape
        beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
        interpolated_images = real * beta + fake.detach() * (1 - beta)
        interpolated_images.requires_grad_(True)

        # Calculate critic scores
        mixed_scores = critic(interpolated_images, alpha, train_step)

        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty

    def train(self):
        self.gen.train()
        self.disc.train()
        step = int(log2(hp.START_TRAIN_AT_IMG_SIZE / 4))
        FIXED_NOISE = torch.randn(
            hp.SAMPLE_SIZE, hp.NOISE_DIM, 1, 1, device=self.device
        )

        # Training
        for epochs in hp.PROGRESSIVE_EPOCHS:
            alpha = 1e-5
            hp.DATASET.IMAGE_SIZE = 4 * 2 ** step
            loader, _ = DataLoader(hp.DATASET).create_dataloader(
                hp.BATCH_SIZES[step],
                hp.SAMPLE_SIZE,
            )

            for epoch in range(epochs):
                pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
                for batch_idx, (real, _) in pbar:
                    self.itr += 1
                    real = real.to(self.device)
                    batch_size = real.shape[0]
                    noise = torch.randn(
                        batch_size, hp.NOISE_DIM, 1, 1, device=self.device
                    )

                    # Train Discriminator
                    with torch.cuda.amp.autocast():
                        fake = self.gen(noise, alpha, step)
                        disc_real = self.disc(real, alpha, step)
                        disc_fake = self.disc(fake.detach(), alpha, step)
                        lossD_wgan = torch.mean(disc_real) - torch.mean(disc_fake)
                        lossD_gp = self.gradient_penalty(
                            self.disc, real, fake, alpha, step, device=self.device
                        )
                        lossD = (
                            -lossD_wgan
                            + lossD_gp * hp.LAMBDA_GP
                            + (0.001) * torch.mean(real ** 2)
                        )

                    self.opt_disc.zero_grad()
                    self.scaler_dis.scale(lossD).backward()
                    self.scaler_dis.step(self.opt_disc)
                    self.scaler_dis.update()

                    # Train Generator
                    with torch.cuda.amp.autocast():
                        disc_output = self.disc(fake, alpha, step)
                        lossG = -torch.mean(disc_output)

                    self.opt_gen.zero_grad()
                    self.scaler_gen.scale(lossG).backward()
                    self.scaler_gen.step(self.opt_gen)
                    self.scaler_gen.update()

                    self.lossD.update_state(lossD)
                    self.lossG.update_state(lossG)
                    self.lossD_gp.update_state(lossD_gp)
                    self.lossD_wgan.update_state(lossD_wgan)

                    if batch_idx % hp.LOG_INTERVAL == 0:
                        lossD = self.logging_scaler(self.lossD)
                        lossG = self.logging_scaler(self.lossG)
                        lossD_gp = self.logging_scaler(self.lossD_gp)
                        lossD_wgan = self.logging_scaler(self.lossD_wgan)
                        pbar.set_description_str(
                            f"Step[{step}], "
                            f"Epoch[{epoch+1} / {epochs}], "
                            f"lossD:{lossD: .2f}, "
                            f"lossG:{lossG: .2f}, "
                            f"wgan:{lossD_wgan: .4f}, "
                            f"gp:{lossD_gp: .4f}, "
                        )

                    if batch_idx % hp.TEST_INTERVAL == 0:
                        test_image = self.test(real, FIXED_NOISE, alpha, step)

                    # Batch END
                    alpha += batch_size / (
                        (hp.PROGRESSIVE_EPOCHS[step] * 0.5) * len(loader)
                    )
                    alpha = min(alpha, 1)

                # Epoch process
                test_image = self.test(real, FIXED_NOISE, alpha, step)
                self.save_image_to_logdir(test_image, epoch + 1)

            # Progressive process
            self.save_checkpoint(
                self.gen, self.disc, self.opt_gen, self.opt_disc, epoch + 1
            )
            step += 1

    @torch.no_grad()
    def test(self, real, noise, alpha, step):
        self.gen.train(False)
        fake = self.gen(noise, alpha, step)
        nrow = int(noise.shape[0] ** 0.5)
        img_grid_fake = torchvision.utils.make_grid(fake, nrow, normalize=True)
        img_grid_real = torchvision.utils.make_grid(real, nrow, normalize=True)
        self.tb.add_image("Fake Images", img_grid_fake, global_step=self.itr)
        self.tb.add_image("Real Images", img_grid_real, global_step=self.itr)
        self.gen.train(True)
        return img_grid_fake


if __name__ == "__main__":
    trainer = ProGAN().train()
    print("Train Done")
