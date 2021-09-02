import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

import hyperparameters as hp
from dataset import DataLoader
from model import Discriminator, Generator
from losses import VGGPerceptualLoss
from opt import Metrics, Template


class SRGAN(Template):
    def __init__(self):
        super().__init__(
            device=hp.DEVICE,
            seed=hp.SEED,
            model_name=self.__class__.__name__,
        )

        self.dataset = DataLoader(hp.DATASET)
        self.gen = Generator(
            hp.INPUT_CHANNELS, hp.OUTPUT_CHANNELS, hp.GEN_DIM, hp.NUM_RESIDUALS
        ).to(self.device)
        self.disc = Discriminator(hp.OUTPUT_CHANNELS, hp.DISC_DIM).to(self.device)

        # optimizer
        self.opt_gen = optim.Adam(
            self.gen.parameters(), lr=hp.LEARNING_RATE, betas=hp.BETAS
        )
        self.opt_disc = optim.Adam(
            self.disc.parameters(), lr=hp.LEARNING_RATE, betas=hp.BETAS
        )

        self.bce_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.vgg_loss = VGGPerceptualLoss().to(self.device)

        self.restore_checkpoint(
            hp.RESTORE_CKPT_PATH,
            self.gen,
            self.disc,
            self.opt_gen,
            self.opt_disc,
            hp.LEARNING_RATE,
            self.device,
        )
        self.real_prob = Metrics("TRAIN/Real Prob")
        self.fake_prob = Metrics("TRAIN/Fake Prob")
        self.lossD_real = Metrics("TRAIN/lossD_real")
        self.lossD_fake = Metrics("TRAIN/lossD_fake")
        self.lossD = Metrics("TRAIN/lossD")
        self.lossG_vgg = Metrics("TRAIN/lossG_vgg")

    def train(self):
        loader, _ = self.dataset.create_dataloader(hp.BATCH_SIZE, hp.SAMPLE_SIZE)
        self.gen.train()
        self.disc.train()

        for lr_image, hr_image in loader:
            FIXED_DATA = {
                "lr_image": lr_image[: hp.SAMPLE_SIZE].to(self.device),
                "hr_image": hr_image[: hp.SAMPLE_SIZE].to(self.device),
            }
            break

        # Training
        for epoch in range(hp.NUM_EPOCHS):
            pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
            for batch_idx, (lr_image, hr_image) in pbar:
                self.itr += 1
                lr_image = lr_image.to(self.device)
                hr_image = hr_image.to(self.device)

                lr_image_fake = self.gen(lr_image)
                disc_real = self.disc(hr_image)
                lossD_real = self.bce_loss(
                    disc_real,
                    torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real),
                )
                disc_fake = self.disc(lr_image_fake.detach())
                lossD_fake = self.bce_loss(disc_fake, torch.zeros_like(disc_fake))
                lossD = lossD_fake + lossD_real

                self.disc.zero_grad()
                lossD.backward()
                self.opt_disc.step()

                # Train Generator
                output = self.disc(lr_image_fake)
                lossG_adv = self.bce_loss(output, torch.ones_like(output))
                lossG_vgg = self.vgg_loss(lr_image_fake, hr_image)
                lossG = lossG_adv * hp.LAMBDA_ADV + lossG_vgg * hp.LAMBDA_VGG

                self.gen.zero_grad()

                lossG.backward()
                self.opt_gen.step()

                # Logging
                self.real_prob.update_state(self.gan_prob(disc_real))
                self.fake_prob.update_state(self.gan_prob(disc_fake))
                self.lossD_real.update_state(lossD_real)
                self.lossD_fake.update_state(lossD_fake)
                self.lossD.update_state(lossD)
                self.lossG_vgg.update_state(lossG_vgg)

                with torch.no_grad():
                    if batch_idx % hp.LOG_INTERVAL == 0:
                        real_prob = self.logging_scaler(self.real_prob)
                        fake_prob = self.logging_scaler(self.fake_prob)
                        lossD_real = self.logging_scaler(self.lossD_real)
                        lossD_fake = self.logging_scaler(self.lossD_fake)
                        lossD = self.logging_scaler(self.lossD)
                        lossG_vgg = self.logging_scaler(self.lossG_vgg)

                        pbar.set_description_str(
                            f"Epoch[{epoch+1} / {hp.NUM_EPOCHS}], "
                            f"real:{real_prob: .2f}, "
                            f"fake:{fake_prob: .2f}, "
                            f"lossD:{lossD: .2f}, "
                            f"lossG_vgg:{lossG_vgg: .2f} "
                        )

                    if batch_idx % hp.TEST_INTERVAL == 0:
                        self.test(FIXED_DATA["lr_image"], FIXED_DATA["hr_image"])

            # Epoch process
            self.logging_weight_and_gradient("GEN", self.gen, self.itr)
            self.logging_weight_and_gradient("DISC", self.disc, self.itr)

            test_image = self.test(FIXED_DATA["lr_image"], FIXED_DATA["hr_image"])
            self.save_image_to_logdir(test_image, epoch + 1)

            self.save_checkpoint(
                self.gen, self.disc, self.opt_gen, self.opt_disc, epoch + 1
            )

    def test(self, lr_image, hr_image):
        return super().test(self.gen, hr_image, lr_image)


if __name__ == "__main__":
    trainer = SRGAN()
    trainer.train()
    print("Train Done")
