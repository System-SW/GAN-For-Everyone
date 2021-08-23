import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

import hyperparameters as hp
from dataset import Dataset
from model import Discriminator, Generator
from losses import VGGLoss
from opt import Template


class SRGAN(Template):
    def __init__(self):
        super().__init__(device=hp.DEVICE, seed=hp.SEED, model_name="SRGAN")

        self.dataset = Dataset(hp.DATASET)
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
        self.vgg_loss = VGGLoss().to(self.device)

    def train(self):
        loader = self.dataset.create_dataloader(hp.BATCH_SIZE)
        self.gen.train()
        self.disc.train()

        for lr_image, hr_image in loader:
            FIX_DATA = {
                "lr_image": lr_image[: hp.SAMPLE_SIZE].to(self.device),
                "hr_image": hr_image[: hp.SAMPLE_SIZE].to(self.device),
            }
            break

        # Training
        for epoch in range(hp.NUM_EPOCHS):
            pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
            for batch_idx, (lr_image, hr_image) in pbar:
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

                with torch.no_grad():
                    if batch_idx % hp.LOG_INTERVAL == 0:
                        log_disc_real = self.gan_prob(disc_real).item()
                        log_disc_fake = self.gan_prob(disc_fake).item()
                        log_vgg_loss = lossG_vgg.item()
                        lossD = lossD.item()
                        lossG = lossG.item()

                        self.tb.add_scalar(
                            "TRAIN/Real Prob",
                            log_disc_real,
                            global_step=self.itr,
                        )
                        self.tb.add_scalar(
                            "TRAIN/Fake Prob",
                            log_disc_fake,
                            global_step=self.itr,
                        )
                        self.tb.add_scalar(
                            "TRAIN/VGG Loss",
                            log_vgg_loss,
                            global_step=self.itr,
                        )
                        self.tb.add_scalar("TRAIN/LossD", lossD, global_step=self.itr)
                        self.tb.add_scalar("TRAIN/LossG", lossG, global_step=self.itr)

                    if batch_idx % hp.TEST_INTERVAL == 0:
                        pbar.set_description_str(
                            f"Epoch[{epoch+1} / {hp.NUM_EPOCHS}], "
                            f"real:{log_disc_real: .2f}, "
                            f"fake:{log_disc_fake: .2f}, "
                            f"lossD:{lossD: .2f}, "
                            f"lossG_VGG:{log_vgg_loss: .2f} "
                            f"lossG:{lossG: .2f} "
                        )
                        self.test(FIX_DATA["lr_image"], FIX_DATA["hr_image"])
                self.itr += 1
            self.save_checkpoint(
                self.gen, self.disc, self.opt_gen, self.opt_disc, epoch
            )

    def test(self, lr_image, hr_image):
        self.gen.eval()
        fake = self.gen(lr_image)
        img_grid_inputs = torchvision.utils.make_grid(lr_image, normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
        img_grid_real = torchvision.utils.make_grid(hr_image, normalize=True)
        self.tb.add_image("Input Images", img_grid_inputs, global_step=self.itr)
        self.tb.add_image("Fake Images", img_grid_fake, global_step=self.itr)
        self.tb.add_image("Real Images", img_grid_real, global_step=self.itr)
        self.gen.train()


if __name__ == "__main__":
    trainer = SRGAN()
    trainer.train()
    print("Train Done")
