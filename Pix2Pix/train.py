from typing import overload
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

import hyperparameters as hp
from dataset import DataLoader
from model import Discriminator, Generator
from opt import Metrics, Template


class Pix2Pix(Template):
    def __init__(self):
        super().__init__(
            device=hp.DEVICE,
            seed=hp.SEED,
            model_name=self.__class__.__name__,
        )

        self.dataloader = DataLoader(hp.DATASET)
        self.gen = Generator(hp.INPUT_CHANNELS, hp.IMAGE_CHANNELS, hp.GEN_DIM).to(
            self.device
        )
        self.disc = Discriminator(hp.INPUT_CHANNELS, hp.IMAGE_CHANNELS, hp.DISC_DIM).to(
            self.device
        )

        # Optimizer
        self.opt_gen = optim.Adam(
            self.gen.parameters(), lr=hp.LEARNING_RATE, betas=hp.BETAS
        )
        self.opt_disc = optim.Adam(
            self.disc.parameters(), lr=hp.LEARNING_RATE, betas=hp.BETAS
        )

        self.scaler_gen = torch.cuda.amp.GradScaler()
        self.scaler_dis = torch.cuda.amp.GradScaler()

        self.BCE_LOSS = nn.BCEWithLogitsLoss().to(self.device)
        self.L1_LOSS = nn.L1Loss().to(self.device)

        self.restore_checkpoint(
            ckpt_path=hp.RESTORE_CKPT_PATH,
            gen=self.gen,
            disc=self.disc,
            opt_gen=self.opt_gen,
            opt_disc=self.disc,
            lr=hp.LEARNING_RATE,
        )
        self.real_prob = Metrics("TRAIN/Real Prob")
        self.fake_prob = Metrics("TRAIN/Fake Prob")
        self.lossD_real = Metrics("TRAIN/lossD_real")
        self.lossD_fake = Metrics("TRAIN/lossD_fake")
        self.lossD = Metrics("TRAIN/lossD")
        self.lossG_l1 = Metrics("TRAIN/lossG_l1")

    def train(self):
        train_loader, val_loader = self.dataloader.create_dataloader(
            hp.BATCH_SIZE, hp.SAMPLE_SIZE, hp.NUM_WORKERS
        )
        for inputs, targets in val_loader:
            FIXED_DATA = {
                "inputs": inputs.to(self.device),
                "targets": targets.to(self.device),
            }
            break

        self.gen.train()
        self.disc.train()

        # Training
        for epoch in range(hp.NUM_EPOCHS):
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
            for batch_idx, (inputs, targets) in pbar:
                self.itr += 1
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                with torch.cuda.amp.autocast():
                    fake = self.gen(inputs)
                    disc_real = self.disc(inputs, targets)
                    lossD_real = self.BCE_LOSS(disc_real, torch.ones_like(disc_real))

                    disc_fake = self.disc(inputs, fake.detach())
                    lossD_fake = self.BCE_LOSS(disc_fake, torch.zeros_like(disc_fake))
                    lossD = (lossD_fake + lossD_real) / 2

                self.disc.zero_grad()
                self.scaler_dis.scale(lossD).backward()
                self.scaler_dis.step(self.opt_disc)
                self.scaler_dis.update()

                # Train Generator
                with torch.cuda.amp.autocast():
                    output = self.disc(inputs, fake)
                    lossG_adv = self.BCE_LOSS(output, torch.ones_like(output))
                    lossG_l1 = self.L1_LOSS(fake, targets)
                    lossG = lossG_adv + lossG_l1 * hp.LAMBDA_L1

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
                self.lossG_l1.update_state(lossG_l1)

                if batch_idx % hp.LOG_INTERVAL == 0:
                    real_prob = self.logging_scaler(self.real_prob)
                    fake_prob = self.logging_scaler(self.fake_prob)
                    lossG_l1 = self.logging_scaler(self.lossG_l1)
                    self.logging_scaler(self.lossD_real)
                    self.logging_scaler(self.lossD_fake)
                    self.logging_scaler(self.lossD)

                    pbar.set_description_str(
                        f"Epoch[{epoch+1} / {hp.NUM_EPOCHS}], "
                        f"real:{real_prob: .2f}, "
                        f"fake:{fake_prob: .2f}, "
                        f"l1_loss:{lossG_l1: .2f} "
                    )

                if batch_idx % hp.TEST_INTERVAL == 0:
                    test_image = self.test(FIXED_DATA["inputs"], FIXED_DATA["targets"])
                    self.save_image_to_logdir(test_image, self.itr)

            # Epoch process
            self.logging_weight_and_gradient("GEN", self.gen, self.itr)
            self.logging_weight_and_gradient("DISC", self.disc, self.itr)

            test_image = self.test(FIXED_DATA["inputs"], FIXED_DATA["targets"])
            super().save_image_to_logdir(test_image, epoch)
            self.save_checkpoint(
                gen=self.gen,
                disc=self.disc,
                opt_gen=self.opt_gen,
                opt_disc=self.opt_disc,
                epoch=epoch,
            )

    @torch.no_grad()
    def test(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self.gen.eval()
        fake = self.gen(inputs)
        nrow = fake.shape[0]
        img_grid_fake = torchvision.utils.make_grid(fake, nrow, normalize=True)
        img_grid_real = torchvision.utils.make_grid(targets, nrow, normalize=True)
        img_grid_input = torchvision.utils.make_grid(inputs, nrow, normalize=True)
        test_image = torch.cat([img_grid_fake, img_grid_real, img_grid_input], dim=1)
        self.tb.add_image("TEST/images", test_image, global_step=self.itr)
        self.gen.train()
        return test_image

    @torch.no_grad()
    def save_image_to_logdir(
        self, image: torch.Tensor, gs: int, image_format="JPEG"
    ) -> None:
        fp = os.path.join(
            self.image_dir, f"{self.__class__.__name__}_GS:{str(gs).zfill(10)}.jpg"
        )
        torchvision.utils.save_image(image, fp, image_format)


if __name__ == "__main__":
    trainer = Pix2Pix().train()
    print("Train Done")
