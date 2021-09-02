import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

import hyperparameters as hp
from dataset import DataLoader
from model import Discriminator, Generator, initialize_weights
from opt import Metrics, Template


class EBGAN(Template):
    def __init__(self):
        super().__init__(
            device=hp.DEVICE, seed=hp.SEED, model_name=self.__class__.__name__
        )

        self.dataloader = DataLoader(hp.DATASET)
        self.gen = Generator(hp.NOISE_DIM, hp.IMAGE_CHANNELS, hp.GEN_DIM).to(
            self.device
        )
        self.disc = Discriminator(hp.IMAGE_CHANNELS, hp.DISC_DIM).to(self.device)
        initialize_weights(self.gen)
        initialize_weights(self.disc)

        # optimizer
        self.opt_gen = optim.Adam(
            self.gen.parameters(), lr=hp.LEARNING_RATE, betas=hp.BETAS
        )
        self.opt_disc = optim.Adam(
            self.disc.parameters(), lr=hp.LEARNING_RATE, betas=hp.BETAS
        )

        self.scaler_gen = torch.cuda.amp.GradScaler()
        self.scaler_dis = torch.cuda.amp.GradScaler()
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.add_image = lambda tag, image: self.tb.add_image(
            tag,
            torchvision.utils.make_grid(image[: hp.SAMPLE_SIZE], normalize=True),
            self.itr,
        )

        self.lossD_real = Metrics("TRAIN/lossD_real")
        self.lossD_fake = Metrics("TRAIN/lossD_fake")
        self.lossD = Metrics("TRAIN/lossD")
        self.lossG = Metrics("TRAIN/lossG")

        self.restore_checkpoint(
            hp.RESTORE_CKPT_PATH,
            self.gen,
            self.disc,
            self.opt_gen,
            self.opt_disc,
            hp.LEARNING_RATE,
            self.device,
        )

    def pullaway_loss(self, embeddings):
        norm = torch.norm(embeddings, 1)
        normalized_embeddings = embeddings / norm
        similarity = (
            torch.matmul(normalized_embeddings, normalized_embeddings.transpose(1, 0))
            ** 2
        )
        batch_size = embeddings.size()[0]
        pt_loss = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
        return pt_loss

    def train(self):
        loader, _ = self.dataloader.create_dataloader(hp.BATCH_SIZE, hp.SAMPLE_SIZE)
        FIXED_NOISE = torch.randn(
            hp.SAMPLE_SIZE, hp.NOISE_DIM, 1, 1, device=self.device
        )
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
                    # FAKE Disc is AutoEncoder output is fake`
                    fake = self.gen(noise)
                    fake_, fake_hidden = self.disc(fake.detach())
                    lossD_fake = self.MSE_loss(fake_, fake.detach())
                    # REAL
                    real_, real_hidden = self.disc(real)
                    lossD_real = self.MSE_loss(real_, real)
                    lossD = lossD_real + torch.clip(hp.MARGIN - lossD_fake, min=0)

                self.disc.zero_grad()
                self.scaler_dis.scale(lossD).backward()
                self.scaler_dis.step(self.opt_disc)
                self.scaler_dis.update()

                # Train Generator
                with torch.cuda.amp.autocast():
                    output_, output_hidden = self.disc(fake)
                    output_hidden = output_hidden.view(batch_size, -1)
                    lossG = (
                        self.MSE_loss(output_, fake.detach())
                        + self.pullaway_loss(output_hidden) * hp.PT_WEIGHT
                    )

                self.gen.zero_grad()
                self.scaler_gen.scale(lossG).backward()
                self.scaler_gen.step(self.opt_gen)
                self.scaler_gen.update()

                self.lossD_real.update_state(lossD_real)
                self.lossD_fake.update_state(lossD_fake)
                self.lossD.update_state(lossD)
                self.lossG.update_state(lossG)

                if batch_idx % hp.LOG_INTERVAL == 0:
                    lossD_real = self.logging_scaler(self.lossD_real)
                    lossD_fake = self.logging_scaler(self.lossD_fake)
                    lossD = self.logging_scaler(self.lossD)
                    lossG = self.logging_scaler(self.lossG)
                    pbar.set_description_str(
                        f"Epoch[{epoch+1} / {hp.NUM_EPOCHS}], "
                        f"lossD_real:{lossD_real: .2f}, "
                        f"lossD_fake:{lossD_fake: .2f}, "
                        f"lossD:{lossD: .2f}, "
                        f"lossG:{lossG: .2f} "
                    )

                if batch_idx % hp.TEST_INTERVAL == 0:
                    test_image = self.test(
                        self.gen, real[: hp.SAMPLE_SIZE], FIXED_NOISE
                    )
                    self.add_image("Train/disc_fake_", fake_)
                    self.add_image("Train/disc_real_", real_)
                    self.add_image("Train/disc_real", real)
            # logging model parameters
            self.logging_weight_and_gradient("GEN", self.gen, self.itr)
            self.logging_weight_and_gradient("DISC", self.disc, self.itr)

            self.save_image_to_logdir(test_image, epoch + 1)
            self.save_checkpoint(
                self.gen, self.disc, self.opt_gen, self.opt_disc, epoch + 1
            )


if __name__ == "__main__":
    trainer = EBGAN().train()
    print("Train Done")
