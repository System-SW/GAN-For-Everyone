import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

import hyperparameters as hp
from dataset import Dataset
from model import Discriminator, Generator, initialize_weights
from opt import Template


class EBGAN(Template):
    def __init__(self):
        super().__init__(device=hp.DEVICE, seed=hp.SEED, model_name="EBGAN")

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
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.add_image = lambda tag, image: self.tb.add_image(
            tag,
            torchvision.utils.make_grid(image[: hp.SAMPLE_SIZE], normalize=True),
            global_step=self.itr,
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

                # Train Discriminator
                with torch.cuda.amp.autocast():
                    # FAKE Disc is AutoEncoder output is fake`
                    fake = self.gen(noise)
                    fake_, fake_hidden = self.disc(fake.detach())
                    lossD_fake = self.MSE_loss(fake_, fake.detach())
                    # REAL
                    real_, real_hidden = self.disc(real)
                    lossD_real = self.MSE_loss(real_, real)

                    lossD = lossD_real + torch.clip(
                        hp.MARGIN - lossD_fake, min=0
                    )  # max(a,0)

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

                with torch.no_grad():
                    if batch_idx % hp.LOG_INTERVAL == 0:
                        self.tb.add_scalar("LossD", lossD.item(), global_step=self.itr)
                        self.tb.add_scalar(
                            "LossD/real", lossD_real.item(), global_step=self.itr
                        )
                        self.tb.add_scalar(
                            "LossD/fake", lossD_fake.item(), global_step=self.itr
                        )

                        self.tb.add_scalar("LossG", lossG.item(), global_step=self.itr)

                    if batch_idx % hp.TEST_INTERVAL == 0:
                        pbar.set_description_str(
                            f"Epoch[{epoch+1} / {hp.NUM_EPOCHS}], "
                            f"lossD:{lossD.item(): .2f}, "
                            f"lossG:{lossG.item(): .2f} "
                        )
                        self.test(real)
                        self.add_image("Train/disc_fake_", fake_)
                        self.add_image("Train/disc_real_", real_)
                        self.add_image("Train/disc_real", real)

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
    trainer = EBGAN().train()
    print("Train Done")
