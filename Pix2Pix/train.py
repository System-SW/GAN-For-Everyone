import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.serialization import save
import torchvision
from tqdm import tqdm

import hyperparameters as hp
from dataset import Dataset
from model import Discriminator, Generator
from opt import Template


class Pix2Pix(Template):
    def __init__(self):
        super().__init__(
            device=hp.DEVICE, seed=hp.SEED, model_name='Pix2Pix')

        self.dataset = Dataset(hp.DATASET)
        self.gen = Generator(hp.INPUT_CHANNELS,
                             hp.IMAGE_CHANNELS,
                             hp.GEN_DIM).to(self.device)

        self.disc = Discriminator(hp.INPUT_CHANNELS,
                                  hp.IMAGE_CHANNELS,
                                  hp.DISC_DIM).to(self.device)

        # Optimizer
        self.opt_gen = optim.Adam(self.gen.parameters(),
                                  lr=hp.LEARNING_RATE,
                                  betas=hp.BETAS)
        self.opt_disc = optim.Adam(self.disc.parameters(),
                                   lr=hp.LEARNING_RATE,
                                   betas=hp.BETAS)

        self.scaler_gen = torch.cuda.amp.GradScaler()
        self.scaler_dis = torch.cuda.amp.GradScaler()

        self.BCE_LOSS = nn.BCEWithLogitsLoss().to(self.device)
        self.L1_LOSS = nn.L1Loss().to(self.device)

        self.restore_checkopint(cktp_path=hp.CKPT_PATH,
                                gen=self.gen,
                                disc=self.disc,
                                opt_gen=self.opt_gen,
                                opt_disc=self.disc,
                                lr=hp.LEARNING_RATE)

    def train(self):
        train_loader, val_loader = self.dataset.create_dataloader(
            hp.BATCH_SIZE, hp.NUM_WORKERS)

        FIX_DATA = None
        for inputs, targets in val_loader:
            FIX_DATA = (inputs.to(self.device)[:32],
                        targets.to(self.device)[:32])
            break

        self.gen.train()
        self.disc.train()

        # Training
        for epoch in range(hp.NUM_EPOCHS):
            pbar = tqdm(enumerate(train_loader),
                        total=len(train_loader), leave=False)
            for batch_idx, (inputs, targets) in pbar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                with torch.cuda.amp.autocast():
                    fake = self.gen(inputs)
                    disc_real = self.disc(inputs, targets)
                    lossD_real = self.BCE_LOSS(
                        disc_real, torch.ones_like(disc_real))

                    disc_fake = self.disc(inputs, fake.detach())
                    lossD_fake = self.BCE_LOSS(
                        disc_fake, torch.zeros_like(disc_fake))
                    lossD = (lossD_fake + lossD_real) / 2

                self.disc.zero_grad()
                self.scaler_dis.scale(lossD).backward()
                self.scaler_dis.step(self.opt_disc)
                self.scaler_dis.update()

                # Train Generator
                with torch.cuda.amp.autocast():
                    output = self.disc(inputs, fake)
                    adv_loss = self.BCE_LOSS(output, torch.ones_like(output))
                    l1_loss = self.L1_LOSS(fake, targets)
                    lossG = adv_loss + l1_loss * hp.LRAMBDA_L1

                self.gen.zero_grad()
                self.scaler_gen.scale(lossG).backward()
                self.scaler_gen.step(self.opt_gen)
                self.scaler_gen.update()

                with torch.no_grad():
                    if batch_idx % hp.LOG_INTERVAL == 0:
                        self.tb.add_scalar(
                            'TRAIN/Real Prob', self.gan_prob(disc_real).item(),
                            global_step=self.itr)
                        self.tb.add_scalar(
                            'TRAIN/Fake Prob', self.gan_prob(disc_fake).item(),
                            global_step=self.itr)
                        self.tb.add_scalar(
                            'TRAIN/L1_LOSS', l1_loss.item(),
                            global_step=self.itr)
                        self.tb.add_scalar(
                            'TRAIN/ADV_LOSS', adv_loss.item(),
                            global_step=self.itr)

                    if batch_idx % hp.TEST_INTERVAL == 0:
                        pbar.set_description_str(
                            f'Epoch[{epoch+1} / {hp.NUM_EPOCHS}], '
                            f'real:{self.gan_prob(disc_real).item(): .2f}, '
                            f'fake:{self.gan_prob(disc_fake).item(): .2f}, '
                            f'lossD:{lossD.item(): .2f}, '
                            f'lossG:{lossG.item(): .2f} ')
                        self.test(FIX_DATA[0], FIX_DATA[1])
                self.itr += 1

            # Epoch ckpt save
            self.save_checkopint(
                gen=self.gen,
                disc=self.disc,
                opt_gen=self.opt_gen,
                opt_disc=self.opt_disc,
                epoch=epoch)

    def test(self, inputs, targets):
        self.gen.eval()
        fake = self.gen(inputs)
        batch_size = fake.shape[0]

        img_grid_fake = torchvision.utils.make_grid(
            fake, nrow=batch_size, normalize=True)
        img_grid_real = torchvision.utils.make_grid(
            targets, nrow=batch_size, normalize=True)
        img_grid_input = torchvision.utils.make_grid(
            inputs, nrow=batch_size, normalize=True)

        save_image = torch.cat(
            [img_grid_fake, img_grid_real, img_grid_input],
            dim=1)

        self.tb.add_image("TEST/images", save_image,
                          global_step=self.itr)
        torchvision.utils.save_image(
            save_image,
            os.path.join(self.image_dir,
                         f'GS: { str(self.itr).zfill(8)}.jpg'))
        self.gen.train()


if __name__ == '__main__':
    trainer = Pix2Pix().train()
    print('Train Done')
