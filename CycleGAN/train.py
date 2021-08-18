import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

import hyperparameters as hp
from dataset import Dataset
from model import Discriminator, Generator
from opt import Template


class CycleGAN(Template):
    def __init__(self):
        super().__init__(
            device=hp.DEVICE, seed=hp.SEED, model_name='CycleGAN')

        self.dataset = Dataset(hp.DATASET)
        self.gen_B, self.gen_A = [Generator(
            hp.INPUT_CHANNELS,
            hp.OUPUT_CHANNELS,
            hp.GEN_DIM,
            hp.NUM_RESIDUALS).to(self.device) for _ in range(2)]

        self.disc_z, self.disc_A = [Discriminator(
            hp.INPUT_CHANNELS,
            hp.DISC_DIM).to(self.device) for _ in range(2)]

        # optimizer
        self.opt_gen = optim.Adam(
            list(self.gen_A.parameters()) + list(self.gen_B.parameters()),
            lr=hp.LEARNING_RATE,
            betas=hp.BETAS)

        self.opt_disc = optim.Adam(
            list(self.disc_A.parameters()) + list(self.disc_z.parameters()),
            lr=hp.LEARNING_RATE,
            betas=hp.BETAS)

        self.scaler_gen = torch.cuda.amp.GradScaler()
        self.scaler_dis = torch.cuda.amp.GradScaler()

        self.mse_loss = nn.MSELoss().to(self.device)
        self.L1_loss = nn.L1Loss().to(self.device)

    def switch_mode(self, train=True):
        self.gen_A.train(train)
        self.gen_B.train(train)
        self.disc_A.train(train)
        self.disc_z.train(train)

    def train(self):
        self.switch_mode(train=True)
        train_loader, val_loader = self.dataset.create_dataloader(
            hp.BATCH_SIZE, hp.NUM_WORKERS)
        for A, B in val_loader:
            FIX_DATA = {"A": A.to(self.device)[:hp.SAMPLE_SIZE],
                        "B": B.to(self.device)[:hp.SAMPLE_SIZE]}
            break

        # Training
        for epoch in range(hp.NUM_EPOCHS):
            pbar = tqdm(enumerate(train_loader),
                        total=len(train_loader), leave=False)
            for batch_idx, (A, B) in pbar:
                '''
                A : horse
                B : zebra
                '''
                # Train discriminator
                A = A.to(self.device)
                B = B.to(self.device)

                with torch.cuda.amp.autocast():
                    # Train horse models
                    A_fake = self.gen_A(B)
                    disc_A_real = self.disc_A(A)
                    disc_A_fake = self.disc_A(A_fake.detach())
                    lossD_A_real = self.mse_loss(
                        disc_A_real, torch.ones_like(disc_A_real))
                    lossD_A_fake = self.mse_loss(
                        disc_A_fake, torch.zeros_like(disc_A_fake))
                    lossD_A = lossD_A_real + lossD_A_fake

                    # Train zebra models
                    B_fake = self.gen_B(A)
                    disc_B_real = self.disc_z(B)
                    disc_B_fake = self.disc_z(B_fake.detach())
                    lossD_B_real = self.mse_loss(
                        disc_B_real, torch.ones_like(disc_B_real))
                    lossD_B_fake = self.mse_loss(
                        disc_B_fake, torch.zeros_like(disc_B_fake))
                    lossD_B = lossD_B_real + lossD_B_fake

                    lossD = (lossD_A + lossD_B) / 2

                # Update Discriminator
                self.opt_disc.zero_grad()
                self.scaler_dis.scale(lossD).backward()
                self.scaler_dis.step(self.opt_disc)
                self.scaler_dis.update()

                # Train Generator
                with torch.cuda.amp.autocast():
                    # calc ADV loss
                    disc_A_output = self.disc_A(A_fake)
                    disc_B_output = self.disc_z(B_fake)
                    lossG_adv_h = self.mse_loss(
                        disc_A_output, torch.ones_like(disc_A_output))
                    lossG_adv_z = self.mse_loss(
                        disc_B_output, torch.ones_like(disc_B_output))
                    # calc cycle loss
                    B_cycle = self.gen_B(A_fake)
                    A_cycle = self.gen_A(B_fake)
                    lossG_cycle_A = self.L1_loss(A, A_cycle)
                    lossG_cycle_B = self.L1_loss(B, B_cycle)

                    A_id = self.gen_A(A)
                    B_id = self.gen_B(B)
                    lossG_id_A = self.L1_loss(A, A_id)
                    lossG_id_B = self.L1_loss(B, B_id)

                    lossG = lossG_adv_h + lossG_adv_z \
                        + lossG_cycle_A * hp.LAMBDA_CYCLE \
                        + lossG_cycle_B * hp.LAMBDA_CYCLE \
                        + lossG_id_A * hp.LAMBDA_IDENTITY \
                        + lossG_id_B * hp.LAMBDA_IDENTITY \

                self.opt_gen.zero_grad()
                self.scaler_gen.scale(lossG).backward()
                self.scaler_gen.step(self.opt_gen)
                self.scaler_gen.update()

                with torch.no_grad():
                    if batch_idx % hp.LOG_INTERVAL == 0:
                        real_A, fake_A, real_B, fake_B = self.add_scalar(
                            disc_A_real, disc_A_fake,
                            disc_B_real, disc_B_fake, lossD,
                            lossG_cycle_A, lossG_cycle_B,
                            lossG_id_A, lossG_id_B, lossG)

                        pbar.set_description_str(
                            f'Epoch[{epoch+1} / {hp.NUM_EPOCHS}], '
                            f'real(A):{real_A: .2f}, '
                            f'fake(A):{fake_A: .2f}, '
                            f'real(B):{real_B: .2f}, '
                            f'fake(B):{fake_B: .2f}, ')

                    if batch_idx % hp.TEST_INTERVAL == 0:
                        self.test(FIX_DATA['A'], FIX_DATA['B'])
                self.itr += 1

    def add_scalar(self, disc_A_real, disc_A_fake, disc_B_real, disc_B_fake, lossD,
                   lossG_cycle_A, lossG_cycle_B, lossG_id_A, lossG_id_B, lossG):
        real_A = disc_A_real.mean().item()
        fake_A = disc_A_fake.mean().item()
        real_B = disc_B_real.mean().item()
        fake_B = disc_B_fake.mean().item()
        cycle_A = lossG_cycle_A.item()
        cycle_B = lossG_cycle_B.item()
        id_A = lossG_id_A.item()
        id_B = lossG_id_B.item()

        self.tb.add_scalar('TRAIN/Real (A)', real_A, self.itr)
        self.tb.add_scalar('TRAIN/Fake (A)', fake_A, self.itr)
        self.tb.add_scalar('TRAIN/Real (B)', real_B, self.itr)
        self.tb.add_scalar('TRAIN/Fake (B)', fake_B, self.itr)
        self.tb.add_scalar('TRAIN/Cycle (A)', cycle_A, self.itr)
        self.tb.add_scalar('TRAIN/Cycle (B)', cycle_B, self.itr)
        self.tb.add_scalar('TRAIN/ID (A)', id_A, self.itr)
        self.tb.add_scalar('TRAIN/ID (B)', id_B, self.itr)
        self.tb.add_scalar('TRAIN/LossD', lossD.item(), self.itr)
        self.tb.add_scalar('TRAIN/LossG', lossG.item(), self.itr)
        return real_A, fake_A, real_B, fake_B

    def test(self, A, B):
        self.switch_mode(False)
        size = A.shape[0]
        A_ = self.gen_A(B)
        B_ = self.gen_B(A)

        save_image = torch.cat([
            torchvision.utils.make_grid(A, size, 1, normalize=True),
            torchvision.utils.make_grid(B_, size, 1, normalize=True),
            torchvision.utils.make_grid(B, size, 1, normalize=True),
            torchvision.utils.make_grid(A_, size, 1, normalize=True),
        ], dim=1)

        self.tb.add_image("TEST/images", save_image, self.itr)
        torchvision.utils.save_image(
            save_image,
            os.path.join(self.image_dir, f'GS: { str(self.itr).zfill(8)}.jpg'))
        self.switch_mode(True)


if __name__ == '__main__':
    trainer = CycleGAN().train()
    print('Train Done')
