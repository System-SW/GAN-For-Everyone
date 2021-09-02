import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

import hyperparameters as hp
from dataset import DataLoader
from model import Discriminator, Generator
from opt import Metrics, Template


class CycleGAN(Template):
    def __init__(self):
        super().__init__(
            device=hp.DEVICE,
            seed=hp.SEED,
            model_name=self.__class__.__name__,
        )

        self.dataset = DataLoader(hp.DATASET)
        self.gen_B, self.gen_A = [
            Generator(
                hp.INPUT_CHANNELS, hp.OUTPUT_CHANNELS, hp.GEN_DIM, hp.NUM_RESIDUALS
            ).to(self.device)
            for _ in range(2)
        ]
        self.disc_B, self.disc_A = [
            Discriminator(hp.INPUT_CHANNELS, hp.DISC_DIM).to(self.device)
            for _ in range(2)
        ]

        # optimizer
        self.opt_gen = optim.Adam(
            list(self.gen_A.parameters()) + list(self.gen_B.parameters()),
            lr=hp.LEARNING_RATE,
            betas=hp.BETAS,
        )

        self.opt_disc = optim.Adam(
            list(self.disc_A.parameters()) + list(self.disc_B.parameters()),
            lr=hp.LEARNING_RATE,
            betas=hp.BETAS,
        )

        self.scaler_gen = torch.cuda.amp.GradScaler()
        self.scaler_dis = torch.cuda.amp.GradScaler()

        self.mse_loss = nn.MSELoss().to(self.device)
        self.L1_loss = nn.L1Loss().to(self.device)

        self.restore_checkpoint(
            ckpt_path=hp.RESTORE_CKPT_PATH,
            gen_A=self.gen_A,
            gen_B=self.gen_B,
            disc_A=self.disc_A,
            disc_B=self.disc_B,
            opt_gen=self.opt_gen,
            opt_disc=self.opt_disc,
            lr=hp.LEARNING_RATE,
            device=self.device,
        )

        self.lossD_real = Metrics("TRAIN/lossD_real")
        self.lossD_fake = Metrics("TRAIN/lossD_fake")
        self.disc_A_real = Metrics("Train.A/disc_real")
        self.disc_A_fake = Metrics("Train.A/disc_fake")
        self.disc_B_real = Metrics("Train.B/disc_real")
        self.disc_B_fake = Metrics("Train.B/disc_fake")
        self.lossG_cycle_A = Metrics("Train.A/LossG_cycle")
        self.lossG_cycle_B = Metrics("Train.B/LossG_cycle")
        self.lossG_id_A = Metrics("Train.A/LossG_id")
        self.lossG_id_B = Metrics("Train.B/LossG_id")
        self.lossG = Metrics("Train/LossG")
        self.lossD = Metrics("Train/LossD")

    def switch_mode(self, train=True):
        self.gen_A.train(train)
        self.gen_B.train(train)
        self.disc_A.train(train)
        self.disc_B.train(train)

    def train(self):
        self.switch_mode(train=True)
        train_loader, val_loader = self.dataset.create_dataloader(
            hp.BATCH_SIZE, hp.SAMPLE_SIZE, hp.NUM_WORKERS
        )
        for A, B in val_loader:
            FIX_DATA = {
                "A": A.to(self.device),
                "B": B.to(self.device),
            }
            break

        # Training
        for epoch in range(hp.NUM_EPOCHS):
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
            for batch_idx, (A, B) in pbar:
                """
                A : horse
                B : zebra
                """
                self.itr += 1
                # Train discriminator
                A = A.to(self.device)
                B = B.to(self.device)

                with torch.cuda.amp.autocast():

                    # calc A disc loss
                    A_fake = self.gen_A(B)
                    disc_A_real = self.disc_A(A)
                    disc_A_fake = self.disc_A(A_fake.detach())
                    lossD_A_real = self.mse_loss(
                        disc_A_real, torch.ones_like(disc_A_real)
                    )
                    lossD_A_fake = self.mse_loss(
                        disc_A_fake, torch.zeros_like(disc_A_fake)
                    )
                    lossD_A = lossD_A_real + lossD_A_fake

                    # calc B disc loss
                    B_fake = self.gen_B(A)
                    disc_B_real = self.disc_B(B)
                    disc_B_fake = self.disc_B(B_fake.detach())
                    lossD_B_real = self.mse_loss(
                        disc_B_real, torch.ones_like(disc_B_real)
                    )
                    lossD_B_fake = self.mse_loss(
                        disc_B_fake, torch.zeros_like(disc_B_fake)
                    )
                    lossD_B = lossD_B_real + lossD_B_fake
                    # conbine losses
                    lossD = (lossD_A + lossD_B) / 2

                # Update disc optimizer
                self.opt_disc.zero_grad()
                self.scaler_dis.scale(lossD).backward()
                self.scaler_dis.step(self.opt_disc)
                self.scaler_dis.update()

                # Train Generator
                with torch.cuda.amp.autocast():
                    # calc ADV loss
                    disc_A_output = self.disc_A(A_fake)
                    disc_B_output = self.disc_B(B_fake)
                    lossG_adv_h = self.mse_loss(
                        disc_A_output, torch.ones_like(disc_A_output)
                    )
                    lossG_adv_z = self.mse_loss(
                        disc_B_output, torch.ones_like(disc_B_output)
                    )
                    # calc cycle loss
                    B_cycle = self.gen_B(A_fake)
                    A_cycle = self.gen_A(B_fake)
                    lossG_cycle_A = self.L1_loss(A, A_cycle)
                    lossG_cycle_B = self.L1_loss(B, B_cycle)
                    # calc identity loss
                    A_id = self.gen_A(A)
                    B_id = self.gen_B(B)
                    lossG_id_A = self.L1_loss(A, A_id)
                    lossG_id_B = self.L1_loss(B, B_id)
                    # conbine losses
                    lossG = (
                        lossG_adv_h
                        + lossG_adv_z
                        + lossG_cycle_A * hp.LAMBDA_CYCLE
                        + lossG_cycle_B * hp.LAMBDA_CYCLE
                        + lossG_id_A * hp.LAMBDA_IDENTITY
                        + lossG_id_B * hp.LAMBDA_IDENTITY
                    )
                # update gen optimizer
                self.opt_gen.zero_grad()
                self.scaler_gen.scale(lossG).backward()
                self.scaler_gen.step(self.opt_gen)
                self.scaler_gen.update()

                # Logging
                self.disc_A_real.update_state(disc_A_real.mean())
                self.disc_A_fake.update_state(disc_A_fake.mean())
                self.disc_B_real.update_state(disc_B_real.mean())
                self.disc_B_fake.update_state(disc_B_fake.mean())
                self.lossG_cycle_A.update_state(lossG_cycle_A)
                self.lossG_cycle_B.update_state(lossG_cycle_B)
                self.lossG_id_A.update_state(lossG_id_A)
                self.lossG_id_B.update_state(lossG_id_B)
                self.lossD.update_state(lossD)
                self.lossG.update_state(lossG)

                with torch.no_grad():
                    if batch_idx % hp.LOG_INTERVAL == 0:
                        disc_A_real = self.logging_scaler(self.disc_A_real)
                        disc_A_fake = self.logging_scaler(self.disc_A_fake)
                        disc_B_real = self.logging_scaler(self.disc_B_real)
                        disc_B_fake = self.logging_scaler(self.disc_B_fake)
                        self.logging_scaler(self.lossG_cycle_A)
                        self.logging_scaler(self.lossG_cycle_B)
                        self.logging_scaler(self.lossG_id_A)
                        self.logging_scaler(self.lossG_id_B)
                        self.logging_scaler(self.lossD)
                        self.logging_scaler(self.lossG)

                        pbar.set_description_str(
                            f"Epoch[{epoch+1} / {hp.NUM_EPOCHS}], "
                            f"real(A):{disc_A_real: .2f}, "
                            f"fake(A):{disc_A_fake: .2f}, "
                            f"real(B):{disc_B_real: .2f}, "
                            f"fake(B):{disc_B_fake: .2f}, "
                        )
                    if batch_idx % hp.TEST_INTERVAL == 0:
                        test_image = self.test(FIX_DATA["A"], FIX_DATA["B"])
                        self.save_image_to_logdir(test_image, self.itr)

            self.logging_weight_and_gradient("GEN_A", self.gen_A, self.itr)
            self.logging_weight_and_gradient("GEN_B", self.gen_B, self.itr)
            self.logging_weight_and_gradient("DIS_A", self.disc_A, self.itr)
            self.logging_weight_and_gradient("DIS_A", self.disc_B, self.itr)

            test_image = self.test(FIX_DATA["A"], FIX_DATA["B"])
            super().save_image_to_logdir(test_image, epoch)

            self.save_checkpoint(
                gen_A=self.gen_A,
                gen_B=self.gen_A,
                disc_A=self.disc_A,
                disc_B=self.disc_B,
                opt_gen=self.opt_gen,
                opt_disc=self.opt_disc,
                epoch=epoch,
            )

    @torch.no_grad()
    def test(self, A, B):
        self.switch_mode(False)
        size = A.shape[0]
        A_ = self.gen_A(B)
        B_ = self.gen_B(A)

        test_image = torch.cat(
            [
                torchvision.utils.make_grid(A, size, 1, normalize=True),
                torchvision.utils.make_grid(B_, size, 1, normalize=True),
                torchvision.utils.make_grid(B, size, 1, normalize=True),
                torchvision.utils.make_grid(A_, size, 1, normalize=True),
            ],
            dim=1,
        )

        self.tb.add_image("TEST/images", test_image, self.itr)
        self.switch_mode(True)
        return test_image

    @torch.no_grad()
    def save_image_to_logdir(
        self, image: torch.Tensor, gs: int, image_format="JPEG"
    ) -> None:
        fp = os.path.join(
            self.image_dir, f"{self.__class__.__name__}_GS:{str(gs).zfill(10)}.jpg"
        )
        torchvision.utils.save_image(image, fp, image_format)

    @staticmethod
    def restore_checkpoint(
        ckpt_path: str,
        gen_A: nn.Module,
        gen_B: nn.Module,
        disc_A: nn.Module,
        disc_B: nn.Module,
        opt_gen: torch.optim.Optimizer,
        opt_disc: torch.optim.Optimizer,
        lr: float,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        if ckpt_path is None:
            return -1

        if not os.path.exists(ckpt_path):
            return -1

        ckpt = torch.load(ckpt_path, map_location=device)

        gen_A.load_state_dict(ckpt["gen_A_state_dict"])
        gen_B.load_state_dict(ckpt["gen_B_state_dict"])
        opt_gen.load_state_dict(ckpt["opt_gen"])
        disc_A.load_state_dict(ckpt["disc_A_state_dict"])
        disc_B.load_state_dict(ckpt["disc_B_state_dict"])
        opt_disc.load_state_dict(ckpt["opt_disc"])

        for pg in opt_gen.param_groups:
            pg["lr"] = lr

        for pg in opt_disc.param_groups:
            pg["lr"] = lr

        print("Restore Ckpt FROM :", ckpt_path)

    @staticmethod
    def save_checkpoint(
        gen_A: nn.Module,
        gen_B: nn.Module,
        disc_A: nn.Module,
        disc_B: nn.Module,
        opt_gen: torch.optim.Optimizer,
        opt_disc: torch.optim.Optimizer,
        epoch=0,
    ):
        checkpoint = {
            "gen_A_state_dict": gen_A.state_dict(),
            "gen_B_state_dict": gen_B.state_dict(),
            "opt_gen": opt_gen.state_dict(),
            "disc_A_state_dict": disc_A.state_dict(),
            "disc_B_state_dict": disc_B.state_dict(),
            "opt_disc": opt_disc.state_dict(),
        }
        save_path = os.path.join("checkpoint.pth.tar")
        torch.save(checkpoint, save_path)
        print(f"Epoch:{epoch} ckpt save => {save_path}")


if __name__ == "__main__":
    trainer = CycleGAN().train()
    print("Train Done")
