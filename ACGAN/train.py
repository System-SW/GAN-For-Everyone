import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

import hyperparameters as hp
from dataset import DataLoader
from model import Discriminator, Generator, initialize_weights
from opt import Metrics, Template


class ACGAN(Template):
    def __init__(self):
        super().__init__(
            device=hp.DEVICE, seed=hp.SEED, model_name=self.__class__.__name__
        )

        self.dataset = DataLoader(hp.DATASET)
        self.gen = Generator(
            hp.IMAGE_SIZE, hp.NOISE_DIM, hp.IMAGE_CHANNELS, hp.GEN_DIM, hp.NUM_CLASSES
        ).to(self.device)
        self.disc = Discriminator(
            hp.IMAGE_SIZE, hp.IMAGE_CHANNELS, hp.DISC_DIM, hp.NUM_CLASSES
        ).to(self.device)
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
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.real_prob = Metrics("TRAIN/Real Prob")
        self.fake_prob = Metrics("TRAIN/Fake Prob")
        self.lossD_real_class = Metrics("TRAIN/lossD_real_class")
        self.lossD_fake_class = Metrics("TRAIN/lossD_fake_class")
        self.lossD = Metrics("TRAIN/lossD")
        self.lossG = Metrics("TRAIN/lossG")
        self.lossG_class = Metrics("TRAIN/lossG_class")

        self.restore_checkpoint(
            hp.RESTORE_CKPT_PATH,
            self.gen,
            self.disc,
            self.opt_gen,
            self.opt_disc,
            hp.LEARNING_RATE,
            self.device,
        )

    def train(self):
        loader, _ = self.dataset.create_dataloader(hp.BATCH_SIZE, hp.SAMPLE_SIZE)
        FIXED_DATA = {
            "noise": torch.randn(
                hp.SAMPLE_SIZE, hp.NOISE_DIM, 1, 1, device=self.device
            ),
            "label": torch.tensor(
                [
                    i
                    for _ in range(int(hp.SAMPLE_SIZE ** 0.5))
                    for i in range(int(hp.SAMPLE_SIZE ** 0.5))
                ],
                device=self.device,
            ),
        }
        self.gen.train()
        self.disc.train()

        # Training
        for epoch in range(hp.NUM_EPOCHS):
            pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
            for batch_idx, (real, label) in pbar:
                self.itr += 1
                real = real.to(self.device)
                label = label.to(self.device)
                batch_size = real.shape[0]
                noise = torch.randn(batch_size, hp.NOISE_DIM, 1, 1, device=self.device)
                gen_labels = torch.randint_like(label, low=0, high=hp.NUM_CLASSES)

                # Train Discriminator
                with torch.cuda.amp.autocast():
                    fake = self.gen(noise, gen_labels)
                    disc_real_logits, disc_real_aux = self.disc(real)
                    lossD_real_logits = self.bce_loss(
                        disc_real_logits, torch.ones_like(disc_real_logits)
                    )
                    lossD_real_class = self.cross_entropy_loss(disc_real_aux, label)
                    lossD_real = (lossD_real_logits + lossD_real_class) / 2

                    disc_fake_logits, disc_fake_aux = self.disc(fake.detach())
                    lossD_fake_logits = self.bce_loss(
                        disc_fake_logits, torch.zeros_like(disc_fake_logits)
                    )
                    lossD_fake_class = self.cross_entropy_loss(
                        disc_fake_aux, gen_labels
                    )
                    lossD_fake = (lossD_fake_logits + lossD_fake_class) / 2

                    lossD = (lossD_real + lossD_fake) / 2
                # Update Discriminator
                self.disc.zero_grad()
                self.scaler_dis.scale(lossD).backward()
                self.scaler_dis.step(self.opt_disc)
                self.scaler_dis.update()

                # Train Generator
                with torch.cuda.amp.autocast():
                    output, output_label = self.disc(fake)
                    lossG_adv = self.bce_loss(output, torch.ones_like(output))
                    lossG_class = self.cross_entropy_loss(output_label, gen_labels)
                    lossG = (lossG_adv + lossG_class) / 2
                # Update Generator
                self.gen.zero_grad()
                self.scaler_gen.scale(lossG).backward()
                self.scaler_gen.step(self.opt_gen)
                self.scaler_gen.update()

                # logging
                self.real_prob.update_state(self.gan_prob(disc_real_logits))
                self.fake_prob.update_state(self.gan_prob(disc_fake_logits))
                self.lossD_real_class.update_state(lossD_real_class)
                self.lossD_fake_class.update_state(lossD_fake_class)
                self.lossD.update_state(lossD)
                self.lossG.update_state(lossG)
                self.lossG_class.update_state(lossG_class)

                if batch_idx % hp.LOG_INTERVAL == 0:
                    real_prob = self.logging_scaler(self.real_prob)
                    fake_prob = self.logging_scaler(self.fake_prob)
                    lossD = self.logging_scaler(self.lossD)
                    lossG = self.logging_scaler(self.lossG)
                    self.logging_scaler(self.lossD_real_class)
                    self.logging_scaler(self.lossD_fake_class)
                    self.logging_scaler(self.lossG_class)

                    pbar.set_description_str(
                        f"Epoch[{epoch+1} / {hp.NUM_EPOCHS}], "
                        f"real:{real_prob: .2f}, "
                        f"fake:{fake_prob: .2f}, "
                        f"lossD:{lossD: .2f}, "
                        f"lossG:{lossG: .2f} "
                    )
                if batch_idx % hp.TEST_INTERVAL == 0:
                    self.test(real, FIXED_DATA["noise"], FIXED_DATA["label"])

            # Epoch process
            self.logging_weight_and_gradient("GEN", self.gen, self.itr)
            self.logging_weight_and_gradient("DISC", self.disc, self.itr)

            test_image = self.test(real, FIXED_DATA["noise"], FIXED_DATA["label"])
            self.save_image_to_logdir(test_image, epoch + 1)

            self.save_checkpoint(
                self.gen, self.disc, self.opt_gen, self.opt_disc, epoch + 1
            )

    @torch.no_grad()
    def test(self, real, noise, label):
        self.gen.eval()
        fake = self.gen(noise, label)
        img_grid_fake = torchvision.utils.make_grid(
            fake, int(hp.SAMPLE_SIZE ** 0.5), normalize=True
        )
        img_grid_real = torchvision.utils.make_grid(
            real, int(hp.SAMPLE_SIZE ** 0.5), normalize=True
        )
        self.tb.add_image("Fake Images", img_grid_fake, global_step=self.itr)
        self.tb.add_image("Real Images", img_grid_real, global_step=self.itr)
        self.gen.train()
        return img_grid_fake


if __name__ == "__main__":
    trainer = ACGAN().train()
    print("Train Done")
