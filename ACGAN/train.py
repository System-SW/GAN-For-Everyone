import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm
import hyperparameters as hp
from dataset import Dataset
from model import Discriminator, Generator, initialize_weights
from opt import Template


class ACGAN(Template):
    def __init__(self):
        super().__init__(
            device=hp.DEVICE, seed=hp.SEED, model_name=self.__class__.__name__
        )

        self.dataset = Dataset(hp.DATASET)
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

        self.FIX_DATA = {
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

    def train(self):
        loader = self.dataset.create_dataloader(hp.BATCH_SIZE)
        self.gen.train()
        self.disc.train()

        # Training
        for epoch in range(hp.NUM_EPOCHS):
            pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
            for batch_idx, (real, label) in pbar:
                real = real.to(self.device)
                label = label.to(self.device)
                batch_size = real.shape[0]
                noise = torch.randn(batch_size, hp.NOISE_DIM, 1, 1, device=self.device)
                gen_labels = torch.randint_like(label, low=0, high=hp.NUM_CLASSES)

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

                self.disc.zero_grad()
                self.scaler_dis.scale(lossD).backward()
                self.scaler_dis.step(self.opt_disc)
                self.scaler_dis.update()

                # Train Generator
                with torch.cuda.amp.autocast():
                    output, output_label = self.disc(fake)
                    lossG_adv = self.bce_loss(output, torch.ones_like(output))
                    lossG_auxiliary = self.cross_entropy_loss(output_label, gen_labels)
                    lossG = (lossG_adv + lossG_auxiliary) / 2

                self.gen.zero_grad()
                self.scaler_gen.scale(lossG).backward()
                self.scaler_gen.step(self.opt_gen)
                self.scaler_gen.update()

                with torch.no_grad():
                    if batch_idx % hp.LOG_INTERVAL == 0:
                        log_real_prob = self.gan_prob(disc_real_logits).item()
                        log_fake_prob = self.gan_prob(disc_fake_logits).item()
                        log_lossD_real_class = lossD_real_class.item()
                        log_lossD_fake_class = lossD_fake_class.item()
                        log_lossG_class = lossG_auxiliary.item()

                        self.logging(
                            log_real_prob,
                            log_fake_prob,
                            log_lossD_real_class,
                            log_lossD_fake_class,
                            log_lossG_class,
                        )
                        pbar.set_description_str(
                            f"Epoch[{epoch+1} / {hp.NUM_EPOCHS}], "
                            f"real:{log_real_prob: .2f}, "
                            f"fake:{log_fake_prob: .2f}, "
                            f"lossD:{lossD.item(): .2f}, "
                            f"lossG:{lossG.item(): .2f} "
                        )

                self.itr += 1
            self.test(real)
            self.save_checkpoint(
                self.gen, self.disc, self.opt_gen, self.opt_disc, epoch
            )

    def logging(
        self,
        log_real_prob,
        log_fake_prob,
        log_lossD_real_class,
        log_lossD_fake_class,
        log_lossG_class,
    ):
        self.tb.add_scalar("TRAIN/Real Prob", log_real_prob, self.itr)
        self.tb.add_scalar("TRAIN/Real Cross Entropy", log_lossD_real_class, self.itr)
        self.tb.add_scalar("TRAIN/Fake Prob", log_fake_prob, self.itr)
        self.tb.add_scalar("TRAIN/Fake Cross Entropy", log_lossD_fake_class, self.itr)
        self.tb.add_scalar("TRAIN/lossG Cross Entropy", log_lossG_class, self.itr)

    @torch.no_grad()
    def test(self, real):
        self.gen.eval()
        noise = self.FIX_DATA["noise"]
        label = self.FIX_DATA["label"]
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


if __name__ == "__main__":
    trainer = ACGAN().train()
    print("Train Done")
