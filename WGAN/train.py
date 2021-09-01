import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

import hyperparameters as hp
from dataset import DataLoader
from model import Critic, Generator, initialize_weights
from opt import Template, Metrics


class WGAN(Template):
    def __init__(self):
        super().__init__(
            device=hp.DEVICE,
            seed=hp.SEED,
            model_name=self.__class__.__name__,
        )

        self.dataloader = DataLoader(hp.DATASET)
        self.gen = Generator(hp.NOISE_DIM, hp.IMAGE_CHANNELS, hp.GEN_DIM).to(
            self.device
        )
        self.critic = Critic(hp.IMAGE_CHANNELS, hp.DISC_DIM).to(self.device)
        initialize_weights(self.gen)
        initialize_weights(self.critic)

        # optimizer
        self.opt_gen = optim.RMSprop(self.gen.parameters(), lr=hp.LEARNING_RATE)
        self.opt_critic = optim.RMSprop(self.critic.parameters(), lr=hp.LEARNING_RATE)

        self.scaler_gen = torch.cuda.amp.GradScaler()
        self.scaler_critic = torch.cuda.amp.GradScaler()
        self.loss_critic = Metrics("TRAIN/loss critic")
        self.lossG = Metrics("TRAIN/lossG")

        self.restore_checkpoint(
            hp.RESTORE_CKPT_PATH,
            self.gen,
            self.disc,
            self.opt_gen,
            self.opt_critic,
            hp.LEARNING_RATE,
            self.device,
        )

    def train(self):
        loader, _ = self.dataloader.create_dataloader(hp.BATCH_SIZE, hp.SAMPLE_SIZE)
        FIXED_NOISE = torch.randn(
            hp.SAMPLE_SIZE, hp.NOISE_DIM, 1, 1, device=self.device
        )
        self.gen.train()
        self.critic.train()

        # Training
        for epoch in range(hp.NUM_EPOCHS):
            pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
            for batch_idx, (real, _) in pbar:
                self.itr += 1
                real = real.to(self.device)
                batch_size = real.shape[0]
                noise = torch.randn(batch_size, hp.NOISE_DIM, 1, 1, device=self.device)

                # Train Critic
                for _ in range(hp.CRITIC_ITERATIONS):
                    with torch.cuda.amp.autocast():
                        fake = self.gen(noise)
                        critic_real = self.critic(real).reshape(-1)
                        critic_fake = self.critic(fake.detach()).reshape(-1)

                        loss_critic = -(
                            torch.mean(critic_real) - torch.mean(critic_fake)
                        )

                    self.critic.zero_grad()
                    self.scaler_critic.scale(loss_critic).backward()
                    self.scaler_critic.step(self.opt_critic)
                    self.scaler_critic.update()

                    # weight clipping for Lipschitz-continuous
                    for p in self.critic.parameters():
                        p.data.clamp_(-hp.WEIGHT_CLIP, hp.WEIGHT_CLIP)

                # Train Generator
                with torch.cuda.amp.autocast():
                    output = self.critic(fake).reshape(-1)
                    lossG = -torch.mean(output)

                self.gen.zero_grad()
                self.scaler_gen.scale(lossG).backward()
                self.scaler_gen.step(self.opt_gen)
                self.scaler_gen.update()

                # Logging
                self.loss_critic.update_state(loss_critic)
                self.lossG.update_state(lossG)

                if batch_idx % hp.LOG_INTERVAL == 0:
                    loss_critic = self.logging_scaler(self.loss_critic)
                    lossG = self.logging_scaler(self.lossG)
                    pbar.set_description_str(
                        f"Epoch[{epoch+1} / {hp.NUM_EPOCHS}], "
                        f"lossD:{loss_critic: .2f}, "
                        f"lossG:{lossG: .2f} "
                    )

                if batch_idx % hp.TEST_INTERVAL == 0:
                    self.test(self.gen, real[: hp.SAMPLE_SIZE], FIXED_NOISE)

            # Epoch process
            self.logging_weight_and_gradient("GEN", self.gen, self.itr)
            self.logging_weight_and_gradient("CRITIC", self.critic, self.itr)

            test_image = self.test(self.gen, real[: hp.SAMPLE_SIZE], FIXED_NOISE)
            self.save_image_to_logdir(test_image, epoch)
            self.save_checkpoint(
                self.gen, self.critic, self.opt_gen, self.opt_critic, epoch
            )


if __name__ == "__main__":
    trainer = WGAN().train()
    print("Train Done")
