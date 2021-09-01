import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

import hyperparameters as hp
from dataset import DataLoader
from model import Critic, Generator, initialize_weights
from opt import Metrics, Template


class WGAN_GP(Template):
    def __init__(self):
        super().__init__(device=hp.DEVICE, seed=hp.SEED, model_name="WGAN-GP")

        self.dataloader = DataLoader(hp.DATASET)
        self.gen = Generator(hp.NOISE_DIM, hp.IMAGE_CHANNELS, hp.GEN_DIM).to(
            self.device
        )
        self.critic = Critic(hp.IMAGE_CHANNELS, hp.DISC_DIM).to(self.device)
        initialize_weights(self.gen)
        initialize_weights(self.critic)

        # optimizer
        self.opt_gen = optim.Adam(
            self.gen.parameters(), lr=hp.LEARNING_RATE, betas=hp.BETAS
        )
        self.opt_critic = optim.Adam(
            self.critic.parameters(), lr=hp.LEARNING_RATE, betas=hp.BETAS
        )

        self.loss_critic = Metrics("TRAIN/Critic loss")
        self.lossG = Metrics("TRAIN/Generator loss")
        self.gp = Metrics("TRAIN/Critic gradient penalty")

    def gradient_penalty(self, real, fake):
        BATCH_SIZE, C, H, W = real.shape
        alpha = torch.rand(BATCH_SIZE, 1, 1, 1).repeat(1, C, H, W).to(self.device)
        interpolated_images = real * alpha + fake * (1 - alpha)

        mixed_scores = self.critic(interpolated_images)
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        return torch.mean((gradient_norm - 1) ** 2)

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

                # Train Critic
                for _ in range(hp.CRITIC_ITERATIONS):
                    noise = torch.randn(
                        batch_size, hp.NOISE_DIM, 1, 1, device=self.device
                    )

                    fake = self.gen(noise)
                    critic_real = self.critic(real).reshape(-1)
                    critic_fake = self.critic(fake).reshape(-1)
                    gradient_penalty = self.gradient_penalty(real, fake)
                    loss_critic = (
                        -(torch.mean(critic_real) - torch.mean(critic_fake))
                        + gradient_penalty * hp.LAMBDA_GP
                    )
                    self.critic.zero_grad()
                    loss_critic.backward(retain_graph=True)
                    self.opt_critic.step()

                # Train Generator
                output = self.critic(fake).reshape(-1)
                lossG = -torch.mean(output)

                self.gen.zero_grad()
                lossG.backward()
                self.opt_gen.step()

                # Logging
                self.loss_critic.update_state(loss_critic)
                self.lossG.update_state(lossG)
                self.gp.update_state(gradient_penalty)

                if batch_idx % hp.LOG_INTERVAL == 0:
                    loss_critic = self.logging_scaler(self.loss_critic)
                    lossG = self.logging_scaler(self.lossG)
                    gradient_penalty = self.logging_scaler(self.gp)
                    pbar.set_description_str(
                        f"Epoch[{epoch+1} / {hp.NUM_EPOCHS}], "
                        f"lossD:{loss_critic: .2f}, "
                        f"lossG:{lossG: .2f}, "
                        f"GP:{gradient_penalty: .2f} "
                    )
                if batch_idx % hp.TEST_INTERVAL == 0:
                    self.test(self.gen, real[: hp.SAMPLE_SIZE], FIXED_NOISE)
            # Epoch process
            self.logging_weight_and_gradient("GEN", self.gen, self.itr)
            self.logging_weight_and_gradient("CRITIC", self.critic, self.itr)

            test_image = self.test(self.gen, real[: hp.SAMPLE_SIZE], FIXED_NOISE)
            self.save_image_to_logdir(test_image, epoch + 1)

            self.save_checkpoint(
                self.gen, self.critic, self.opt_gen, self.opt_critic, epoch + 1
            )

    @torch.no_grad()
    def add_scaler(self, loss_critic, lossG, gradient_penalty):
        loss_critic = loss_critic.item()
        lossG = lossG.item()
        gradient_penalty = gradient_penalty.item()
        self.tb.add_scalar("TRAIN/Critic loss", -loss_critic, self.itr)
        self.tb.add_scalar("TRAIN/Generator loss", lossG, self.itr)
        self.tb.add_scalar("TRAIN/Critic gradient penalty", gradient_penalty, self.itr)
        return loss_critic, lossG, gradient_penalty


if __name__ == "__main__":
    trainer = WGAN_GP().train()
    print("Train Done")
