import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

import hyperparameters as hp
from dataset import Dataset
from model import Critic, Generator, initialize_weights
from opt import Template


class WGAN(Template):
    def __init__(self):
        super().__init__(
            device=hp.DEVICE, seed=hp.SEED, model_name='WGAN')

        self.dataset = Dataset(hp.DATASET)
        self.gen = Generator(hp.NOISE_DIM, hp.IMAGE_CHANNELS,
                             hp.GEN_DIM).to(self.device)
        self.critic = Critic(hp.IMAGE_CHANNELS,
                             hp.DISC_DIM).to(self.device)
        initialize_weights(self.gen)
        initialize_weights(self.critic)

        self.FIXED_NOISE = torch.randn(32, hp.NOISE_DIM, 1, 1,
                                       device=self.device)
        # optimizer
        self.opt_gen = optim.RMSprop(
            self.gen.parameters(), lr=hp.LEARNING_RATE)
        self.opt_disc = optim.RMSprop(
            self.critic.parameters(), lr=hp.LEARNING_RATE)

        self.scaler_gen = torch.cuda.amp.GradScaler()
        self.scaler_critic = torch.cuda.amp.GradScaler()

    def train(self):
        loader = self.dataset.create_dataloader(
            hp.BATCH_SIZE)
        self.gen.train()
        self.critic.train()

        # Training
        for epoch in range(hp.NUM_EPOCHS):
            pbar = tqdm(enumerate(loader), total=len(loader))
            for batch_idx, (real, _) in pbar:
                real = real.to(self.device)
                batch_size = real.shape[0]
                noise = torch.randn(
                    batch_size, hp.NOISE_DIM, 1, 1, device=self.device)

                # Train Critic
                for _ in range(hp.CRITIC_ITERATIONS):
                    with torch.cuda.amp.autocast():
                        fake = self.gen(noise)
                        critic_real = self.critic(real).reshape(-1)
                        critic_fake = self.critic(fake.detach()).reshape(-1)

                        loss_critic = -(torch.mean(critic_real) -
                                        torch.mean(critic_fake))

                    self.critic.zero_grad()
                    self.scaler_critic.scale(loss_critic).backward()
                    self.scaler_critic.step(self.opt_disc)
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

                with torch.no_grad():
                    if batch_idx % hp.LOG_INTERVAL == 0:
                        self.tb.add_scalar(
                            'TRAIN/Critic loss', -loss_critic, global_step=self.itr)
                        self.tb.add_scalar(
                            'TRAIN/Generator loss', lossG, global_step=self.itr)

                    if batch_idx % hp.TEST_INTERVAL == 0:
                        pbar.set_description_str(
                            f'Epoch[{epoch+1} / {hp.NUM_EPOCHS}], '
                            f'lossD:{-loss_critic.item(): .2f}, '
                            f'lossG:{lossG.item(): .2f} ')
                        self.test(real)
                self.itr += 1

    def test(self, real):
        self.gen.eval()
        fake = self.gen(self.FIXED_NOISE)
        img_grid_fake = torchvision.utils.make_grid(
            fake, normalize=True)
        img_grid_real = torchvision.utils.make_grid(
            real[:32], normalize=True)
        self.tb.add_image(
            "Fake Images", img_grid_fake,
            global_step=self.itr
        )
        self.tb.add_image(
            "Real Images", img_grid_real,
            global_step=self.itr
        )
        self.gen.train()


if __name__ == '__main__':
    trainer = WGAN().train()
