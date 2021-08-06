import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
import hyperparameters as hp
from dataset import build_dataloader

device = hp.device
dataloader = build_dataloader(hp.DATASET_CAT)
gen = Generator(hp.NOISE_DIM, hp.IMAGE_CHANNELS, hp.GEN_DIM).to(device)
disc = Discriminator(hp.IMAGE_CHANNELS, hp.DISC_DIM).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(),
                     lr=hp.LEARNING_RATE, betas=hp.BETAS)
opt_disc = optim.Adam(disc.parameters(),
                      lr=hp.LEARNING_RATE, betas=hp.BETAS)
scaler_gen = torch.cuda.amp.GradScaler()
scaler_dis = torch.cuda.amp.GradScaler()
gan_loss = nn.BCEWithLogitsLoss()

FIXED_NOISE = torch.randn(32, hp.NOISE_DIM, 1, 1, device=device)
tb = SummaryWriter('logs/DCGAN')
step = 0


for epoch in range(hp.NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        batch_size = real.shape[0]
        noise = torch.randn(batch_size, hp.NOISE_DIM, 1, 1, device=device)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            fake = gen(noise)
            disc_real = disc(real).reshape(-1)
            lossD_real = gan_loss(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            lossD_fake = gan_loss(disc_fake, torch.zeros_like(disc_fake))
            loss_D = (lossD_fake + lossD_real) / 2

        disc.zero_grad()
        scaler_dis.scale(loss_D).backward()
        scaler_dis.step(opt_disc)
        scaler_dis.update()

        # Train Generator
        with torch.cuda.amp.autocast():
            output = disc(fake).reshape(-1)
            loss_G = gan_loss(output, torch.ones_like(output))

        gen.zero_grad()
        scaler_gen.scale(loss_G).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        if batch_idx % 100 == 0:
            real_logit = torch.sigmoid(disc_real).mean()
            fake_logit = torch.sigmoid(disc_fake).mean()
            print(
                f"Epoch [{epoch+1}/{hp.NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} Loss D: {loss_D.item(): .4f}, \
                loss G: {loss_G.item(): .4f} real: {real_logit.item(): .4f}, fake: {fake_logit.item(): .4f} "
            )

            with torch.no_grad():
                fake = gen(FIXED_NOISE)
                img_real = torchvision.utils.make_grid(
                    real, normalize=True)
                img_fake = torchvision.utils.make_grid(
                    fake, normalize=True)
                tb.add_image('Real', img_real, global_step=step)
                tb.add_image('Fake', img_fake, global_step=step)
            step += 1
