import os
import random
import time
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils import tensorboard


class Template(metaclass=ABCMeta):
    """Abstract Class for Trainer"""

    def __init__(self, **kwargs):
        """
        Template initializer
        kwargs: it has to be defined {device, seed, model_name}

        1. set Training seed {random, numpy, torch, cuda}
        2. create logging directory {add timestamp, image dir, checkpoint dir, ...}
        """

        self.itr = 0
        self.device = kwargs["device"]
        self.gan_prob = lambda x: torch.mean(torch.sigmoid(x))

        # Set Training Seed
        random.seed(kwargs["seed"])
        np.random.seed(kwargs["seed"])
        torch.manual_seed(kwargs["seed"])
        torch.cuda.manual_seed(kwargs["seed"])

        torch.backends.cudnn.benchmark = True

        # Create TensorBoard Dir
        timestamp = time.strftime("%Y-%m-%d-%H:00", time.localtime(time.time()))
        logdir = os.path.join("log", kwargs["model_name"], timestamp)

        self.tb = tensorboard.SummaryWriter(logdir)
        self.image_dir = os.path.join(logdir, "image")
        os.makedirs(self.tb.log_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

    @abstractmethod
    def train(self):
        """Model Train Methods"""
        ...

    def test(
        self, gen: nn.Module, real: torch.Tensor, fixed_noise: torch.Tensor
    ) -> torch.Tensor:
        """Base TEST Method, if u need change override this method

        Args:
            gen (nn.Module): generator
            real (torch.Tenser): real data batch
            fixed_noise (torch.Tensor): fixed latent vector

        Returns:
            [Torch.Tensor]: generated fake image tensor
        """
        gen.eval()
        fake = gen(fixed_noise)
        nrow = int(fixed_noise.shape[0] ** 0.5)
        img_grid_fake = torchvision.utils.make_grid(fake, nrow, normalize=True)
        img_grid_real = torchvision.utils.make_grid(real, nrow, normalize=True)
        self.tb.add_image("Fake Images", img_grid_fake, global_step=self.itr)
        self.tb.add_image("Real Images", img_grid_real, global_step=self.itr)
        gen.train()
        return img_grid_fake

    @staticmethod
    def save_checkpoint(
        gen: nn.Module,
        disc: nn.Module,
        opt_gen: torch.optim.Optimizer,
        opt_disc: torch.optim.Optimizer,
        epoch=0,
    ):
        """SAVE training state to checkpoint

        Args:
            gen (nn.Module): self.gen, Generator
            disc (nn.Module): self.disc, Discrimiantor
            opt_gen (torch.optim.Optimizer): self.opt_gen, Generator Optimaier
            opt_disc (torch.optim.Optimizer): self.opt_disc, Discriminator Optimaier
            epoch (int, optional): current epoch from epoch loop. Defaults to 0.
        """
        checkpoint = {
            "gen_state_dict": gen.state_dict(),
            "opt_gen": opt_gen.state_dict(),
            "disc_state_dict": disc.state_dict(),
            "opt_disc": opt_disc.state_dict(),
        }
        save_path = os.path.join("checkpoint.pth.tar")
        torch.save(checkpoint, save_path)
        print(f"Epoch:{epoch} ckpt save => {save_path}")

    @staticmethod
    def restore_checkpoint(
        ckpt_path: str,
        gen: nn.Module,
        disc: nn.Module,
        opt_gen: torch.optim.Optimizer,
        opt_disc: torch.optim.Optimizer,
        lr: float,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        """Restore parameters from checkpoint path

        Args:
            ckpt_path (str): Check point path
            gen (nn.Module): self.gen, Generator
            disc (nn.Module): self.disc, Discrimiantor
            opt_gen (torch.optim.Optimizer): self.opt_gen, Generator Optimaier
            opt_disc (torch.optim.Optimizer): self.opt_disc, Discriminator Optimaier
            lr (float): init Learning rate from hyperparameter
            device ([type], optional): Defaults to torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), checkpoint load device location

        Returns:
            [type]: -1(skip)/None(loaded and print message)
        """

        if ckpt_path is None:
            return -1

        if not os.path.exists(ckpt_path):
            return -1

        ckpt = torch.load(ckpt_path, map_location=device)

        gen.load_state_dict(ckpt["gen_state_dict"])
        opt_gen.load_state_dict(ckpt["opt_gen"])
        disc.load_state_dict(ckpt["disc_state_dict"])
        opt_disc.load_state_dict(ckpt["opt_disc"])

        for pg in opt_gen.param_groups:
            pg["lr"] = lr

        for pg in opt_disc.param_groups:
            pg["lr"] = lr

        print("Restore Ckpt FROM :", ckpt_path)

    @staticmethod
    def model_to_torchscript(model, save_path):
        device = torch.device("cpu")
        model = model.to(device).eval()
        model_ts = torch.jit.script(model)
        model_ts.save(save_path)

    def logging_weight_and_gradient(
        self,
        model_name: str,
        model: nn.Module,
        itr: int,
        weight=True,
        gradient=True,
    ):
        """logging gradients
        Args:
            model_name (str): Model name ex) Gen/Disc
            model (nn.Module): target Model
            itr (int): logging itr/step
            weight (bool, optional): [bool]. Defaults to True. weight logging flag
            gradient (bool, optional): [bool]. Defaults to True. grad logging flag
        """
        if weight == False and gradient == False:
            return -1

        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace(".weight", "")
            g = p.grad
            try:
                if weight:
                    self.tb.add_histogram(f"{model_name}_w/{i}_{name}", p, itr)
                if gradient:
                    self.tb.add_histogram(f"{model_name}_g/{i}_{name}", g, itr)
            except ValueError:
                pass

    @torch.no_grad()
    def logging_scaler(self, metrics, reset=True):
        data = metrics.result_and_reset() if reset else metrics.result()
        self.tb.add_scalar(metrics.name, data, self.itr)
        return data

    @staticmethod
    def generate_latent_vactor(batch: int, dim: int, device: torch.device):


class Metrics:
    """mean metrics class for logging scaler"""

    __slots__ = ["name", "_container"]

    def __init__(self, name: str):
        """metrics initializer
        Args:
            name (str): metrics object name tag

        container is float list
        """
        self.name = name
        self._container = list()

    def reset_state(self):
        """clear container"""
        self._container.clear()

    def update_state(self, data: any):
        """append data in container

        Args:
            data (torch.Tensor | float): logging data
        """
        if isinstance(data, torch.Tensor):
            data = data.item()
        self._container.append(data)

    def result(self) -> float:
        """calculate container and return

        Returns:
            [float]: calculated(mean) logging data
        """
        return np.mean(self._container)

    def result_and_reset(self) -> float:
        """calculate container and reset

        Returns:
            [float]: calculated(mean) logging data
        """
        data = self.result()
        self.reset_state()
        return data
