import os
import re
import time
from abc import ABCMeta, abstractmethod

import torch
from torch import nn
from torch.utils import tensorboard


class Template(metaclass=ABCMeta):
    """Abstract Class for Trainer"""

    def __init__(self, **kwargs):
        """
        Args:
            hp (dict): hyperparameters for train
            model_name (str): train model name (for log dir name)
        """

        self.itr = 0
        self.device = kwargs["device"]
        self.gan_prob = lambda x: torch.mean(torch.sigmoid(x))

        # Set Training Seed
        torch.manual_seed(kwargs["seed"])
        torch.cuda.manual_seed(kwargs["seed"])
        torch.backends.cudnn.benchmark = True

        # Create TensorBoard Dir
        start_time = time.strftime("%Y-%m-%d-%H:00", time.localtime(time.time()))
        logdir = os.path.join("log", kwargs["model_name"], start_time)
        self.tb = tensorboard.SummaryWriter(logdir)
        self.image_dir = os.path.join(logdir, "image")
        os.makedirs(self.tb.log_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def test(self):
        ...

    def save_checkopint(self, gen, disc, opt_gen, opt_disc, epoch=0):

        checkpoint = {
            "gen_state_dict": gen.state_dict(),
            "opt_gen": opt_gen.state_dict(),
            "disc_state_dict": disc.state_dict(),
            "opt_disc": opt_disc.state_dict(),
        }
        save_path = os.path.join("checkpoint.pth.tar")
        torch.save(checkpoint, save_path)
        print(f"Epoch:{epoch+1} ckpt save => {save_path}")

    def restore_checkopint(self, cktp_path, gen, disc, opt_gen, opt_disc, lr):

        if cktp_path is None:
            return -1

        if not os.path.exists(cktp_path):
            return -1

        ckpt = torch.load(cktp_path, map_location=self.device)

        gen.load_state_dict(ckpt["gen_state_dict"])
        opt_gen.load_state_dict(ckpt["opt_gen"])
        disc.load_state_dict(ckpt["disc_state_dict"])
        opt_disc.load_state_dict(ckpt["opt_disc"])

        for pg in opt_gen.param_groups:
            pg["lr"] = lr

        for pg in opt_disc.param_groups:
            pg["lr"] = lr

        print("Restore Ckpt FROM :", cktp_path)
