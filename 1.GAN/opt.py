import os
import time
from abc import ABCMeta, abstractmethod

import torch
from torch import nn
from torch.utils import tensorboard


class Template(metaclass=ABCMeta):
    """ Abstract Class for Trainer """

    def __init__(self,
                 hp: dict,
                 model_name: str):
        """ 
        Args:
            hp (dict): hyperparameters for train
            model_name (str): train model name (for log dir name)
        """

        self.itr = 0
        self.device = hp.DEVICE
        self.gan_prob = lambda x: torch.mean(torch.sigmoid(x))

        # Set Training Seed
        torch.manual_seed(hp.SEED)
        torch.cuda.manual_seed(hp.SEED)
        torch.backends.cudnn.benchmark = True

        # Create TensorBoard Dir
        start_time = time.strftime(
            '%Y-%m-%d-%H:00',
            time.localtime(time.time()))
        logdir = os.path.join(
            'log', model_name, start_time)
        self.tb = tensorboard.SummaryWriter(logdir)
        os.makedirs(self.tb.log_dir, exist_ok=True)
        os.makedirs(os.path.join(logdir, 'image'),
                    exist_ok=True)
        os.makedirs(os.path.join(logdir, 'ckpt'),
                    exist_ok=True)

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def test(self):
        ...
