import torch

from dataset import *
from dataset.dataset import CelebaHQ

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 2333


class DATASET_CELEBAHQ:
    DS = CelebaHQ
    DIR = "../DATASET/celeba_hq"
    IMAGE_SIZE = None
    IMAGE_CHANNELS = 3


DATASET = DATASET_CELEBAHQ
IMAGE_SIZE = DATASET.IMAGE_SIZE
IMAGE_CHANNELS = DATASET.IMAGE_CHANNELS

START_TRAIN_AT_IMG_SIZE = 4
NOISE_DIM = 512
SAMPLE_SIZE = 10 ** 2
GEN_DIM = 512
DISC_DIM = 512
BATCH_SIZES = [32, 32, 32, 16, 16, 16, 16, 8, 4]
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)
LAMBDA_GP = 10

LEARNING_RATE = 1e-3
BETAS = (0.0, 0.99)

LOG_INTERVAL = 5
TEST_INTERVAL = 200
