import torch
from dataset import *

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 2333
RESTORE_CKPT_PATH = None


class DATASET_MAP:
    DS = MapDataset
    DIR = "../DATASET/Pix2Pix/maps/train"
    VAL_DIR = "../DATASET/Pix2Pix/maps/val"
    IMAGE_SIZE = 256
    IMAGE_CHANNELS = 3
    INPUT_CHANNELS = 3


class DATASET_ANIME(DATASET_MAP):
    DS = AnimeDataset
    DIR = "../DATASET/shuushuu/train"
    VAL_DIR = "../DATASET/shuushuu/val"
    INPUT_CHANNELS = 1


class DATASET_BW(DATASET_ANIME):
    DS = BlackWhiteDataset


DATASET = DATASET_BW
IMAGE_SIZE = DATASET.IMAGE_SIZE
IMAGE_CHANNELS = DATASET.IMAGE_CHANNELS
INPUT_CHANNELS = DATASET.INPUT_CHANNELS
NUM_WORKERS = 16

GEN_DIM = 64
DISC_DIM = 64

BATCH_SIZE = 16
SAMPLE_SIZE = 4
LEARNING_RATE = 2e-4
BETAS = (0.5, 0.999)

NUM_EPOCHS = 100
LAMBDA_L1 = 100

LOG_INTERVAL = 10
TEST_INTERVAL = 100
