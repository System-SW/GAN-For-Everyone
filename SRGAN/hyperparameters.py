from pickle import TRUE
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 2333


class DATASET_CELEBS:
    CAT = "celebs"
    DIR = "../dataset/celebA"
    IMAGE_SIZE = 96
    LR_SIZE = 96
    OUTPUT_CHANNELS = 3
    INPUT_CHANNELS = 3


DATASET = DATASET_CELEBS
IMAGE_SIZE = DATASET.IMAGE_SIZE
OUTPUT_CHANNELS = DATASET.OUTPUT_CHANNELS
INPUT_CHANNELS = DATASET.INPUT_CHANNELS
NUM_RESIDUALS = 16
NUM_WORKERS = 16

GEN_DIM = 64
DISC_DIM = 64

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
BETAS = (0.9, 0.999)

NUM_EPOCHS = 100
LAMBDA_ADV = 1e-3
LAMBDA_VGG = 0.006

GAN_MODE = True
GENERATOR_WIGHT_PATH = 'generator_mse.pth'

SAMPLE_SIZE = 8
LOG_INTERVAL = 10
TEST_INTERVAL = 100
CKPT_PATH = None  # U R checkpoint pth path