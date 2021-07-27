import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = 'dataset/'
SEED = 2333
LR = 2e-4
Z_DIM = 64
CHANNELS_IMAGE = 3
IMAGE_DIM = 28 * 28 * 1  # 784
BATCH_SIZE = 64
NUM_EPOCHS = 50
LOG_INTERVAL = 10
