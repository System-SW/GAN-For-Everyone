import os

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2 as ToTensor
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader


class HoreZebraDataset(data.Dataset):
    def __init__(self, dataset) -> None:
        self.zebra_image_paths = []
        self.horse_image_paths = []
        for root, _, files in os.walk(os.path.join(dataset.DIR, 'Horse')):
            for file in files:
                self.horse_image_paths.append(os.path.join(root, file))

        for root, _, files in os.walk(os.path.join(dataset.DIR, 'Zebra')):
            for file in files:
                self.zebra_image_paths.append(os.path.join(root, file))

        self.horse_data_size = len(self.horse_image_paths)
        self.zebra_data_size = len(self.zebra_image_paths)
        self.datasize = max(self.horse_data_size,
                            self.zebra_data_size)

        self.both_transfrom = A.Compose(
            [
                A.Resize(dataset.IMAGE_SIZE,
                         dataset.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.5 for _ in range(dataset.INPUT_CHANNELS)],
                            std=[0.5 for _ in range(dataset.INPUT_CHANNELS)],
                            max_pixel_value=255.0),
                ToTensor()
            ],
            additional_targets={'image0': 'image'}
        )

    def __len__(self):
        return self.datasize

    def __getitem__(self, index):
        # get PATH
        horse_path = self.horse_image_paths[index % self.horse_data_size]
        zebra_path = self.zebra_image_paths[index % self.zebra_data_size]

        # image load
        horse = Image.open(horse_path).convert('RGB')
        zebra = Image.open(zebra_path).convert('RGB')

        # preprocessing
        augmentations = self.both_transfrom(
            image0=np.array(horse),
            image=np.array(zebra))

        # return Horse, Zebra
        return augmentations['image'], augmentations['image0']


class Dataset:
    def __init__(self, dataset) -> None:
        DATASET = {
            'HZ': HoreZebraDataset
        }

        print(f'{dataset.CAT.upper()} Dataset')

        self._dataset_train = DATASET[dataset.CAT](dataset)
        dataset.DIR = dataset.VAL_DIR
        self._dataset_val = DATASET[dataset.CAT](dataset)

    def create_dataloader(self, batch_size: int, num_workers=16):
        return (DataLoader(self._dataset_train,
                           batch_size=batch_size,
                           shuffle=True, num_workers=num_workers),
                DataLoader(self._dataset_val,
                           batch_size=batch_size,
                           shuffle=False))


if __name__ == "__main__":
    import torch
    from torchvision.utils import make_grid, save_image

    import hyperparameters as hp

    size = 32
    hp.DATASET = hp.DATASET_HZ
    hp.BATCH_SIZE = size

    dataloader, _ = Dataset(hp.DATASET).create_dataloader(hp.BATCH_SIZE)

    for horse, zebra in dataloader:
        print('horse', horse.shape)
        print('zebra', zebra.shape)

        horse = make_grid(horse, size, normalize=True)
        zebra = make_grid(zebra, size, normalize=True)

        image = torch.cat([horse, zebra], dim=1)
        save_image([image], 'temp.jpg')

        break
