import os

import albumentations as A
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2 as ToTensor
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader


class CelebADataset(data.Dataset):
    def __init__(self, dataset, ext_candidates=["jpg", "png", "bmp"]):
        self.image_paths = []
        for root, dirs, files in os.walk(dataset.DIR):
            for filename in files:
                if os.path.splitext(filename)[1].lower()[1:] in ext_candidates:
                    self.image_paths.append(os.path.join(root, filename))

        self.transform = A.Compose(
            [
                A.RandomCrop(width=dataset.IMAGE_SIZE, height=dataset.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        )
        lr_size = dataset.IMAGE_SIZE // 4
        self.lr_transform = A.Compose(
            [
                A.Resize(lr_size, lr_size, interpolation=Image.BICUBIC),
                A.Normalize(
                    mean=[0.0 for _ in range(dataset.INPUT_CHANNELS)],
                    std=[1.0 for _ in range(dataset.INPUT_CHANNELS)],
                ),
                ToTensor(),
            ]
        )
        self.hr_transform = A.Compose(
            [
                A.Normalize(
                    mean=[0.5 for _ in range(dataset.OUTPUT_CHANNELS)],
                    std=[0.5 for _ in range(dataset.OUTPUT_CHANNELS)],
                ),
                ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        image = np.array(image)
        image = self.transform(image=image)["image"]
        hr_image = self.hr_transform(image=image)["image"]
        lr_image = self.lr_transform(image=image)["image"]
        return lr_image, hr_image


class Dataset:
    def __init__(self, dataset) -> None:
        DATASET = {
            "celebs": CelebADataset,
        }

        print(f"{dataset.CAT.upper()} Dataset")
        self._dataset_train = DATASET[dataset.CAT](dataset)

    def create_dataloader(self, batch_size: int):
        return DataLoader(self._dataset_train, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    import hyperparameters as hp

    dataloader = Dataset(hp.DATASET).create_dataloader(hp.BATCH_SIZE)

    for lr, hr in dataloader:
        print(lr.shape)
        print(hr.shape)
        break
