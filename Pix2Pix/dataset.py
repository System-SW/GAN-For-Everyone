import os

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2 as ToTensor
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader


class MapDataset(data.Dataset):
    def __init__(self, dataset) -> None:
        self.root_dir = dataset.DIR
        self.image_paths = []

        for root, dirs, files in os.walk(dataset.DIR):
            for filename in files:
                self.image_paths.append(os.path.join(root, filename))

        self.both_transform = A.Compose(
            [A.Resize(dataset.IMAGE_SIZE, dataset.IMAGE_SIZE), A.HorizontalFlip(p=0.5)],
            additional_targets={"image0": "image"},
        )

        self.transform_input = A.Compose(
            [
                A.ColorJitter(p=0.2),
                A.Normalize(
                    mean=[0.5 for _ in range(dataset.INPUT_CHANNELS)],
                    std=[0.5 for _ in range(dataset.INPUT_CHANNELS)],
                    max_pixel_value=255.0,
                ),
                ToTensor(),
            ]
        )

        self.transform_mask = A.Compose(
            [
                A.Normalize(
                    mean=[0.5 for _ in range(dataset.IMAGE_CHANNELS)],
                    std=[0.5 for _ in range(dataset.IMAGE_CHANNELS)],
                    max_pixel_value=255.0,
                ),
                ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        input_image = image.crop((0, 0, 600, 600))
        target_image = image.crop((600, 0, 1200, 600))

        augmentations = self.both_transform(
            image=np.array(input_image), image0=np.array(target_image)
        )

        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = self.transform_input(image=input_image)["image"]
        target_image = self.transform_mask(image=target_image)["image"]

        return input_image, target_image


class AnimeDataset(data.Dataset):
    def __init__(self, dataset) -> None:
        self.root_dir = dataset.DIR

        for root, _, files in os.walk(os.path.join(dataset.DIR, "line")):
            self.line_paths = sorted(
                [os.path.join(root, filename) for filename in files]
            )
        for root, _, files in os.walk(os.path.join(dataset.DIR, "color")):
            self.color_paths = sorted(
                [os.path.join(root, filename) for filename in files]
            )

        self.both_transform = A.Compose(
            [
                A.RandomCrop(dataset.IMAGE_SIZE, dataset.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
            ],
            additional_targets={"image0": "image"},
        )

        self.trainsform_line = A.Compose(
            [
                A.RandomBrightnessContrast(brightness_limit=0.1, p=0.3),
                A.Normalize(
                    mean=[0.5 for _ in range(dataset.INPUT_CHANNELS)],
                    std=[0.5 for _ in range(dataset.INPUT_CHANNELS)],
                    max_pixel_value=255.0,
                ),
                ToTensor(),
            ]
        )
        self.trainsform_color = A.Compose(
            [
                A.ColorJitter(p=0.2),
                A.Normalize(
                    mean=[0.5 for _ in range(dataset.IMAGE_CHANNELS)],
                    std=[0.5 for _ in range(dataset.IMAGE_CHANNELS)],
                    max_pixel_value=255.0,
                ),
                ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.color_paths)

    def __getitem__(self, index):
        line_path = self.line_paths[index]
        color_path = self.color_paths[index]
        line_image = Image.open(line_path).convert("L")
        color_image = Image.open(color_path).convert("RGB")

        augmentations = self.both_transform(
            image=np.array(line_image), image0=np.array(color_image)
        )

        line_image = augmentations["image"]
        color_image = augmentations["image0"]

        line_image = self.trainsform_line(image=line_image)["image"]
        color_image = self.trainsform_color(image=color_image)["image"]

        return line_image, color_image


class BlackWhightDataset(data.Dataset):
    def __init__(self, dataset) -> None:
        for root, _, files in os.walk(os.path.join(dataset.DIR, "color")):
            self.image_paths = sorted(
                [os.path.join(root, filename) for filename in files]
            )

        self.both_transform = A.Compose(
            [
                A.RandomCrop(dataset.IMAGE_SIZE, dataset.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
            ],
            additional_targets={"image0": "image"},
        )

        self.trainsform_input = A.Compose(
            [
                A.RandomBrightnessContrast(brightness_limit=0.1, p=0.3),
                A.Normalize(
                    mean=[0.5 for _ in range(dataset.INPUT_CHANNELS)],
                    std=[0.5 for _ in range(dataset.INPUT_CHANNELS)],
                    max_pixel_value=255.0,
                ),
                ToTensor(),
            ]
        )
        self.trainsform_color = A.Compose(
            [
                A.ColorJitter(p=0.2),
                A.Normalize(
                    mean=[0.5 for _ in range(dataset.IMAGE_CHANNELS)],
                    std=[0.5 for _ in range(dataset.IMAGE_CHANNELS)],
                    max_pixel_value=255.0,
                ),
                ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        color_image = Image.open(image_path).convert("RGB")
        input_image = color_image.copy().convert("L")

        augmentations = self.both_transform(
            image=np.array(input_image), image0=np.array(color_image)
        )

        input_image = augmentations["image"]
        color_image = augmentations["image0"]

        input_image = self.trainsform_input(image=input_image)["image"]
        color_image = self.trainsform_color(image=color_image)["image"]

        return input_image, color_image


class Dataset:
    def __init__(self, dataset) -> None:
        DATASET = {
            "MAP": MapDataset,
            "ANIME": AnimeDataset,
            "BlackWhight": BlackWhightDataset,
        }

        print(f"{dataset.CAT.upper()} Dataset")

        self._dataset_train = DATASET[dataset.CAT](dataset)
        dataset.DIR = dataset.VAL_DIR
        self._dataset_val = DATASET[dataset.CAT](dataset)

    def create_dataloader(self, batch_size: int, num_workers=16):
        return (
            DataLoader(
                self._dataset_train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            ),
            DataLoader(self._dataset_val, batch_size=batch_size, shuffle=False),
        )


if __name__ == "__main__":
    import torch
    from torchvision.utils import make_grid, save_image

    import hyperparameters as hp

    size = 32
    hp.DATASET = hp.DATASET_ANIME
    hp.BATCH_SIZE = size

    dataloader, _ = Dataset(hp.DATASET).create_dataloader(hp.BATCH_SIZE)

    for inputs, targets in dataloader:
        print("inputs", inputs.shape)
        print("target", targets.shape)

        inputs = make_grid(inputs, size, normalize=True)
        targets = make_grid(targets, size, normalize=True)

        image = torch.cat([inputs, targets], dim=1)
        save_image([image], "temp.jpg")

        break
