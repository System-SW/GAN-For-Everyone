import os

import albumentations as A
import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2 as ToTensor
from PIL import Image
from torch.utils import data


class MnistDataset(datasets.MNIST):
    def __init__(self, dataset) -> None:
        transform = transforms.Compose(
            [
                transforms.Resize((dataset.IMAGE_SIZE, dataset.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5 for _ in range(dataset.IMAGE_CHANNELS)],
                    [0.5 for _ in range(dataset.IMAGE_CHANNELS)],
                ),
            ]
        )
        super().__init__(dataset.DIR, transform=transform, download=True, train=True)


class Cifar10Dataset(datasets.CIFAR10):
    def __init__(self, dataset) -> None:
        transform = transforms.Compose(
            [
                transforms.Resize((dataset.IMAGE_SIZE, dataset.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5 for _ in range(dataset.IMAGE_CHANNELS)],
                    [0.5 for _ in range(dataset.IMAGE_CHANNELS)],
                ),
            ]
        )
        super().__init__(dataset.DIR, transform=transform, download=True, train=True)


class FashionMNISTDataset(datasets.FashionMNIST):
    def __init__(self, dataset) -> None:
        transform = transforms.Compose(
            [
                transforms.Resize((dataset.IMAGE_SIZE, dataset.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5 for _ in range(dataset.IMAGE_CHANNELS)],
                    [0.5 for _ in range(dataset.IMAGE_CHANNELS)],
                ),
            ]
        )
        super().__init__(dataset.DIR, transform=transform, download=True, train=True)


class CelebADataset(data.Dataset):
    def __init__(self, dataset, ext_candidates=["jpg", "png", "bmp"]):

        self.image_paths = []

        for root, dirs, files in os.walk(dataset.DIR):
            for filename in files:
                if os.path.splitext(filename)[1].lower()[1:] in ext_candidates:
                    self.image_paths.append(os.path.join(root, filename))

        size = dataset.IMAGE_SIZE

        self.transform = A.Compose(
            [
                A.CenterCrop(150, 150, p=1),
                A.Resize(size, size, cv2.INTER_NEAREST),
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
        image = np.array(Image.open(self.image_paths[index]).convert("RGB"))
        image = self.transform(image=image)["image"]
        return image, 0


class CelebADatasetSRGAN(data.Dataset):
    def __init__(self, dataset, ext_candidates=["jpg", "png", "bmp"]):
        self.image_paths = []
        for root, _, files in os.walk(dataset.DIR):
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


class BlackWhiteDataset(data.Dataset):
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


class HorseZebraDataset(data.Dataset):
    def __init__(self, dataset) -> None:
        self.zebra_image_paths = []
        self.horse_image_paths = []
        for root, _, files in os.walk(os.path.join(dataset.DIR, "Horse")):
            for file in files:
                self.horse_image_paths.append(os.path.join(root, file))

        for root, _, files in os.walk(os.path.join(dataset.DIR, "Zebra")):
            for file in files:
                self.zebra_image_paths.append(os.path.join(root, file))

        self.horse_data_size = len(self.horse_image_paths)
        self.zebra_data_size = len(self.zebra_image_paths)
        self.datasize = max(self.horse_data_size, self.zebra_data_size)

        self.both_transfrom = A.Compose(
            [
                A.Resize(dataset.IMAGE_SIZE, dataset.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.5 for _ in range(dataset.INPUT_CHANNELS)],
                    std=[0.5 for _ in range(dataset.INPUT_CHANNELS)],
                    max_pixel_value=255.0,
                ),
                ToTensor(),
            ],
            additional_targets={"image0": "image"},
        )

    def __len__(self):
        return self.datasize

    def __getitem__(self, index):
        # get PATH
        horse_path = self.horse_image_paths[index % self.horse_data_size]
        zebra_path = self.zebra_image_paths[index % self.zebra_data_size]

        # image load
        horse = Image.open(horse_path).convert("RGB")
        zebra = Image.open(zebra_path).convert("RGB")

        # preprocessing
        augmentations = self.both_transfrom(
            image0=np.array(horse), image=np.array(zebra)
        )

        # return Horse, Zebra
        return augmentations["image"], augmentations["image0"]
