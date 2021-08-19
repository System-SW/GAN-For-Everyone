import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader


class CelabADataset(data.Dataset):
    def __init__(self, dataset, ext_cndidates=["jpg", "png", "bmp"]):

        self.image_paths = []

        for root, dirs, files in os.walk(dataset.DIR):
            for filename in files:
                if os.path.splitext(filename)[1].lower()[1:] in ext_cndidates:
                    self.image_paths.append(os.path.join(root, filename))

        self.transform = transforms.Compose(
            [
                transforms.Resize((dataset.IMAGE_SIZE, dataset.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5 for _ in range(dataset.IMAGE_CHANNELS)],
                    [0.5 for _ in range(dataset.IMAGE_CHANNELS)],
                ),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        image = image.convert("RGB")
        image = self.transform(image)
        return image, 0


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
        super().__init__(dataset.DIR, transform=transform, download=True)


class Dataset:
    def __init__(self, dataset) -> None:
        DATASET = {
            "mnist": MnistDataset,
            "celaba": CelabADataset,
        }

        print(f"{dataset.CAT.upper()} Dataset")
        self._dataset_train = DATASET[dataset.CAT](dataset)
        self._dataset_val = DATASET[dataset.CAT](dataset)

    def create_dataloader(self, batch_size: int):
        return DataLoader(self._dataset_train, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    import DCGAN.hyperparameters as hp

    dataloader = Dataset(hp.DATASET).create_dataloader(hp.BATCH_SIZE)

    for image, _ in dataloader:
        print(image.shape)
