import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Dataset:
    def __init__(self, dataset) -> None:
        transform = transforms.Compose(
            [
                transforms.Resize(dataset.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5 for _ in range(dataset.IMAGE_CHANNELS)],
                    [0.5 for _ in range(dataset.IMAGE_CHANNELS)]
                ),
            ]
        )

        if dataset.CAT == 'mnist':
            ds = datasets.MNIST
            print('MNIST Dataset')
        elif dataset.CAT == 'celaba':
            ds = datasets.CelebA
            print('celabA Dataset')
        else:
            ds = datasets.MNIST
            print('MNIST Dataset')

        self._dataset = ds(
            root=dataset.DIR,
            transform=transform,
            download=True)

    def create_dataloader(self, batch_size: int):
        return DataLoader(self._dataset,
                          batch_size=batch_size,
                          shuffle=True)