import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Dataset:
    def __init__(self, dataset_dir) -> None:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))]
        )
        self._dataset = datasets.MNIST(
            root=dataset_dir, transform=transform, download=True)

    def create_dataloader(self, batch_size: int):
        return DataLoader(self._dataset,
                          batch_size=batch_size,
                          shuffle=True)
