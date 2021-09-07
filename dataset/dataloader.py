# Copyright (C) 2021 yslee
#
# This file is part of GAN_For_Everyone.
#
# GAN_For_Everyone is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GAN_For_Everyone is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GAN_For_Everyone.  If not, see <http://www.gnu.org/licenses/>.

from torch.utils import data


class DataLoader:
    def __init__(self, dataset) -> None:
        print(f"Select {dataset.DS.__name__}")

        self._dataset_train = dataset.DS(dataset)

        if hasattr(dataset, "VAL_DIR"):
            dataset.DIR = dataset.VAL_DIR
            self._dataset_val = dataset.DS(dataset)
        else:
            self._dataset_val = None

        if hasattr(dataset, "VAL_DIR"):
            dataset.DIR = dataset.VAL_DIR
            self._dataset_test = dataset.DS(dataset)
        else:
            self._dataset_test = None

    def create_dataloader(
        self,
        batch_size: int,
        sample_size,
        num_warkers=16,
    ):
        return (
            self.create_train_data_loader(batch_size, num_warkers),
            self.create_val_data_loader(sample_size, 1),
        )

    def create_train_data_loader(self, batch_size: int, num_warkers: int = 16):
        return data.DataLoader(
            self._dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_warkers,
        )

    def create_val_data_loader(self, batch_size: int, num_warkers: int):
        if self._dataset_val is None:
            return None
        return data.DataLoader(
            self._dataset_val,
            batch_size=batch_size,
            num_workers=num_warkers,
        )

    def create_test_data_loader(self, batch_size: int, num_warkers: int):
        if self._dataset_val is None:
            return None

        return data.DataLoader(
            self._dataset_test,
            batch_size=batch_size,
            num_workers=num_warkers,
        )
