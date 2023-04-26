from typing import List, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class ExampleDataset(Dataset):
    def __init__(self, output_shape: List[int], num_samples: int = 100):
        self.output_shape = list(output_shape)
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        rand_tens = torch.rand(self.output_shape)
        rand_lab = torch.zeros(1000)
        rand_loc = torch.randint(10, (1,))
        rand_lab[rand_loc] = 1
        return rand_tens, rand_lab


class ExampleDataModule(pl.LightningDataModule):
    def __init__(self, output_shape: List[int], batch_size: int = 10, num_samples: int = 100):
        super().__init__()
        self.batch_size = batch_size

        self._training_data: Optional[Dataset] = ExampleDataset(output_shape=output_shape, num_samples=num_samples)
        self._val_data: Optional[Dataset] = ExampleDataset(output_shape=output_shape, num_samples=num_samples)
        self._testing_data: Optional[Dataset] = ExampleDataset(output_shape=output_shape, num_samples=num_samples)
        self._predict_data: Optional[Dataset] = ExampleDataset(output_shape=output_shape, num_samples=num_samples)

    def train_dataloader(self):
        if self._training_data:
            return DataLoader(self._training_data, batch_size=self.batch_size)

    def val_dataloader(self):
        if self._val_data:
            return DataLoader(self._testing_data, batch_size=self.batch_size)

    def test_dataloader(self):
        if self._testing_data:
            return DataLoader(self._testing_data, batch_size=self.batch_size)

    def predict_dataloader(self):
        if self._predict_data:
            return DataLoader(self._predict_data, batch_size=self.batch_size)
