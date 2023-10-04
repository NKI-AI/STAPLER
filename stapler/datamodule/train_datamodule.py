from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, WeightedRandomSampler

from stapler.datamodule.components.tokenizers import Tokenizer
from stapler.datamodule.components.train_dataset import STAPLERDataset
from stapler.datamodule.dataloader.general_dataloader import create_dataloader


# A minimialistic lightning datamodule for the training
class TrainDataModule(LightningDataModule):
    def __init__(
        self,
        train_data_path: Union[str, Path],
        test_data_path: Union[str, Path],
        tokenizer: Tokenizer,
        transform: Optional[Any] = None,
        padder: Optional[Any] = None,
        fold: Optional[int] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        weighted_class_sampling: bool = False,
        weighted_epitope_sampling: bool = False,
    ) -> None:
        super().__init__()

        # Save the parameters
        self.save_hyperparameters()

        self.train_data_path = Path(train_data_path)
        self.test_data_path = Path(test_data_path)
        self.transform = transform
        self.tokenizer = tokenizer
        self.padder = padder
        self.fold = fold

    def setup(self, stage: Optional[str] = None) -> None:

        if self.fold is not None:
            self.val_data_path_fold = self.train_data_path.parent / f"{self.train_data_path.stem.strip('.csv')}_val-fold{self.fold}.csv"
            self.train_data_path_fold = self.train_data_path.parent / f"{self.train_data_path.stem.strip('.csv')}_train-fold{self.fold}.csv"

            self.val_dataset = STAPLERDataset(
                self.val_data_path_fold,
                self.tokenizer,
                None,
                self.padder,
            )

            self.train_dataset = STAPLERDataset(
                self.train_data_path_fold,
                self.tokenizer,
                self.transform,
                self.padder,
            )

        else:
            self.train_dataset = STAPLERDataset(
                self.train_data_path,
                self.tokenizer,
                self.transform,
                self.padder,
            )

        self.test_dataset = STAPLERDataset(
            self.test_data_path,
            self.tokenizer,
            None,
            self.padder,
        )

    def train_dataloader(self) -> DataLoader:
        train_dataloader = create_dataloader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            weighted_class_sampling=self.hparams.weighted_class_sampling,
            weighted_epitope_sampling=self.hparams.weighted_epitope_sampling
        )
        return train_dataloader

    def predict_dataloader(self) -> DataLoader:
        predict_dataloader = create_dataloader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            weighted_class_sampling=False,
            weighted_epitope_sampling=False
        )
        return predict_dataloader

    def val_dataloader(self) -> DataLoader | None:
        if self.fold is not None:
            # create a validation data_loader using the same parameters as the test data_loader
            val_dataloader = create_dataloader(
                dataset=self.val_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                persistent_workers=self.hparams.num_workers > 0,
                weighted_class_sampling=False,
                weighted_epitope_sampling=False
            )
            return val_dataloader
        else:
            return None
