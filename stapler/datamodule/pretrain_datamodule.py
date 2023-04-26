from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from stapler.datamodule.components.pretrain_dataset import PretrainDatasetTcrEpitope
from stapler.datamodule.components.tokenizers import Tokenizer


# A minimialistic lightning datamodule for the pretraining
class PretrainDataModuleTcrEpitope(LightningDataModule):
    def __init__(
        self,
        tcrs_path: Union[str, Path],
        epitopes_path: Union[str, Path],
        tokenizer: Tokenizer,
        transform: Optional[Any] = None,
        padder: Optional[Any] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__()

        # Save the parameters
        self.save_hyperparameters()

        self.tcrs_path = tcrs_path
        self.epitopes_path = epitopes_path
        self.transform = transform
        self.tokenizer = tokenizer
        self.padder = padder

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = PretrainDatasetTcrEpitope(
            self.tcrs_path,
            self.epitopes_path,
            self.tokenizer,
            self.transform,
            self.padder,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=self.hparams.pin_memory,
        )
