from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Optimizer

# import torchmetrics accurcy
from torchmetrics import Accuracy, AveragePrecision

from stapler.transforms.transforms import Transform
from stapler.utils.io_utils import get_pylogger

logger = get_pylogger(__name__)


class STAPLERLitModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        transforms: dict[str, Transform],
        loss: nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["model"])

        self.model = model
        self.transforms = transforms
        self.loss = loss

        self.optimizer = optimizer
        self.scheduler = scheduler

        # accuracy metric (pretraining and fine-tuning)
        self.train_mlm_acc = Accuracy(task="multiclass", num_classes=25)
        self.val_mlm_acc = Accuracy(task="multiclass", num_classes=25)

    def forward(self, batch: dict[str, torch.Tensor], stage: str, **kwargs):

        if self.transforms and stage in self.transforms:
            batch = self.transforms[stage](batch)

        preds = self.model(batch["input"], return_attn=False, return_embeddings=True, **kwargs)
        loss = self.loss(preds, batch)

        # dict with loss and preds
        output = {"loss": loss, "preds": preds}
        return output

    def training_step(self, batch, batch_idx):
        output_dict = self.forward(batch, stage="fit")
        loss = output_dict["loss"]
        preds = output_dict["preds"]
        batch_size = batch["input"].shape[0]

        # log
        labels_mlm, preds_mlm = self.extract_mlm_labels_and_preds(batch, preds)
        self.train_mlm_acc(labels_mlm, preds_mlm)
        self.log(
            "train_mlm_acc",
            self.train_mlm_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        return loss


    def extract_mlm_labels_and_preds(self, batch, preds):
        mlm_preds = preds["mlm_logits"]
        mlm_labels = batch["mlm_labels"]
        mlm_mask_indices = batch["mlm_mask_indices"]

        # Get the predicted token indices
        preds_indices = torch.argmax(mlm_preds, dim=-1)

        # Extract the labels and predicted tokens for masked positions
        masked_labels = mlm_labels[mlm_mask_indices]
        masked_preds = preds_indices[mlm_mask_indices]

        return masked_labels, masked_preds

    def configure_optimizers(self):
        optim = self.optimizer(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optim)
            return [optim], [scheduler]
        else:
            return optim
