# TODO: make this an actual model, not pl.module
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn as nn
from torch.nn import functional as F

from stapler.utils.model_utils import CosineWarmupScheduler


class FFWD(pl.LightningModule):
    def __init__(
        self,
        model,
        input_size,
        hidden_size,
        num_classes,
        learning_rate,
        # batch_size,
        token_dim,
        max_epochs,
        num_hidden_layers,
        dropout,
    ):
        super().__init__()

        self.model = model
        self.input_size = input_size
        self.token_dim = token_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.max_epochs = max_epochs

        # create an embedding layer
        self.token_emb = nn.Embedding(self.token_dim, self.token_dim)

        # create a fully connected layer
        # first hidden layer (after flattening the input)
        self.l1 = nn.Linear(self.token_dim * self.input_size, self.hidden_size)

        # loop over cfg.num_layers to create the layers 2 until cfg.num_layers
        # and store them in self.linears
        self.linears = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_hidden_layers)]
        )

        # output layer
        self.last_layer = nn.Linear(self.hidden_size, self.num_classes)
        # self.reduce = Reduce(' b l h -> b h', 'mean')

        self.cls_criterion = nn.CrossEntropyLoss()

        # lightning metrics
        metrics_train = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(threshold=0.5),
                torchmetrics.AveragePrecision(pos_label=1),
                torchmetrics.AUROC(pos_label=1),
                torchmetrics.F1Score(threshold=0.5),
            ]
        )

        metrics_val = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(threshold=0.5),
                torchmetrics.AveragePrecision(pos_label=1),
                torchmetrics.AUROC(pos_label=1),
                torchmetrics.F1Score(threshold=0.5),
            ]
        )

        self.train_metrics = metrics_train.clone(prefix="train_")
        self.valid_metrics = metrics_val.clone(prefix="val_")

    def init_(self):
        nn.init.kaiming_normal_(self.token_emb.weight)

    def forward(self, batch):
        if len(batch) == 2:
            inputs, labels = batch
            epitopes_batch = []
            reference_id = []
        elif len(batch) == 4:
            inputs, labels, epitopes_batch, reference_id = batch
        else:
            raise ValueError(
                "Wrong input size (batch should contain 2 (inputs, labels) or 3 elements (inputs, labels, epitopes)"
            )

        # print(sum(labels)/len(labels))

        # embed the inputs
        out = self.token_emb(inputs)
        # flatten all dimensions except the batch dimension (0)
        out = torch.flatten(out, 1)

        # linear layers from input*token_dim to hidden_size
        out = F.relu(self.l1(out))
        # loop over cfg.num_layers until cfg.num_layers
        for i, l in enumerate(self.linears):
            # x = self.linears[i](x) + l(x)
            out = F.relu(self.linears[i](out))

        # dropout before the output layer
        out = F.dropout(out, p=self.dropout, training=self.training)
        # last linear layer from hidden_size to num_classes
        out = F.relu(self.last_layer(out))

        loss = self.cls_criterion(out, torch.squeeze(labels).long())

        # predictions and accuracy

        true_labels = torch.squeeze(labels).int().detach()
        # softmax over the cls_logit
        cls_logit_softmax = torch.nn.functional.softmax(out, dim=-1)
        # extract the postive cls_logit
        cls_logit_pos = cls_logit_softmax[:, 1]

        return {
            "loss": loss,
            "preds": cls_logit_pos,
            "target": true_labels,
            "epitopes": epitopes_batch,
            "reference_id": reference_id,
        }

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        self.log(
            "train_loss",
            outputs["loss"],
            batch_size=len(batch[0]),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

        output = self.train_metrics(outputs["preds"], outputs["target"])
        self.log_dict(output, batch_size=len(batch[0]))

        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        self.log(
            "val_loss",
            outputs["loss"],
            batch_size=len(batch[0]),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        output = self.valid_metrics(outputs["preds"], outputs["target"])
        self.log_dict(output, batch_size=len(batch[0]), logger=True, on_epoch=True)

    def configure_optimizers(self):
        # optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # return optim

        optim = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01
        )  # 0.0001 LEARNING_RATE)
        # https://pytorch-lightning-bolts.readthedocs.io/en/latest/learning_rate_schedulers.html
        # linear warmup and cosine decay
        scheduler = CosineWarmupScheduler(optimizer=optim, warmup=10, max_iters=self.max_epochs)

        return [optim], [scheduler]
