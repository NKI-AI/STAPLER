from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Union

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy


class LossFactory(nn.Module):
    """Loss factory to construct the total loss."""

    def __init__(
        self,
        losses: list,
    ):
        """
        Parameters
        ----------
        losses : list
            List of losses which are functions which accept `(input, batch, weight)`. batch will be a dict(str,Any) containing
            for instance the labels and any other needed data. The weight will be applied per loss.
        """
        super().__init__()

        self._losses = []
        for loss in losses:
            self._losses += list(loss.values())

        self._weights = [torch.tensor(loss.weight) for loss in self._losses]

    def forward(self, input: torch.Tensor, batch: dict[str, Any]):
        total_loss = sum(
            [
                weight.to(batch["input"].device) * curr_loss(input, batch)
                for weight, curr_loss in zip(self._weights, self._losses)
            ]
        )
        return total_loss


# abstract class for losses
class Loss(ABC):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    @abstractmethod
    def __call__(self, input: torch.Tensor, batch: dict[str, Any]):
        pass


class MLMLoss(Loss):
    def __init__(self, weight: float = 1.0, pad_token_id: int = 0):
        super().__init__(weight)
        self.mlm_loss = cross_entropy
        self.pad_token_id = pad_token_id

    def __call__(self, input: dict[str, torch.Tensor], batch: dict[str, Any]):
        pred_mlm = input["mlm_logits"].transpose(1, 2)
        loss = self.mlm_loss(pred_mlm, batch["mlm_labels"], ignore_index=self.pad_token_id)
        return loss


class CLSLoss(Loss):
    def __init__(self, weight: float = 1.0):
        super().__init__(weight)
        self.cls_loss = nn.CrossEntropyLoss(reduction="mean")

    def __call__(self, input: dict[str, torch.Tensor], batch: dict[str, Any]):
        pred_cls = input["cls_logits"]
        loss = self.cls_loss(pred_cls, batch["cls_labels"])  #  torch.squeeze(batch['cls_label']).long()
        return loss


# class TrainLoss(torch.nn.Module):
#     def __init__(self,
#                  hidden_dim: int,
#                  num_tokens: int,
#                  cls_dropout: float,
#                  pad_token_id: int,
#                  mlm_loss_weight: float,
#                  cls_loss_weight: float
#                  ):
#         super().__init__()  # TODO: do i need to call super here?
#
#         # TODO: Would this be more elegant using hparams? The paramaters are somewhat random and not really related to each other
#         #  ; hidden dim is a model parameter, num_tokens is a data parameter, mlm_loss_weight and cls_loss_weight depend on cdr3 or full-seq
#         # TODO: Problem arises as the hidden dim and num_tokens are in the medium_model.yaml (transformer config)
#         self.hidden_dim = hidden_dim
#         self.num_tokens = num_tokens
#         self.cls_dropout = cls_dropout
#         self.pad_token_id = pad_token_id
#         self.mlm_loss_weight = mlm_loss_weight
#         self.cls_loss_weight = cls_loss_weight
#
#         # TODO: do we want to have the logit transform inside the loss?
#         self.to_logits = nn.Linear(self.hidden_dim, self.num_tokens)
#         self.to_cls = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
#                                     nn.Tanh(),
#                                     nn.Dropout(self.cls_dropout),
#                                     nn.Linear(self.hidden_dim, 2))
#
#         self.cls_loss = torch.nn.CrossEntropyLoss()
#         self.mlm_loss = torch.nn.CrossEntropyLoss()
#
#     def __call__(self,
#                  x: torch.Tensor,
#                  batch: dict(str, Any)
#                  ):
#         cls_logit = self.to_cls(x[:, 0, :])
#         logits = self.to_logits(x)
#         cls_loss = self.cls_loss(cls_logit, torch.squeeze(batch['cls_label']).long())
#         mlm_loss = self.mlm_loss(logits.transpose(1, 2),
#                                  batch['mlm_labels'],
#                                  ignore_index=self.pad_token_id
#                                  )
#         weighted_loss = mlm_loss * self.mlm_loss_weight + cls_loss * self.cls_loss_weight
#
#         return weighted_loss
#
#
# class PretrainLoss(torch.nn.Module):
#     def __init__(self,
#                  hidden_dim: int,
#                  num_tokens: int,
#                  pad_token_id: int,
#                  ):
#         super().__init__()
#
#         self.hidden_dim = hidden_dim
#         self.num_tokens = num_tokens
#         self.pad_token_id = pad_token_id
#
#         self.mlm_loss = torch.nn.CrossEntropyLoss()
#         self.to_logits = nn.Linear(self.hidden_dim, self.num_tokens)
#
#     def __call__(self,
#                  x: torch.Tensor,
#                  batch: dict(str, Any)
#                  ):
#
#         logits = self.to_logits(x)
#         mlm_loss = self.mlm_loss(logits.transpose(1, 2),
#                                  batch['mlm_labels'],
#                                  ignore_index=self.pad_token_id
#                                  )
#         return mlm_loss
