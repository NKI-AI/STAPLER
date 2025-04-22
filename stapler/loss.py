# Copyright 2023 Schumacher Lab. All Rights Reserved.
# Copyright 2023 AI for Oncology Research Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
        loss = self.cls_loss(pred_cls, batch["cls_labels"])
        return loss
