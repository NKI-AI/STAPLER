"""Contains the TransformFactory class, which is used to instantiate the correct transform"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch


# Abstract class for the transforms
class Transform(ABC):
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self, data: torch.Tensor) -> Any:
        pass


# Class for padding the sequences to a fixed length
class PadSequence(Transform):
    def __init__(self, pad_token_id: int, max_seq_len: int) -> None:
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # Pad the sequences to the max length
        data = torch.nn.functional.pad(data, (0, self.max_seq_len - data.shape[0]), "constant", self.pad_token_id)
        return data


class TransformFactory:
    def __init__(self, transforms: list, **kwargs) -> None:
        self.transforms = []
        for transform in transforms:
            self.transforms.append(transform(**kwargs))

    def __call__(self, data: torch.Tensor) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data
