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
