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
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from stapler.datamodule.components.tokenizers import Tokenizer


class STAPLERDataset(Dataset):
    def __init__(
        self,
        train_data_path: Union[str, Path],
        tokenizer: Tokenizer,
        transform: Optional[Any] = None,
        padder: Optional[Any] = None,
    ) -> None:
        train_data = pd.read_csv(train_data_path)

        self.tcr_df = train_data[["full_seq_reconstruct_alpha_aa", "full_seq_reconstruct_beta_aa"]]
        self.epitope_df = train_data["epitope_aa"]
        self.labels = train_data["label_true_pair"]

        self.tcrs = self.tcr_df.to_numpy()
        self.epitopes = self.epitope_df.to_numpy()
        self.transform = transform

        self.tokenizer = tokenizer
        self.padder = padder(self.tokenizer.pad_token_id, self.max_seq_len)

    # property for maximum sequence length
    @property
    def max_seq_len(self) -> dict[str, torch.Tensor]:
        # max of tcrs[0] (strings) + max tcrs[1] (strings) + epitope (strings) + 3
        return (
            self.tcr_df["full_seq_reconstruct_alpha_aa"].str.len().max()
            + self.tcr_df["full_seq_reconstruct_beta_aa"].str.len().max()
            + self.epitope_df.str.len().max()
            + 3
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tcr_a, tcr_b = self.tcrs[index]
        epitope = self.epitopes[index]
        label = torch.tensor(int(self.labels[index]))

        sample = "[CLS] " + " ".join(tcr_a) + " [SEP] " + " ".join(epitope) + " [SEP] " + " ".join(tcr_b)
        sample = self.tokenizer.encode(sample)
        sample = torch.from_numpy(np.asarray(sample))

        if self.transform:
            sample = self.transform(sample)
        if self.padder:
            sample = self.padder(sample)

        output_dict = {"input": sample, "cls_labels": label}
        return output_dict

    def __len__(self) -> int:
        return len(self.tcrs)
