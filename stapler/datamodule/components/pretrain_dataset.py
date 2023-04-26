from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from stapler.datamodule.components.tokenizers import Tokenizer


class PretrainDatasetTcrEpitope(Dataset):
    def __init__(
        self,
        tcrs_path: Union[str, Path],
        epitopes_path: Union[str, Path],
        tokenizer: Tokenizer,
        transform: Optional[Any] = None,
        padder: Optional[Any] = None,
    ) -> None:

        tcr_data = pd.read_csv(tcrs_path)
        epitope_data = pd.read_csv(epitopes_path)

        self.tcr_df = tcr_data[["cdr3_alpha_aa", "cdr3_beta_aa"]]
        self.epitope_df = epitope_data["epitope_aa"]

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
            self.tcr_df["cdr3_alpha_aa"].str.len().max()
            + self.tcr_df["cdr3_beta_aa"].str.len().max()
            + self.epitope_df.str.len().max()
            + 3
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tcr_a, tcr_b = self.tcrs[index]
        # Create random index for epitope
        epitope = self.epitopes[np.random.randint(0, len(self.epitopes))]  # TODO check randomness per epoch

        sample = "[CLS] " + " ".join(tcr_a) + " [SEP] " + " ".join(epitope) + " [SEP] " + " ".join(tcr_b)
        sample = self.tokenizer.encode(sample)
        sample = torch.from_numpy(np.asarray(sample))

        if self.transform:
            sample = self.transform(sample)
        if self.padder:
            sample = self.padder(sample)

        output_dict = {"input": sample}
        return output_dict

    def __len__(self) -> int:
        return len(self.tcrs)
