"""Create a class that masks the input data. In particular the input will be a tokenized sequence of amino acids. The masking will be done by replacing the amino acids with a mask token."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from stapler.transforms.transforms import Transform


class Masking(Transform):
    def __init__(
        self,
        mask_token_id: int,
        pad_token_id: int,
        mask_prob: float,
        replace_prob: float,
        mask_ignore_token_ids: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.mask_ignore_token_ids = mask_ignore_token_ids or []
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob

    def __call__(self, input_dict: dict[str, torch.Tensor]):
        """Mask the input data.

        Args:
            x (torch.Tensor): The input data, consisting of batches of tokenized sequences.

        Returns:
            tuple: A tuple containing masked_input, labels, and mask_indices.
        """
        input_dict["original_input"] = input_dict["input"].clone()
        x = input_dict["input"]
        x = x.clone()
        mask = torch.rand(x.shape) < self.mask_prob
        mask = mask.to(x.device)

        # Exclude [PAD] token and tokens in mask_ignore_token_ids from being masked
        for token_id in [self.pad_token_id] + self.mask_ignore_token_ids:
            token_id = torch.tensor(token_id, device=x.device)
            mask &= x != token_id

        # Replace tokens with [MASK] based on replace_prob
        replace = torch.rand(x.shape) < self.replace_prob
        replace = replace.to(x.device)
        masked_input = x.clone()
        masked_input[mask & replace] = self.mask_token_id

        # Create labels tensor with [PAD] token for unmasked positions
        labels = x.clone()
        labels[~mask] = self.pad_token_id

        # Get mask_indices
        mask_indices = torch.nonzero(mask, as_tuple=True)

        input_dict["input"] = masked_input
        input_dict["mlm_labels"] = labels
        input_dict["mlm_mask_indices"] = mask_indices

        return input_dict

