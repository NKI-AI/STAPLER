import math
from functools import reduce

import torch


class InputMasker:
    def __init__(self, mask_prob, replace_prob, mask_token_id, pad_token_id, mask_ignore_token_ids):
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.mask_ignore_token_ids = mask_ignore_token_ids

    def mask_input(self, input):
        # do not mask [PAD] tokens, or any other tokens in the tokens designated to be excluded ([CLS], [SEP])
        no_mask = mask_with_tokens(input, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)

        # get mask indices
        mask_indices = torch.nonzero(mask, as_tuple=True)

        # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
        masked_input = input.clone().detach()

        # [mask] input
        replace_prob = prob_mask_like(input, self.replace_prob)
        masked_input = masked_input.masked_fill(mask * replace_prob, self.mask_token_id)

        # mask out any tokens to padding tokens that were not going to be masked
        labels = input.masked_fill(~mask, self.pad_token_id)

        return masked_input, labels, mask_indices

    def __call__(self, batch):
        batch["original_input"] = batch["input"]
        batch["input"], batch["mlm_labels"], batch["mlm_mask_indices"] = self.mask_input(batch["input"])

        return batch


def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob


def mask_with_tokens(t, token_ids):
    """
    :param t: input tensor with dimensions (BATCH, AA_SEQUENCE, AA_dimension)
    :param token_ids: token ids that are excluded from masking
    :return:
    """

    init_no_mask = torch.full_like(t, False, dtype=torch.bool)  # copy shape and init with False
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)  # get the matching indices and make True
    return mask


def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device  # shape of the mask and the device
    max_masked = math.ceil(prob * seq_len)  # max n of aa masked per seq

    num_tokens = mask.sum(
        dim=-1, keepdim=True
    )  # number of aas that are allowed to be masked (TRUE if allowed to be masked)
    mask_excess = mask.cumsum(dim=-1) > (num_tokens * prob).ceil()
    mask_excess = mask_excess[:, :max_masked]  # prevent masking more than allowed

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)  # indices of the aa's that will be masked

    new_mask = torch.zeros((batch, seq_len + 1), device=device)  # init empty mask
    new_mask.scatter_(-1, sampled_indices, 1)  # 1 for every index that is masked, 0 if not
    return new_mask[:, 1:].bool()
