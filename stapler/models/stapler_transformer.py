from __future__ import annotations

import torch
import torch.nn as nn
from x_transformers.x_transformers import TransformerWrapper


class STAPLERTransformer(TransformerWrapper):
    def __init__(
        self,
        num_tokens: int = 25,
        cls_dropout: float = 0.0,
        checkpoint_path: str | None = None,
        output_classification: bool = True,
        classification_head: nn.Module | None = None,
        **kwargs,
    ):
        """
        STAPLERTransformer extends TransformerWrapper to perform token-level and sequence-level classification tasks.

        Args:
            output_classification (bool): Whether to output logits for classification tasks. Defaults to True.
            classification_head (nn.Module): Custom sequence classification head. Defaults to None.
            **kwargs: Keyword arguments for the TransformerWrapper.
        """
        super().__init__(num_tokens=num_tokens, **kwargs)
        self.output_classification = output_classification
        self.hidden_dim = self.attn_layers.dim
        self.num_tokens = num_tokens

        self.to_logits = nn.Linear(self.hidden_dim, self.num_tokens)

        if checkpoint_path:
            self.load_model(checkpoint_path)

        if self.output_classification:
            if classification_head is not None:  # For custom classification head
                self.to_cls = classification_head
            else:
                self.to_cls = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(cls_dropout),
                    nn.Linear(self.hidden_dim, 2),
                )

    def forward(self, x, **kwargs):
        """
        Forward pass of the STAPLERTransformer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
            **kwargs: Additional keyword arguments for the TransformerWrapper.

        Returns:
            output_dict (dict): Dictionary containing logits for token-level and sequence-level classification tasks.
        """

        x = super().forward(x, **kwargs)
        output_dict = {}
        logits = self.to_logits(x)
        output_dict["mlm_logits"] = logits

        if self.output_classification:
            cls_logit = self.to_cls(x[:, 0, :])
            output_dict["cls_logits"] = cls_logit

        return output_dict

    def load_model(self, checkpoint_path: str):
        """Locate state dict in lightning checkpoint and load into model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["state_dict"]

        # Remove "model." or "transformer." prefix from state dict keys and remove any keys containing 'to_cls'
        try:
            state_dict = {
                key.replace("model.", "").replace("transformer.", ""): value
                for key, value in state_dict.items()
                if "to_cls" not in key
            }
        except Exception as e:
            print("Error loading state dict. Please check the checkpoint file.")
            raise e


        self.load_state_dict(state_dict)
