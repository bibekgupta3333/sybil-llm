"""Supervised autoencoder at T = 64: the `task/encoder` ConvAutoencoder (80,189 params, input 14 x 64, z = 16) plus
the classifier head of `task/autoencoder_st` (Linear 16 -> 32 -> ReLU -> Dropout 0.2 -> Linear 32 -> 1, 577 params).

`forward(x, mask)` returns (reconstruction (B, T, F), z (B, latent), logit (B,)), the interface the
`task/autoencoder_st` trainer expects. Loss = BCEWithLogits (pos_weight from train counts) + recon_weight * masked MSE.
"""

from __future__ import annotations

import torch
from torch import nn

from task.autoencoder_st.model import ClassifierHead, count_params, masked_mse, window_errors
from task.autoencoder_st_t64.config import SupervisedAET64Config
from task.encoder.model import ConvAutoencoder

__all__ = ["SupervisedAutoencoder", "count_params", "masked_mse", "window_errors"]


class SupervisedAutoencoder(nn.Module):
    """`task.encoder.model.ConvAutoencoder` + `task.autoencoder_st.model.ClassifierHead` on z."""

    def __init__(self, settings: SupervisedAET64Config) -> None:
        super().__init__()
        if settings.seq_len != 64 or settings.n_features != 13:
            raise ValueError("task/encoder's ConvAutoencoder is fixed at 64 x 13")
        self.autoencoder = ConvAutoencoder(settings.architecture())
        self.head = ClassifierHead(settings)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon, z = self.autoencoder(x, mask)
        return recon, z, self.head(z)
