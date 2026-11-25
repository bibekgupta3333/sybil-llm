"""Supervised autoencoder: the depth-configurable ConvAutoencoder of `task/autoencoder_benign_t20` (`n_down` stride-2
stages; 2 = the T = 32 model) plus the `task/autoencoder_st_t32` classifier head on z (both imported, not copied).

Head: Linear 16 -> 32 -> ReLU -> Dropout 0.2 -> Linear 32 -> 1 (logit of "GridSybil"). Loss (in the trainer) =
BCEWithLogits (pos_weight from train counts) + recon_weight * masked MSE over real rows. With n_down = 2 and T = 32 the
module tree, parameter names / shapes and initialisation order equal `task.autoencoder_st_t32.model
.SupervisedAutoencoder` (tested).
"""

from __future__ import annotations

import torch
from torch import nn

from task.autoencoder_benign_t20.model import ConvAutoencoder, count_params, masked_mse, window_errors
from task.autoencoder_st_t32.model import ClassifierHead

__all__ = ["ClassifierHead", "ConvAutoencoder", "SupervisedAutoencoder", "count_params", "masked_mse", "window_errors"]


class SupervisedAutoencoder(nn.Module):
    """`forward(x, mask)` returns (reconstruction (B, T, F), z (B, latent), logit (B,))."""

    def __init__(self, settings) -> None:
        super().__init__()
        self.autoencoder = ConvAutoencoder(settings)
        self.head = ClassifierHead(settings)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon, z = self.autoencoder(x, mask)
        return recon, z, self.head(z)
