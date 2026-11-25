"""1D-conv autoencoder with a configurable number of stride-2 stages (`n_down`), for windows of T = 20 / 10 / 5.

The `task/autoencoder_benign_t32` ConvAutoencoder has two stride-2 convolutions (T -> T/2 -> T/4), so it needs T
divisible by 4. Here the same layers (channels 14 -> 32 -> 64 -> 64, kernels 3, BatchNorm + ReLU, z = 16, masked MSE)
are built with only `n_down` of the two later encoder convolutions at stride 2 (the others at stride 1, length kept),
and the decoder mirrors them: an upsampling stage is ConvTranspose1d (k4, s2, p1) as in T = 32, a length-preserving
stage is ConvTranspose1d (k3, s1, p1).

    n_down 2 (T = 20, and T = 32): encoder strides 1, 2, 2; decoder ConvT k4 s2, k4 s2 (the T = 32 model exactly)
    n_down 1 (T = 10):             encoder strides 1, 2, 1; decoder ConvT k3 s1, k4 s2
    n_down 0 (T = 5):              encoder strides 1, 1, 1; decoder ConvT k3 s1, k3 s1

Encoder output 64 x T / 2**n_down -> Linear -> z; decoder Linear z -> 64 x T / 2**n_down + ReLU, the two stages,
Conv1d 32 -> 13 (k3). With n_down = 2 the module tree, parameter names / shapes and initialisation order equal the T = 32
model, so the same seed gives the same weights and outputs (tested).
"""

from __future__ import annotations

import torch
from torch import nn

from task.autoencoder_benign_t32.model import conv_block, count_params, masked_mse, window_errors

__all__ = ["ConvAutoencoder", "conv_block", "count_params", "encoder_strides", "masked_mse", "window_errors"]

MAX_DOWN = 2


def encoder_strides(n_down: int) -> tuple[int, int]:
    """Strides of the 2nd and 3rd encoder convolution: the first `n_down` of them are 2, the rest 1."""
    if n_down not in range(MAX_DOWN + 1):
        raise ValueError(f"n_down must be 0, 1 or 2, got {n_down}")
    return tuple(2 if k < n_down else 1 for k in range(MAX_DOWN))


def up_stage(c_in: int, c_out: int, stride: int) -> nn.Module:
    """Decoder stage that undoes an encoder convolution of this stride: ConvT k4 s2 (doubles T) or ConvT k3 s1 (keeps T)."""
    if stride == 2:
        return nn.ConvTranspose1d(c_in, c_out, 4, stride=2, padding=1)
    return nn.ConvTranspose1d(c_in, c_out, 3, stride=1, padding=1)


class ConvAutoencoder(nn.Module):
    """Encoder + decoder; `forward(x (B, T, F), mask (B, T))` returns (reconstruction (B, T, F), z (B, latent)).

    Reads `n_features`, `seq_len`, `latent_dim` and `n_down` of the settings.
    """

    def __init__(self, settings) -> None:
        super().__init__()
        f, t, n_down = settings.n_features, settings.seq_len, settings.n_down
        s2, s3 = encoder_strides(n_down)
        if t % (2**n_down):
            raise ValueError(f"seq_len must be divisible by 2 ** n_down = {2 ** n_down}, got {t}")
        self.short_len = t // 2**n_down
        self.encoder_conv = nn.Sequential(
            *conv_block(nn.Conv1d(f + 1, 32, 3, stride=1, padding=1), 32),
            *conv_block(nn.Conv1d(32, 64, 3, stride=s2, padding=1), 64),
            *conv_block(nn.Conv1d(64, 64, 3, stride=s3, padding=1), 64),
        )
        self.to_z = nn.Linear(64 * self.short_len, settings.latent_dim)
        self.from_z = nn.Linear(settings.latent_dim, 64 * self.short_len)
        self.decoder_conv = nn.Sequential(
            nn.ReLU(),
            *conv_block(up_stage(64, 64, s3), 64),
            *conv_block(up_stage(64, 32, s2), 32),
            nn.Conv1d(32, f, 3, stride=1, padding=1),
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """z (B, latent). Padding rows are zeroed and the mask is the 14th channel."""
        m = mask.unsqueeze(-1).to(x.dtype)
        h = torch.cat([x * m, m], dim=-1).transpose(1, 2)
        return self.to_z(self.encoder_conv(h).flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruction (B, T, F)."""
        h = self.from_z(z).view(-1, 64, self.short_len)
        return self.decoder_conv(h).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, mask)
        return self.decode(z), z
