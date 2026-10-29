"""1D-conv autoencoder over time: 14 x 48 (13 features + mask) -> z (16) -> 13 x 48 reconstruction.

Encoder: Conv1d 14 -> 32 (k3, s1) -> 64 (k3, s2) -> 64 (k3, s2), each BatchNorm + ReLU; 64 x 12 = 768 -> Linear -> z.
Decoder: Linear 16 -> 768 -> 64 x 12; ConvTranspose1d 64 -> 64 (k4, s2) -> 32 (k4, s2), each BatchNorm + ReLU;
Conv1d 32 -> 13 (k3). The loss is the mean squared error over real rows only.
"""

from __future__ import annotations

import torch
from torch import nn

from task.autoencoder_overlap.config import AutoencoderConfig


def masked_mse(recon: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over real rows and all features (padding rows ignored)."""
    m = mask.unsqueeze(-1).to(recon.dtype)
    return ((recon - x) ** 2 * m).sum() / (m.sum() * x.shape[-1]).clamp_min(1.0)


def window_errors(recon: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-window masked MSE (B,) and per-window, per-feature masked MSE (B, F)."""
    m = mask.unsqueeze(-1).to(recon.dtype)
    per_feature = ((recon - x) ** 2 * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
    return per_feature.mean(dim=1), per_feature


def conv_block(conv: nn.Module, channels: int) -> list[nn.Module]:
    """conv -> BatchNorm -> ReLU."""
    return [conv, nn.BatchNorm1d(channels), nn.ReLU()]


class ConvAutoencoder(nn.Module):
    """Encoder + decoder; `forward(x (B, T, F), mask (B, T))` returns (reconstruction (B, T, F), z (B, latent))."""

    def __init__(self, settings: AutoencoderConfig) -> None:
        super().__init__()
        f, t = settings.n_features, settings.seq_len
        if t % 4:
            raise ValueError(f"seq_len must be divisible by 4, got {t}")
        self.short_len = t // 4
        self.encoder_conv = nn.Sequential(
            *conv_block(nn.Conv1d(f + 1, 32, 3, stride=1, padding=1), 32),
            *conv_block(nn.Conv1d(32, 64, 3, stride=2, padding=1), 64),
            *conv_block(nn.Conv1d(64, 64, 3, stride=2, padding=1), 64),
        )
        self.to_z = nn.Linear(64 * self.short_len, settings.latent_dim)
        self.from_z = nn.Linear(settings.latent_dim, 64 * self.short_len)
        self.decoder_conv = nn.Sequential(
            nn.ReLU(),
            *conv_block(nn.ConvTranspose1d(64, 64, 4, stride=2, padding=1), 64),
            *conv_block(nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1), 32),
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


def count_params(module: nn.Module) -> int:
    """Trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
