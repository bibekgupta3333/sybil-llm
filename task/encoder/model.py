"""Small 1D-conv encoder over time, its mirror decoder, and a supervised head.

Input (B, T, 13) + mask (B, T) -> channels-first (B, C, T) with C = 13 (+ 1 mask channel). Encoder: Conv1d
C -> 32 (k3, s1) -> 64 (k3, s2) -> 64 (k3, s2), each BatchNorm + ReLU, flatten (64 x 16) -> Linear -> z (16).
Decoder: Linear 16 -> 64 x 16 -> ConvTranspose1d 64 -> 64 (k4, s2) -> 32 (k4, s2), BatchNorm + ReLU ->
Conv1d 32 -> 13 (k3) = reconstruction (B, T, 13).
"""

from __future__ import annotations

import torch
from torch import nn

from task.encoder.config import N_FEATURES, SEQ_LEN, EncoderTaskConfig


def masked_mse(recon: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over real rows and all features (padding rows ignored)."""
    m = mask.unsqueeze(-1).to(recon.dtype)
    return ((recon - x) ** 2 * m).sum() / (m.sum() * x.shape[-1]).clamp_min(1.0)


def window_errors(recon: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-window masked MSE (B,) and per-window, per-feature masked MSE (B, 13)."""
    m = mask.unsqueeze(-1).to(recon.dtype)
    rows = m.sum(dim=1).clamp_min(1.0)  # (B, 1)
    per_feat = ((recon - x) ** 2 * m).sum(dim=1) / rows
    return per_feat.mean(dim=1), per_feat


class ConvEncoder(nn.Module):
    """(B, T, 13) + mask -> z (B, latent_dim)."""

    def __init__(self, settings: EncoderTaskConfig) -> None:
        super().__init__()
        self.mask_channel = settings.mask_channel
        c_in = N_FEATURES + int(settings.mask_channel)
        c0, c1, c2 = settings.channels
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, c0, 3, stride=1, padding=1),
            nn.BatchNorm1d(c0),
            nn.ReLU(),
            nn.Conv1d(c0, c1, 3, stride=2, padding=1),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.Conv1d(c1, c2, 3, stride=2, padding=1),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
        )
        self.out_len = SEQ_LEN // 4
        self.fc = nn.Linear(c2 * self.out_len, settings.latent_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(x.dtype)
        h = x * m
        if self.mask_channel:
            h = torch.cat([h, m], dim=-1)
        h = self.conv(h.transpose(1, 2))
        return self.fc(h.flatten(1))


class ConvDecoder(nn.Module):
    """z (B, latent_dim) -> reconstruction (B, T, 13)."""

    def __init__(self, settings: EncoderTaskConfig) -> None:
        super().__init__()
        c0, c1, c2 = settings.channels
        self.c2, self.in_len = c2, SEQ_LEN // 4
        self.fc = nn.Linear(settings.latent_dim, c2 * self.in_len)
        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose1d(c2, c1, 4, stride=2, padding=1),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.ConvTranspose1d(c1, c0, 4, stride=2, padding=1),
            nn.BatchNorm1d(c0),
            nn.ReLU(),
            nn.Conv1d(c0, N_FEATURES, 3, stride=1, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.c2, self.in_len)
        return self.deconv(h).transpose(1, 2)


class ConvAutoencoder(nn.Module):
    """Encoder + decoder; `forward` returns (reconstruction, z)."""

    def __init__(self, settings: EncoderTaskConfig) -> None:
        super().__init__()
        self.encoder = ConvEncoder(settings)
        self.decoder = ConvDecoder(settings)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x, mask)
        return self.decoder(z), z


class SupervisedHead(nn.Module):
    """Encoder + Linear 16 -> 32 -> ReLU -> 1; `forward` returns (logit (B,), z). Sigmoid is applied in scoring."""

    def __init__(self, settings: EncoderTaskConfig) -> None:
        super().__init__()
        self.encoder = ConvEncoder(settings)
        self.head = nn.Sequential(
            nn.Linear(settings.latent_dim, settings.head_hidden), nn.ReLU(), nn.Linear(settings.head_hidden, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x, mask)
        return self.head(z).squeeze(-1), z


def count_params(module: nn.Module) -> int:
    """Trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
