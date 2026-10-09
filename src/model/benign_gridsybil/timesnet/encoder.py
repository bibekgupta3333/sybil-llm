"""Mask-aware TimesNet encoder (same code and checks as `src/model/benign_gridsybil/encoder_T64.ipynb`).

(x (B, T, 13), mask (B, T)) -> (H (B, T, d) per message, z (B, d) per window). Padding rows never influence the
output: they are zeroed after every layer and periods are found on the real rows only. Inner width d_ff = 64 at
d = 128 (D12).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PretrainConfig


class MaskedOps:
    """Mask-aware tensor operations; `mask` is (B, T) bool with True for real rows."""

    @staticmethod
    def fill_padding(x: torch.Tensor, mask: torch.Tensor, value: float = 0.0) -> torch.Tensor:
        """(B, T, C) -> same, every padding row set to `value`."""
        return x.masked_fill(~mask.bool().unsqueeze(-1), value)

    @staticmethod
    def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean over the real rows of each window: (B, T, d) -> (B, d)."""
        mask = mask.bool()
        total = MaskedOps.fill_padding(h, mask).sum(dim=1)
        count = mask.sum(dim=1, keepdim=True).clamp(min=1).to(h.dtype)
        return total / count

    @staticmethod
    def masked_mse(pred: torch.Tensor, target: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Mean squared error over the selected rows (B, T) and all features -> scalar."""
        rows = rows.bool()
        squared = MaskedOps.fill_padding((pred - target) ** 2, rows).sum(dim=-1)
        return squared.sum() / (rows.sum().clamp(min=1).to(pred.dtype) * pred.shape[-1])


class PeriodFinder:
    """Per-window top-k periods from an FFT over the real rows only."""

    def __init__(self, top_k: int, min_freq: int) -> None:
        self.top_k, self.min_freq = top_k, min_freq

    def __call__(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, C) + (B, T) -> periods (B, k) long and weights (B, k) (FFT amplitudes; 0 where none allowed)."""
        batch_size = h.shape[0]
        real_rows = mask.bool().sum(dim=1)
        periods = torch.ones(batch_size, self.top_k, dtype=torch.long, device=h.device)
        weights = torch.zeros(batch_size, self.top_k, dtype=h.dtype, device=h.device)
        for n in real_rows.unique().tolist():  # windows of equal length together
            rows = (real_rows == n).nonzero().squeeze(1)
            amplitude = torch.fft.rfft(h[rows, :n], dim=1).abs().mean(dim=-1)  # (b, n // 2 + 1)
            allowed = torch.arange(amplitude.shape[1], device=h.device) >= self.min_freq
            k = min(self.top_k, int(allowed.sum()))
            if k == 0:  # too short for any rhythm
                continue
            top_scores, top_freqs = amplitude.masked_fill(~allowed, float("-inf")).topk(k, dim=1)
            periods[rows, :k] = n // top_freqs
            weights[rows, :k] = top_scores
        return periods, weights


class InceptionBlock2d(nn.Module):
    """1x1, 3x3, 5x5 Conv2d in parallel ("same" padding), outputs averaged.

    Adapted from `Inception_Block_V1`, THUML Time-Series-Library (MIT licence).
    """

    def __init__(self, in_channels: int, out_channels: int, n_kernels: int) -> None:
        super().__init__()
        self.kernels = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i) for i in range(n_kernels)
        )
        for conv in self.kernels:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(b, C_in, cycles, p) -> (b, C_out, cycles, p)."""
        return torch.stack([conv(x) for conv in self.kernels], dim=-1).mean(dim=-1)


class TimesBlock(nn.Module):
    """Find periods -> fold -> Inception 2-D convs -> unfold -> mix -> residual -> mask."""

    def __init__(self, settings: PretrainConfig) -> None:
        super().__init__()
        self.period_finder = PeriodFinder(settings.top_k, settings.min_freq)
        self.conv = nn.Sequential(
            InceptionBlock2d(settings.d_model, settings.d_ff, settings.n_kernels),
            nn.GELU(),
            InceptionBlock2d(settings.d_ff, settings.d_model, settings.n_kernels),
        )
        self.last_periods: torch.Tensor | None = None  # kept for the period diagnostic

    def fold_conv_unfold(self, x: torch.Tensor, period: int) -> torch.Tensor:
        """(b, T, d) -> grid (b, d, cycles, period) -> conv -> (b, T, d)."""
        b, length, d = x.shape
        padded = math.ceil(length / period) * period
        if padded > length:
            x = torch.cat([x, x.new_zeros(b, padded - length, d)], dim=1)
        grid = x.reshape(b, padded // period, period, d).permute(0, 3, 1, 2).contiguous()
        grid = self.conv(grid)
        return grid.permute(0, 2, 3, 1).reshape(b, padded, d)[:, :length]

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, T, d) with zero padding rows -> (B, T, d) with zero padding rows."""
        periods, weights = self.period_finder(x, mask)
        self.last_periods = periods.detach().cpu()
        branches = []
        for slot in range(periods.shape[1]):
            out = torch.zeros_like(x)
            for period in torch.unique(periods[:, slot]).tolist():  # windows sharing a period together
                members = (periods[:, slot] == period).nonzero(as_tuple=True)[0]
                out[members] = self.fold_conv_unfold(x[members], int(period))
            branches.append(out)
        stacked = torch.stack(branches, dim=-1)  # (B, T, d, k)
        mix = F.softmax(weights, dim=1)[:, None, None, :]  # (B, 1, 1, k)
        out = (stacked * mix).sum(dim=-1) + x  # mix + residual
        return out * mask.unsqueeze(-1).to(out.dtype)


class TokenEmbedding(nn.Module):
    """Conv1d over time (zero padding) + fixed sin/cos positions + dropout; padding rows set to 0."""

    def __init__(self, settings: PretrainConfig, max_len: int = 512) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            settings.n_features, settings.d_model, kernel_size=3, padding=1, padding_mode="zeros", bias=False
        )
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.dropout = nn.Dropout(settings.dropout)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, settings.d_model, 2).float() * (-math.log(10000.0) / settings.d_model))
        table = torch.zeros(max_len, settings.d_model)
        table[:, 0::2] = torch.sin(position * div_term)
        table[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positions", table, persistent=False)  # fixed, not a parameter

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, T, 13) -> (B, T, d), padding rows 0."""
        tokens = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, d, T) -> (B, T, d)
        out = self.dropout(tokens + self.positions[: x.shape[1]])
        return out * mask.unsqueeze(-1).to(out.dtype)


class TimesNetEncoder(nn.Module):
    """Mask-aware TimesNet encoder: (x, mask) -> (H per message, z per window)."""

    def __init__(self, settings: PretrainConfig) -> None:
        super().__init__()
        self.embedding = TokenEmbedding(settings)
        self.blocks = nn.ModuleList(TimesBlock(settings) for _ in range(settings.n_blocks))
        self.norm = nn.LayerNorm(settings.d_model)  # shared by all blocks, as in the reference model

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B, T, 13), mask (B, T) bool -> H (B, T, d) zero at padding, z (B, d)."""
        mask = mask.bool()
        keep = mask.unsqueeze(-1).to(x.dtype)
        h = self.embedding(MaskedOps.fill_padding(x, mask), mask)
        for block in self.blocks:
            h = self.norm(block(h, mask)) * keep
        return h, MaskedOps.masked_mean(h, mask)

    def periods_used(self) -> torch.Tensor:
        """Periods each block chose in the last forward pass: (n_blocks, B, k)."""
        return torch.stack([block.last_periods for block in self.blocks])
