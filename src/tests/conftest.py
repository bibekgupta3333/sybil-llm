"""Shared fixtures: default settings, a synthetic feature scaling and synthetic windows (no data files)."""

from __future__ import annotations

import pytest
import torch

from src.model.benign_gridsybil.timesnet.config import PretrainConfig
from src.model.benign_gridsybil.timesnet.data import FEATURES, FeatureSpace

# Synthetic train statistics in the same order and of the same magnitude as the real metadata.json.
SYNTHETIC_MEAN = [480.0, 650.0, 480.0, 650.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.0, 170.0, -0.25, -0.1]
SYNTHETIC_STD = [480.0, 430.0, 390.0, 300.0, 5.0, 6.0, 7.0, 8.0, 1.0, 1.3, 420.0, 1.7, 0.65]


def synthetic_metadata() -> dict:
    """A metadata.json-like dict with the 13 features and their train normalisation."""
    return {"features": list(FEATURES), "normalisation": {"mean": SYNTHETIC_MEAN, "std": SYNTHETIC_STD}}


def prefix_mask(lengths: list[int], seq_len: int) -> torch.Tensor:
    """(B, T) bool with the first `lengths[i]` rows real (padding at the end, as in the encoder input)."""
    return torch.arange(seq_len).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)


def synthetic_windows(lengths: list[int], seq_len: int = 64, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled windows (B, T, 13) with zero padding rows and their mask."""
    gen = torch.Generator().manual_seed(seed)
    mask = prefix_mask(lengths, seq_len)
    x = torch.randn(len(lengths), seq_len, len(FEATURES), generator=gen)
    return x * mask.unsqueeze(-1), mask


@pytest.fixture
def settings() -> PretrainConfig:
    """Default settings (d = 128, d_ff = 64, D12)."""
    return PretrainConfig()


@pytest.fixture
def space() -> FeatureSpace:
    """Train z-score scaling from synthetic statistics."""
    return FeatureSpace(synthetic_metadata())


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)
