"""Mask-aware TimesNet encoder: shapes, size, padding invariance, batch composition."""

from __future__ import annotations

import pytest
import torch

from src.model.benign_gridsybil.timesnet.config import PretrainConfig
from src.model.benign_gridsybil.timesnet.encoder import TimesNetEncoder

from .conftest import synthetic_windows


@pytest.fixture
def encoder(settings: PretrainConfig) -> TimesNetEncoder:
    return TimesNetEncoder(settings).eval()


@pytest.mark.parametrize("seq_len", [24, 50, 64])
def test_shapes_and_zero_padding(encoder: TimesNetEncoder, settings: PretrainConfig, seq_len: int) -> None:
    lengths = [seq_len, seq_len // 2, 5, 1]
    x, mask = synthetic_windows(lengths, seq_len)
    with torch.no_grad():
        h, z = encoder(x, mask)
    assert h.shape == (len(lengths), seq_len, settings.d_model)
    assert z.shape == (len(lengths), settings.d_model)
    assert torch.all(h[~mask] == 0)
    assert torch.isfinite(h).all() and torch.isfinite(z).all()


def test_parameter_count(encoder: TimesNetEncoder) -> None:
    assert sum(p.numel() for p in encoder.parameters()) == 2_301_312


def test_padding_values_never_matter(encoder: TimesNetEncoder) -> None:
    x, mask = synthetic_windows([64, 40, 17, 4, 2, 1])
    garbage = x + torch.randn_like(x) * 100.0 * (~mask).unsqueeze(-1)
    with torch.no_grad():
        ha, za = encoder(x, mask)
        hb, zb = encoder(garbage, mask)
    assert float((za - zb).abs().max()) == 0.0
    assert float((ha - hb).abs().max()) == 0.0


def test_batch_composition(encoder: TimesNetEncoder) -> None:
    x, mask = synthetic_windows([64, 50, 33, 20, 12, 8, 3, 1], seed=1)
    with torch.no_grad():
        _, z_batch = encoder(x, mask)
        z_alone = torch.cat([encoder(x[i : i + 1], mask[i : i + 1])[1] for i in range(len(x))])
        _, z_reversed = encoder(x.flip(0), mask.flip(0))
    assert float((z_batch - z_alone).abs().max()) <= 1e-5
    assert float((z_batch - z_reversed.flip(0)).abs().max()) <= 1e-5
