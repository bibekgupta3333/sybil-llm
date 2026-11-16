"""The package's own TimesNet encoder and pretraining model: sizes, output shapes, padding invariance (CPU)."""

from __future__ import annotations

import torch

from src.model.all_t32_benign.timesnet.config import T32BenignConfig
from src.model.all_t32_benign.timesnet.encoder import TimesNetEncoder
from src.model.all_t32_benign.timesnet.heads import PretrainingModel

ENCODER_PARAMS = 2_301_312  # d = 128, d_ff = 64 (D12)
MODEL_PARAMS = 2_402_461  # encoder + heads + the 13-number mask token, as the runs report


def _batch(seed: int = 0, batch: int = 6, seq_len: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Random windows with mixed real-row counts (1..T), padding rows zero."""
    gen = torch.Generator().manual_seed(seed)
    lengths = torch.tensor([seq_len, 1, 2, 7, 19, seq_len - 1])[:batch]
    mask = torch.arange(seq_len)[None, :] < lengths[:, None]
    x = torch.randn(batch, seq_len, 13, generator=gen) * mask.unsqueeze(-1)
    return x, mask


def test_parameter_counts_at_d128_dff64_t32() -> None:
    settings = T32BenignConfig()
    assert (settings.d_model, settings.d_ff, settings.seq_len) == (128, 64, 32)
    model = PretrainingModel(settings)
    assert sum(p.numel() for p in model.encoder.parameters()) == ENCODER_PARAMS
    assert sum(p.numel() for p in model.parameters()) == MODEL_PARAMS
    assert sum(p.numel() for p in TimesNetEncoder(settings).parameters()) == ENCODER_PARAMS


def test_output_shapes() -> None:
    settings = T32BenignConfig()
    torch.manual_seed(0)
    encoder = TimesNetEncoder(settings).eval()
    x, mask = _batch()
    with torch.no_grad():
        h, z = encoder(x, mask)
    assert h.shape == (x.shape[0], settings.seq_len, settings.d_model)
    assert z.shape == (x.shape[0], settings.d_model)
    assert float(h[~mask].abs().max()) == 0.0  # padding rows of H are zero


def test_padding_does_not_change_z() -> None:
    torch.manual_seed(0)
    encoder = TimesNetEncoder(T32BenignConfig()).eval()
    x, mask = _batch()
    garbage = x + torch.randn_like(x) * 100.0 * (~mask).unsqueeze(-1)
    with torch.no_grad():
        _, za = encoder(x, mask)
        _, zb = encoder(garbage, mask)
    assert float((za - zb).abs().max()) <= 1e-4


def test_same_seed_same_weights() -> None:
    torch.manual_seed(3)
    first = PretrainingModel(T32BenignConfig()).state_dict()
    torch.manual_seed(3)
    second = PretrainingModel(T32BenignConfig()).state_dict()
    assert list(first) == list(second)
    assert all(torch.equal(first[k], second[k]) for k in first)
