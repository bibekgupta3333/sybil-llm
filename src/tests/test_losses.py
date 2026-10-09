"""Pretraining losses: NT-Xent false-negative masking, frozen loss scales, which losses a batch yields."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.model.benign_gridsybil.timesnet.config import PretrainConfig
from src.model.benign_gridsybil.timesnet.data import FeatureSpace
from src.model.benign_gridsybil.timesnet.heads import PretrainingModel
from src.model.benign_gridsybil.timesnet.losses import LossScales, Objective, nt_xent

from .conftest import synthetic_windows


def projections(m: int, dim: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    return F.normalize(torch.randn(m, dim), dim=-1), F.normalize(torch.randn(m, dim), dim=-1)


def test_nt_xent_without_groups_matches_distinct_groups() -> None:
    p1, p2 = projections(6)
    plain, masked_plain = nt_xent(p1, p2, 0.1)
    distinct, masked_distinct = nt_xent(p1, p2, 0.1, torch.arange(6))
    assert masked_plain == 0 and masked_distinct == 0
    assert torch.allclose(plain, distinct)


def test_nt_xent_masks_same_group_and_keeps_positives() -> None:
    m = 4
    p1, p2 = projections(m)
    loss, masked = nt_xent(p1, p2, 0.1, torch.zeros(m, dtype=torch.long))
    assert masked == (4 * m * m - 2 * m - 2 * m) // 2  # every pair except self and the positive
    assert float(loss) == pytest.approx(0.0, abs=1e-6)  # only the positive is left in each row

    groups = torch.tensor([0, 0, 1, 2])
    loss, masked = nt_xent(p1, p2, 0.1, groups)
    assert masked == 4  # windows 0 and 1: (0,1), (0,1'), (0',1), (0',1')
    reps = torch.cat([p1, p2]) / 0.1**0.5
    sim = reps @ reps.T
    target = torch.cat([torch.arange(m, 2 * m), torch.arange(m)])
    drop = torch.eye(2 * m, dtype=torch.bool)
    for a, b in [(0, 1), (0, 5), (4, 1), (4, 5)]:
        drop[a, b] = drop[b, a] = True
    expected = F.cross_entropy(sim.masked_fill(drop, float("-inf")), target)
    assert torch.allclose(loss, expected, atol=1e-5)


def test_loss_scales_freeze_per_loss() -> None:
    scales = LossScales(warmup=3)
    assert scales.scale("recon") == 1.0
    for step in range(3):
        scales.update({"recon": 2.0 + step}, step)
    assert scales.frozen
    assert scales.scale("recon") == pytest.approx(3.0)
    scales.update({"recon": 100.0, "nce": 4.0}, 3)  # recon frozen; nce counts its first present value
    assert scales.scale("recon") == pytest.approx(3.0)
    assert scales.scale("nce") == pytest.approx(4.0)
    assert not scales.frozen  # nce has not seen `warmup` values yet
    scales.update({"recon": 100.0}, 4)  # nce absent: nothing counted for it
    assert scales.count["nce"] == 1
    scales.update({"nce": 6.0}, 5)
    scales.update({"nce": float("nan")}, 6)  # non-finite values are never counted
    scales.update({"nce": 8.0}, 7)
    assert scales.frozen
    assert scales.scale("nce") == pytest.approx(6.0)
    scales.update({"nce": 1000.0}, 8)
    assert scales.scale("nce") == pytest.approx(6.0)


@pytest.fixture
def model(settings: PretrainConfig) -> PretrainingModel:
    return PretrainingModel(settings).eval()


def test_objective_one_row_windows_give_no_losses(
    settings: PretrainConfig, space: FeatureSpace, model: PretrainingModel
) -> None:
    x, mask = synthetic_windows([1] * 6)
    with torch.no_grad():
        losses, extra = Objective(settings, space)(model, x, mask, torch.Generator().manual_seed(0))
    assert losses == {}
    assert extra["nce_masked_pairs"] == 0


def test_objective_normal_windows_give_all_losses(
    settings: PretrainConfig, space: FeatureSpace, model: PretrainingModel
) -> None:
    x, mask = synthetic_windows([64, 64, 50, 40, 32, 20, 10, 8])
    groups = torch.tensor([0, 0, 1, 2, 3, 4, 5, 6])
    with torch.no_grad():
        losses, extra = Objective(settings, space)(model, x, mask, torch.Generator().manual_seed(0), groups)
    assert set(losses) == {"recon", "nce", "p1", "p2", "p3"}
    assert all(torch.isfinite(v) for v in losses.values())
    assert extra["nce_masked_pairs"] == 4
