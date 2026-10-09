"""Views of a batch: reconstruction masking, augmentations, injected violations touch real rows only."""

from __future__ import annotations

import torch

from src.model.benign_gridsybil.timesnet.config import PretrainConfig
from src.model.benign_gridsybil.timesnet.data import FeatureSpace
from src.model.benign_gridsybil.timesnet.views import Augmenter, ReconMasker, ViolationInjector

from .conftest import synthetic_windows

LENGTHS = [64, 63, 40, 17, 8, 4, 3, 2, 1] * 4


def test_masker_hides_real_rows_only(settings: PretrainConfig) -> None:
    _, mask = synthetic_windows(LENGTHS)
    masker = ReconMasker(settings)
    for seed in range(5):
        hidden = masker(mask, torch.Generator().manual_seed(seed))
        assert hidden.shape == mask.shape
        assert not bool((hidden & ~mask).any())
        n = mask.sum(1)
        assert not bool(hidden[n < 2].any())  # fewer than two real rows: nothing to hide
        assert bool((hidden[n >= 2].sum(1) >= 1).all())
        assert bool((hidden.sum(1) < n.clamp(min=1)).all())  # at least one real row stays visible


def test_augmenter_never_grows_a_window(settings: PretrainConfig, space: FeatureSpace) -> None:
    x, mask = synthetic_windows(LENGTHS)
    augment = Augmenter(settings, space)
    for seed in range(5):
        view, new_mask = augment(x, mask, torch.Generator().manual_seed(seed))
        assert view.shape == x.shape and new_mask.shape == mask.shape
        assert bool((new_mask.sum(1) <= mask.sum(1)).all())
        assert bool((new_mask.sum(1) >= 1).all())
        assert torch.all(view[~new_mask] == 0)
        assert torch.equal(new_mask, synthetic_windows(new_mask.sum(1).tolist())[1])  # still a prefix


def test_injector_touches_real_rows_only(settings: PretrainConfig, space: FeatureSpace) -> None:
    x, mask = synthetic_windows(LENGTHS)
    inject = ViolationInjector(settings, space)
    n = mask.sum(1)
    for seed in range(5):
        xi, targets, valid = inject(x, mask, torch.Generator().manual_seed(seed))
        assert torch.all(xi[~mask] == 0)
        assert torch.equal(valid, n >= settings.min_rows_aux)
        assert not bool(targets[n < 4].any())  # < 4 real rows: never injected
        assert bool((targets.sum(1) <= 1).all())  # at most one violation type per window
        untouched = targets.sum(1) == 0
        assert float((xi[untouched] - x[untouched]).abs().max()) <= 1e-4
        assert float((xi[:, 0] - x[:, 0]).abs().max()) <= 1e-4  # the first row is never changed
    injected = sum(int(inject(x, mask, torch.Generator().manual_seed(s))[1].sum()) for s in range(5))
    assert injected > 0
