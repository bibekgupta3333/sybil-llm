"""Tests of the classical detectors on tiny synthetic windows (CPU, a few hundred windows)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from baselinepaper.experiments import config as cfg
from baselinepaper.experiments.data import FeatureStatistics, LabelledWindowSet, WindowSet
from baselinepaper.experiments.detectors import available, get_detector
from baselinepaper.experiments.detectors import window_statistics as ws
from baselinepaper.experiments.detectors.window_statistics import StatisticsDetector, WindowStatistics

T, F = cfg.SEQ_LEN, cfg.N_FEATURES
UNSUPERVISED = ("pca", "iforest", "ocsvm", "knn")
CLASSICAL = (*UNSUPERVISED, "statslr")


def make_windows(n: int, seed: int, shift: float = 0.0, lengths=None) -> tuple[np.ndarray, np.ndarray]:
    """n windows with random prefix masks and zero padding rows; the first three windows have 1, 2, 3 rows."""
    rng = np.random.default_rng(seed)
    if lengths is None:
        lengths = rng.integers(1, T + 1, size=n)
        lengths[:3] = (1, 2, 3)
    mask = np.arange(T)[None, :] < np.asarray(lengths)[:, None]
    x = (rng.normal(size=(n, T, F)) + shift).astype(np.float32)
    x[~mask] = 0.0
    return x, mask


def fitted(name: str, seed: int = 0, regime: str | None = None, statistics=None) -> StatisticsDetector:
    """A detector fitted on 300 synthetic windows (labelled for the supervised regime)."""
    cls = get_detector(name)
    regime = regime or cls.regimes[0]
    x, mask = make_windows(300, seed=1)
    if regime == cfg.SUPERVISED:
        xa, ma = make_windows(150, seed=2, shift=1.5)
        labels = np.concatenate([np.zeros(len(x)), np.ones(len(xa))]).astype(np.int8)
        train = LabelledWindowSet(np.concatenate([x, xa]), np.concatenate([mask, ma]), labels)
    else:
        train = WindowSet(x, mask)
    detector = cls(regime=regime, device="cpu", statistics=statistics)
    detector.fit(train, None, seed)
    return detector


def test_registry_and_contract():
    """Names, regimes, seeds and score names are the ones frozen in hyperparameters-classical.md."""
    detectors = available()
    assert set(CLASSICAL) <= set(detectors)
    for name in UNSUPERVISED:
        assert detectors[name].regimes == cfg.UNLABELLED_REGIMES
    assert detectors["statslr"].regimes == (cfg.SUPERVISED,)
    assert {n: detectors[n].is_stochastic for n in CLASSICAL} == {
        "pca": False,
        "iforest": True,
        "ocsvm": True,
        "knn": True,
        "statslr": False,
    }
    assert {n for n in CLASSICAL if detectors[n].uses_subsample} == {"ocsvm", "knn"}
    assert ws.SUBSAMPLE_SIZE == 20_000
    for name in CLASSICAL:
        assert len(detectors[name].score_names) == 1
        hyper = detectors[name](regime=detectors[name].regimes[0]).hyperparameters()
        assert hyper["estimator"]["class"].startswith("sklearn.") and hyper["validation_data_used"] is False


def test_statistics_values_and_shape():
    """65 values per window, equal to a plain per-window computation over the real rows."""
    x, mask = make_windows(40, seed=0)
    stats = WindowStatistics(chunk_windows=7)(x, mask)
    assert stats.shape == (40, 5 * F) and np.isfinite(stats).all()
    for i in range(len(x)):
        real = x[i, mask[i]].astype(np.float64)
        expected = np.concatenate(
            [real.mean(axis=0), real.std(axis=0), real.min(axis=0), real.max(axis=0), real[-1] - real[0]]
        )
        np.testing.assert_allclose(stats[i], expected, atol=1e-12)
    np.testing.assert_array_equal(stats, WindowStatistics(chunk_windows=1000)(x, mask))


def test_statistics_ignore_padded_rows():
    """Changing the values of padding rows changes no statistic."""
    x, mask = make_windows(60, seed=3)
    garbage = x.copy()
    garbage[~mask] = np.random.default_rng(4).normal(size=(int((~mask).sum()), F)).astype(np.float32) * 1e3
    np.testing.assert_array_equal(WindowStatistics()(x, mask), WindowStatistics()(garbage, mask))


def test_one_row_window_has_zero_spread():
    """A one-row window: std = 0, last - first = 0, mean = min = max = the row."""
    x, mask = make_windows(5, seed=5, lengths=[1, 1, 1, 1, 1])
    stats = WindowStatistics()(x, mask)
    mean, std, minimum, maximum, delta = (stats[:, k * F : (k + 1) * F] for k in range(5))
    assert (std == 0).all() and (delta == 0).all()
    np.testing.assert_array_equal(mean, x[:, 0].astype(np.float64))
    np.testing.assert_array_equal(minimum, mean)
    np.testing.assert_array_equal(maximum, mean)


def test_statistics_reject_empty_window():
    x, mask = make_windows(3, seed=6)
    mask[1] = False
    with pytest.raises(ValueError):
        WindowStatistics()(x, mask)


@pytest.mark.parametrize("name", CLASSICAL)
def test_scores_finite_and_padding_independent(name):
    """One finite score per window, the same whatever the padding rows hold."""
    detector = fitted(name)
    x, mask = make_windows(80, seed=7)
    scores = detector.score(x, mask)
    assert set(scores) == set(detector.score_names)
    values = scores[detector.score_names[0]]
    assert values.shape == (80,) and values.dtype == np.float64 and np.isfinite(values).all()
    garbage = x.copy()
    garbage[~mask] = 123.0
    np.testing.assert_array_equal(detector.score(garbage, mask)[detector.score_names[0]], values)


@pytest.mark.parametrize("name", UNSUPERVISED)
def test_far_away_window_scores_higher(name):
    """Windows far from the training distribution score above every in-distribution window."""
    detector = fitted(name)
    lengths = [T] * 20
    near = detector.score(*make_windows(20, seed=8, lengths=lengths))[detector.score_names[0]]
    far = detector.score(*make_windows(20, seed=9, shift=25.0, lengths=lengths))[detector.score_names[0]]
    assert far.min() > near.max()


def test_statslr_direction_and_probability():
    """P(attack) is in [0, 1] and higher for windows shifted like the training attack class."""
    detector = fitted("statslr")
    benign = detector.score(*make_windows(100, seed=10))["probability"]
    attack = detector.score(*make_windows(100, seed=11, shift=1.5))["probability"]
    assert 0 <= min(benign.min(), attack.min()) and max(benign.max(), attack.max()) <= 1
    assert attack.mean() > benign.mean() + 0.3
    assert detector.fit_info()["converged"]


def test_statslr_score_takes_no_labels():
    """`score` has exactly (x, mask); the unsupervised regimes are refused; an unlabelled set cannot be fitted."""
    cls = get_detector("statslr")
    assert list(inspect.signature(cls.score).parameters) == ["self", "x", "mask"]
    assert list(inspect.signature(cls.score_model).parameters) == ["self", "features"]
    with pytest.raises(ValueError):
        cls(regime=cfg.BENIGN_ONLY)
    x, mask = make_windows(50, seed=12)
    with pytest.raises(ValueError):
        cls(regime=cfg.SUPERVISED).fit(WindowSet(x, mask), None, 0)


@pytest.mark.parametrize("name", CLASSICAL)
def test_seed_reproducibility(name):
    """The same seed gives identical scores."""
    x, mask = make_windows(60, seed=13)
    a, b = fitted(name, seed=3), fitted(name, seed=3)
    key = a.score_names[0]
    np.testing.assert_array_equal(a.score(x, mask)[key], b.score(x, mask)[key])


def test_seed_changes_stochastic_detectors(monkeypatch):
    """A stochastic detector depends on the seed: the forest's trees, the subsample of ocsvm / knn."""
    x, mask = make_windows(60, seed=14)
    a, b = fitted("iforest", seed=0), fitted("iforest", seed=1)
    assert not np.array_equal(a.score(x, mask)["isolation"], b.score(x, mask)["isolation"])
    monkeypatch.setattr(ws, "SUBSAMPLE_SIZE", 100)
    rows0, rows1 = StatisticsDetector.subsample_rows(300, 0), StatisticsDetector.subsample_rows(300, 1)
    assert len(rows0) == 100 == len(np.unique(rows0)) and not np.array_equal(rows0, rows1)
    np.testing.assert_array_equal(rows0, StatisticsDetector.subsample_rows(300, 0))
    np.testing.assert_array_equal(StatisticsDetector.subsample_rows(80, 0), np.arange(80))
    for name, key in (("ocsvm", "decision"), ("knn", "distance")):
        a, b = fitted(name, seed=0), fitted(name, seed=1)
        assert a.fit_info()["subsample_size"] == 100 and a.fit_info()["training_windows"] == 300
        assert not np.array_equal(a.score(x, mask)[key], b.score(x, mask)[key])


def test_knn_score_is_mean_distance_to_k_nearest():
    """knn against a direct computation on the scaled statistics."""
    detector = fitted("knn")
    x, mask = make_windows(15, seed=15)
    xt, mt = make_windows(300, seed=1)
    reference = detector.scaled_statistics(xt, mt)
    query = detector.scaled_statistics(x, mask)
    distances = np.sqrt(((query[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2))
    expected = np.sort(distances, axis=1)[:, :5].mean(axis=1)
    np.testing.assert_allclose(detector.score(x, mask)["distance"], expected, rtol=1e-6, atol=1e-6)


def test_pca_keeps_95_percent_and_scores_residual():
    detector = fitted("pca")
    info = detector.fit_info()
    assert 0.95 <= info["explained_variance_ratio_sum"] <= 1.0 and 1 <= info["n_components"] < 5 * F
    xt, mt = make_windows(300, seed=1)
    features = detector.scaled_statistics(xt, mt)
    np.testing.assert_allclose(features.mean(axis=0), 0.0, atol=1e-9)
    components = detector._pca.components_
    centred = features - detector._pca.mean_
    residual = centred - centred @ components.T @ components
    np.testing.assert_allclose(detector.score(xt, mt)["reconstruction"], (residual**2).mean(axis=1), atol=1e-10)


def test_regime_statistics_are_applied():
    """In a re-standardised regime the detector sees standardise(x, mask), in fit and in score."""
    stats = FeatureStatistics(np.full(F, 0.7), np.full(F, 2.5), np.full(F, 1000))
    plain = fitted("knn", regime=cfg.MIX_ALL)
    restandardised = fitted("knn", regime=cfg.MIX_ALL, statistics=stats)
    x, mask = make_windows(30, seed=16)
    # An affine map of the inputs is undone by the scaler fitted on the statistics, so the scores agree ...
    np.testing.assert_allclose(
        restandardised.score(x, mask)["distance"], plain.score(x, mask)["distance"], rtol=1e-4, atol=1e-4
    )
    # ... while the unscaled statistics differ by exactly that map.
    raw = WindowStatistics()(x, mask, transform=restandardised.standardise)
    np.testing.assert_allclose(raw[:, :F], (WindowStatistics()(x, mask)[:, :F] - 0.7) / 2.5, atol=1e-5)
