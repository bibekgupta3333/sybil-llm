"""Tests for `src/model/all_t32_benign/timesnet/scoring.py` (synthetic arrays, no data reads)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.model.all_t32_benign.timesnet import scoring


def test_mahalanobis_matches_numpy_and_no_shrinkage_when_well_conditioned():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(500, 4)) * np.array([1.0, 2.0, 0.5, 3.0])
    maha = scoring.Mahalanobis().fit(z)
    q = rng.normal(size=(7, 4))
    inv = np.linalg.inv(np.cov(z, rowvar=False))
    dz = q - z.mean(0)
    expected = np.sqrt(np.einsum("nd,de,ne->n", dz, inv, dz))
    assert maha.shrinkage == 0.0
    np.testing.assert_allclose(maha.score(q), expected, rtol=1e-8)


def test_mahalanobis_shrinks_when_ill_conditioned():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(200, 2))
    z = np.c_[a, a[:, 0]]  # exactly collinear column -> singular covariance
    maha = scoring.Mahalanobis(max_cond=1e6, shrinkage=0.01).fit(z)
    assert maha.shrinkage == 0.01
    assert np.isfinite(maha.score(z)).all()


def test_knn_cosine_scores_on_toy_points():
    bank = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # two along x, one along y
    knn = scoring.KNNScorer(k_values=(1, 2), max_bank=10, seed=0).fit(bank)
    out = knn.score(np.array([[5.0, 0.0], [0.0, 3.0], [1.0, 1.0]]))
    np.testing.assert_allclose(out[1], [0.0, 0.0, 1 - np.sqrt(0.5)], atol=1e-6)
    np.testing.assert_allclose(out[2], [0.0, 0.5, 1 - np.sqrt(0.5)], atol=1e-6)  # (0, 1) then 1 - 0 = 1 -> mean 0.5


def test_knn_bank_subsample_is_deterministic_and_bounded():
    z = np.random.default_rng(2).normal(size=(100, 3))
    a = scoring.KNNScorer((5,), max_bank=30, seed=7).fit(z)
    b = scoring.KNNScorer((5,), max_bank=30, seed=7).fit(z)
    c = scoring.KNNScorer((5,), max_bank=30, seed=8).fit(z)
    assert len(a.bank) == 30
    np.testing.assert_array_equal(a.bank_rows, b.bank_rows)
    assert not np.array_equal(a.bank_rows, c.bank_rows)
    with pytest.raises(ValueError):
        scoring.KNNScorer((50,), max_bank=10)


def test_binary_metrics_match_sklearn_and_threshold_is_strict():
    rng = np.random.default_rng(3)
    benign, attack = rng.normal(size=50), rng.normal(loc=1.0, size=30)
    m = scoring.BinaryMetrics.compute(benign, attack, {"p95": 1.0})
    y = np.r_[np.zeros(50), np.ones(30)]
    s = np.r_[benign, attack]
    assert m.auroc == pytest.approx(roc_auc_score(y, s))
    assert m.pr_auc == pytest.approx(average_precision_score(y, s))
    assert m.chance == pytest.approx(30 / 80)
    assert m.at_threshold["p95"]["recall"] == pytest.approx((attack > 1.0).mean())
    assert m.at_threshold["p95"]["benign_fpr"] == pytest.approx((benign > 1.0).mean())
    tie = scoring.BinaryMetrics.compute(np.array([1.0]), np.array([1.0]), {"t": 1.0})
    assert tie.at_threshold["t"]["recall"] == 0.0  # equal to the threshold is not an alarm
    empty = scoring.BinaryMetrics.compute(np.array([]), np.array([1.0, 2.0]))
    assert empty.auroc is None and empty.pr_auc is None


def test_quantile_thresholds_names_and_values():
    th = scoring.quantile_thresholds(np.arange(101, dtype=float), (0.95, 0.99))
    assert th == {"p95": pytest.approx(95.0), "p99": pytest.approx(99.0)}


def test_track_scores_max_mean_and_mixed_labels():
    ids = ["f#1-2#0", "f#1-2#1", "f#3-4#0", "g#1-2#0"]
    tracks = scoring.TrackScores(ids)
    assert list(tracks.tracks) == ["f#1-2", "f#3-4", "g#1-2"]
    agg = tracks.aggregate(np.array([1.0, 3.0, 5.0, 2.0]))
    np.testing.assert_allclose(agg["max"], [3.0, 5.0, 2.0])
    np.testing.assert_allclose(agg["mean"], [2.0, 5.0, 2.0])
    _, mixed = tracks.track_labels(np.array([0, 16, 0, 0]))
    assert mixed == 1


def test_length_score_direction_is_fixed():
    s = scoring.length_score(np.array([1, 2, 32]))
    assert s[0] > s[1] > s[2]  # shorter = more suspicious, whatever the labels


def _split(ids, n_rows, recon):
    return scoring.ScoredSplit(np.asarray(ids), np.asarray(n_rows), {"recon": np.asarray(recon, dtype=float)})


def test_detection_report_subsets_tracks_and_length_reference():
    val = _split([f"v#1-{i}#0" for i in range(20)], np.arange(1, 21), np.linspace(0, 1, 20))
    report = scoring.DetectionReport(val, (0.95, 0.99))
    assert set(report.thresholds["window"]) == {"recon", "length_only"}
    assert report.thresholds["window"]["recon"]["p95"] == pytest.approx(np.quantile(np.linspace(0, 1, 20), 0.95))
    benign = _split(["b#1-1#0", "b#1-1#1", "b#2-2#0", "b#3-3#0"], [32, 10, 2, 20], [0.1, 0.2, 0.3, 0.1])
    attack = _split(["a#1-1#0", "a#2-2#0", "a#3-3#0"], [1, 2, 8], [0.9, 0.8, 0.05])
    out = report.compare(benign, attack, np.zeros(4, int), np.full(3, 16))
    w = out["window"]
    assert w["recon"]["all"]["auroc"] == pytest.approx(
        roc_auc_score([0] * 4 + [1] * 3, [0.1, 0.2, 0.3, 0.1, 0.9, 0.8, 0.05])
    )
    assert out["counts"]["n_le_3"] == {"benign_windows": 1, "attack_windows": 2}
    assert out["counts"]["n_ge_2"] == {"benign_windows": 4, "attack_windows": 2}
    assert out["counts"]["recon_filled_share"] == {"benign": 0.0, "attack": pytest.approx(1 / 3)}
    assert set(w["recon"]) == {"all", "n_ge_2", "n_le_3", "n_ge_4"}
    assert set(w["length_only"]) == {"all", "n_ge_2", "n_le_3", "n_ge_4"}  # every score on the same subsets
    assert w["recon"]["n_ge_2"]["n_attack"] == 2
    assert out["counts"]["n_ge_4"] == {"benign_windows": 3, "attack_windows": 1}
    assert w["length_only"]["all"]["auroc"] > 0.5  # attack windows are shorter here
    assert out["counts"]["tracks"] == {"benign": 3, "attack": 3, "mixed_label_tracks": 0, "shared_track_ids": 0}
    assert out["track"]["recon"]["max"]["n_benign"] == 3
    assert set(out["track"]["recon"]) == {"max", "mean"}
