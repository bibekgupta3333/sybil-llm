"""Tests of the evaluation on synthetic score files that follow the contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from task.encoder.evaluate import EvalConfig, Evaluator, VehicleBootstrap, binary_metrics, parse_args

N = 600


def make_test_set(rng: np.random.Generator) -> dict:
    """Shared ids / labels / lengths / senders of the synthetic test split."""
    label = (rng.random(N) < 0.6).astype(np.int64)
    return {
        "id": np.array([f"w{i}" for i in range(N)]),
        "label": label,
        "n_messages": rng.integers(1, 65, size=N),
        "sender": rng.integers(0, 80, size=N),
    }


def write_baselines(root: Path, test: dict, rng: np.random.Generator) -> Path:
    out = root / "baselines"
    out.mkdir()
    np.savez(
        out / "scores_test.npz",
        id=test["id"],
        label=test["label"],
        n_messages=test["n_messages"],
        sender=test["sender"],
        length_score=-test["n_messages"].astype(float),
        stats_prob=np.clip(0.3 * test["label"] + rng.random(N) * 0.7, 0, 1),
    )
    (out / "summary.json").write_text(json.dumps({"length": {"direction": "-n_messages"}}))
    return out


def write_run(root: Path, name: str, mode: str, seed: int, test: dict, rng: np.random.Generator) -> Path:
    """One run folder in the contract format; windows are stored in a shuffled order."""
    run = root / name
    run.mkdir()
    perm = rng.permutation(N)
    y = test["label"][perm]
    feat = rng.gamma(2.0, 0.1, size=(N, 13)) * (1 + y[:, None])
    if mode == "ae":
        recon, maha, sup = feat.mean(axis=1), rng.gamma(2.0, 1.0, N) + y, np.full(N, np.nan)
    else:
        recon, maha, sup = np.full(N, np.nan), np.full(N, np.nan), np.clip(0.5 * y + 0.5 * rng.random(N), 0, 1)
        feat = np.full((N, 13), np.nan)
    np.savez(
        run / "scores_test.npz",
        id=test["id"][perm],
        label=y,
        n_messages=test["n_messages"][perm],
        recon_error=recon,
        recon_error_feat=feat,
        mahalanobis=maha,
        supervised_prob=sup,
        z=rng.normal(size=(N, 16)) + y[:, None],
    )
    if mode == "ae":
        np.savez(run / "scores_val.npz", recon_error=rng.gamma(2.0, 0.1, 200), mahalanobis=rng.gamma(2.0, 1.0, 200))
    (run / "config.json").write_text(json.dumps({"mode": mode, "seed": seed, "run_id": name}))
    return run


@pytest.fixture()
def setup(tmp_path: Path) -> EvalConfig:
    rng = np.random.default_rng(1)
    test = make_test_set(rng)
    base = write_baselines(tmp_path, test, rng)
    runs = [write_run(tmp_path, f"ae-s{s}", "ae", s, test, rng) for s in range(3)]
    runs += [write_run(tmp_path, f"sup-s{s}", "supervised", s, test, rng) for s in range(2)]
    return EvalConfig(
        runs=tuple(str(r) for r in runs), baselines_dir=str(base), out_dir=str(tmp_path / "evaluation"), n_boot=50
    )


def test_binary_metrics_confusion():
    m = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.9, 0.8, 0.2]), 0.5)
    assert m["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == m["recall"] == m["f1"] == m["fpr"] == 0.5


def test_bootstrap_ci_brackets_perfect_score():
    senders = np.repeat(np.arange(20), 5)
    y = np.tile([0, 1], 50)
    ci = VehicleBootstrap(senders, 30, 0).auroc_ci(y, y.astype(float))
    assert ci == [1.0, 1.0]


def test_evaluator_end_to_end(setup: EvalConfig):
    results = Evaluator(setup).run()
    methods = results["methods"]
    assert set(methods) == {
        "ae:recon_error",
        "ae:mahalanobis",
        "supervised:supervised_prob",
        "baseline:length",
        "baseline:stats_lr",
    }
    ae = methods["ae:recon_error"]
    assert ae["summary"]["auroc"]["n"] == 3
    assert set(ae["per_seed"][0]["thresholds"]) == {"val_p95", "val_p99"}
    assert methods["supervised:supervised_prob"]["summary"]["auroc"]["n"] == 2
    assert set(methods["supervised:supervised_prob"]["per_seed"][0]["thresholds"]) == {"p0.5"}
    # The AE recon error was built to separate the classes; alignment after the shuffle must keep that.
    assert ae["summary"]["auroc"]["mean"] > 0.8
    lo, hi = ae["per_seed"][0]["auroc_ci95_vehicle_bootstrap"]
    assert lo <= ae["per_seed"][0]["auroc"] <= hi
    assert set(results["strata"]) == {"n<=3", "4-16", "17-48", "49-64"}
    assert sum(s["n_windows"] for s in results["strata"].values()) == N
    assert len(results["per_feature_recon_error"]) == 13
    out = Path(setup.out_dir)
    for name in (
        "results.json",
        "roc_curves.png",
        "strata_auroc.png",
        "error_histograms.png",
        "per_feature_error.png",
        "z_pca.png",
    ):
        assert (out / name).exists(), name


def test_misaligned_ids_are_rejected(setup: EvalConfig, tmp_path: Path):
    run = Path(setup.runs[0])
    data = dict(np.load(run / "scores_test.npz"))
    data["id"] = data["id"].copy()
    data["id"][0] = "unknown"
    np.savez(run / "scores_test.npz", **data)
    with pytest.raises(ValueError, match="not in the baseline file"):
        Evaluator(setup).run()


def test_cli_parses_runs():
    cfg = parse_args(["--runs", "a", "b", "--n-boot", "10"])
    assert cfg.runs == ("a", "b") and cfg.n_boot == 10
