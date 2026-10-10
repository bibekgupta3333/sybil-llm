"""Unit tests for the detection exporter (tiny synthetic encoder, run folder and shards; CPU; no src/data access)."""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import export_detection_sample as exp  # noqa: E402
from src.model.benign_gridsybil.timesnet.config import PretrainConfig  # noqa: E402
from src.model.benign_gridsybil.timesnet.data import FEATURES  # noqa: E402
from src.model.benign_gridsybil.timesnet.encoder import TimesNetEncoder  # noqa: E402

_MEAN = np.linspace(10.0, 130.0, len(FEATURES))
_STD = np.linspace(1.0, 13.0, len(FEATURES))
_CODES = {"Benign": 0, "GridSybil": 16, "DataReplaySybil": 17, "DoSRandomSybil": 18, "DoSDisruptiveSybil": 19}


def _write_gz(path: pathlib.Path, obj: dict) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(obj, f)


def _shard(path: pathlib.Path, split: str, windows: list[dict]) -> None:
    """Feature shard + `_info` companion; x is stored as train z-scores (4 decimals), like the real input."""
    feats = []
    for w in windows:
        z = (np.asarray(w["raw"]) - _MEAN) / _STD
        feats.append({"id": w["id"], "n": len(w["raw"]), "x": np.round(z, 4).tolist()})
    _write_gz(path, {"seq_len": 24, "features": FEATURES, "windows": feats})
    info = [
        {
            "id": w["id"],
            "scenario": w["id"].split("/", 1)[0],
            "label": _CODES[w["cls"]],
            "label_name": w["cls"],
            "is_attack": int(w["cls"] != "Benign"),
        }
        for w in windows
    ]
    _write_gz(path.with_name(path.name.replace(".json.gz", "_info.json.gz")), {"split": split, "windows": info})


def _window(rng: np.random.Generator, scenario: str, sender: int, k: int, cls: str) -> dict:
    n = int(rng.integers(1, 25)) if cls in ("Benign", "GridSybil") else int(rng.integers(1, 4))
    raw = _MEAN + _STD * rng.normal(size=(n, len(FEATURES))) + (3.0 if cls != "Benign" else 0.0)
    return {"id": f"{scenario}/run0/traceJSON-1.json#{5000 + sender}-{sender}#{k}", "raw": raw, "cls": cls}


@pytest.fixture()
def world(tmp_path: pathlib.Path) -> dict:
    """A run folder (tiny encoder), an encoder input with train + test shards, and an output folder."""
    rng = np.random.default_rng(3)
    data = tmp_path / "input"
    (data / "train").mkdir(parents=True)
    (data / "test").mkdir()
    metadata = {"features": FEATURES, "normalisation": {"mean": _MEAN.tolist(), "std": _STD.tolist()}}
    (data / "metadata.json").write_text(json.dumps(metadata))
    train = [
        _window(rng, f"{fam}_{grp}", s, k, "Benign" if s % 3 else fam)
        for s in range(60)
        for k in range(4)
        for fam, grp in [(("GridSybil", "DoSRandomSybil")[s % 2], ("0709", "1416")[k % 2])]
    ]
    _shard(data / "train" / "part-00000.json.gz", "train", train[:120])
    _shard(data / "train" / "part-00001.json.gz", "train", train[120:])
    test = []
    for i, cls in enumerate(exp.CLASSES * 8):
        family = "GridSybil" if cls == "Benign" else cls
        for grp in ("0709", "1416"):
            test.append(_window(rng, f"{family}_{grp}", 1000 + i, 0, cls))
    _shard(data / "test" / "part-00000.json.gz", "test", test)

    settings = PretrainConfig(
        dataset="all", data_dir=str(data), seq_len=24, d_model=8, d_ff=4, n_blocks=1, holdout_frac=0.3
    )
    run = tmp_path / "model-all"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(dataclasses.asdict(settings)))
    scaling = {name: {"method": "train z-score", "mean": float(m)} for name, m in zip(FEATURES, _MEAN)}
    scaling["claimed_pos_x"] = {"method": "median / IQR + soft tail", "median": 9.0, "iqr": 2.0, "soft_clip": 5.0}
    (run / "env.json").write_text(json.dumps({"scaling": scaling}))
    torch.manual_seed(0)
    encoder = TimesNetEncoder(settings)
    torch.save({"encoder": encoder.state_dict(), "epoch": 3, "best": 0.5}, run / "best.pt")
    digest = hashlib.sha256((run / "best.pt").read_bytes()).hexdigest()
    (run / "SHA256SUMS").write_text(f"{digest}  best.pt\n")
    out = tmp_path / "out"
    cfg = exp.DetectionConfig(
        run_dir=run,
        out_dir=out,
        allowed_out_root=out,
        per_class=5,
        bank=40,
        calib=10,
        k=3,
        device="cpu",
        workers=1,
        candidate_rate=1.0,
        batch_size=16,
    )
    return {"cfg": cfg, "out": out, "test": test}


def test_auroc_known_cases() -> None:
    assert exp.auroc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert exp.auroc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0
    assert exp.auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5
    assert exp.auroc(np.array([2.0, 0.0]), np.array([1.0])) == 0.5
    assert exp.auroc(np.array([]), np.array([1.0])) is None


def test_knn_score_matches_brute_force() -> None:
    gen = torch.Generator().manual_seed(1)
    bank = torch.nn.functional.normalize(torch.randn(50, 6, generator=gen), dim=1)
    query = torch.nn.functional.normalize(torch.randn(7, 6, generator=gen), dim=1)
    got = exp.KnnScorer(bank, 4, torch.device("cpu"), batch_size=3)(query)
    sims = (query @ bank.T).numpy()
    want = 1.0 - np.sort(sims, axis=1)[:, -4:].mean(axis=1)
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_knn_refuses_k_above_bank() -> None:
    with pytest.raises(exp.ExportError):
        exp.KnnScorer(torch.zeros(2, 3), 3, torch.device("cpu"))


def test_threshold_rule_and_rank() -> None:
    calib = np.arange(1, 101, dtype=float)
    th = exp.Threshold(calib, 0.95)
    assert th.theta == pytest.approx(np.quantile(calib, 0.95))
    assert th.flags(np.array([th.theta, th.theta + 1e-9])).tolist() == [False, True]
    np.testing.assert_allclose(th.rank_pct(np.array([0.0, 50.0, 1000.0])), [0.0, 50.0, 100.0])


def test_label_vault_refuses_before_scoring(tmp_path: pathlib.Path) -> None:
    vault = exp.LabelVault([tmp_path / "missing_info.json.gz"])
    with pytest.raises(PermissionError):
        vault.reveal(["a", "b"], None)
    with pytest.raises(PermissionError):
        vault.reveal(["a", "b"], np.array([0.1]))
    with pytest.raises(PermissionError):
        vault.reveal(["a"], np.array([np.nan]))
    assert not vault.opened


def test_test_reader_refuses_label_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(PermissionError):
        exp.GroupTestReader(24, "1416")(str(tmp_path / "part-00000_info.json.gz"))


def test_scaling_from_report_uses_robust_entries() -> None:
    metadata = {"features": FEATURES, "normalisation": {"mean": _MEAN.tolist(), "std": _STD.tolist()}}
    report = {"range": {"method": "median / IQR + soft tail", "median": 7.0, "iqr": 3.0, "soft_clip": 4.0}}
    space = exp.TrainedRun.scaling_from_report(metadata, report)
    j = FEATURES.index("range")
    assert float(space.mean[j]) == 7.0 and float(space.std[j]) == 3.0 and bool(space.robust[j])
    assert space.soft_clip == 4.0 and int(space.robust.sum()) == 1
    bad = {"bearing": {"method": "train z-score", "mean": 999.0}}
    with pytest.raises(exp.ExportError):
        exp.TrainedRun.scaling_from_report(metadata, bad)


def test_writer_refuses_outside_root(tmp_path: pathlib.Path) -> None:
    with pytest.raises(exp.ExportError):
        exp.BundleWriter(tmp_path / "elsewhere", tmp_path / "out")
    with pytest.raises(exp.ExportError):
        exp.BundleWriter(exp.REPO_ROOT / "src" / "data" / "x", exp.REPO_ROOT / "src")


def test_checkpoint_sum_mismatch_fails(world: dict) -> None:
    run = world["cfg"].run_dir
    (run / "SHA256SUMS").write_text("0" * 64 + "  best.pt\n")
    with pytest.raises(exp.ExportError):
        exp.DetectionExport(world["cfg"])


def test_export_end_to_end(world: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    real_score, real_reveal = exp.KnnScorer.__call__, exp.LabelVault.reveal

    def score(self, z):  # noqa: ANN001
        events.append("score")
        return real_score(self, z)

    def reveal(self, ids, scores):  # noqa: ANN001
        events.append("labels")
        return real_reveal(self, ids, scores)

    monkeypatch.setattr(exp.KnnScorer, "__call__", score)
    monkeypatch.setattr(exp.LabelVault, "reveal", reveal)
    summary = exp.DetectionExport(world["cfg"]).run_all()
    assert events == ["score", "score", "labels"]  # calibration, test, then labels

    manifest = json.loads((world["out"] / "manifest.json").read_text())
    assert set(manifest) == {
        "schema",
        "created",
        "git_commit",
        "model",
        "method",
        "scope",
        "features",
        "units",
        "metrics",
        "caveats",
        "windows",
    }
    assert manifest["schema"] == 1
    assert set(manifest["model"]) == {
        "run_id",
        "checkpoint",
        "sha256",
        "epoch",
        "best_val_joint",
        "dataset",
        "seq_len",
        "d_model",
    }
    assert manifest["model"]["epoch"] == 3 and manifest["model"]["best_val_joint"] == 0.5
    assert set(manifest["method"]) == {"name", "k", "bank_size", "theta", "theta_rule", "seed"}
    assert set(manifest["scope"]) == {"split", "group", "note", "per_class"}
    assert set(manifest["metrics"]) == {"sample", "full_test_1416", "benign_fpr_at_theta"}
    assert manifest["features"] == FEATURES and set(manifest["units"]) == set(FEATURES)
    for name in exp.CLASSES:
        entry = manifest["metrics"]["sample"][name]
        assert set(entry) == {"n", "auroc_vs_benign", "flag_rate", "length_baseline_auroc", "mean_rows"}
        assert entry["n"] == 5
        assert manifest["metrics"]["full_test_1416"][name]["n"] == 8
    assert manifest["metrics"]["sample"]["Benign"]["auroc_vs_benign"] is None
    assert manifest["method"]["theta"] == pytest.approx(summary["theta"])

    windows = manifest["windows"]
    assert len(windows) == 25
    assert all(w["scenario"].endswith("_1416") for w in windows)
    assert set(windows[0]) == {"i", "id", "scenario", "class", "label", "n", "offset", "score", "flag", "rank_pct"}
    offsets = np.cumsum([0] + [w["n"] for w in windows[:-1]])
    assert [w["offset"] for w in windows] == offsets.tolist()
    assert all(w["flag"] == (w["score"] > manifest["method"]["theta"]) for w in windows)
    assert all(w["label"] == int(w["class"] != "Benign") for w in windows)

    blob = (world["out"] / "x.f32").read_bytes()
    assert len(blob) == sum(w["n"] for w in windows) * 13 * 4
    rows = np.frombuffer(blob, dtype="<f4").reshape(-1, 13)
    raw_by_id = {w["id"]: w["raw"] for w in world["test"]}
    for w in windows[:6]:
        np.testing.assert_allclose(rows[w["offset"] : w["offset"] + w["n"]], raw_by_id[w["id"]], atol=2e-3)
