"""Unit tests for the frozen-encoder evaluation (tiny synthetic encoder, run folder and shards; CPU; no src/data)."""

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

import evaluate_frozen_encoder as ev  # noqa: E402
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
    n = int(rng.integers(1, 25)) if cls in ("Benign", "GridSybil") else int(rng.integers(1, 6))
    raw = _MEAN + _STD * rng.normal(size=(n, len(FEATURES))) + (3.0 if cls != "Benign" else 0.0)
    return {"id": f"{scenario}/run0/traceJSON-1.json#{5000 + sender}-{sender}#{k}", "raw": raw, "cls": cls}


@pytest.fixture()
def world(tmp_path: pathlib.Path) -> dict:
    rng = np.random.default_rng(5)
    data = tmp_path / "input"
    (data / "train").mkdir(parents=True)
    (data / "test").mkdir()
    train = []
    for s in range(100):
        for k in range(4):
            cls = ev.CLASSES[s % 5]
            family = "GridSybil" if cls == "Benign" else cls
            train.append(_window(rng, f"{family}_{('0709', '1416')[k % 2]}", s, k, cls))
    counts: dict = {}
    for w in train:
        fam = w["id"].split("_", 1)[0]
        counts.setdefault(fam, {}).setdefault(w["cls"], {"windows": 0})["windows"] += 1
    metadata = {
        "features": FEATURES,
        "normalisation": {"mean": _MEAN.tolist(), "std": _STD.tolist()},
        "counts": {"train": counts},
    }
    (data / "metadata.json").write_text(json.dumps(metadata))
    _shard(data / "train" / "part-00000.json.gz", "train", train[:200])
    _shard(data / "train" / "part-00001.json.gz", "train", train[200:])
    test = []
    for i, cls in enumerate(ev.CLASSES * 10):
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
    (run / "env.json").write_text(json.dumps({"scaling": scaling}))
    torch.manual_seed(0)
    torch.save({"encoder": TimesNetEncoder(settings).state_dict(), "epoch": 2, "best": 0.4}, run / "best.pt")
    digest = hashlib.sha256((run / "best.pt").read_bytes()).hexdigest()
    (run / "SHA256SUMS").write_text(f"{digest}  best.pt\n")
    out = tmp_path / "evaluation" / "model-all"
    cfg = ev.EvalConfig(
        run_dir=run,
        out_dir=out,
        allowed_out_root=tmp_path / "evaluation",
        per_class=15,
        seeds=(0, 1),
        knn_k=3,
        lr_max_iter=200,
        bank=40,
        calib=10,
        anomaly_k=3,
        bootstrap=20,
        candidate_margin=3.0,
        bank_candidate_rate=1.0,
        device="cpu",
        batch_size=16,
        knn_batch=7,
        workers=1,
    )
    return {"cfg": cfg, "out": out, "test": test}


def test_auroc_spec_matches_exporter_and_weights_equal_repeats() -> None:
    rng = np.random.default_rng(0)
    scores = np.round(rng.normal(size=60), 1)  # ties on purpose
    pos, neg = np.arange(0, 25), np.arange(25, 60)
    assert ev.AurocSpec(scores, pos, neg)() == pytest.approx(exp.auroc(scores[pos], scores[neg]))
    w = rng.integers(0, 3, size=60).astype(float)
    rep_pos = np.repeat(scores[pos], w[pos].astype(int))
    rep_neg = np.repeat(scores[neg], w[neg].astype(int))
    assert ev.AurocSpec(scores, pos, neg)(w) == pytest.approx(exp.auroc(rep_pos, rep_neg))
    assert ev.Metrics.auroc(scores, np.array([], int), neg) is None


def test_f1_and_recall_from_confusion() -> None:
    true = np.array([0, 0, 1, 1, 1, 2])
    pred = np.array([0, 1, 1, 1, 0, 2])
    conf = ev.Metrics.confusion(true, pred, 3)
    assert conf.tolist() == [[1, 1, 0], [1, 2, 0], [0, 0, 1]]
    f1 = ev.Metrics.f1_from_confusion(conf)
    np.testing.assert_allclose(f1, [0.5, 2 / 3, 1.0])
    assert ev.Metrics.recall(conf) == [0.5, 2 / 3, 1.0]


def test_vehicle_bootstrap_keeps_vehicles_together() -> None:
    vehicles = ["a", "a", "b", "c", "c", "c"]
    for w in ev.VehicleBootstrap(vehicles, 10, 0).weights():
        assert w[0] == w[1] and w[3] == w[4] == w[5]
        assert w[[0, 2, 3]].sum() == 3  # three vehicles drawn


def test_strata_split_by_length() -> None:
    rows = ev.FrozenEncoderEvaluation.strata_rows(np.array([1, 3, 4, 24]))
    assert rows["n<=3"].tolist() == [True, True, False, False]
    assert rows["n>=4"].tolist() == [False, False, True, True]
    assert rows["all"].all()


def test_stats_features_ignore_padding(world: dict) -> None:
    run = exp.TrainedRun(world["cfg"].run_dir, "best.pt", None)
    x = torch.zeros(1, 24, len(FEATURES))
    x[0, :2] = torch.tensor([[1.0] * 13, [3.0] * 13])
    x[0, 2:] = 99.0  # padding garbage must not count
    mask = torch.zeros(1, 24, dtype=torch.bool)
    mask[0, :2] = True
    feats = ev.StatsFeatures(run)(exp.WindowSet(x, mask, ["w"]))
    assert feats.shape == (1, 53)
    np.testing.assert_allclose(feats[0, :13], 2.0, atol=1e-5)
    np.testing.assert_allclose(feats[0, 13:26], 1.0, atol=1e-5)
    np.testing.assert_allclose(feats[0, 26:39], 1.0, atol=1e-5)
    np.testing.assert_allclose(feats[0, 39:52], 3.0, atol=1e-5)
    assert feats[0, 52] == 2.0


def test_knn_probe_votes() -> None:
    x = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    y = np.array([0, 0, 1, 1])
    votes = ev.KnnProbe(2, "cosine", torch.device("cpu"), 1).fit(x, y).votes(np.array([[1.0, 0.05]]), 2)
    assert votes.tolist() == [[1.0, 0.0]]


def test_writer_refuses_protected(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ev.EvalError):
        ev.ResultWriter(tmp_path / "x", tmp_path / "y")
    with pytest.raises(ev.EvalError):
        ev.ResultWriter(ev.REPO_ROOT / "src/runs/pretraining/x", ev.REPO_ROOT / "src/runs")


def test_end_to_end_labels_after_predictions(world: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    real_probe, real_reveal = ev.FrozenEncoderEvaluation._probe, exp.LabelVault.reveal

    def probe(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        events.append("predict")
        return real_probe(self, *args, **kwargs)

    def reveal(self, ids, scores):  # noqa: ANN001
        events.append("labels")
        return real_reveal(self, ids, scores)

    monkeypatch.setattr(ev.FrozenEncoderEvaluation, "_probe", probe)
    monkeypatch.setattr(exp.LabelVault, "reveal", reveal)
    results = ev.FrozenEncoderEvaluation(world["cfg"]).run_all()
    assert events.index("labels") == len(events) - 1 and events.count("predict") == 8

    out = world["out"]
    assert {p.name for p in out.iterdir()} == {"results.json", "config.json", "env.json", "report_data.json"}
    on_disk = json.loads((out / "results.json").read_text())
    assert set(on_disk) == {"schema", "created", "scope", "data", "probes", "metrics", "bootstrap", "timings_s"}
    assert results["data"]["test_windows"] == 50  # 1416 only
    assert all(results["data"]["test"][c]["n"] == 10 for c in ev.CLASSES)
    for seed in ("seed0", "seed1"):
        assert all(results["data"]["fit"][seed][c]["n"] == 15 for c in ev.CLASSES)
    for rep in ev.REPRESENTATIONS:
        for probe_name in ("lr", "knn", "anomaly"):
            summary = on_disk["metrics"][rep][probe_name]["summary"]
            assert len(summary["binary_auroc"]["values"]) == 2
            n_all = sum(summary["by_length"]["all"][c]["n"] for c in ev.CLASSES)
            n_split = sum(summary["by_length"][s][c]["n"] for s in ("n<=3", "n>=4") for c in ev.CLASSES)
            assert n_all == n_split == 50
        assert "macro_f1" in on_disk["metrics"][rep]["lr"]["summary"]
        ci = on_disk["bootstrap"]["ci"][rep]["lr"]["binary_auroc"]
        assert ci["n"] == 20 and ci["lo"] <= ci["hi"]
    report = json.loads((out / "report_data.json").read_text())
    assert [r["representation"] for r in report["main_table"]] == list(ev.REPRESENTATIONS)
    assert len(report["per_family"]) == 4 * 3 * 4
    config = json.loads((out / "config.json").read_text())
    assert config["evaluation"]["seeds"] == [0, 1] and config["run"]["epoch"] == 2
    env = json.loads((out / "env.json").read_text())
    assert {"python", "torch", "sklearn", "device", "git_commit"} <= set(env)
