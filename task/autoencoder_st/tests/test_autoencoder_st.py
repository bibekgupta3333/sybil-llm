"""Tests for the supervised T48 autoencoder on synthetic shards (CPU only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from task.autoencoder_st.config import N_FEATURES, SEQ_LEN, TOTAL_PARAMS, SupervisedAEConfig
from task.autoencoder_st.data import WindowDataset, track_id
from task.autoencoder_st.metrics import (
    TrackScores,
    epoch_metrics,
    fpr_threshold,
    max_f1_threshold,
    threshold_metrics,
    val_thresholds,
)
from task.autoencoder_st.model import SupervisedAutoencoder, count_params, masked_mse, window_errors
from task.autoencoder_st.train import Experiment
from task.autoencoder_st.trainer import EarlyStopping, Trainer


def write_shards(root: Path, split: str, n_links: int = 24, seed: int = 0, sender_base: int = 0) -> None:
    """Synthetic `<split>/b48/part-XXXXX.json` + `_info.json`: links cut at 48 (no overlap), junk in padding rows.

    GridSybil windows get a feature offset so the classifier has something to learn.
    """
    rng = np.random.default_rng(seed)
    folder = root / split / "b48"
    folder.mkdir(parents=True, exist_ok=True)
    windows, info = [], []
    for j in range(n_links):
        n_link = int(rng.choice([5, 30, 60, 100]))
        label = 16 if j % 3 else 0
        link = rng.normal(size=(n_link, N_FEATURES)) + (1.5 if label else 0.0)
        scenario = "GridSybil_0709" if j % 2 else "GridSybil_1416"
        sender = sender_base + j
        for k, start in enumerate(range(0, n_link, SEQ_LEN)):
            n = min(n_link - start, SEQ_LEN)
            x = np.full((SEQ_LEN, N_FEATURES), 7.0)
            x[:n] = link[start : start + n]
            wid = f"{scenario}/run/trace-{j}.json#1-{sender}#{k}"
            windows.append({"id": wid, "x": x.round(4).tolist(), "mask": [1] * n + [0] * (SEQ_LEN - n)})
            info.append(
                {
                    "id": wid,
                    "scenario": scenario,
                    "sender": sender,
                    "receiver": 1,
                    "link_window_index": k,
                    "n_messages": n,
                    "label": label,
                    "label_name": "GridSybil" if label else "Benign",
                    "is_attack": int(label > 0),
                }
            )
    half = len(windows) // 2
    for part, (lo, hi) in enumerate([(0, half), (half, len(windows))]):
        (folder / f"part-{part:05d}.json").write_text(json.dumps({"windows": windows[lo:hi]}))
        (folder / f"part-{part:05d}_info.json").write_text(json.dumps({"windows": info[lo:hi]}))


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    root = tmp_path / "T48_tvt"
    write_shards(root, "train", n_links=36, seed=0, sender_base=0)
    write_shards(root, "val", n_links=12, seed=1, sender_base=1000)
    return root


def small_settings(data_dir: Path, runs: Path, **changes) -> SupervisedAEConfig:
    return SupervisedAEConfig(
        data_dir=str(data_dir), runs_dir=str(runs), batch_size=16, device="cpu", workers=2, **changes
    )


def test_track_id() -> None:
    assert track_id("G_0709/r/t.json#1-10635#2") == "G_0709/r/t.json#1-10635"
    assert track_id("G_0709/r/t.json#1-10635#0") == track_id("G_0709/r/t.json#1-10635#1")
    for bad in ("G_0709/r/t.json#1-10635", "G_0709/r/t.json"):
        with pytest.raises(ValueError):
            track_id(bad)


def test_dataset_labels_padding_and_tracks(data_dir: Path) -> None:
    data = WindowDataset.load(data_dir, "train", workers=2)
    assert data.x.shape[1:] == (SEQ_LEN, N_FEATURES) and data.mask.shape == (len(data), SEQ_LEN)
    assert np.all(data.x[~data.mask] == 0.0)
    assert len(np.unique(data.tracks)) == 36 and len(data) > 36
    assert set(np.unique(data.y)) == {0.0, 1.0} and data.y.dtype == np.float32
    counts = data.check_labels()
    assert counts["benign"] + counts["gridsybil"] == len(data)
    assert TrackScores(data.tracks).track_labels(data.y)[1] == 0


def test_check_labels_rejects_one_class(data_dir: Path) -> None:
    data = WindowDataset.load(data_dir, "train", workers=2)
    with pytest.raises(ValueError):
        data.take(np.flatnonzero(data.y == 1)).check_labels()


def test_model_shapes_and_params() -> None:
    model = SupervisedAutoencoder(SupervisedAEConfig())
    x, m = torch.randn(4, SEQ_LEN, N_FEATURES), torch.ones(4, SEQ_LEN, dtype=torch.bool)
    recon, z, logit = model(x, m)
    assert recon.shape == (4, SEQ_LEN, N_FEATURES) and z.shape == (4, 16) and logit.shape == (4,)
    assert count_params(model.autoencoder) == 71_741 and count_params(model.head) == 577
    assert count_params(model) == TOTAL_PARAMS == 72_318


def test_masked_loss_ignores_padding() -> None:
    x, recon = torch.randn(3, SEQ_LEN, N_FEATURES), torch.randn(3, SEQ_LEN, N_FEATURES)
    mask = torch.zeros(3, SEQ_LEN, dtype=torch.bool)
    mask[:, :10] = True
    changed = recon.clone()
    changed[:, 10:] += 100.0
    assert torch.allclose(masked_mse(recon, x, mask), masked_mse(changed, x, mask))
    e1, f1 = window_errors(recon, x, mask)
    e2, f2 = window_errors(changed, x, mask)
    assert torch.allclose(e1, e2) and torch.allclose(f1, f2)


def test_padding_input_does_not_change_outputs() -> None:
    model = SupervisedAutoencoder(SupervisedAEConfig()).eval()
    x = torch.randn(2, SEQ_LEN, N_FEATURES)
    mask = torch.zeros(2, SEQ_LEN, dtype=torch.bool)
    mask[:, :5] = True
    junk = x.clone()
    junk[:, 5:] = 50.0
    with torch.no_grad():
        for a, b in zip(model(x, mask), model(junk, mask)):
            assert torch.allclose(a, b)


def test_pos_weight_in_loss() -> None:
    settings = SupervisedAEConfig(recon_weight=0.0)
    trainer = Trainer(SupervisedAutoencoder(settings), settings, torch.device("cpu"), pos_weight=0.25)
    x, m = torch.zeros(2, SEQ_LEN, N_FEATURES), torch.ones(2, SEQ_LEN, dtype=torch.bool)
    logit, y = torch.zeros(2), torch.tensor([1.0, 0.0])
    total, bce, _ = trainer.losses(x, logit, x, m, y)
    assert torch.isclose(bce, torch.tensor((0.25 * np.log(2) + np.log(2)) / 2, dtype=torch.float32))
    assert torch.isclose(total, bce)


def test_pos_weight_from_train_counts(data_dir: Path, tmp_path: Path) -> None:
    summary = Experiment(small_settings(data_dir, tmp_path / "runs", max_epochs=1)).run()
    c = summary["class_counts"]["train"]
    assert summary["pos_weight"] == pytest.approx(c["benign"] / c["gridsybil"])


def test_metrics_on_toy_data() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.4, 0.6, 0.35, 0.8, 0.9])
    m = epoch_metrics(y, s)
    assert m["auroc"] == pytest.approx(roc_auc_score(y, s)) == pytest.approx(7 / 9)
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, s))
    t = threshold_metrics(y, s, 0.5)
    assert t["confusion"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}
    assert t["precision"] == pytest.approx(2 / 3) and t["recall"] == pytest.approx(2 / 3)
    assert t["f1"] == pytest.approx(2 / 3) and t["benign_fpr"] == pytest.approx(1 / 3)
    assert t["balanced_accuracy"] == pytest.approx(2 / 3) == m["balanced_accuracy"]


def test_threshold_picks() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.4, 0.6, 0.35, 0.8, 0.9])
    assert max_f1_threshold(y, s) == pytest.approx(0.8)  # precision 1, recall 2/3 -> F1 0.8 (0.35 gives 0.75)
    benign = np.linspace(0.0, 1.0, 101)
    assert fpr_threshold(benign, 0.05) == pytest.approx(0.95)
    picks = val_thresholds(y, s, 0.05)
    assert set(picks) == {"max_f1", "benign_fpr_5pct"}
    assert picks["max_f1"]["f1"] == pytest.approx(0.8)
    assert picks["benign_fpr_5pct"]["benign_fpr"] <= 1 / 3


def test_track_aggregation_and_labels() -> None:
    tracks = np.array(["b", "a", "b", "c", "a", "b"])
    agg = TrackScores(tracks)
    out = agg.aggregate(np.array([1.0, 5.0, 3.0, 2.0, 1.0, 2.0]))
    assert list(agg.tracks) == ["a", "b", "c"] and list(agg.n_windows) == [2, 3, 1]
    assert np.allclose(out["max"], [5.0, 3.0, 2.0]) and np.allclose(out["mean"], [3.0, 2.0, 2.0])
    labels, mixed = agg.track_labels(np.array([1, 0, 1, 1, 0, 1]))
    assert list(labels) == [0, 1, 1] and mixed == 0
    assert agg.track_labels(np.array([1, 0, 0, 1, 0, 1]))[1] == 1


def test_early_stopping_absolute_min_delta() -> None:
    stop = EarlyStopping(patience=2, min_delta=1e-2)
    assert stop.update(1, 0.70)
    assert not stop.update(2, 0.705)  # +0.005 < 0.01 absolute
    assert stop.update(3, 0.72) and stop.best_epoch == 3 and stop.waited == 0
    assert not stop.update(4, 0.71) and not stop.update(5, 0.729)
    assert stop.stop


def test_one_epoch_smoke_writes_run_folder(data_dir: Path, tmp_path: Path) -> None:
    settings = small_settings(data_dir, tmp_path / "runs", max_epochs=2)
    summary = Experiment(settings).run()
    run = tmp_path / "runs" / settings.run_id
    for name in ("config.json", "env.json", "history.jsonl", "train.log", "curves.png", "best.pt", "last.pt"):
        assert (run / name).exists(), name
    rows = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    assert len(rows) == summary["epochs_run"] == 2
    for key in ("loss", "bce", "recon", "auroc", "pr_auc", "f1", "balanced_accuracy"):
        assert f"train_{key}" in rows[0] and f"val_{key}" in rows[0]
    assert all(r["grad_norm_mean"] > 0 and r["grad_norm_max"] >= r["grad_norm_mean"] and r["lr"] > 0 for r in rows)
    assert "*best" in (run / "train.log").read_text()
    assert summary["params"] == TOTAL_PARAMS and summary["best_epoch"] >= 1
    assert summary["best_val_auroc"] == max(r["val_auroc"] for r in rows[: summary["best_epoch"]])
    assert set(summary["thresholds"]["window"]) == {"max_f1", "benign_fpr_5pct"}
    assert set(summary["thresholds"]["track"]) == {"max", "mean"}
    assert summary["threshold_source"] == "val split" and summary["val_tracks"] == 12

    trainer = Trainer.from_checkpoint(run / "best.pt", device="cpu")
    out = trainer.predict(np.zeros((3, SEQ_LEN, N_FEATURES), np.float32), np.ones((3, SEQ_LEN), bool))
    assert out["prob"].shape == (3,) and out["z"].shape == (3, 16) and np.all((out["prob"] >= 0) & (out["prob"] <= 1))
    with pytest.raises(FileExistsError):
        Experiment(settings).run()


def test_resume_matches_uninterrupted_run(data_dir: Path, tmp_path: Path) -> None:
    full = Experiment(small_settings(data_dir, tmp_path / "full", max_epochs=3)).run()
    part_settings = small_settings(data_dir, tmp_path / "part", max_epochs=2)
    Experiment(part_settings).run()
    run = tmp_path / "part" / part_settings.run_id
    (run / "summary.json").unlink()
    resumed_settings = small_settings(data_dir, tmp_path / "part", max_epochs=3)
    config = json.loads((run / "config.json").read_text())
    config["max_epochs"] = 3
    (run / "config.json").write_text(json.dumps(config))
    resumed = Experiment(resumed_settings, resume=True).run()
    rows_full = [json.loads(x) for x in (tmp_path / "full" / "seed0" / "history.jsonl").read_text().splitlines()]
    rows_resumed = [json.loads(x) for x in (run / "history.jsonl").read_text().splitlines()]
    assert len(rows_resumed) == resumed["epochs_run"] == 3
    for a, b in zip(rows_full, rows_resumed):
        assert a["val_loss"] == pytest.approx(b["val_loss"], rel=1e-5)
        assert a["train_loss"] == pytest.approx(b["train_loss"], rel=1e-5)
    assert resumed["best_val_auroc"] == pytest.approx(full["best_val_auroc"])
