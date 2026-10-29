"""Tests for the T48 overlap autoencoder on synthetic shards (CPU only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from task.autoencoder_overlap.config import N_FEATURES, SEQ_LEN, AutoencoderConfig
from task.autoencoder_overlap.data import WindowDataset, track_id
from task.autoencoder_overlap.model import ConvAutoencoder, count_params, masked_mse, window_errors
from task.autoencoder_overlap.scoring import Mahalanobis, TrackScores, quantile_thresholds
from task.autoencoder_overlap.train import Experiment
from task.autoencoder_overlap.trainer import EarlyStopping, Trainer


def link_starts(n: int, t: int = SEQ_LEN, stride: int = 24) -> list[int]:
    """Window starts of an n-message link (stride 24, end-aligned last window)."""
    if n <= t:
        return [0]
    starts = list(range(0, n - t + 1, stride))
    return starts + ([n - t] if starts[-1] + t < n else [])


def write_shards(root: Path, split: str, n_links: int = 24, seed: int = 0, sender_base: int = 0) -> None:
    """Synthetic `<split>/b48/part-XXXXX.json` + `_info.json`: overlapping windows with `@start` ids, junk padding."""
    rng = np.random.default_rng(seed)
    folder = root / split / "b48"
    folder.mkdir(parents=True, exist_ok=True)
    windows, info = [], []
    for j in range(n_links):
        n_link = int(rng.choice([5, 30, 60, 100]))
        link = rng.normal(size=(n_link, N_FEATURES))
        scenario = "GridSybil_0709" if j % 2 else "GridSybil_1416"
        sender = sender_base + j
        label = 16 if j % 3 else 0
        for k, start in enumerate(link_starts(n_link)):
            n = min(n_link - start, SEQ_LEN)
            x = np.full((SEQ_LEN, N_FEATURES), 7.0)
            x[:n] = link[start : start + n]
            wid = f"{scenario}/run/trace-{j}.json#1-{sender}#{k}@{start}"
            windows.append({"id": wid, "x": x.round(4).tolist(), "mask": [1] * n + [0] * (SEQ_LEN - n)})
            info.append(
                {
                    "id": wid,
                    "scenario": scenario,
                    "sender": sender,
                    "receiver": 1,
                    "link_window_index": k,
                    "start": start,
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
    root = tmp_path / "T48_tvt_s24"
    write_shards(root, "train", seed=0, sender_base=0)
    write_shards(root, "val", n_links=8, seed=1, sender_base=1000)
    return root


def test_track_id_strips_index_and_start() -> None:
    assert track_id("G_0709/r/t.json#1-10635#2@48") == "G_0709/r/t.json#1-10635"
    assert track_id("G_0709/r/t.json#1-10635#0@0") == track_id("G_0709/r/t.json#1-10635#1@24")
    with pytest.raises(ValueError):
        track_id("G_0709/r/t.json#1-10635#0")


def test_dataset_shapes_padding_and_tracks(data_dir: Path) -> None:
    data = WindowDataset.load(data_dir, "train", workers=2)
    assert data.x.shape[1:] == (SEQ_LEN, N_FEATURES) and data.mask.shape == (len(data), SEQ_LEN)
    assert np.all(data.x[~data.mask] == 0.0)
    assert len(np.unique(data.tracks)) == 24 and len(data) > 24
    assert set(data.info["label_name"]) == {"Benign", "GridSybil"}


def test_overlapping_windows_share_rows(data_dir: Path) -> None:
    data = WindowDataset.load(data_dir, "train", workers=2)
    rows = np.flatnonzero(data.info["start"] == 24)
    assert len(rows)
    r = rows[0]
    prev = np.flatnonzero((data.tracks == data.tracks[r]) & (data.info["start"] == 0))[0]
    assert np.allclose(data.x[prev, 24:48], data.x[r, :24])


def test_model_shapes_and_params() -> None:
    model = ConvAutoencoder(AutoencoderConfig())
    x, m = torch.randn(4, SEQ_LEN, N_FEATURES), torch.ones(4, SEQ_LEN, dtype=torch.bool)
    recon, z = model(x, m)
    assert recon.shape == (4, SEQ_LEN, N_FEATURES) and z.shape == (4, 16)
    assert count_params(model) == 71_741


def test_masked_loss_ignores_padding() -> None:
    x, recon = torch.randn(3, SEQ_LEN, N_FEATURES), torch.randn(3, SEQ_LEN, N_FEATURES)
    mask = torch.zeros(3, SEQ_LEN, dtype=torch.bool)
    mask[:, :10] = True
    changed = recon.clone()
    changed[:, 10:] += 100.0
    assert torch.allclose(masked_mse(recon, x, mask), masked_mse(changed, x, mask))
    e1, f1 = window_errors(recon, x, mask)
    e2, f2 = window_errors(changed, x, mask)
    assert torch.allclose(e1, e2) and torch.allclose(f1, f2) and f1.shape == (3, N_FEATURES)


def test_padding_input_does_not_change_output() -> None:
    model = ConvAutoencoder(AutoencoderConfig()).eval()
    x = torch.randn(2, SEQ_LEN, N_FEATURES)
    mask = torch.zeros(2, SEQ_LEN, dtype=torch.bool)
    mask[:, :5] = True
    junk = x.clone()
    junk[:, 5:] = 50.0
    with torch.no_grad():
        assert torch.allclose(model(x, mask)[0], model(junk, mask)[0])


def test_early_stopping_min_delta() -> None:
    stop = EarlyStopping(patience=2, min_delta=1e-2)
    assert stop.update(1, 1.0)
    assert not stop.update(2, 0.995)  # 0.5% better: below the 1% relative min_delta
    assert stop.best == 1.0 and stop.waited == 1 and not stop.stop
    assert stop.update(3, 0.98)  # 2% better: resets the counter
    assert stop.best_epoch == 3 and stop.waited == 0
    assert not stop.update(4, 0.979) and not stop.update(5, 0.99)
    assert stop.stop


def test_mahalanobis_on_toy_gaussian(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    cov = np.array([[4.0, 1.2, 0.0], [1.2, 1.0, 0.3], [0.0, 0.3, 0.5]])
    mean = np.array([1.0, -2.0, 0.5])
    z = rng.multivariate_normal(mean, cov, size=200_000)
    maha = Mahalanobis().fit(z)
    assert maha.shrinkage == 0.0
    assert np.allclose(maha.mean, mean, atol=0.02)
    assert np.allclose(maha.precision, np.linalg.inv(cov), atol=0.03)
    d = maha.score(z)
    assert abs(np.mean(d**2) - 3.0) < 0.05  # E[d^2] = dimension
    probe = mean + np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert np.allclose(maha.score(probe), [0.0, np.sqrt(4.0 * np.linalg.inv(cov)[0, 0])], atol=0.03)
    np.savez(tmp_path / "m.npz", **maha.state())
    restored = Mahalanobis.from_state(dict(np.load(tmp_path / "m.npz")))
    assert np.allclose(restored.score(probe), maha.score(probe))


def test_mahalanobis_shrinks_singular_covariance() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=(5000, 2))
    z = np.column_stack([a, a[:, 0] + a[:, 1]])  # rank 2
    maha = Mahalanobis(max_cond=1e6, shrinkage=0.01).fit(z)
    assert maha.shrinkage == 0.01 and np.all(np.isfinite(maha.score(z)))


def test_track_aggregation_and_labels() -> None:
    tracks = np.array(["b", "a", "b", "c", "a", "b"])
    scores = np.array([1.0, 5.0, 3.0, 2.0, 1.0, 2.0])
    agg = TrackScores(tracks)
    assert list(agg.tracks) == ["a", "b", "c"] and list(agg.n_windows) == [2, 3, 1]
    out = agg.aggregate(scores)
    assert np.allclose(out["max"], [5.0, 3.0, 2.0]) and np.allclose(out["mean"], [3.0, 2.0, 2.0])
    labels, mixed = agg.track_labels(np.array([16, 0, 16, 16, 0, 16]))
    assert list(labels) == [0, 16, 16] and mixed == 0
    labels, mixed = agg.track_labels(np.array([16, 0, 0, 16, 0, 16]))
    assert labels[1] == 16 and mixed == 1


def test_quantile_thresholds() -> None:
    th = quantile_thresholds(np.arange(101.0), (0.95, 0.99))
    assert th == {"p95": 95.0, "p99": 99.0}


def test_experiment_writes_run_folder(data_dir: Path, tmp_path: Path) -> None:
    settings = AutoencoderConfig(
        data_dir=str(data_dir), runs_dir=str(tmp_path / "runs"), max_epochs=2, batch_size=16, device="cpu", workers=2
    )
    summary = Experiment(settings).run()
    run = tmp_path / "runs" / settings.run_id
    for name in (
        "config.json",
        "env.json",
        "history.jsonl",
        "train.log",
        "loss_curve.png",
        "best.pt",
        "mahalanobis.npz",
        "summary.json",
    ):
        assert (run / name).exists(), name
    rows = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    assert len(rows) == summary["epochs_run"] == 2
    assert all("grad_norm" in r and r["grad_norm"] > 0 and "lr" in r for r in rows)
    assert "grad_norm" in (run / "train.log").read_text()
    assert summary["stop_reason"] == "max_epochs" and summary["best_epoch"] >= 1
    assert set(summary["thresholds"]["window"]) == {"reconstruction", "mahalanobis"}
    assert set(summary["thresholds"]["track"]) == {
        "reconstruction_max",
        "reconstruction_mean",
        "mahalanobis_max",
        "mahalanobis_mean",
    }
    assert set(summary["thresholds"]["window"]["mahalanobis"]) == {"p95", "p99"}
    assert summary["val_tracks"] == 8

    trainer = Trainer.from_checkpoint(run / "best.pt", device="cpu")
    scores = trainer.score(np.zeros((3, SEQ_LEN, N_FEATURES), np.float32), np.ones((3, SEQ_LEN), bool))
    assert scores["error"].shape == (3,) and scores["z"].shape == (3, 16)
    with pytest.raises(FileExistsError):
        Experiment(settings).run()
