"""Tests for the one-class T = 32 autoencoder on synthetic shards (CPU only)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from task.autoencoder_benign_t32.config import N_FEATURES, SEQ_LEN, TOTAL_PARAMS, AutoencoderConfig
from task.autoencoder_benign_t32.data import WindowDataset, track_id
from task.autoencoder_benign_t32.model import ConvAutoencoder, count_params, masked_mse, window_errors
from task.autoencoder_benign_t32.scoring import Mahalanobis, TrackScores, quantile_thresholds
from task.autoencoder_benign_t32.train import Experiment
from task.autoencoder_benign_t32.trainer import EarlyStopping, Trainer


def write_shards(root: Path, split: str, n_links: int = 24, seed: int = 0, sender_base: int = 0, label: int = 0):
    """Synthetic `<split>/b32/part-XXXXX.json` + `_info.json`: consecutive 32-row crops of links, junk padding."""
    rng = np.random.default_rng(seed)
    folder = root / split / "b32"
    folder.mkdir(parents=True, exist_ok=True)
    windows, info = [], []
    for j in range(n_links):
        n_link = int(rng.choice([2, 20, 40, 70]))
        link = rng.normal(size=(n_link, N_FEATURES))
        scenario = "DataReplaySybil_0709" if j % 2 else "GridSybil_1416"
        sender = sender_base + j
        for k, start in enumerate(range(0, n_link, SEQ_LEN)):
            n = min(n_link - start, SEQ_LEN)
            x = np.full((SEQ_LEN, N_FEATURES), 7.0)
            x[:n] = link[start : start + n]
            wid = f"{scenario}/run/traceJSON-{j}-1-A0-0-7.json#{100 + j}-{sender}#{k}"
            windows.append({"id": wid, "x": x.round(4).tolist(), "mask": [1] * n + [0] * (SEQ_LEN - n)})
            info.append(
                {
                    "id": wid,
                    "family": scenario.split("_")[0],
                    "scenario": scenario,
                    "receiver": j,
                    "sender": sender,
                    "sender_pseudo": 100 + j,
                    "link_window_index": k,
                    "n_messages": n,
                    "label": label,
                    "label_name": "Benign" if label == 0 else "Attack",
                    "is_attack": int(label > 0),
                }
            )
    half = len(windows) // 2
    for part, (lo, hi) in enumerate([(0, half), (half, len(windows))]):
        (folder / f"part-{part:05d}.json").write_text(json.dumps({"windows": windows[lo:hi]}))
        (folder / f"part-{part:05d}_info.json").write_text(json.dumps({"windows": info[lo:hi]}))


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    root = tmp_path / "T32_benign"
    write_shards(root, "train", seed=0, sender_base=0)
    write_shards(root, "val", n_links=8, seed=1, sender_base=1000)
    return root


def settings_for(data_dir: Path, tmp_path: Path, **changes) -> AutoencoderConfig:
    base = AutoencoderConfig(
        data_dir=str(data_dir),
        runs_dir=str(tmp_path / "runs"),
        cache_dir="",
        max_epochs=2,
        batch_size=16,
        device="cpu",
        workers=2,
    )
    return dataclasses.replace(base, **changes)


def test_track_id_strips_window_index() -> None:
    wid = "DataReplaySybil_0709/r/traceJSON-10005-10003-A0-28468-7.json#1094892-9489#2"
    assert track_id(wid) == "DataReplaySybil_0709/r/traceJSON-10005-10003-A0-28468-7.json#1094892-9489"
    assert track_id(wid[:-1] + "0") == track_id(wid)
    with pytest.raises(ValueError):
        track_id("G_0709/r/t.json#1-10635")
    with pytest.raises(ValueError):
        track_id("G_0709/r/t.json#1-10635#0@0")


def test_dataset_shapes_padding_tracks_and_cache(data_dir: Path, tmp_path: Path) -> None:
    data = WindowDataset.load(data_dir, "train", workers=2)
    assert data.x.shape[1:] == (SEQ_LEN, N_FEATURES) and data.mask.shape == (len(data), SEQ_LEN)
    assert np.all(data.x[~data.mask] == 0.0)
    assert len(np.unique(data.tracks)) == 24 and len(data) > 24
    cache = tmp_path / "cache"
    first = WindowDataset.load(data_dir, "train", workers=2, cache_dir=cache)
    assert (cache / "train" / "manifest.json").exists()
    again = WindowDataset.load(data_dir, "train", workers=2, cache_dir=cache)
    assert np.array_equal(first.x, again.x) and np.array_equal(first.ids, again.ids)
    assert np.array_equal(again.x, data.x) and list(again.info["scenario"]) == list(data.info["scenario"])


def test_model_shapes_and_params() -> None:
    model = ConvAutoencoder(AutoencoderConfig())
    x, m = torch.randn(4, SEQ_LEN, N_FEATURES), torch.ones(4, SEQ_LEN, dtype=torch.bool)
    recon, z = model(x, m)
    assert recon.shape == (4, SEQ_LEN, N_FEATURES) and z.shape == (4, 16)
    assert count_params(model) == TOTAL_PARAMS == 63_293
    assert model.to_z.in_features == 512


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
    assert abs(np.mean(maha.score(z) ** 2) - 3.0) < 0.05  # E[d^2] = dimension
    probe = mean + np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert np.allclose(maha.score(probe), [0.0, np.sqrt(4.0 * np.linalg.inv(cov)[0, 0])], atol=0.03)
    np.savez(tmp_path / "m.npz", **maha.state())
    restored = Mahalanobis.from_state(dict(np.load(tmp_path / "m.npz")))
    assert np.allclose(restored.score(probe), maha.score(probe))


def test_track_aggregation_and_labels() -> None:
    agg = TrackScores(np.array(["b", "a", "b", "c", "a", "b"]))
    assert list(agg.tracks) == ["a", "b", "c"] and list(agg.n_windows) == [2, 3, 1]
    out = agg.aggregate(np.array([1.0, 5.0, 3.0, 2.0, 1.0, 2.0]))
    assert np.allclose(out["max"], [5.0, 3.0, 2.0]) and np.allclose(out["mean"], [3.0, 2.0, 2.0])
    labels, mixed = agg.track_labels(np.array([16, 0, 16, 16, 0, 16]))
    assert list(labels) == [0, 16, 16] and mixed == 0


def test_quantile_thresholds() -> None:
    assert quantile_thresholds(np.arange(101.0), (0.95, 0.99)) == {"p95": 95.0, "p99": 99.0}


def test_experiment_smoke_writes_run_folder(data_dir: Path, tmp_path: Path) -> None:
    write_shards(data_dir, "test_gridsybil", n_links=4, seed=2, sender_base=2000, label=16)  # must stay unread
    settings = settings_for(data_dir, tmp_path, max_epochs=1)
    summary = Experiment(settings).run()
    run = tmp_path / "runs" / settings.run_id
    for name in (
        "config.json",
        "env.json",
        "history.jsonl",
        "train.log",
        "curves.png",
        "best.pt",
        "last.pt",
        "mahalanobis.npz",
        "summary.json",
    ):
        assert (run / name).exists(), name
    rows = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    assert len(rows) == summary["epochs_run"] == 1
    assert all(r["grad_norm_mean"] > 0 and r["grad_norm_max"] >= r["grad_norm_mean"] and "lr" in r for r in rows)
    assert "*best" in (run / "train.log").read_text()
    assert summary["params"] == TOTAL_PARAMS and summary["stop_reason"] == "max_epochs"
    assert set(summary["thresholds"]["track"]) == {
        "reconstruction_max",
        "reconstruction_mean",
        "mahalanobis_max",
        "mahalanobis_mean",
    }
    assert set(summary["thresholds"]["window"]["mahalanobis"]) == {"p95", "p99"} and summary["val_tracks"] == 8
    trainer = Trainer.from_checkpoint(run / "best.pt", device="cpu")
    scores = trainer.score(np.zeros((3, SEQ_LEN, N_FEATURES), np.float32), np.ones((3, SEQ_LEN), bool))
    assert scores["error"].shape == (3,) and scores["z"].shape == (3, 16)
    with pytest.raises(FileExistsError):
        Experiment(settings).run()
    with pytest.raises(ValueError):
        Experiment(settings).load("test_gridsybil")


def test_attack_window_in_train_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "bad"
    write_shards(root, "train", seed=0, label=17)
    write_shards(root, "val", n_links=8, seed=1, sender_base=1000)
    with pytest.raises(ValueError, match="attack windows"):
        Experiment(settings_for(root, tmp_path)).run()


def test_resume_matches_uninterrupted_run(data_dir: Path, tmp_path: Path) -> None:
    def fit(run_dir: Path, epochs: int, resume: bool = False) -> tuple[dict, ConvAutoencoder]:
        s = settings_for(data_dir, tmp_path, max_epochs=epochs, patience=50)
        torch.manual_seed(0)
        model = ConvAutoencoder(s)
        train = WindowDataset.load(data_dir, "train", workers=2)
        val = WindowDataset.load(data_dir, "val", workers=2)
        run_dir.mkdir(parents=True, exist_ok=True)
        trainer = Trainer(model, s, torch.device("cpu"))
        state = torch.load(run_dir / "last.pt", weights_only=False) if resume else None
        result = trainer.fit((train.x, train.mask), (val.x, val.mask), run_dir, resume=resume)
        if state is not None:
            assert result["epochs_run"] == state["epoch"] + 1
        last = torch.load(run_dir / "last.pt", weights_only=False)
        model.load_state_dict(last["model"])
        return last, model

    straight, _ = fit(tmp_path / "straight", 3)
    fit(tmp_path / "split", 2)
    resumed, _ = fit(tmp_path / "split", 3, resume=True)
    assert [r["val_loss"] for r in resumed["history"]] == pytest.approx([r["val_loss"] for r in straight["history"]])
    for k, v in straight["model"].items():
        assert torch.allclose(v.float(), resumed["model"][k].float(), atol=1e-6), k
    assert len((tmp_path / "split" / "history.jsonl").read_text().splitlines()) == 3
