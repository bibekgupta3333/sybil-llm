"""Tests for the T48 autoencoder on synthetic shards (CPU only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from task.autoencoder.config import N_FEATURES, SEQ_LEN, AutoencoderConfig
from task.autoencoder.data import WindowDataset, vehicle_split
from task.autoencoder.model import ConvAutoencoder, count_params, masked_mse, window_errors
from task.autoencoder.train import Experiment
from task.autoencoder.trainer import Trainer


def write_shards(root: Path, split: str, n_shards: int = 2, per_shard: int = 30, seed: int = 0) -> None:
    """Synthetic `<split>/b48/part-XXXXX.json` + `_info.json` with mixed lengths and junk in padding rows."""
    rng = np.random.default_rng(seed)
    folder = root / split / "b48"
    folder.mkdir(parents=True, exist_ok=True)
    for k in range(n_shards):
        windows, info = [], []
        for j in range(per_shard):
            n = int(rng.integers(1, SEQ_LEN + 1))
            x = np.full((SEQ_LEN, N_FEATURES), 7.0)
            x[:n] = rng.normal(size=(n, N_FEATURES))
            scenario = "GridSybil_0709" if j % 2 else "GridSybil_1416"
            sender = int(rng.integers(0, 20))
            wid = f"{scenario}/run/trace-{k}-{j}.json#1-{sender}#0"
            windows.append({"id": wid, "x": x.round(4).tolist(), "mask": [1] * n + [0] * (SEQ_LEN - n)})
            label = 16 if j % 3 else 0
            info.append(
                {
                    "id": wid,
                    "scenario": scenario,
                    "sender": sender,
                    "receiver": 1,
                    "n_messages": n,
                    "label": label,
                    "label_name": "GridSybil" if label else "Benign",
                    "is_attack": int(label > 0),
                }
            )
        (folder / f"part-{k:05d}.json").write_text(json.dumps({"windows": windows}))
        (folder / f"part-{k:05d}_info.json").write_text(json.dumps({"windows": info}))


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    root = tmp_path / "T48"
    write_shards(root, "train")
    return root


def test_dataset_shapes_and_padding(data_dir: Path) -> None:
    data = WindowDataset.load(data_dir, "train", workers=2)
    assert data.x.shape == (60, SEQ_LEN, N_FEATURES) and data.mask.shape == (60, SEQ_LEN)
    assert np.all(data.x[~data.mask] == 0.0)
    assert set(data.info["label_name"]) == {"Benign", "GridSybil"}


def test_vehicle_split_is_disjoint(data_dir: Path) -> None:
    data = WindowDataset.load(data_dir, "train", workers=2)
    fit, val = vehicle_split(data.vehicles, 0.1, seed=0)
    assert len(fit) + len(val) == len(data) and len(val) > 0
    assert not set(data.vehicles[fit]) & set(data.vehicles[val])


def test_model_shapes() -> None:
    model = ConvAutoencoder(AutoencoderConfig())
    x, m = torch.randn(4, SEQ_LEN, N_FEATURES), torch.ones(4, SEQ_LEN, dtype=torch.bool)
    recon, z = model(x, m)
    assert recon.shape == (4, SEQ_LEN, N_FEATURES) and z.shape == (4, 16)
    assert count_params(model) > 0


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


def test_experiment_writes_run_folder(data_dir: Path, tmp_path: Path) -> None:
    settings = AutoencoderConfig(
        data_dir=str(data_dir), runs_dir=str(tmp_path / "runs"), max_epochs=2, batch_size=16, device="cpu", workers=2
    )
    summary = Experiment(settings).run()
    run = tmp_path / "runs" / settings.run_id
    for name in ("config.json", "env.json", "history.jsonl", "train.log", "loss_curve.png", "best.pt", "summary.json"):
        assert (run / name).exists(), name
    assert len((run / "history.jsonl").read_text().splitlines()) == summary["epochs_run"] == 2
    assert summary["stop_reason"] == "max_epochs" and summary["best_epoch"] >= 1

    scores = Trainer.from_checkpoint(run / "best.pt", device="cpu").score(
        np.zeros((3, SEQ_LEN, N_FEATURES), np.float32), np.ones((3, SEQ_LEN), bool)
    )
    assert scores["error"].shape == (3,) and scores["z"].shape == (3, 16)
    with pytest.raises(FileExistsError):
        Experiment(settings).run()
