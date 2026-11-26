"""Tests for the supervised T64 autoencoder (CPU only, synthetic cache; the real data is never read)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from task.autoencoder_st_t64 import data as data_module
from task.autoencoder_st_t64.config import TOTAL_PARAMS, SupervisedAET64Config
from task.autoencoder_st_t64.evaluate import evaluate
from task.autoencoder_st_t64.model import SupervisedAutoencoder, count_params
from task.autoencoder_st_t64.train import Experiment
from task.autoencoder_st_t64.trainer import Trainer
from task.encoder.data import CACHE_VERSION, VehicleSplit

T, F = 64, 13


def write_cache(root: Path, split: str, n_vehicles: int = 60, seed: int = 0) -> None:
    """A complete `task/encoder`-style npy cache with a manifest that matches a fake data dir."""
    rng = np.random.default_rng(seed)
    data_dir = root / "data"
    (data_dir / split / "b64").mkdir(parents=True, exist_ok=True)
    (data_dir / split / "b64" / "part-00000.json").write_text("{}")
    ids, vehicles, n_msg, labels, xs, masks = [], [], [], [], [], []
    for v in range(n_vehicles):
        label = v % 3 != 0
        for k in range(3):
            n = int(rng.integers(1, T + 1))
            x = np.zeros((T, F), np.float32)
            x[:n] = rng.normal(size=(n, F)) + (1.5 if label else 0.0)
            m = np.zeros(T, bool)
            m[:n] = True
            ids.append(f"{split}/run/trace-{v}.json#{v}-{v}#{k}")
            vehicles.append(f"{'0709' if v % 2 else '1416'}:{v + (1000 if split == 'test' else 0)}")
            n_msg.append(n)
            labels.append(int(label))
            xs.append(x)
            masks.append(m)
    folder = root / "cache" / split
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "x.npy", np.stack(xs))
    np.save(folder / "mask.npy", np.stack(masks))
    np.savez(folder / "meta.npz", ids=np.asarray(ids), vehicles=np.asarray(vehicles), n_messages=np.asarray(n_msg))
    np.save(folder / "labels.npy", np.asarray(labels, np.int8))
    source = {"data_dir": str(data_dir), "shards": ["part-00000.json"], "version": CACHE_VERSION}
    (folder / "manifest.json").write_text(json.dumps({"source": source, "windows": len(ids)}))


@pytest.fixture()
def settings(tmp_path: Path) -> SupervisedAET64Config:
    for split in ("train", "test"):
        write_cache(tmp_path, split)
    return SupervisedAET64Config(
        data_dir=str(tmp_path / "data"),
        cache_dir=str(tmp_path / "cache"),
        runs_dir=str(tmp_path / "runs"),
        max_epochs=3,
        batch_size=32,
        eval_batch=64,
        device="cpu",
    )


def test_param_count_and_shapes() -> None:
    model = SupervisedAutoencoder(SupervisedAET64Config())
    assert count_params(model) == TOTAL_PARAMS == 80_766
    assert count_params(model.autoencoder) == 80_189 and count_params(model.head) == 577
    x, m = torch.randn(5, T, F), torch.ones(5, T, dtype=torch.bool)
    recon, z, logit = model(x, m)
    assert recon.shape == (5, T, F) and z.shape == (5, 16) and logit.shape == (5,)


def test_padding_does_not_change_z() -> None:
    model = SupervisedAutoencoder(SupervisedAET64Config()).eval()
    x, m = torch.randn(2, T, F), torch.zeros(2, T, dtype=torch.bool)
    m[:, :10] = True
    x2 = x.clone()
    x2[:, 10:] = 99.0
    with torch.no_grad():
        assert torch.allclose(model(x, m)[1], model(x2, m)[1])


def test_val_is_the_encoder_vehicle_draw(settings: SupervisedAET64Config) -> None:
    train, val, reads = data_module.load_train_val(settings)
    full_vehicles = np.load(Path(settings.cache_dir) / "train" / "meta.npz")["vehicles"]
    _, val_rows = VehicleSplit(0.1, 0).split(full_vehicles)
    assert set(val.vehicles) == set(full_vehicles[val_rows])
    assert not set(train.vehicles) & set(val.vehicles)
    assert len(train) + len(val) == len(full_vehicles)
    assert reads == ["train: supervised targets"]


def test_training_never_opens_test(settings: SupervisedAET64Config, monkeypatch: pytest.MonkeyPatch) -> None:
    from task.encoder.data import WindowStore

    opened: list[str] = []
    original = WindowStore.open

    def spy(self, split, *a, **k):
        opened.append(split)
        return original(self, split, *a, **k)

    monkeypatch.setattr(WindowStore, "open", spy)
    summary = Experiment(settings).run()
    assert opened == ["train"]
    assert summary["splits_opened"] == ["train"] and summary["params"] == TOTAL_PARAMS
    run = Path(settings.runs_dir) / "seed0"
    for name in ("config.json", "env.json", "history.jsonl", "train.log", "curves.png", "best.pt", "last.pt"):
        assert (run / name).is_file(), name
    assert set(summary["thresholds"]["window"]) == {"max_f1", "benign_fpr_5pct"}

    trainer = Trainer.from_checkpoint(run / "best.pt", device="cpu")
    assert isinstance(trainer.model, SupervisedAutoencoder)
    results, _ = evaluate(run)
    assert opened[-1] == "test"
    assert (run / "test_results.json").is_file()
    for key in ("window/max_f1", "window/benign_fpr_5pct", "track max/max_f1", "track mean/max_f1"):
        assert 0.0 <= results[key]["auroc"] <= 1.0
    assert 0.5 <= results["length_only_auroc"] <= 1.0
