"""Tests for the baseline encoder task on synthetic T64 shards (CPU only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from task.encoder.config import N_FEATURES, SEQ_LEN, EncoderTaskConfig
from task.encoder.data import ShardDecoder, VehicleSplit, WindowStore, vehicle_key
from task.encoder.model import ConvAutoencoder, SupervisedHead, count_params, masked_mse, window_errors
from task.encoder.train import Experiment, Mahalanobis

CONTRACT_TEST = {"id", "label", "n_messages", "recon_error", "recon_error_feat", "mahalanobis", "supervised_prob", "z"}


def write_shards(root: Path, split: str, n_shards: int = 2, per_shard: int = 40, seed: int = 0) -> None:
    """Synthetic `<split>/b64/part-XXXXX.json` + `_info.json` with mixed lengths and both classes."""
    rng = np.random.default_rng(seed)
    folder = root / split / "b64"
    folder.mkdir(parents=True, exist_ok=True)
    for k in range(n_shards):
        windows, info = [], []
        for j in range(per_shard):
            n = int(rng.integers(1, SEQ_LEN + 1))
            attack = int(rng.random() < 0.5)
            x = np.zeros((SEQ_LEN, N_FEATURES))
            x[:n] = rng.normal(size=(n, N_FEATURES)) + 2.0 * attack
            x[n:] = 7.0  # padding junk: the reader must zero it
            mask = [1] * n + [0] * (SEQ_LEN - n)
            scenario = "GridSybil_0709" if j % 2 else "GridSybil_1416"
            sender = int(rng.integers(0, 30)) + 1000 * split.startswith("test")
            wid = f"{scenario}/run/trace-{k}-{j}.json#1-{sender}#0"
            windows.append({"id": wid, "x": x.round(4).tolist(), "mask": mask})
            info.append(
                {
                    "id": wid,
                    "scenario": scenario,
                    "sender": sender,
                    "receiver": 1,
                    "n_messages": n,
                    "label": 16 if attack else 0,
                }
            )
        (folder / f"part-{k:05d}.json").write_text(json.dumps({"windows": windows}))
        (folder / f"part-{k:05d}_info.json").write_text(json.dumps({"split": split, "windows": info}))


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    root = tmp_path / "T64"
    write_shards(root, "train", seed=0)
    write_shards(root, "test", n_shards=1, seed=1)
    return root


def test_vehicle_key() -> None:
    assert vehicle_key("GridSybil_0709", 10551) == "0709:10551"


def test_decoder_zeroes_padding_and_maps_labels(data_dir: Path) -> None:
    shard = ShardDecoder().decode(data_dir / "train/b64/part-00000.json")
    assert shard.x.shape == (40, SEQ_LEN, N_FEATURES)
    assert np.all(shard.x[~shard.mask] == 0.0)
    assert set(np.unique(shard.labels)) <= {0, 1}
    assert np.array_equal(shard.mask.sum(1), shard.n_messages)


def test_store_cache_and_label_gate(data_dir: Path, tmp_path: Path) -> None:
    store = WindowStore(data_dir, tmp_path / "cache", workers=2)
    data = store.open("train")
    assert len(data) == 80 and data.x.shape == (80, SEQ_LEN, N_FEATURES)
    assert data.label_reads == []
    sub = data.take(np.array([3, 1]))
    assert sub.label_reads == []  # taking a subset does not read labels
    assert np.array_equal(sub.labels("check"), data.labels("check")[[1, 3]])
    assert store.build("train") == tmp_path / "cache" / "train"  # cached, not rebuilt


def test_vehicle_split_is_disjoint() -> None:
    vehicles = np.array([f"0709:{i % 50}" for i in range(1000)])
    fit, val = VehicleSplit(0.1, 0).split(vehicles)
    assert len(fit) + len(val) == 1000
    assert not set(vehicles[fit]) & set(vehicles[val])
    assert len(set(vehicles[val])) == 5
    fit2, val2 = VehicleSplit(0.1, 0).split(vehicles)
    assert np.array_equal(val, val2)


def test_model_shapes_and_param_counts() -> None:
    s = EncoderTaskConfig()
    ae, sup = ConvAutoencoder(s), SupervisedHead(s)
    x, m = torch.randn(5, SEQ_LEN, N_FEATURES), torch.ones(5, SEQ_LEN, dtype=torch.bool)
    recon, z = ae(x, m)
    assert recon.shape == x.shape and z.shape == (5, 16)
    logit, z2 = sup(x, m)
    assert logit.shape == (5,) and z2.shape == (5, 16)
    assert count_params(ae) == 80189
    assert 50_000 <= count_params(ae) <= 150_000


def test_masked_losses_ignore_padding() -> None:
    x = torch.zeros(2, SEQ_LEN, N_FEATURES)
    recon = torch.zeros_like(x)
    mask = torch.zeros(2, SEQ_LEN, dtype=torch.bool)
    mask[:, :4] = True
    recon[:, 4:] = 100.0  # error only on padding rows
    assert float(masked_mse(recon, x, mask)) == 0.0
    recon[0, 0, 0] = 2.0
    err, feat = window_errors(recon, x, mask)
    assert feat.shape == (2, N_FEATURES)
    assert float(err[0]) == pytest.approx(4.0 / 4 / N_FEATURES)
    assert float(err[1]) == 0.0


def test_encoder_ignores_padding_values() -> None:
    torch.manual_seed(0)
    enc = ConvAutoencoder(EncoderTaskConfig()).encoder.eval()
    x = torch.randn(3, SEQ_LEN, N_FEATURES)
    m = torch.zeros(3, SEQ_LEN, dtype=torch.bool)
    m[:, :10] = True
    x2 = x.clone()
    x2[:, 10:] = 50.0
    assert torch.allclose(enc(x, m), enc(x2, m))


def test_mahalanobis_matches_definition() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(2000, 3)) * [1.0, 2.0, 3.0]
    maha = Mahalanobis(1e-9).fit(z)
    assert maha.score(maha.mean[None])[0] == pytest.approx(0.0, abs=1e-9)
    assert np.mean(maha.score(z) ** 2) == pytest.approx(3.0, rel=0.01)


@pytest.mark.parametrize("mode", ["ae", "supervised"])
def test_end_to_end_run_writes_contract(data_dir: Path, tmp_path: Path, mode: str) -> None:
    s = EncoderTaskConfig(
        mode=mode,
        seed=0,
        data_dir=str(data_dir),
        runs_dir=str(tmp_path / "runs"),
        cache_dir=str(tmp_path / "cache"),
        max_epochs=2,
        batch_size=16,
        eval_batch=32,
        val_frac=0.2,
        device="cpu",
    )
    summary = Experiment(s).run()
    run = tmp_path / "runs" / f"{mode}-seed0"
    for name in ("config.json", "env.json", "history.jsonl", "scores_test.npz", "scores_val.npz", "summary.json"):
        assert (run / name).exists(), name
    test = np.load(run / "scores_test.npz")
    assert set(test.files) == CONTRACT_TEST
    n = len(test["id"])
    assert n == 40 and test["id"].dtype.kind == "U"
    assert set(np.unique(test["label"])) <= {0, 1}
    assert test["recon_error_feat"].shape == (n, N_FEATURES) and test["z"].shape == (n, 16)
    assert np.isfinite(test["mahalanobis"]).all()
    if mode == "ae":
        assert np.isfinite(test["recon_error"]).all() and np.isnan(test["supervised_prob"]).all()
    else:
        assert np.isnan(test["recon_error"]).all()
        assert ((test["supervised_prob"] >= 0) & (test["supervised_prob"] <= 1)).all()
    val = np.load(run / "scores_val.npz")
    assert set(val.files) == {"recon_error", "mahalanobis"}
    assert summary["label_reads"][-1].startswith("test: evaluation labels")
    assert len(open(run / "history.jsonl").read().splitlines()) == summary["epochs_run"]
    assert json.loads((run / "config.json").read_text())["seed"] == 0
