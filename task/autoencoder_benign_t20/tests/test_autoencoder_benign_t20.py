"""Tests for the one-class T = 20 autoencoder and the depth-configurable ConvAutoencoder (synthetic shards, CPU only).

The helpers without a `test_` prefix (`write_shards`, `settings_for`, `check_*`) are shared with the T = 10 / 5 tests.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from task.autoencoder_benign_t20.config import N_DOWN, N_FEATURES, SEQ_LEN, TOTAL_PARAMS, AutoencoderConfig
from task.autoencoder_benign_t20.data import WindowDataset
from task.autoencoder_benign_t20.model import ConvAutoencoder, count_params, encoder_strides
from task.autoencoder_benign_t20.train import Experiment, parse_args
from task.autoencoder_benign_t20.trainer import Trainer
from task.autoencoder_benign_t32 import config as t32_config
from task.autoencoder_benign_t32 import model as t32_model

EXPECTED_PARAMS = {(32, 2): 63_293, (20, 2): 56_957, (10, 1): 52_861, (5, 0): 50_813}


def write_shards(
    root: Path, split: str, seq_len: int, n_links: int = 24, seed: int = 0, sender_base: int = 0, label: int = 0
) -> None:
    """Synthetic `<split>/b<T>/part-XXXXX.json` + `_info.json`: consecutive T-row crops of links, junk padding."""
    rng = np.random.default_rng(seed)
    folder = root / split / f"b{seq_len:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    windows, info = [], []
    for j in range(n_links):
        n_link = int(rng.choice([2, seq_len + 3, 2 * seq_len + 1, 70]))
        link = rng.normal(size=(n_link, N_FEATURES))
        scenario = "DataReplaySybil_0709" if j % 2 else "GridSybil_1416"
        sender = sender_base + j
        for k, start in enumerate(range(0, n_link, seq_len)):
            n = min(n_link - start, seq_len)
            x = np.full((seq_len, N_FEATURES), 7.0)
            x[:n] = link[start : start + n]
            wid = f"{scenario}/run/traceJSON-{j}-1-A0-0-7.json#{100 + j}-{sender}#{k}"
            windows.append({"id": wid, "x": x.round(4).tolist(), "mask": [1] * n + [0] * (seq_len - n)})
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


def make_data_dir(tmp_path: Path, seq_len: int) -> Path:
    root = tmp_path / f"T{seq_len}_benign"
    write_shards(root, "train", seq_len, seed=0, sender_base=0)
    write_shards(root, "val", seq_len, n_links=8, seed=1, sender_base=1000)
    return root


def settings_for(config_cls: type, data_dir: Path, tmp_path: Path, **changes):
    base = config_cls(
        data_dir=str(data_dir),
        runs_dir=str(tmp_path / "runs"),
        cache_dir="",
        max_epochs=2,
        batch_size=16,
        device="cpu",
        workers=1,
    )
    return dataclasses.replace(base, **changes)


def check_config(config_cls: type, seq_len: int, n_down: int, package: str) -> None:
    """Every setting except the paths, seq_len and n_down equals the T = 32 package's."""
    ours, theirs = config_cls().to_dict(), t32_config.AutoencoderConfig().to_dict()
    differ = {k for k in theirs if ours[k] != theirs[k]}
    assert differ == {"data_dir", "runs_dir", "cache_dir", "seq_len"}, differ
    assert set(ours) - set(theirs) == {"n_down"}
    s = config_cls()
    assert (s.seq_len, s.n_down) == (seq_len, n_down)
    assert s.data_dir == f"src/data/encoder_input/all/T{seq_len}_benign"
    assert s.runs_dir == f"task/{package}/runs" and s.cache_dir == f"task/{package}/cache"
    assert config_cls.from_dict(json.loads(json.dumps(s.to_dict()))) == s


def check_model(config_cls: type, total_params: int) -> None:
    s = config_cls()
    model = ConvAutoencoder(s)
    assert count_params(model) == total_params == EXPECTED_PARAMS[(s.seq_len, s.n_down)]
    assert model.short_len == s.seq_len // 2**s.n_down and model.to_z.in_features == 64 * model.short_len
    x, m = torch.randn(4, s.seq_len, N_FEATURES), torch.ones(4, s.seq_len, dtype=torch.bool)
    recon, z = model(x, m)
    assert recon.shape == (4, s.seq_len, N_FEATURES) and z.shape == (4, 16)
    model.eval()
    mask = torch.zeros(2, s.seq_len, dtype=torch.bool)
    mask[:, :3] = True
    junk = x[:2].clone()
    junk[:, 3:] = 50.0
    with torch.no_grad():
        assert torch.allclose(model(x[:2], mask)[0], model(junk, mask)[0])


def check_smoke(experiment_cls: type, config_cls: type, trainer_cls: type, tmp_path: Path) -> None:
    """One short run writes the full run folder, asserts the parameter count, reloads, refuses test splits."""
    seq_len = config_cls().seq_len
    data_dir = make_data_dir(tmp_path, seq_len)
    write_shards(data_dir, "test_gridsybil", seq_len, n_links=4, seed=2, sender_base=2000, label=16)  # stays unread
    settings = settings_for(config_cls, data_dir, tmp_path, max_epochs=1)
    summary = experiment_cls(settings).run()
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
    assert summary["params"] == experiment_cls.total_params and summary["stop_reason"] == "max_epochs"
    assert summary["epochs_run"] == 1 and summary["val_tracks"] == 8
    assert set(summary["thresholds"]["track"]) == {
        "reconstruction_max",
        "reconstruction_mean",
        "mahalanobis_max",
        "mahalanobis_mean",
    }
    assert json.loads((run / "config.json").read_text())["n_down"] == settings.n_down
    trainer = trainer_cls.from_checkpoint(run / "best.pt", device="cpu")
    assert type(trainer.settings) is config_cls and trainer.settings == settings
    scores = trainer.score(np.zeros((3, seq_len, N_FEATURES), np.float32), np.ones((3, seq_len), bool))
    assert scores["error"].shape == (3,) and scores["z"].shape == (3, 16)
    with pytest.raises(FileExistsError):
        experiment_cls(settings).run()
    with pytest.raises(ValueError):
        experiment_cls(settings).load("test_gridsybil")
    wrong = make_data_dir(tmp_path / "wrong", seq_len + 1 if seq_len % 2 else seq_len * 2)
    with pytest.raises(ValueError, match="rows, the config says"):
        experiment_cls(settings_for(config_cls, wrong, tmp_path / "wrong")).run()


def check_resume(experiment_cls: type, config_cls: type, tmp_path: Path) -> None:
    """A run stopped after 2 epochs and resumed to 3 equals the uninterrupted 3-epoch run."""
    data_dir = make_data_dir(tmp_path, config_cls().seq_len)
    full = experiment_cls(settings_for(config_cls, data_dir, tmp_path, runs_dir=str(tmp_path / "full"), max_epochs=3))
    full_summary = full.run()
    part = settings_for(config_cls, data_dir, tmp_path, runs_dir=str(tmp_path / "part"), max_epochs=2)
    experiment_cls(part).run()
    (tmp_path / "part" / "seed0" / "summary.json").unlink()
    resumed = experiment_cls(dataclasses.replace(part, max_epochs=3), resume=True).run()
    assert resumed["epochs_run"] == 3
    assert np.isclose(resumed["best_val_loss"], full_summary["best_val_loss"], rtol=1e-5)
    a = torch.load(tmp_path / "full" / "seed0" / "last.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "part" / "seed0" / "last.pt", weights_only=False)["model"]
    for k, v in a.items():
        assert torch.allclose(v.float(), b[k].float(), atol=1e-6), k


def check_feature_log(experiment_cls: type, config_cls: type, tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path, config_cls().seq_len)
    experiment_cls(settings_for(config_cls, data_dir, tmp_path, max_epochs=1)).run()
    assert not (tmp_path / "runs" / "seed0" / "feature_history.jsonl").exists()
    settings = settings_for(config_cls, data_dir, tmp_path, log_feature_error=True)
    experiment_cls(settings).run()
    run = tmp_path / "runs" / "seed0_featlog"
    rows = [json.loads(line) for line in (run / "feature_history.jsonl").read_text().splitlines()]
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [h["epoch"] for h in history] == [1, 2]
    for row, h in zip(rows, history):
        assert len(row["val_row_pooled"]) == N_FEATURES
        assert np.isclose(np.mean(row["val_row_pooled"]), h["val_loss"], rtol=1e-4)


# ---------------------------------------------------------------------------------------------- the shared model


def test_n_down_2_is_the_t32_model_bit_for_bit() -> None:
    """With n_down = 2 the new model has the T = 32 model's parameter names, shapes and count, the same weights for the
    same seed, and the same outputs (train and eval mode), at T = 32 and at T = 20; the T = 32 model stays 63,293."""
    assert t32_config.TOTAL_PARAMS == count_params(t32_model.ConvAutoencoder(t32_config.AutoencoderConfig())) == 63_293
    for seq_len in (32, 20):
        old_cfg = dataclasses.replace(t32_config.AutoencoderConfig(), seq_len=seq_len)
        new_cfg = dataclasses.replace(AutoencoderConfig(), seq_len=seq_len, n_down=2)
        torch.manual_seed(123)
        old = t32_model.ConvAutoencoder(old_cfg)
        torch.manual_seed(123)
        new = ConvAutoencoder(new_cfg)
        a, b = old.state_dict(), new.state_dict()
        assert list(a) == list(b)
        for k in a:
            assert a[k].shape == b[k].shape and torch.equal(a[k], b[k]), k
        assert [type(m) for m in old.modules()][1:] == [type(m) for m in new.modules()][1:]
        assert count_params(new) == EXPECTED_PARAMS[(seq_len, 2)]
        gen = torch.Generator().manual_seed(7)
        x = torch.randn(5, seq_len, N_FEATURES, generator=gen)
        mask = torch.arange(seq_len).unsqueeze(0) < torch.tensor([[seq_len], [1], [3], [seq_len - 2], [9]])
        for mode in ("train", "eval"):
            getattr(old, mode)()
            getattr(new, mode)()
            with torch.no_grad():
                (ra, za), (rb, zb) = old(x, mask), new(x, mask)
            assert torch.equal(ra, rb) and torch.equal(za, zb), mode


def test_encoder_strides_and_divisibility() -> None:
    assert encoder_strides(2) == (2, 2) and encoder_strides(1) == (2, 1) and encoder_strides(0) == (1, 1)
    with pytest.raises(ValueError):
        encoder_strides(3)
    with pytest.raises(ValueError, match="divisible"):
        ConvAutoencoder(dataclasses.replace(AutoencoderConfig(), seq_len=10, n_down=2))
    for (seq_len, n_down), params in EXPECTED_PARAMS.items():
        model = ConvAutoencoder(dataclasses.replace(AutoencoderConfig(), seq_len=seq_len, n_down=n_down))
        assert count_params(model) == params
        recon, _ = model(torch.randn(2, seq_len, N_FEATURES), torch.ones(2, seq_len, dtype=torch.bool))
        assert recon.shape == (2, seq_len, N_FEATURES)
        kinds = [(m.kernel_size[0], m.stride[0]) for m in model.decoder_conv if isinstance(m, torch.nn.ConvTranspose1d)]
        strides = encoder_strides(n_down)
        assert kinds == [(4, 2) if s == 2 else (3, 1) for s in (strides[1], strides[0])]


# ---------------------------------------------------------------------------------------------- the T = 20 package


def test_config_equals_t32_except_input_length_and_depth() -> None:
    check_config(AutoencoderConfig, 20, 2, "autoencoder_benign_t20")
    assert (SEQ_LEN, N_DOWN, N_FEATURES) == (20, 2, 13)


def test_model_shapes_params_and_padding() -> None:
    check_model(AutoencoderConfig, TOTAL_PARAMS)


def test_dataset_reads_b20_shards(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path, SEQ_LEN)
    assert (data_dir / "train" / "b20").is_dir()
    data = WindowDataset.load(data_dir, "train", workers=1)
    assert data.x.shape[1:] == (SEQ_LEN, N_FEATURES) and np.all(data.x[~data.mask] == 0.0)
    assert len(np.unique(data.tracks)) == 24 and len(data) > 24


def test_parse_args_seed_workers_and_defaults() -> None:
    settings, max_shards, resume = parse_args(["--seed", "3", "--workers", "2", "--epochs", "5"])
    assert settings == dataclasses.replace(AutoencoderConfig(), seed=3, workers=2, max_epochs=5)
    assert settings.run_id == "seed3" and max_shards == 0 and not resume


def test_experiment_smoke_writes_run_folder(tmp_path: Path) -> None:
    check_smoke(Experiment, AutoencoderConfig, Trainer, tmp_path)


def test_attack_window_in_train_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "bad"
    write_shards(root, "train", SEQ_LEN, seed=0, label=17)
    write_shards(root, "val", SEQ_LEN, n_links=8, seed=1, sender_base=1000)
    with pytest.raises(ValueError, match="attack windows"):
        Experiment(settings_for(AutoencoderConfig, root, tmp_path)).run()


def test_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    check_resume(Experiment, AutoencoderConfig, tmp_path)


def test_feature_error_log_is_opt_in(tmp_path: Path) -> None:
    check_feature_log(Experiment, AutoencoderConfig, tmp_path)
