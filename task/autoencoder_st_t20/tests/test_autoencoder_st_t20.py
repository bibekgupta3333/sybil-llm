"""Tests for the supervised T = 20 autoencoder on synthetic shards (CPU only).

The helpers without a `test_` prefix (`write_shards`, `make_data_dir`, `small_settings`, `check_*`) are shared with
the T = 10 / 5 tests. Metrics, thresholds and the training loop are `task/autoencoder_st_t32`'s and tested there.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from task.autoencoder_st_t20.config import N_DOWN, N_FEATURES, SEQ_LEN, TOTAL_PARAMS, SupervisedAEConfig
from task.autoencoder_st_t20.data import WindowDataset
from task.autoencoder_st_t20.model import SupervisedAutoencoder, count_params
from task.autoencoder_st_t20.train import Experiment
from task.autoencoder_st_t20.trainer import Trainer
from task.autoencoder_st_t32 import config as t32_config
from task.autoencoder_st_t32 import model as t32_model

FAMILIES = ("GridSybil", "DataReplaySybil", "DoSRandomSybil", "DoSDisruptiveSybil")
EXPECTED_PARAMS = {(32, 2): 63_870, (20, 2): 57_534, (10, 1): 53_438, (5, 0): 51_390}


def write_shards(root: Path, split: str, seq_len: int, n_links: int = 24, seed: int = 0, sender_base: int = 0) -> None:
    """Synthetic `<split>/b<T>/part-XXXXX.json` + `_info.json`: links cut at T (no overlap), junk in padding rows.

    Benign links come from all four families, GridSybil links only from the GridSybil family; GridSybil windows get a
    feature offset so the classifier has something to learn.
    """
    rng = np.random.default_rng(seed)
    folder = root / split / f"b{seq_len:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    windows, info = [], []
    for j in range(n_links):
        n_link = int(rng.choice([2, seq_len + 3, 2 * seq_len + 1, 70]))
        label = 16 if j % 3 else 0
        family = "GridSybil" if label else FAMILIES[j % len(FAMILIES)]
        link = rng.normal(size=(n_link, N_FEATURES)) + (1.5 if label else 0.0)
        scenario = f"{family}_{'0709' if j % 2 else '1416'}"
        sender = sender_base + j
        for k, start in enumerate(range(0, n_link, seq_len)):
            n = min(n_link - start, seq_len)
            x = np.full((seq_len, N_FEATURES), 7.0)
            x[:n] = link[start : start + n]
            wid = f"{scenario}/run/trace-{j}.json#{100 + j}-{sender}#{k}"
            windows.append({"id": wid, "x": x.round(4).tolist(), "mask": [1] * n + [0] * (seq_len - n)})
            info.append(
                {
                    "id": wid,
                    "family": family,
                    "scenario": scenario,
                    "receiver": 1,
                    "sender": sender,
                    "sender_pseudo": 100 + j,
                    "link_window_index": k,
                    "n_messages": n,
                    "label": int(label > 0),
                    "label_name": "GridSybil" if label else "Benign",
                    "is_attack": int(label > 0),
                }
            )
    half = len(windows) // 2
    for part, (lo, hi) in enumerate([(0, half), (half, len(windows))]):
        (folder / f"part-{part:05d}.json").write_text(json.dumps({"windows": windows[lo:hi]}))
        (folder / f"part-{part:05d}_info.json").write_text(json.dumps({"windows": info[lo:hi]}))


def make_data_dir(tmp_path: Path, seq_len: int) -> Path:
    root = tmp_path / f"T{seq_len}_benign_grid"
    write_shards(root, "train", seq_len, n_links=36, seed=0, sender_base=0)
    write_shards(root, "val", seq_len, n_links=12, seed=1, sender_base=1000)
    return root


def small_settings(config_cls: type, data_dir: Path, runs: Path, **changes):
    return config_cls(
        data_dir=str(data_dir), runs_dir=str(runs), cache_dir="", batch_size=16, device="cpu", workers=1, **changes
    )


def check_config(config_cls: type, seq_len: int, n_down: int, package: str) -> None:
    """Every setting except the paths, seq_len and n_down equals the T = 32 supervised package's."""
    ours, theirs = config_cls().to_dict(), t32_config.SupervisedAEConfig().to_dict()
    assert {k for k in theirs if ours[k] != theirs[k]} == {"data_dir", "runs_dir", "cache_dir", "seq_len"}
    assert set(ours) - set(theirs) == {"n_down"}
    s = config_cls()
    assert (s.seq_len, s.n_down) == (seq_len, n_down)
    assert s.data_dir == f"src/data/encoder_input/all/T{seq_len}_benign_grid"
    assert s.runs_dir == f"task/{package}/runs" and s.cache_dir == f"task/{package}/cache"
    assert config_cls.from_dict(json.loads(json.dumps(s.to_dict()))) == s


def check_model(config_cls: type, total_params: int) -> None:
    s = config_cls()
    model = SupervisedAutoencoder(s)
    assert count_params(model) == total_params == EXPECTED_PARAMS[(s.seq_len, s.n_down)]
    assert count_params(model.head) == 577
    x, m = torch.randn(4, s.seq_len, N_FEATURES), torch.ones(4, s.seq_len, dtype=torch.bool)
    recon, z, logit = model(x, m)
    assert recon.shape == (4, s.seq_len, N_FEATURES) and z.shape == (4, 16) and logit.shape == (4,)
    model.eval()
    mask = torch.zeros(2, s.seq_len, dtype=torch.bool)
    mask[:, :3] = True
    junk = x[:2].clone()
    junk[:, 3:] = 50.0
    with torch.no_grad():
        for a, b in zip(model(x[:2], mask), model(junk, mask)):
            assert torch.allclose(a, b)


def check_smoke(experiment_cls: type, config_cls: type, trainer_cls: type, tmp_path: Path) -> None:
    seq_len = config_cls().seq_len
    data_dir = make_data_dir(tmp_path, seq_len)
    write_shards(data_dir, "test", seq_len, n_links=6, seed=2, sender_base=2000)
    read: list[str] = []
    original = WindowDataset.load.__func__
    mp = pytest.MonkeyPatch()
    mp.setattr(
        WindowDataset,
        "load",
        classmethod(lambda cls, r, split, *a, **k: (read.append(split), original(cls, r, split, *a, **k))[1]),
    )
    try:
        settings = small_settings(config_cls, data_dir, tmp_path / "runs", max_epochs=2)
        summary = experiment_cls(settings).run()
    finally:
        mp.undo()
    assert sorted(read) == ["train", "val"]
    run = tmp_path / "runs" / settings.run_id
    for name in ("config.json", "env.json", "history.jsonl", "train.log", "curves.png", "best.pt", "last.pt"):
        assert (run / name).exists(), name
    rows = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    assert len(rows) == summary["epochs_run"] == 2
    assert summary["params"] == experiment_cls.total_params and summary["best_epoch"] >= 1
    c = summary["class_counts"]["train"]
    assert summary["pos_weight"] == pytest.approx(c["benign"] / c["gridsybil"])
    assert set(summary["thresholds"]["window"]) == {"max_f1", "benign_fpr_5pct"}
    assert set(summary["thresholds"]["track"]) == {"max", "mean"} and summary["val_tracks"] == 12
    trainer = trainer_cls.from_checkpoint(run / "best.pt", device="cpu")
    assert type(trainer.settings) is config_cls and trainer.settings == settings
    out = trainer.predict(np.zeros((3, seq_len, N_FEATURES), np.float32), np.ones((3, seq_len), bool))
    assert out["prob"].shape == (3,) and out["z"].shape == (3, 16)
    with pytest.raises(FileExistsError):
        experiment_cls(settings).run()
    wrong = make_data_dir(tmp_path / "wrong", seq_len * 2)
    with pytest.raises(ValueError, match="rows, the config says"):
        experiment_cls(small_settings(config_cls, wrong, tmp_path / "wrong_runs", max_epochs=1)).run()


def check_resume(experiment_cls: type, config_cls: type, tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path, config_cls().seq_len)
    full = experiment_cls(small_settings(config_cls, data_dir, tmp_path / "full", max_epochs=3)).run()
    part = small_settings(config_cls, data_dir, tmp_path / "part", max_epochs=2)
    experiment_cls(part).run()
    run = tmp_path / "part" / part.run_id
    (run / "summary.json").unlink()
    config = json.loads((run / "config.json").read_text())
    config["max_epochs"] = 3
    (run / "config.json").write_text(json.dumps(config))
    resumed = experiment_cls(dataclasses.replace(part, max_epochs=3), resume=True).run()
    rows_full = [json.loads(x) for x in (tmp_path / "full" / "seed0" / "history.jsonl").read_text().splitlines()]
    rows_resumed = [json.loads(x) for x in (run / "history.jsonl").read_text().splitlines()]
    assert len(rows_resumed) == resumed["epochs_run"] == 3
    for a, b in zip(rows_full, rows_resumed):
        assert a["val_loss"] == pytest.approx(b["val_loss"], rel=1e-5)
    assert resumed["best_val_auroc"] == pytest.approx(full["best_val_auroc"])


def check_feature_log(experiment_cls: type, config_cls: type, tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path, config_cls().seq_len)
    experiment_cls(small_settings(config_cls, data_dir, tmp_path / "runs", max_epochs=1)).run()
    assert not (tmp_path / "runs" / "seed0" / "feature_history.jsonl").exists()
    settings = small_settings(config_cls, data_dir, tmp_path / "runs", max_epochs=2, log_feature_error=True)
    experiment_cls(settings).run()
    run = tmp_path / "runs" / "seed0_featlog"
    rows = [json.loads(line) for line in (run / "feature_history.jsonl").read_text().splitlines()]
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [h["epoch"] for h in history] == [1, 2]
    for row, h in zip(rows, history):
        assert np.isclose(np.mean(row["val_row_pooled"]), h["val_recon"], rtol=1e-4)


def test_n_down_2_is_the_st_t32_model_bit_for_bit() -> None:
    """With n_down = 2 at T = 32 the supervised model equals `task.autoencoder_st_t32`'s: names, shapes, weights for
    the same seed, outputs; that model stays 63,870 parameters."""
    old_cfg = t32_config.SupervisedAEConfig()
    assert count_params(t32_model.SupervisedAutoencoder(old_cfg)) == t32_config.TOTAL_PARAMS == 63_870
    new_cfg = dataclasses.replace(SupervisedAEConfig(), seq_len=32, n_down=2)
    torch.manual_seed(5)
    old = t32_model.SupervisedAutoencoder(old_cfg).eval()
    torch.manual_seed(5)
    new = SupervisedAutoencoder(new_cfg).eval()
    a, b = old.state_dict(), new.state_dict()
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    x = torch.randn(3, 32, N_FEATURES, generator=torch.Generator().manual_seed(1))
    mask = torch.arange(32).unsqueeze(0) < torch.tensor([[32], [2], [17]])
    with torch.no_grad():
        for p, q in zip(old(x, mask), new(x, mask)):
            assert torch.equal(p, q)


def test_config_equals_t32_except_input_length_and_depth() -> None:
    check_config(SupervisedAEConfig, 20, 2, "autoencoder_st_t20")
    assert (SEQ_LEN, N_DOWN, N_FEATURES) == (20, 2, 13)


def test_model_shapes_params_and_padding() -> None:
    check_model(SupervisedAEConfig, TOTAL_PARAMS)


def test_one_run_smoke_reads_only_train_and_val(tmp_path: Path) -> None:
    check_smoke(Experiment, SupervisedAEConfig, Trainer, tmp_path)


def test_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    check_resume(Experiment, SupervisedAEConfig, tmp_path)


def test_feature_error_log_is_opt_in(tmp_path: Path) -> None:
    check_feature_log(Experiment, SupervisedAEConfig, tmp_path)
