"""Tests for the unlabelled T = 64 autoencoder on synthetic shards (CPU only)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from task.autoencoder_t64.config import AutoencoderT64Config
from task.autoencoder_t64.evaluate import Evaluation, Metrics
from task.autoencoder_t64.scoring import Mahalanobis, ScoreFiles, TrackScores, track_id, track_ids
from task.autoencoder_t64.train import Experiment, ValidationThresholds, parse_args
from task.autoencoder_t64.trainer import Trainer
from task.encoder.config import N_FEATURES, SEQ_LEN, EncoderTaskConfig
from task.encoder.data import VehicleSplit, WindowStore
from task.encoder.model import ConvAutoencoder, count_params


def write_shards(root: Path, split: str, n_links: int, seed: int, sender_base: int) -> None:
    """Synthetic `<split>/b64/part-XXXXX.json` + `_info.json` (two shards): links cut at 64, junk in padding rows."""
    rng = np.random.default_rng(seed)
    folder = root / split / "b64"
    folder.mkdir(parents=True, exist_ok=True)
    windows, info = [], []
    for j in range(n_links):
        n_link = int(rng.choice([2, 9, 40, 64, 100]))
        scenario = "GridSybil_0709" if j % 2 else "GridSybil_1416"
        sender = sender_base + j // 2  # two links per sender vehicle
        label = 16 if sender % 3 else 0
        for k, start in enumerate(range(0, n_link, SEQ_LEN)):
            n = min(n_link - start, SEQ_LEN)
            x = np.full((SEQ_LEN, N_FEATURES), 7.0)
            x[:n] = rng.normal(size=(n, N_FEATURES)) + (1.5 if label else 0.0)
            wid = f"{scenario}/run/trace-{j}.json#{900 + j}-{sender}#{k}"
            windows.append({"id": wid, "x": x.round(4).tolist(), "mask": [1] * n + [0] * (SEQ_LEN - n)})
            info.append({"id": wid, "scenario": scenario, "sender": sender, "n_messages": n, "label": label})
    half = len(windows) // 2
    for part, sl in enumerate((slice(0, half), slice(half, None))):
        (folder / f"part-{part:05d}.json").write_text(json.dumps({"windows": windows[sl]}))
        (folder / f"part-{part:05d}_info.json").write_text(json.dumps({"windows": info[sl]}))


def make_settings(root: Path, **changes) -> AutoencoderT64Config:
    values = {
        "data_dir": str(root / "data"),
        "runs_dir": str(root / "runs"),
        "cache_dir": str(root / "cache"),
        "max_epochs": 3,
        "batch_size": 16,
        "eval_batch": 32,
        "device": "cpu",
    }
    return dataclasses.replace(AutoencoderT64Config(), **(values | changes))


@pytest.fixture(scope="module")
def finished(tmp_path_factory) -> tuple[AutoencoderT64Config, dict, dict]:
    """A trained + evaluated run on synthetic data. Training runs with the label file of the train cache removed."""
    root = tmp_path_factory.mktemp("ae_t64")
    write_shards(root / "data", "train", n_links=80, seed=1, sender_base=100)
    write_shards(root / "data", "test", n_links=30, seed=2, sender_base=900)
    settings = make_settings(root)
    store = WindowStore(settings.data_dir, settings.cache_dir)
    labels_path = store.build("train") / "labels.npy"
    hidden = labels_path.with_suffix(".hidden")
    labels_path.rename(hidden)
    summary = Experiment(settings).run()
    test_cache_built = (Path(settings.cache_dir) / "test").exists()
    hidden.rename(labels_path)
    results = Evaluation(Path(settings.runs_dir) / settings.run_id, device="cpu").run()
    return settings, {**summary, "test_cache_built_by_training": test_cache_built}, results


def test_config_and_architecture() -> None:
    s = AutoencoderT64Config()
    assert s.run_id == "seed0" and s.data_dir.endswith("benign_gridsybil/T64")
    assert (s.patience, s.plateau_patience, s.max_epochs, s.min_delta) == (10, 5, 300, 0.0)
    assert AutoencoderT64Config.from_dict(json.loads(json.dumps(s.to_dict()))) == s
    arch = s.architecture()
    assert (arch.latent_dim, arch.channels, arch.mask_channel) == (16, (32, 64, 64), True)
    assert count_params(ConvAutoencoder(arch)) == count_params(ConvAutoencoder(EncoderTaskConfig(mode="ae"))) == 80189


def test_parse_args_defaults_and_overrides() -> None:
    assert parse_args([]) == AutoencoderT64Config()
    s = parse_args(["--seed", "3", "--epochs", "7", "--device", "cpu", "--runs-dir", "x"])
    assert (s.seed, s.max_epochs, s.device, s.runs_dir, s.run_id) == (3, 7, "cpu", "x", "seed3")


def test_track_id_groups_windows_of_one_link() -> None:
    ids = np.asarray(["A/r/t.json#5-7#0", "A/r/t.json#5-7#1", "A/r/t.json#6-7#0"])
    assert track_id(ids[0]) == "A/r/t.json#5-7"
    tracks = TrackScores(track_ids(ids))
    assert len(tracks) == 2 and tracks.n_windows.tolist() == [2, 1]
    agg = tracks.aggregate(np.asarray([1.0, 3.0, 5.0]))
    assert agg["max"].tolist() == [3.0, 5.0] and agg["mean"].tolist() == [2.0, 5.0]


def test_validation_thresholds_window_and_track() -> None:
    ids = np.asarray([f"A/r/t.json#{j // 2}-1#{j % 2}" for j in range(200)])
    scores = {"reconstruction": np.arange(200, dtype=float)}
    out = ValidationThresholds((0.95, 0.99)).compute(ids, scores)
    assert out["window"]["reconstruction"]["p95"] == pytest.approx(np.quantile(np.arange(200), 0.95))
    assert out["track"]["reconstruction_max"]["p99"] == pytest.approx(np.quantile(np.arange(1, 200, 2), 0.99))
    assert set(out["track"]) == {"reconstruction_max", "reconstruction_mean"}


def test_metrics_at_threshold() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.3, 0.9, 0.2, 0.8, 0.9, 1.0])
    m = Metrics.at_threshold(labels, scores, 0.8)
    assert m["confusion"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 3}
    assert m["precision"] == 0.75 and m["recall"] == 0.75 and m["benign_fpr"] == 0.25
    assert Metrics.auroc(labels[:4], scores[:4]) is None


def test_val_vehicles_equal_task_encoder_rule(finished) -> None:
    """The held-out vehicles are those of `task.encoder`'s VehicleSplit on all train windows (its supervised pool)."""
    settings, _, _ = finished
    run = Path(settings.runs_dir) / settings.run_id
    train = WindowStore(settings.data_dir, settings.cache_dir).open("train")
    pool = train.take(np.arange(len(train)))  # the pool `task.encoder.train` builds in supervised mode
    _, val_rows = VehicleSplit(EncoderTaskConfig().val_frac, EncoderTaskConfig().split_seed).split(pool.vehicles)
    expected = np.unique(pool.vehicles[val_rows])
    assert np.array_equal(np.load(run / "val_vehicles.npy"), expected)
    assert np.array_equal(np.load(run / "val_ids.npy"), pool.ids[np.sort(val_rows)])
    fit, val = Experiment(settings).fit_val(train)
    assert not set(fit.vehicles) & set(val.vehicles) and len(fit) + len(val) == len(train)


def test_training_reads_no_labels_and_no_test_split(finished) -> None:
    settings, summary, _ = finished
    assert summary["label_reads"] == [] and summary["splits_opened"] == ["train"]
    assert summary["test_cache_built_by_training"] is False  # the fixture trained with labels.npy removed, too
    source = Path("task/autoencoder_t64/train.py").read_text() + Path("task/autoencoder_t64/trainer.py").read_text()
    assert ".labels(" not in source and '"test"' not in source and "labels.npy" not in source


def test_run_folder_files_and_alignment(finished) -> None:
    settings, summary, _ = finished
    run = Path(settings.runs_dir) / settings.run_id
    for name in ("config.json", "env.json", "history.jsonl", "train.log", "best.pt", "summary.json", "results.json"):
        assert (run / name).is_file(), name
    assert json.loads((run / "config.json").read_text())["seed"] == 0
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    assert len(history) == summary["epochs_run"] <= settings.max_epochs
    assert summary["best_val_loss"] == pytest.approx(min(h["val_loss"] for h in history))
    for split, n in (("val", summary["val_windows"]), ("test", None)):
        files = ScoreFiles(run, split).load()
        n = len(files["ids"]) if n is None else n
        assert len(set(files["ids"])) == n
        assert files["recon_error"].shape == files["mahalanobis"].shape == files["n_messages"].shape == (n,)
        assert files["recon_error_feat"].shape == (n, N_FEATURES)
        assert np.allclose(files["recon_error_feat"].mean(axis=1), files["recon_error"], atol=1e-5)
        assert np.isfinite(files["recon_error"]).all() and np.isfinite(files["mahalanobis"]).all()


def test_stored_scores_reproduce_from_checkpoint(finished) -> None:
    """Scores in the files = scores recomputed from best.pt + mahalanobis.npz, window by window in file order."""
    settings, summary, _ = finished
    run = Path(settings.runs_dir) / settings.run_id
    trainer = Trainer.from_checkpoint(run / "best.pt", "cpu")
    test = WindowStore(settings.data_dir, settings.cache_dir).open("test")
    stored = ScoreFiles(run, "test").load()
    assert np.array_equal(stored["ids"], np.asarray(test.ids).astype(str))
    out = trainer.score(test.x, test.mask)
    assert np.allclose(out["error"], stored["recon_error"], atol=1e-5)
    state = np.load(run / "mahalanobis.npz")
    maha = Mahalanobis(settings.maha_ridge)
    maha.mean, maha.precision = state["mean"], state["precision"]
    assert np.allclose(maha.score(out["z"]), stored["mahalanobis"], rtol=1e-4, atol=1e-4)
    val = ScoreFiles(run, "val").load()
    assert summary["thresholds"]["window"]["reconstruction"]["p95"] == pytest.approx(
        np.quantile(val["recon_error"], 0.95)
    )


def test_padding_rows_do_not_change_scores(finished) -> None:
    settings, _, _ = finished
    trainer = Trainer.from_checkpoint(Path(settings.runs_dir) / settings.run_id / "best.pt", "cpu")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, SEQ_LEN, N_FEATURES)).astype(np.float32)
    mask = np.zeros((6, SEQ_LEN), dtype=bool)
    for i, n in enumerate((1, 3, 10, 33, 63, 64)):
        mask[i, :n] = True
    junk = x.copy()
    junk[~mask] = 99.0
    a, b = trainer.score(x, mask), trainer.score(junk, mask)
    assert np.allclose(a["error"], b["error"], atol=1e-6) and np.allclose(a["z"], b["z"], atol=1e-6)


def test_results_match_independent_recomputation(finished) -> None:
    settings, _, results = finished
    run = Path(settings.runs_dir) / settings.run_id
    stored = ScoreFiles(run, "test").load()
    test = WindowStore(settings.data_dir, settings.cache_dir).open("test")
    y = np.asarray(test.labels("test")).astype(int)
    n = stored["n_messages"]
    assert results["test"]["windows"] == len(y) and results["test"]["gridsybil"] == int(y.sum())
    for name, key in (("reconstruction", "recon_error"), ("mahalanobis", "mahalanobis")):
        assert results["window"][name]["auroc"] == pytest.approx(roc_auc_score(y, stored[key]))
        assert results["window"][name]["pr_auc"] == pytest.approx(average_precision_score(y, stored[key]))
        p95 = results["window"][name]["thresholds"]["val_p95"]
        assert sum(p95["confusion"].values()) == len(y)
        assert p95["confusion"]["tp"] == int(((stored[key] >= p95["threshold"]) & (y == 1)).sum())
    assert results["length_only"]["auroc"] == pytest.approx(roc_auc_score(y, -n.astype(float)))
    short, long_ = results["strata_short_long"]["1-3 rows"], results["strata_short_long"][">=4 rows"]
    assert short["windows"] == int((n <= 3).sum()) and short["windows"] + long_["windows"] == len(y)
    assert long_["auroc_reconstruction"] == pytest.approx(roc_auc_score(y[n >= 4], stored["recon_error"][n >= 4]))
    assert sum(v["windows"] for v in results["strata_encoder_bins"].values()) == len(y)
    track = results["track"]
    assert track["mixed_label_tracks"] == 0 and track["tracks"] == len(set(track_ids(stored["ids"])))
    assert set(results["training_set"]) == {"train", "fit", "val"}
    counts = results["training_set"]
    assert counts["fit"]["windows"] + counts["val"]["windows"] == counts["train"]["windows"]
    assert 0 < counts["fit"]["gridsybil_share"] < 1 and len(results["per_feature"]) == N_FEATURES


def test_finished_run_is_not_overwritten(finished) -> None:
    settings, _, _ = finished
    with pytest.raises(FileExistsError):
        Experiment(settings).run()


def test_early_stopping_and_lr_schedule(tmp_path) -> None:
    """A 99% relative decrease is never reached after the first epoch: the run stops at 1 + patience epochs."""
    settings = make_settings(tmp_path, min_delta=0.99, max_epochs=20, patience=3, plateau_patience=1)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, SEQ_LEN, N_FEATURES)).astype(np.float32)
    mask = np.ones((40, SEQ_LEN), dtype=bool)
    mask[:, 20:] = False
    trainer = Trainer(ConvAutoencoder(settings.architecture()), settings, torch.device("cpu"))
    out = trainer.fit((x[:32], mask[:32]), (x[32:], mask[32:]), tmp_path)
    assert out["stop_reason"] == "early_stopping" and out["best_epoch"] == 1 and out["epochs_run"] == 4
    assert (tmp_path / "best.pt").is_file() and (tmp_path / "loss_curve.png").is_file()
