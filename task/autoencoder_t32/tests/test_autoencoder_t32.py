"""Tests for the unlabelled T = 32 autoencoder on synthetic shards (CPU only)."""

from __future__ import annotations

import ast
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from task.autoencoder_benign_t32.model import ConvAutoencoder, count_params
from task.autoencoder_t32 import data as data_module
from task.autoencoder_t32.config import N_FEATURES, SEQ_LEN, TOTAL_PARAMS, AutoencoderT32Config
from task.autoencoder_t32.data import UnlabelledWindows, track_id, vehicle_key
from task.autoencoder_t32.evaluate import TestEvaluation
from task.autoencoder_t32.labels import WindowLabels
from task.autoencoder_t32.scoring import Evaluation, ScoreFiles, at_threshold, detection, ranking
from task.autoencoder_t32.train import Experiment
from task.autoencoder_t32.trainer import Trainer

PACKAGE = Path(data_module.__file__).parent
REPO_ROOT = PACKAGE.parents[1]
LABEL_FREE_MODULES = ("data.py", "trainer.py", "train.py", "config.py")


def write_shards(root: Path, split: str, n_links: int = 24, seed: int = 0, sender_base: int = 0, info: str = "ok"):
    """Synthetic `<split>/b32/part-XXXXX.json` + `_info.json`: consecutive 32-row crops of links, junk padding.

    Links alternate benign (DataReplaySybil_0709) and GridSybil (GridSybil_1416, shifted features). `info`: "ok"
    writes valid info files, "corrupt" writes text that is not JSON, "none" writes no info file.
    """
    rng = np.random.default_rng(seed)
    folder = root / split / "b32"
    folder.mkdir(parents=True, exist_ok=True)
    windows, infos = [], []
    for j in range(n_links):
        n_link = int(rng.choice([2, 20, 40, 70]))
        attack = j % 2 == 0
        link = rng.normal(size=(n_link, N_FEATURES)) + (3.0 if attack else 0.0)
        scenario = "GridSybil_1416" if attack else "DataReplaySybil_0709"
        sender = sender_base + j
        for k, start in enumerate(range(0, n_link, SEQ_LEN)):
            n = min(n_link - start, SEQ_LEN)
            x = np.full((SEQ_LEN, N_FEATURES), 7.0)
            x[:n] = link[start : start + n]
            wid = f"{scenario}/run/traceJSON-{j}-1-A0-0-7.json#{100 + j}-{sender}#{k}"
            windows.append({"id": wid, "x": x.round(4).tolist(), "mask": [1] * n + [0] * (SEQ_LEN - n)})
            infos.append(
                {
                    "id": wid,
                    "family": scenario.split("_")[0],
                    "scenario": scenario,
                    "sender": sender,
                    "n_messages": n,
                    "label": int(attack),
                    "label_name": "GridSybil" if attack else "Benign",
                    "attack_code": 16 if attack else 0,
                    "is_attack": int(attack),
                }
            )
    half = len(windows) // 2
    for part, (lo, hi) in enumerate([(0, half), (half, len(windows))]):
        (folder / f"part-{part:05d}.json").write_text(json.dumps({"windows": windows[lo:hi]}))
        if info == "ok":
            (folder / f"part-{part:05d}_info.json").write_text(json.dumps({"windows": infos[lo:hi]}))
        elif info == "corrupt":
            (folder / f"part-{part:05d}_info.json").write_text("not json: the training side must never parse this")


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """Train and val with unreadable label files, test with valid ones."""
    root = tmp_path / "T32_benign_grid"
    write_shards(root, "train", seed=0, sender_base=0, info="corrupt")
    write_shards(root, "val", n_links=8, seed=1, sender_base=1000, info="corrupt")
    write_shards(root, "test", n_links=12, seed=2, sender_base=2000, info="ok")
    return root


def settings_for(data_dir: Path, tmp_path: Path, **changes) -> AutoencoderT32Config:
    base = AutoencoderT32Config(
        data_dir=str(data_dir),
        runs_dir=str(tmp_path / "runs"),
        cache_dir=str(tmp_path / "cache"),
        max_epochs=3,
        batch_size=16,
        eval_batch=32,
        device="cpu",
        workers=1,
    )
    return dataclasses.replace(base, **changes)


def test_settings_equal_the_one_class_package():
    from task.autoencoder_benign_t32.config import AutoencoderConfig

    ours, theirs = AutoencoderT32Config().to_dict(), AutoencoderConfig().to_dict()
    shared = set(ours) & set(theirs) - {"data_dir", "runs_dir", "cache_dir"}
    assert {
        "lr",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
        "plateau_factor",
        "plateau_patience",
    } <= shared
    assert {"min_delta", "grad_clip", "latent_dim", "threshold_quantiles"} <= shared
    assert {k: ours[k] for k in shared} == {k: theirs[k] for k in shared}
    assert AutoencoderT32Config.from_dict(ours) == AutoencoderT32Config()


def test_model_is_the_sibling_autoencoder_with_its_parameter_count():
    model = ConvAutoencoder(AutoencoderT32Config())
    assert count_params(model) == TOTAL_PARAMS == 63_293
    recon, z = model(torch.zeros(2, SEQ_LEN, N_FEATURES), torch.ones(2, SEQ_LEN, dtype=torch.bool))
    assert recon.shape == (2, SEQ_LEN, N_FEATURES) and z.shape == (2, 16)


def test_ids_give_track_and_vehicle():
    wid = "GridSybil_0709/VeReMi_x/traceJSON-10005-10003-A0-28468-7.json#1094892-9489#2"
    assert track_id(wid) == wid[: -len("#2")]
    assert vehicle_key(wid) == "0709:9489"
    with pytest.raises(ValueError):
        track_id("no-suffix")


def test_training_dataset_exposes_no_labels(data_dir: Path):
    """The object training gets holds x, mask, ids and the row count, nothing else, and is built with the label
    files unreadable."""
    train = UnlabelledWindows.load(data_dir, "train", workers=1)
    assert set(UnlabelledWindows.__slots__) == {"split", "x", "mask", "ids", "n_messages"}
    assert not hasattr(train, "__dict__")
    for name in ("info", "y", "label", "labels", "is_attack", "attack_code", "label_name", "family", "scenario"):
        assert not hasattr(train, name), name
    with pytest.raises(AttributeError):
        train.y = np.zeros(len(train))
    assert train.x.shape[1:] == (SEQ_LEN, N_FEATURES) and train.mask.dtype == bool
    assert np.all(train.x[~train.mask] == 0.0)
    assert np.array_equal(train.n_messages, train.mask.sum(axis=1))


def test_reader_never_touches_info_files(data_dir: Path, tmp_path: Path, monkeypatch):
    """Every file the reader opens is a feature shard or its own cache, with and without the cache."""
    opened: list[str] = []
    real_read_text = Path.read_text

    def spy(self, *args, **kwargs):
        opened.append(str(self))
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    cache = tmp_path / "cache"
    first = UnlabelledWindows.load(data_dir, "train", workers=1, cache_dir=cache)
    again = UnlabelledWindows.load(data_dir, "train", workers=1, cache_dir=cache)
    assert opened and not [p for p in opened if p.endswith("_info.json")]
    assert sorted(p.name for p in (cache / "train").iterdir()) == ["ids.npy", "manifest.json", "mask.npy", "x.npy"]
    assert np.array_equal(first.x, again.x) and np.array_equal(first.ids, again.ids)


def test_feature_shard_with_a_label_field_is_rejected(tmp_path: Path):
    root = tmp_path / "bad"
    write_shards(root, "train", n_links=4, info="none")
    shard = root / "train" / "b32" / "part-00000.json"
    payload = json.loads(shard.read_text())
    payload["windows"][0]["label"] = 1
    shard.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fields other than"):
        UnlabelledWindows.load(root, "train", workers=1)


def test_label_free_modules_do_not_import_the_labels_module():
    for name in LABEL_FREE_MODULES:
        tree = ast.parse((PACKAGE / name).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported.update(f"{node.module}.{a.name}" for a in node.names)
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
        assert not [m for m in imported if "labels" in m or "autoencoder_st_t32" in m], name
        assert not [m for m in imported if m.endswith("evaluate")], name


def test_train_runs_with_unreadable_label_files_and_reads_no_test(data_dir: Path, tmp_path: Path):
    settings = settings_for(data_dir, tmp_path)
    experiment = Experiment(settings)
    with pytest.raises(ValueError, match="reads only"):
        experiment.load("test")
    summary = experiment.run()
    run_dir = Path(settings.runs_dir) / "seed0"
    for name in ("config.json", "env.json", "history.jsonl", "best.pt", "last.pt", "summary.json", "mahalanobis.npz"):
        assert (run_dir / name).is_file(), name
    assert summary["labels_read"] is False and summary["epochs_run"] == 3
    assert set(summary["thresholds"]["window"]) == {"reconstruction", "mahalanobis"}
    assert set(summary["thresholds"]["track"]) == {
        "reconstruction_max",
        "reconstruction_mean",
        "mahalanobis_max",
        "mahalanobis_mean",
    }
    assert not (Path(settings.cache_dir) / "test").exists()
    val = UnlabelledWindows.load(data_dir, "val", workers=1)
    scores = ScoreFiles(run_dir).load("val")
    assert np.array_equal(scores["ids"], val.ids) and len(scores["reconstruction"]) == len(val)
    p95 = summary["thresholds"]["window"]["reconstruction"]["p95"]
    assert np.isclose(p95, np.quantile(scores["reconstruction"].astype(np.float64), 0.95), rtol=1e-5)
    with pytest.raises(FileExistsError):
        Experiment(settings).run()


def test_shared_vehicle_between_train_and_val_is_refused(tmp_path: Path):
    root = tmp_path / "leaky"
    write_shards(root, "train", seed=0, sender_base=0, info="none")
    write_shards(root, "val", n_links=8, seed=0, sender_base=0, info="none")
    with pytest.raises(ValueError, match="both train and val"):
        Experiment(settings_for(root, tmp_path)).run()


def test_command_line_run_installs_the_guard_and_the_guard_blocks_label_files(data_dir: Path, tmp_path: Path):
    """`python -m ...train` finishes with the guard on; the same guard raises when a label file is opened."""
    args = ["--seed", "0", "--epochs", "1", "--device", "cpu", "--workers", "2", "--cache-dir", ""]
    args += ["--data-dir", str(data_dir), "--runs-dir", str(tmp_path / "cli_runs")]
    done = subprocess.run(
        [sys.executable, "-m", "task.autoencoder_t32.train", *args], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr[-2000:]
    summary = json.loads((tmp_path / "cli_runs" / "seed0" / "summary.json").read_text())
    assert summary["label_file_guard"] == {"process": True, "workers": True}
    info = data_dir / "test" / "b32" / "part-00000_info.json"
    code = (
        "from task.autoencoder_t32.data import LabelFileGuard\n"
        "LabelFileGuard.install()\n"
        f"open({str(info)!r}).read()\n"
    )
    blocked = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
    assert blocked.returncode != 0 and "PermissionError" in blocked.stderr


def test_resume_continues_the_same_run(data_dir: Path, tmp_path: Path):
    full = Experiment(settings_for(data_dir, tmp_path, runs_dir=str(tmp_path / "full"))).run()
    part_settings = settings_for(data_dir, tmp_path, runs_dir=str(tmp_path / "part"), max_epochs=2)
    part_dir = Path(part_settings.runs_dir) / "seed0"
    Experiment(part_settings).run()
    (part_dir / "summary.json").unlink()
    resumed = Experiment(dataclasses.replace(part_settings, max_epochs=3), resume=True).run()
    assert resumed["epochs_run"] == 3
    assert np.isclose(resumed["best_val_loss"], full["best_val_loss"], rtol=1e-4)


def test_threshold_and_ranking_metrics():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.95])
    r = ranking(y, s)
    assert np.isclose(r["auroc"], roc_auc_score(y, s)) and r["n_benign"] == 4 and np.isclose(r["chance"], 1 / 3)
    t = at_threshold(y, s, 0.85)
    assert t["recall"] == 0.5 and t["benign_fpr"] == 0.25 and t["precision"] == 0.5
    assert at_threshold(y, s, 0.95)["recall"] == 0.0  # strict >
    assert np.isnan(ranking(np.zeros(3), np.arange(3.0))["auroc"])
    assert set(detection(y, s, {"p95": 0.5, "p99": 0.9})) >= {"auroc", "pr_auc", "p95", "p99"}


def make_evaluation() -> Evaluation:
    """400 windows in 200 two-window tracks; every second track is GridSybil (higher reconstruction score)."""
    rng = np.random.default_rng(0)
    t = np.arange(200)
    track_y = (t % 2 == 0).astype(int)
    track_scenario = np.where(
        track_y == 1,
        np.where(t % 4 == 0, "GridSybil_0709", "GridSybil_1416"),
        np.where(t % 3 == 0, "GridSybil_1416", "DoSRandomSybil_0709"),
    )
    y, scenario = np.repeat(track_y, 2), np.repeat(track_scenario, 2)
    tracks = np.repeat(np.asarray([f"{s}/f.json#{i}-{i}" for i, s in zip(t, track_scenario)]), 2)
    n = len(y)
    lengths = np.where(y == 1, rng.integers(1, 12, n), rng.integers(1, 33, n))
    group = np.asarray([s.rsplit("_", 1)[-1] for s in scenario])
    scores = {"reconstruction": rng.normal(size=n) + 2.0 * y, "mahalanobis": rng.normal(size=n)}
    thresholds = {
        "window": {k: {"p95": 1.5, "p99": 2.5} for k in scores},
        "track": {f"{k}_{how}": {"p95": 1.5, "p99": 2.5} for k in scores for how in ("max", "mean")},
    }
    return Evaluation(scores, y, lengths, tracks, group, scenario, thresholds)


def test_evaluation_report_matches_direct_computation():
    ev = make_evaluation()
    rep = ev.report()
    s, y, n = ev.scores["reconstruction"], ev.y, ev.n_messages
    w = rep["window"]["reconstruction"]
    assert np.isclose(w["auroc"], roc_auc_score(y, s))
    assert np.isclose(w["p95"]["recall"], (s[y == 1] > 1.5).mean())
    assert np.isclose(w["p95"]["benign_fpr"], (s[y == 0] > 1.5).mean())
    assert np.isclose(w["rows_1_3"]["auroc"], roc_auc_score(y[n <= 3], s[n <= 3]))
    assert np.isclose(w["rows_ge_4"]["auroc"], roc_auc_score(y[n >= 4], s[n >= 4]))
    assert w["rows_1_3"]["n_benign"] + w["rows_ge_4"]["n_benign"] == rep["benign"]
    for g in ("0709", "1416"):
        rows = ev.group == g
        assert np.isclose(w["by_group"][g]["auroc"], roc_auc_score(y[rows], s[rows]))
    track = rep["track"]["reconstruction"]
    assert track["tracks"] == 200 and track["mixed_label_tracks"] == 0
    pair_max = s.reshape(-1, 2).max(axis=1)
    assert np.isclose(track["max"]["auroc"], roc_auc_score(y[::2], pair_max))
    assert np.isclose(track["mean"]["auroc"], roc_auc_score(y[::2], s.reshape(-1, 2).mean(axis=1)))
    folder = rep["benign_fpr_by_folder"]["reconstruction"]["DoSRandomSybil_0709"]
    rows = (ev.scenario == "DoSRandomSybil_0709") & (y == 0)
    assert folder["n_benign"] == rows.sum() and np.isclose(folder["p95"], (s[rows] > 1.5).mean())


def test_length_rule_is_fixed_and_not_flipped():
    ev = make_evaluation()
    fixed = ev.report()["length_only"]
    assert np.isclose(fixed["auroc"], roc_auc_score(ev.y, -ev.n_messages.astype(float)))
    flipped = Evaluation(ev.scores, 1 - ev.y, ev.n_messages, ev.tracks, ev.group, ev.scenario, ev.thresholds)
    other = flipped.report()["length_only"]
    assert np.isclose(other["auroc"], 1.0 - fixed["auroc"]) and other["auroc"] < 0.5  # reported as is, below chance
    assert np.isclose(other["auroc_best_direction"], fixed["auroc_best_direction"])


def test_report_restricted_to_gridsybil_folders():
    ev = make_evaluation()
    rows = np.char.startswith(ev.scenario.astype(str), "GridSybil")
    rep = ev.report(rows)
    assert rep["windows"] == rows.sum() and rep["gridsybil"] == ev.y.sum()
    assert set(rep["benign_fpr_by_folder"]["reconstruction"]) <= {"GridSybil_0709", "GridSybil_1416"}
    s = ev.scores["reconstruction"]
    assert np.isclose(rep["window"]["reconstruction"]["auroc"], roc_auc_score(ev.y[rows], s[rows]))


def test_labels_align_with_scored_windows(data_dir: Path):
    test = UnlabelledWindows.load(data_dir, "test", workers=1)
    labels = WindowLabels.load(data_dir, "test", workers=1)
    labels.check(test.ids, test.n_messages)
    assert labels.class_counts()["gridsybil"] > 0 and set(labels.group) == {"0709", "1416"}
    with pytest.raises(ValueError, match="ids differ"):
        labels.check(test.ids[::-1], test.n_messages[::-1])


def test_end_to_end_evaluation_scores_before_labels(tmp_path: Path, monkeypatch):
    root = tmp_path / "T32_benign_grid"
    write_shards(root, "train", seed=0, sender_base=0)
    write_shards(root, "val", n_links=8, seed=1, sender_base=1000)
    write_shards(root, "test", n_links=12, seed=2, sender_base=2000)
    settings = settings_for(root, tmp_path)
    Experiment(settings).run()
    run_dir = Path(settings.runs_dir) / "seed0"
    order: list[str] = []
    real_save, real_load = ScoreFiles.save, WindowLabels.load.__func__
    monkeypatch.setattr(
        ScoreFiles, "save", lambda self, split, *a: (order.append(f"scores:{split}"), real_save(self, split, *a))[1]
    )
    monkeypatch.setattr(
        WindowLabels,
        "load",
        classmethod(lambda cls, d, split, w=8: (order.append(f"labels:{split}"), real_load(cls, d, split, w))[1]),
    )
    result = TestEvaluation(run_dir, "cpu").run()
    assert order[0] == "scores:test" and order[1] == "labels:test"
    assert (run_dir / "evaluation.json").is_file()
    scores = ScoreFiles(run_dir).load("test")
    labels = real_load(WindowLabels, root, "test", 1)
    assert np.array_equal(scores["ids"], labels.ids)
    auroc = result["test_all"]["window"]["reconstruction"]["auroc"]
    assert np.isclose(auroc, roc_auc_score(labels.y, scores["reconstruction"]))
    grid = result["test_gridsybil_folders"]
    assert grid["windows"] == int((labels.family == "GridSybil").sum()) < result["test_all"]["windows"]
    assert result["class_counts"]["train"]["gridsybil"] > 0
    trainer = Trainer.from_checkpoint(run_dir / "best.pt", "cpu")
    assert isinstance(trainer.settings, AutoencoderT32Config)
