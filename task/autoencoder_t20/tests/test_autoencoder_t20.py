"""Tests for the unlabelled T = 20 autoencoder on synthetic shards (CPU only).

The helpers without a `test_` prefix (`write_shards`, `make_data_dir`, `settings_for`, `check_*`) are shared with the
T = 10 / 5 tests. The scoring / evaluation maths is `task/autoencoder_t32`'s and tested there.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from task.autoencoder_benign_t20.model import ConvAutoencoder, count_params
from task.autoencoder_t20.config import N_DOWN, N_FEATURES, SEQ_LEN, TOTAL_PARAMS, AutoencoderT20Config
from task.autoencoder_t20.data import UnlabelledWindows
from task.autoencoder_t20.evaluate import TestEvaluation
from task.autoencoder_t20.labels import WindowLabels
from task.autoencoder_t20.scoring import ScoreFiles
from task.autoencoder_t20.train import Experiment
from task.autoencoder_t20.trainer import Trainer
from task.autoencoder_t32 import config as t32_config

TASK = Path(__file__).resolve().parents[2]
REPO_ROOT = TASK.parent
LABEL_FREE_MODULES = ("data.py", "trainer.py", "train.py", "config.py")


def write_shards(
    root: Path, split: str, seq_len: int, n_links: int = 24, seed: int = 0, sender_base: int = 0, info: str = "ok"
) -> None:
    """Synthetic `<split>/b<T>/part-XXXXX.json` + `_info.json`: consecutive T-row crops of links, junk padding.

    Links alternate benign (DataReplaySybil_0709) and GridSybil (GridSybil_1416, shifted features). `info`: "ok"
    writes valid info files, "corrupt" writes text that is not JSON, "none" writes no info file.
    """
    rng = np.random.default_rng(seed)
    folder = root / split / f"b{seq_len:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    windows, infos = [], []
    for j in range(n_links):
        n_link = int(rng.choice([2, seq_len + 3, 2 * seq_len + 1, 70]))
        attack = j % 2 == 0
        link = rng.normal(size=(n_link, N_FEATURES)) + (3.0 if attack else 0.0)
        scenario = "GridSybil_1416" if attack else "DataReplaySybil_0709"
        sender = sender_base + j
        for k, start in enumerate(range(0, n_link, seq_len)):
            n = min(n_link - start, seq_len)
            x = np.full((seq_len, N_FEATURES), 7.0)
            x[:n] = link[start : start + n]
            wid = f"{scenario}/run/traceJSON-{j}-1-A0-0-7.json#{100 + j}-{sender}#{k}"
            windows.append({"id": wid, "x": x.round(4).tolist(), "mask": [1] * n + [0] * (seq_len - n)})
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


def make_data_dir(tmp_path: Path, seq_len: int, train_info: str = "corrupt") -> Path:
    """Train and val with unreadable label files (by default), test with valid ones."""
    root = tmp_path / f"T{seq_len}_benign_grid"
    write_shards(root, "train", seq_len, seed=0, sender_base=0, info=train_info)
    write_shards(root, "val", seq_len, n_links=8, seed=1, sender_base=1000, info=train_info)
    write_shards(root, "test", seq_len, n_links=12, seed=2, sender_base=2000, info="ok")
    return root


def settings_for(config_cls: type, data_dir: Path, tmp_path: Path, **changes):
    base = config_cls(
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


def check_config(config_cls: type, seq_len: int, n_down: int, package: str) -> None:
    """Every setting except the paths, seq_len and n_down equals the T = 32 unlabelled package's."""
    ours, theirs = config_cls().to_dict(), t32_config.AutoencoderT32Config().to_dict()
    assert {k for k in theirs if ours[k] != theirs[k]} == {"data_dir", "runs_dir", "cache_dir", "seq_len"}
    assert set(ours) - set(theirs) == {"n_down"}
    s = config_cls()
    assert (s.seq_len, s.n_down) == (seq_len, n_down)
    assert s.data_dir == f"src/data/encoder_input/all/T{seq_len}_benign_grid"
    assert s.runs_dir == f"task/{package}/runs" and s.cache_dir == f"task/{package}/cache"
    assert config_cls.from_dict(json.loads(json.dumps(s.to_dict()))) == s


def check_label_free_imports(package: str) -> None:
    """The training-side modules import neither a labels module, an evaluate module nor a supervised package."""
    for name in LABEL_FREE_MODULES:
        tree = ast.parse((TASK / package / name).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported.update(f"{node.module}.{a.name}" for a in node.names)
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
        assert not [m for m in imported if "labels" in m or "autoencoder_st_" in m], (package, name)
        assert not [m for m in imported if m.endswith("evaluate")], (package, name)


def check_train_reads_no_label_and_no_test(experiment_cls: type, config_cls: type, tmp_path: Path) -> None:
    seq_len = config_cls().seq_len
    data_dir = make_data_dir(tmp_path, seq_len)
    settings = settings_for(config_cls, data_dir, tmp_path)
    experiment = experiment_cls(settings)
    with pytest.raises(ValueError, match="reads only"):
        experiment.load("test")
    summary = experiment.run()
    run_dir = Path(settings.runs_dir) / "seed0"
    for name in ("config.json", "env.json", "history.jsonl", "best.pt", "last.pt", "summary.json", "mahalanobis.npz"):
        assert (run_dir / name).is_file(), name
    assert summary["labels_read"] is False and summary["epochs_run"] == 3
    assert summary["params"] == experiment_cls.total_params
    assert not (Path(settings.cache_dir) / "test").exists()
    val = UnlabelledWindows.load(data_dir, "val", workers=1)
    scores = ScoreFiles(run_dir).load("val")
    assert np.array_equal(scores["ids"], val.ids) and len(scores["reconstruction"]) == len(val)
    p95 = summary["thresholds"]["window"]["reconstruction"]["p95"]
    assert np.isclose(p95, np.quantile(scores["reconstruction"].astype(np.float64), 0.95), rtol=1e-5)
    with pytest.raises(FileExistsError):
        experiment_cls(settings).run()


def check_cli_guard(module: str, seq_len: int, tmp_path: Path) -> None:
    """`python -m <module>` finishes with the label-file guard on in the process and the workers."""
    data_dir = make_data_dir(tmp_path, seq_len)
    args = ["--seed", "0", "--epochs", "1", "--device", "cpu", "--workers", "2", "--cache-dir", ""]
    args += ["--data-dir", str(data_dir), "--runs-dir", str(tmp_path / "cli_runs")]
    done = subprocess.run([sys.executable, "-m", module, *args], cwd=REPO_ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    summary = json.loads((tmp_path / "cli_runs" / "seed0" / "summary.json").read_text())
    assert summary["label_file_guard"] == {"process": True, "workers": True}


def check_resume(experiment_cls: type, config_cls: type, tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path, config_cls().seq_len)
    full = experiment_cls(settings_for(config_cls, data_dir, tmp_path, runs_dir=str(tmp_path / "full"))).run()
    part = settings_for(config_cls, data_dir, tmp_path, runs_dir=str(tmp_path / "part"), max_epochs=2)
    experiment_cls(part).run()
    (Path(part.runs_dir) / "seed0" / "summary.json").unlink()
    resumed = experiment_cls(dataclasses.replace(part, max_epochs=3), resume=True).run()
    assert resumed["epochs_run"] == 3
    assert np.isclose(resumed["best_val_loss"], full["best_val_loss"], rtol=1e-4)


def check_end_to_end(experiment_cls: type, evaluation_cls: type, config_cls: type, trainer_cls: type, tmp_path: Path):
    """Train, then evaluate: test scores are written before any label file is opened; metrics match sklearn."""
    seq_len = config_cls().seq_len
    root = make_data_dir(tmp_path, seq_len, train_info="ok")
    settings = settings_for(config_cls, root, tmp_path)
    experiment_cls(settings).run()
    run_dir = Path(settings.runs_dir) / "seed0"
    order: list[str] = []
    real_save, real_load = ScoreFiles.save, WindowLabels.load.__func__
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(
            ScoreFiles, "save", lambda self, split, *a: (order.append(f"scores:{split}"), real_save(self, split, *a))[1]
        )
        mp.setattr(
            WindowLabels,
            "load",
            classmethod(lambda cls, d, split, w=8: (order.append(f"labels:{split}"), real_load(cls, d, split, w))[1]),
        )
        result = evaluation_cls(run_dir, "cpu").run()
    finally:
        mp.undo()
    assert order[0] == "scores:test" and order[1] == "labels:test"
    scores = ScoreFiles(run_dir).load("test")
    labels = WindowLabels.load(root, "test", 1)
    assert np.array_equal(scores["ids"], labels.ids)
    auroc = result["test_all"]["window"]["reconstruction"]["auroc"]
    assert np.isclose(auroc, roc_auc_score(labels.y, scores["reconstruction"]))
    assert np.isclose(result["test_all"]["length_only"]["auroc"], roc_auc_score(labels.y, -scores["n_messages"]))
    assert result["class_counts"]["train"]["gridsybil"] > 0
    trainer = trainer_cls.from_checkpoint(run_dir / "best.pt", "cpu")
    assert type(trainer.settings) is config_cls and trainer.settings.seq_len == seq_len


def test_config_equals_t32_except_input_length_and_depth() -> None:
    check_config(AutoencoderT20Config, 20, 2, "autoencoder_t20")
    assert (SEQ_LEN, N_DOWN, N_FEATURES) == (20, 2, 13)


def test_model_parameter_count() -> None:
    model = ConvAutoencoder(AutoencoderT20Config())
    assert count_params(model) == TOTAL_PARAMS == 56_957


def test_training_dataset_exposes_no_labels(tmp_path: Path) -> None:
    train = UnlabelledWindows.load(make_data_dir(tmp_path, SEQ_LEN), "train", workers=1)
    assert set(UnlabelledWindows.__slots__) == {"split", "x", "mask", "ids", "n_messages"}
    assert train.x.shape[1:] == (SEQ_LEN, N_FEATURES) and np.all(train.x[~train.mask] == 0.0)


def test_label_free_modules_do_not_import_labels() -> None:
    check_label_free_imports("autoencoder_t20")


def test_train_reads_no_label_and_no_test(tmp_path: Path) -> None:
    check_train_reads_no_label_and_no_test(Experiment, AutoencoderT20Config, tmp_path)


def test_wrong_window_length_is_refused(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path, 32)
    with pytest.raises(ValueError, match="rows, the config says 20"):
        Experiment(settings_for(AutoencoderT20Config, data_dir, tmp_path)).run()


def test_shared_vehicle_between_train_and_val_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "leaky"
    write_shards(root, "train", SEQ_LEN, seed=0, sender_base=0, info="none")
    write_shards(root, "val", SEQ_LEN, n_links=8, seed=0, sender_base=0, info="none")
    with pytest.raises(ValueError, match="both train and val"):
        Experiment(settings_for(AutoencoderT20Config, root, tmp_path)).run()


def test_command_line_run_installs_the_guard(tmp_path: Path) -> None:
    check_cli_guard("task.autoencoder_t20.train", SEQ_LEN, tmp_path)


def test_resume_continues_the_same_run(tmp_path: Path) -> None:
    check_resume(Experiment, AutoencoderT20Config, tmp_path)


def test_end_to_end_evaluation_scores_before_labels(tmp_path: Path) -> None:
    check_end_to_end(Experiment, TestEvaluation, AutoencoderT20Config, Trainer, tmp_path)
