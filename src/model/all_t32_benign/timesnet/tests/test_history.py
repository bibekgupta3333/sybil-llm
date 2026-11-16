"""Tests of the training logs (history.jsonl, feature_history.jsonl, steps.jsonl, summary.json) and --short (CPU)."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import numpy as np
import pytest
import torch

from src.model.all_t32_benign.timesnet.config import T32BenignConfig
from src.model.all_t32_benign.timesnet.history import (
    EVAL_KEYS,
    SUMMARY_KEYS,
    TRAIN_KEYS,
    EpochStats,
    JsonLines,
    round_sig,
)
from src.model.all_t32_benign.timesnet.tests.test_train import tiny_settings
from src.model.all_t32_benign.timesnet.train import T32Trainer, TrainCommand, parse_args
from src.model.all_t32_benign.timesnet.base_data import FEATURES

HISTORY_KEYS = (
    "epoch",
    "step",
    "lr",
    *TRAIN_KEYS,
    "train_steps",
    "train_skipped",
    "grad_norm_mean",
    "grad_norm_max",
    "epoch_seconds",
    *EVAL_KEYS,
    "is_best",
    "bad_epochs",
    "scale_recon",
    "scale_nce",
    "scale_p1",
    "scale_p2",
    "scale_p3",
)


def read(path: Path) -> list[dict]:
    return JsonLines(path).read()


@pytest.fixture
def finished_run(encoder_input: Path, tmp_path: Path) -> tuple[T32Trainer, Path]:
    s = tiny_settings(encoder_input, tmp_path, max_epochs=2)
    run_dir = Path(s.runs_dir) / "hist"
    trainer = T32Trainer(s, run_dir, resume=False)
    trainer.run()
    return trainer, run_dir


def test_history_has_one_line_per_epoch_and_the_init_line(finished_run: tuple[T32Trainer, Path]) -> None:
    trainer, run_dir = finished_run
    lines = read(run_dir / "history.jsonl")
    assert [r["epoch"] for r in lines] == [0, 1, 2]
    for r in lines:
        assert set(HISTORY_KEYS) <= set(r), set(HISTORY_KEYS) - set(r)
        assert isinstance(r["is_best"], bool) and r["partial"] is False
    init = lines[0]
    assert all(init[k] is None for k in TRAIN_KEYS) and init["train_steps"] == 0 and init["step"] == 0
    assert init["epoch_seconds"] is None and init["val_joint"] is not None and init["is_best"] is False
    for r, planned in zip(lines[1:], trainer.epoch_batches):
        assert r["train_steps"] + r["train_skipped"] == planned
        assert r["train_joint"] > 0 and r["grad_norm_max"] >= r["grad_norm_mean"] > 0 and r["epoch_seconds"] > 0
        assert r["scale_recon"] is not None and r["lr"] >= 0
    assert lines[-1]["step"] == trainer.step and lines[-1]["lr"] == pytest.approx(0.0, abs=1e-12)
    best = [r["epoch"] for r in lines if r["is_best"]]
    assert best and best[-1] == trainer.best_epoch
    evals = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    evals = [r for r in evals if r["kind"] == "eval"]
    assert [r["val_joint"] for r in evals] == pytest.approx([r["val_joint"] for r in lines])


def test_epoch_means_use_only_updates_with_that_loss(finished_run: tuple[T32Trainer, Path]) -> None:
    _, run_dir = finished_run
    steps = read(run_dir / "steps.jsonl")
    for line in read(run_dir / "history.jsonl")[1:]:
        epoch_steps = [r for r in steps if r["epoch"] == line["epoch"] - 1]
        assert len(epoch_steps) == line["train_steps"]
        for name in ("joint", "recon", "nce", "p1", "p2", "p3"):
            present = [r[name] for r in epoch_steps if r[name] is not None]
            if present:
                assert line[f"train_{name}"] == pytest.approx(np.mean(present), rel=1e-5)
            else:
                assert line[f"train_{name}"] is None
        assert line["grad_norm_mean"] == pytest.approx(np.mean([r["grad_norm"] for r in epoch_steps]), rel=1e-5)


def test_epoch_stats_skip_absent_losses() -> None:
    stats = EpochStats(13, torch.device("cpu"))
    stats.add_update(1.0, {"recon": 2.0, "nce": 4.0}, 0.5)
    stats.add_update(3.0, {"recon": 4.0}, 1.5)
    stats.add_skip()
    means = stats.means()
    assert means["train_joint"] == 2.0 and means["train_recon"] == 3.0 and means["train_nce"] == 4.0
    assert means["train_p1"] is None
    assert (stats.steps, stats.skipped, stats.grad_max, stats.grad_sum) == (2, 1, 1.5, 2.0)
    again = EpochStats(13, torch.device("cpu"))
    again.load(json.loads(json.dumps(stats.state())))
    assert again.means() == means and again.skipped == 1


def test_row_pooled_error_pools_hidden_rows_over_batches() -> None:
    gen = torch.Generator().manual_seed(0)
    stats = EpochStats(13, torch.device("cpu"))
    square, rows = torch.zeros(13, dtype=torch.float64), 0
    for b in (5, 3):
        pred, target = torch.randn(b, 32, 13, generator=gen), torch.randn(b, 32, 13, generator=gen)
        hidden = torch.rand(b, 32, generator=gen) < 0.3
        stats.add_reconstruction(pred, target, hidden)
        square += ((pred - target) ** 2)[hidden].sum(0).double()
        rows += int(hidden.sum())
    pooled = np.array(stats.row_pooled())
    assert pooled.shape == (13,) and pooled == pytest.approx((square / rows).numpy(), rel=1e-6)
    assert EpochStats(13, torch.device("cpu")).row_pooled() is None


def test_feature_history_shapes_and_val_relation(finished_run: tuple[T32Trainer, Path]) -> None:
    _, run_dir = finished_run
    history = read(run_dir / "history.jsonl")
    features = read(run_dir / "feature_history.jsonl")
    assert [r["epoch"] for r in features] == [r["epoch"] for r in history]
    assert features[0]["train_recon_row_pooled"] is None
    for f, h in zip(features, history):
        assert f["features"] == list(FEATURES)
        assert len(f["val_recon"]) == 13 and len(f["val_interp_baseline"]) == 13
        # Evaluator: per-batch per-feature MSE averaged over batches -> mean over features = val_recon exactly.
        assert np.mean(f["val_recon"]) == pytest.approx(h["val_recon"], rel=1e-5)
        assert np.mean(f["val_interp_baseline"]) == pytest.approx(h["val_recon_interp_baseline"], rel=1e-9)
        if f["epoch"] > 0:
            assert len(f["train_recon_row_pooled"]) == 13 and min(f["train_recon_row_pooled"]) >= 0


def test_steps_file_has_one_line_per_real_update(finished_run: tuple[T32Trainer, Path]) -> None:
    trainer, run_dir = finished_run
    steps = read(run_dir / "steps.jsonl")
    assert len(steps) == trainer.step - trainer.skipped == trainer.total_steps
    assert [r["step"] for r in steps] == list(range(1, trainer.step + 1))
    assert set(steps[0]) == {"step", "epoch", "lr", "grad_norm", "joint", "recon", "nce", "p1", "p2", "p3",
                             "step_seconds"}  # fmt: skip
    train = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    for rec in (r for r in train if r["kind"] == "train"):
        line = steps[rec["step"] - 1]
        assert line["lr"] == round_sig(rec["lr"]) and line["grad_norm"] == round_sig(rec["grad_norm"])
    assert round_sig(1.23456789) == 1.23457 and round_sig(float("nan")) is None


def test_summary_json(finished_run: tuple[T32Trainer, Path]) -> None:
    trainer, run_dir = finished_run
    summary = json.loads((run_dir / "summary.json").read_text())
    assert set(SUMMARY_KEYS) <= set(summary)
    assert summary["run_id"] == "hist" and summary["seed"] == 0 and summary["short"] is False
    assert summary["epochs_run"] == 2 and summary["steps"] == trainer.step and summary["stop_reason"] == "max_epochs"
    assert summary["params_total"] > summary["params_encoder"] > 0
    assert summary["train_windows"] == 200 and summary["val_windows"] == 60 and summary["device"] == "cpu"
    assert summary["best_epoch"] == trainer.best_epoch and summary["best_val_joint"] == pytest.approx(trainer.best)


def test_resume_after_budget_stop_does_not_duplicate(encoder_input: Path, tmp_path: Path) -> None:
    s = tiny_settings(encoder_input, tmp_path, max_epochs=2, max_hours=1e-9)  # stops after the first update
    run_dir = Path(s.runs_dir) / "budget"
    first = T32Trainer(s, run_dir, resume=False)
    first.run()
    assert first.step == 1 and json.loads((run_dir / "summary.json").read_text())["stop_reason"] == "time_budget"
    partial = read(run_dir / "history.jsonl")
    assert [r["epoch"] for r in partial] == [0, 1] and partial[1]["partial"] is True
    settings = T32BenignConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    settings.max_hours = 24.0
    again = T32Trainer(settings, run_dir, resume=True)
    assert again.epoch_stats.steps == 1  # the interrupted epoch keeps its first update
    again.run()
    lines = read(run_dir / "history.jsonl")
    assert [r["epoch"] for r in lines] == [0, 1, 2] and not any(r["partial"] for r in lines)
    assert [r["epoch"] for r in read(run_dir / "feature_history.jsonl")] == [0, 1, 2]
    assert lines[1]["train_steps"] + lines[1]["train_skipped"] == again.epoch_batches[0]
    steps = [r["step"] for r in read(run_dir / "steps.jsonl")]
    assert steps == list(range(1, again.step + 1)) and len(steps) == again.total_steps
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["stop_reason"] == "max_epochs" and summary["epochs_run"] == 2


@pytest.mark.parametrize("how", ["file", "signal"])
def test_pause_then_resume_finishes_the_same_updates(
    encoder_input: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    s = tiny_settings(encoder_input, tmp_path, max_epochs=2)
    run_dir = Path(s.runs_dir) / f"pause-{how}"
    first = T32Trainer(s, run_dir, resume=False)
    original = T32Trainer._train_step

    def step_then_pause(self: T32Trainer, *args: object) -> None:
        original(self, *args)
        if self.step == 1:
            if how == "file":
                (self.run_dir / "PAUSE").touch()
            else:
                os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(T32Trainer, "_train_step", step_then_pause)
    first.run()
    monkeypatch.undo()
    assert first.step == 1 and json.loads((run_dir / "summary.json").read_text())["stop_reason"] == "paused"
    assert not (run_dir / "PAUSE").exists() and (run_dir / "last.pt").is_file()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # handlers restored
    settings = T32BenignConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    again = T32Trainer(settings, run_dir, resume=True)
    again.run()
    steps = [r["step"] for r in read(run_dir / "steps.jsonl")]
    assert steps == list(range(1, again.total_steps + 1))
    assert json.loads((run_dir / "summary.json").read_text())["stop_reason"] == "max_epochs"


def test_short_mode_settings(encoder_input: Path, tmp_path: Path) -> None:
    base = ["--runs-dir", str(tmp_path), "--data-dir", str(encoder_input)]
    s, run_dir, resume = TrainCommand(parse_args(["--short", "--seed", "1", *base])).settings()
    assert run_dir.name == "model-t32b-short-seed1" and not resume
    assert s.short and s.shards_per_split == 2 and s.val_shards_per_split == 1 and s.max_epochs == 3
    assert s.eval_windows == 4096 and s.max_steps is None and s.warmup_steps <= 25 and s.norm_warmup_steps <= 25
    s2, _, _ = TrainCommand(parse_args(["--short", "--short-shards", "1", "--max-epochs", "2", *base])).settings()
    assert s2.shards_per_split == 1 and s2.max_epochs == 2
    full, full_dir, _ = TrainCommand(parse_args(["--seed", "1", *base])).settings()
    assert not full.short and full.val_shards_per_split is None and full_dir.name == "model-t32b-seed1"
    with pytest.raises(SystemExit):
        TrainCommand(parse_args(["--short", "--smoke", *base])).settings()


def test_short_run_loads_few_shards_and_records_short(encoder_input: Path, tmp_path: Path) -> None:
    s = tiny_settings(encoder_input, tmp_path, max_epochs=1, shards_per_split=2, val_shards_per_split=1, short=True)
    run_dir = Path(s.runs_dir) / "short"
    trainer = T32Trainer(s, run_dir, resume=False)
    assert len(trainer.train) == 80 and len(trainer.val) == 40  # synthetic shards hold 40 windows
    trainer.run()
    assert json.loads((run_dir / "config.json").read_text())["short"] is True
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["short"] is True and summary["train_windows"] == 80 and summary["val_windows"] == 40
    for name in ("history.jsonl", "feature_history.jsonl", "steps.jsonl", "best.pt", "last.pt"):
        assert (run_dir / name).is_file(), name
    assert len(read(run_dir / "steps.jsonl")) == trainer.step - trainer.skipped  # 0 here: the first 80 have 1 row


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_epoch_stats_on_mps_have_no_float64() -> None:
    stats = EpochStats(13, torch.device("mps"))
    pred = torch.ones(2, 32, 13, device="mps")
    stats.add_reconstruction(pred, torch.zeros_like(pred), torch.ones(2, 32, dtype=torch.bool, device="mps"))
    assert stats.row_pooled() == pytest.approx([1.0] * 13)
    stats.load(stats.state())
    assert stats.square_sum.device.type == "mps"
