"""Tests of the T32_benign trainer: settings, the X1 learning-rate horizon and a tiny end-to-end run (CPU)."""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import pytest
import torch

from src.model.all_t32_benign.timesnet.config import T32BenignConfig
from src.model.all_t32_benign.timesnet.data import restore_space
from src.model.all_t32_benign.timesnet.train import BatchPlan, T32Trainer, TrainCommand, guard_budget_gb, parse_args
from src.model.all_t32_benign.timesnet.base_config import PretrainConfig
from src.pipeline.all_t32_benign.cache import SplitCacheBuilder


def tiny_settings(encoder_input: Path, tmp_path: Path, **changes: object) -> T32BenignConfig:
    values = dict(
        data_dir=str(encoder_input),
        cache_dir=str(tmp_path / "cache"),
        runs_dir=str(tmp_path / "runs"),
        d_model=16,
        d_ff=8,
        n_blocks=2,
        proj_dim=16,
        batch_size=16,
        eval_windows=64,
        eval_batch_size=64,
        max_epochs=2,
        warmup_steps=3,
        norm_warmup_steps=3,
        log_every=1,
        device="cpu",
        workers=1,
        prefetch_batches=0,
        mem_gb=8.0,
        resource_guard=False,
    )
    values.update(changes)
    settings = T32BenignConfig(**values)
    settings.validate()
    return settings


def test_config_defaults_and_validation() -> None:
    s = T32BenignConfig()
    s.validate()
    base = PretrainConfig()
    assert (s.d_model, s.d_ff, s.seq_len, s.dataset) == (128, 64, 32, "t32b")
    assert (s.lambda_recon, s.lambda_nce, s.block_ratio, s.min_rows_aux) == (
        base.lambda_recon,
        base.lambda_nce,
        base.block_ratio,
        base.min_rows_aux,
    )
    assert s.length_bucketed and s.min_train_rows == 1 and s.check_split == "val"
    assert T32BenignConfig.from_dict(json.loads(json.dumps({**dataclasses.asdict(s), "config_class": "x"}))) == s
    for bad in ({"min_train_rows": 0}, {"seq_len": 64}, {"batch_size": 1}, {"check_split": "test"}):
        with pytest.raises(ValueError):
            T32BenignConfig(**bad).validate()
    assert PretrainConfig().dataset == "grid"  # the parent settings are untouched


def test_batch_plan_drops_batches_without_any_loss() -> None:
    s = T32BenignConfig(batch_size=4)
    lengths = torch.tensor([1] * 12 + [2, 5, 9, 32] * 2)
    plan = BatchPlan(s, lengths)
    kept, dropped = plan.epoch(0)
    assert dropped == 3 and len(kept) == 2  # three batches of only 1-row windows
    assert all(plan.applicable(lengths[b]) for b in kept)
    assert not plan.applicable(torch.tensor([1, 1, 1]))
    assert not plan.applicable(torch.tensor([9]))  # a lone window is never a batch
    nce_only = BatchPlan(T32BenignConfig(use_recon=False, use_physics=False), lengths)
    assert not nce_only.applicable(torch.tensor([4, 3, 3]))  # one window survives the shortest crop
    assert nce_only.applicable(torch.tensor([4, 5, 1]))
    assert plan.counts(3)[0] == [2, 2, 2]


def test_x1_lr_reaches_floor_at_last_real_update(encoder_input: Path, tmp_path: Path) -> None:
    s = tiny_settings(encoder_input, tmp_path)
    trainer = T32Trainer(s, Path(s.runs_dir) / "x1", resume=False)
    assert sum(trainer.epoch_dropped) > 0  # the synthetic train set has batches of 1-row windows only
    naive_horizon = s.max_epochs * math.ceil(len(trainer.train) / s.batch_size)
    assert trainer.total_steps == sum(trainer.epoch_batches) < naive_horizon
    factors = [trainer._lr_factor(k) for k in range(trainer.total_steps + 1)]
    assert factors[: s.warmup_steps] == pytest.approx([(k + 1) / s.warmup_steps for k in range(s.warmup_steps)])
    assert factors[-1] == pytest.approx(0.0, abs=1e-12)
    trainer.run()
    records = [json.loads(line) for line in (Path(s.runs_dir) / "x1" / "metrics.jsonl").read_text().splitlines()]
    steps = [r for r in records if r["kind"] == "train"]
    summary = records[-1]
    assert summary["real_updates"] == trainer.total_steps == len(steps)
    assert summary["skipped_steps"] == 0
    assert steps[-1]["lr"] == pytest.approx(0.0, abs=1e-12)  # floor after the last real update
    lrs = [r["lr"] for r in steps]
    peak = lrs.index(max(lrs))
    assert lrs[0] == pytest.approx(s.lr * 2 / s.warmup_steps)  # warm-up: LR after the first update
    assert peak == s.warmup_steps - 2  # LR logged after update k is factor(k): warm-up ends at the full LR
    assert lrs[s.warmup_steps - 1] == pytest.approx(s.lr)  # first cosine value = full LR
    assert all(a >= b for a, b in zip(lrs[peak:], lrs[peak + 1 :]))  # cosine decays monotonically


def test_unexpected_skip_still_advances_the_schedule(encoder_input: Path, tmp_path: Path) -> None:
    s = tiny_settings(encoder_input, tmp_path)
    trainer = T32Trainer(s, Path(s.runs_dir) / "skip", resume=False)
    ones = torch.nonzero(trainer.train_lengths == 1).squeeze(1)[:4]
    before = trainer.sched.last_epoch
    trainer._train_step(ones, trainer.train.x[ones], trainer.train.mask[ones])
    assert trainer.skipped == 1 and trainer.sched.last_epoch == before + 1


def test_end_to_end_run_writes_provenance_and_space(encoder_input: Path, tmp_path: Path) -> None:
    SplitCacheBuilder(encoder_input, tmp_path / "cache", 32, workers=1).build("train")
    SplitCacheBuilder(encoder_input, tmp_path / "cache", 32, workers=1).build("val")
    s = tiny_settings(encoder_input, tmp_path, max_epochs=1, min_train_rows=2)
    run_dir = Path(s.runs_dir) / "e2e"
    trainer = T32Trainer(s, run_dir, resume=False)
    assert trainer.train_rows_filter["windows_removed"] == 80 and len(trainer.train) == 120
    assert trainer.train.reader.startswith("npy cache") and trainer.val.split == "val"
    trainer.run()
    for name in ("config.json", "env.json", "metrics.jsonl", "best.pt", "last.pt"):
        assert (run_dir / name).is_file(), name
    config = json.loads((run_dir / "config.json").read_text())
    assert config["config_class"] == "T32BenignConfig" and config["seed"] == 0
    env = json.loads((run_dir / "env.json").read_text())
    assert env["label_files_opened"] == 0 and env["splits_opened"] == ["train", "val"]
    assert "git_commit" in env and "git_dirty" in env and "cuda_available" in env and env["packages"]
    assert env["batch_plan"]["planned_updates"] == trainer.total_steps
    evals = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert all("max_rss_gb" in r for r in evals if r["kind"] == "eval")
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    assert state["config_class"] == "T32BenignConfig" and {"model", "encoder", "opt", "sched"} <= set(state)
    metadata = json.loads((encoder_input / "metadata.json").read_text())
    space = restore_space(metadata, state["space"])
    assert torch.equal(space.mean, trainer.space.mean.cpu()) and torch.equal(space.std, trainer.space.std.cpu())
    assert int(space.robust.sum()) == 3


def test_resume_continues_the_same_plan(encoder_input: Path, tmp_path: Path) -> None:
    s = tiny_settings(encoder_input, tmp_path, max_epochs=1)
    run_dir = Path(s.runs_dir) / "resume"
    T32Trainer(s, run_dir, resume=False).run()
    again = T32Trainer(T32BenignConfig.from_dict(json.loads((run_dir / "config.json").read_text())), run_dir, True)
    assert again.step == again.total_steps and again.epoch == 1


def test_command_line_defaults(encoder_input: Path, tmp_path: Path) -> None:
    cmd = TrainCommand(parse_args(["--seed", "2", "--runs-dir", str(tmp_path), "--data-dir", str(encoder_input)]))
    settings, run_dir, resume = cmd.settings()
    assert run_dir.name == "model-t32b-seed2" and not resume and settings.seed == 2
    smoke = TrainCommand(parse_args(["--smoke", "--runs-dir", str(tmp_path)]))
    s, smoke_dir, _ = smoke.settings()
    assert s.max_steps == 30 and s.shards_per_split == 1 and smoke_dir.name.startswith("model-t32b-smoke-")
    s3, _, _ = TrainCommand(parse_args(["--min-train-rows", "2"])).settings()
    assert s3.min_train_rows == 2


def test_resume_of_a_finished_run_does_nothing(encoder_input: Path, tmp_path: Path) -> None:
    s = tiny_settings(encoder_input, tmp_path, max_epochs=1)
    run_dir = Path(s.runs_dir) / "done"
    T32Trainer(s, run_dir, resume=False).run()
    before = (run_dir / "metrics.jsonl").read_text()
    again = T32Trainer(T32BenignConfig.from_dict(json.loads((run_dir / "config.json").read_text())), run_dir, True)
    assert again.finished_reason() is not None
    again.run()
    assert (run_dir / "metrics.jsonl").read_text() == before


def test_resume_after_early_stopping_does_nothing(encoder_input: Path, tmp_path: Path) -> None:
    s = tiny_settings(encoder_input, tmp_path, max_epochs=3)
    run_dir = Path(s.runs_dir) / "early"
    trainer = T32Trainer(s, run_dir, resume=False)
    trainer.step, trainer.epoch, trainer.bad_epochs = 5, 1, s.patience
    trainer._save("last.pt")
    again = T32Trainer(s, run_dir, resume=True)
    assert again.finished_reason() == "early_stopping"
    again.run()
    assert not (run_dir / "metrics.jsonl").exists()


def test_checkpoints_always_hold_the_space(encoder_input: Path, tmp_path: Path) -> None:
    s = tiny_settings(encoder_input, tmp_path, max_epochs=2)
    run_dir = Path(s.runs_dir) / "ckpt"
    T32Trainer(s, run_dir, resume=False).run()
    for name in ("best.pt", "last.pt"):
        state = torch.load(run_dir / name, map_location="cpu", weights_only=False)
        assert {"space", "config_class", "skipped", "total_steps", "model", "opt", "torch_rng"} <= set(state)
    assert not list(run_dir.glob("*.tmp"))
    env = json.loads((run_dir / "env.json").read_text())
    assert env["label_files_opened"] == 0 and env["label_paths_refused"] == 0 and env["files_opened"] > 0


def test_guard_budget_caps_only_on_cuda() -> None:
    assert guard_budget_gb(22.0, "cuda", 16.0) == pytest.approx(13.6)
    assert guard_budget_gb(10.0, "cuda", 16.0) == 10.0
    assert guard_budget_gb(22.0, "mps", 16.0) == 22.0
    assert guard_budget_gb(22.0, "cuda", None) == 22.0


def test_command_writes_train_log(encoder_input: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    args = parse_args(
        ["--runs-dir", str(tmp_path / "runs"), "--data-dir", str(encoder_input), "--cache-dir", str(tmp_path / "c")]
        + ["--run-id", "logged", "--max-steps", "3", "--batch-size", "16", "--device", "cpu", "--workers", "1"]
    )
    cmd = TrainCommand(args)
    settings, run_dir, _ = cmd.settings()
    tiny = tiny_settings(encoder_input, tmp_path, max_steps=3, runs_dir=str(tmp_path / "runs"))
    cmd.settings = lambda: (tiny, run_dir, False)  # tiny model, same command path
    stdout = sys.stdout
    cmd.run()
    assert sys.stdout is stdout
    log = (run_dir / "train.log").read_text()
    assert "preflight ok" in log and "step 1/3" in log and "done:" in log
