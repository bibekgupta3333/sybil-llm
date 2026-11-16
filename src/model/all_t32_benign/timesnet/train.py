"""TimesNet pretraining on all/T32_benign (WBS S1.3.J3 tooling for EC2): the run command and its trainer.

Builds on this package's own copies of the original TimesNet pretraining code (`encoder`, `heads`, `views`, `losses`,
`monitors`, `base_train.Trainer`, `preflight`, `BatchPrefetcher`; copied from `src/model/benign_gridsybil/timesnet/`
on 2026-11-15, nothing imported from there) and changes only what this run needs:

- data: the numpy feature cache of `train` and the on-disk benign `val` split (`load_split`, `PretrainFirewall`);
  label-free checkpoint selection on `val` with the existing `Evaluator` (X5); optional `min_train_rows` filter (X3);
- X1: the learning-rate horizon equals the number of real optimiser updates. Batches that cannot produce any loss are
  removed when an epoch is planned (`BatchPlan`), and the scheduler advances once per planned batch even if a batch
  unexpectedly yields no loss, so the cosine reaches its floor exactly at the last planned update;
- X9: env.json records the CUDA device name, total memory, CUDA / cuDNN versions, git commit + dirty flag and the
  installed packages; every eval record in metrics.jsonl carries the peak accelerator memory since the previous eval;
- checkpoints best.pt / last.pt hold everything the existing trainer stores plus `space` (`space_state`) and
  `config_class`; config.json and env.json are written before the first step (RULE 4);
- CUDA: batches are gathered into pinned memory by the prefetch thread and copied with non_blocking transfers.
  Mixed precision is not offered: the per-window FFT (`torch.fft.rfft` over 1..32 real rows) is not known to be safe
  in float16 for non-power-of-two lengths, so the run stays in float32.

Outputs go to `<runs_dir>/<run_id>/` (default `src/runs/pretraining/all/T32_benign/model-t32b-seed<seed>/`):
config.json, env.json, metrics.jsonl, steps.jsonl, history.jsonl, feature_history.jsonl, summary.json (`history.py`),
best.pt, last.pt and train.log (console copy; not for --check). Resuming a
finished run (early stopping, all planned updates or all epochs done) returns without training. On CUDA the memory
guard watches host RAM only and its budget is capped at 85% of system RAM.

Usage (repository root, training environment):

    .venv-train/bin/python -m src.model.all_t32_benign.timesnet.train --check
    .venv-train/bin/python -m src.model.all_t32_benign.timesnet.train --smoke
    .venv-train/bin/python -m src.model.all_t32_benign.timesnet.train --short --seed 0
    .venv-train/bin/python -m src.model.all_t32_benign.timesnet.train --seed 0
    .venv-train/bin/python -m src.model.all_t32_benign.timesnet.train --resume src/runs/pretraining/all/T32_benign/<run_id>
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import importlib.metadata
import json
import logging
import os
import platform
import resource
import signal
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

import numpy as np
import torch
from torch import nn

from src.model.all_t32_benign.timesnet.base_data import FeatureShards, FeatureSpace
from src.model.all_t32_benign.timesnet.heads import PretrainingModel
from src.model.all_t32_benign.timesnet.losses import LossScales, Objective
from src.model.all_t32_benign.timesnet.monitors import Evaluator
from src.model.all_t32_benign.timesnet.base_train import (
    BatchPrefetcher,
    ResourceGuard,
    Trainer,
    git,
    pick_device,
    preflight,
    same_link_pairs,
)
from src.pipeline.all_t32_benign.cache import SplitFiles, file_sha256

from .config import DATA_SELECTION_NOTE, T32BenignConfig
from .data import FeatureFirewall, PretrainFirewall, load_split, space_state
from .history import EpochStats, TrainingHistory

try:
    import psutil
except ImportError:  # pragma: no cover - psutil ships with the pinned training env
    psutil = None

_LOG = logging.getLogger(__name__)
CONFIG_CLASS = "T32BenignConfig"
CUDA_RAM_SHARE = 0.85  # on CUDA the memory guard watches host RAM: cap its budget at this share of system RAM
SHORT_EPOCHS = 3  # --short: whole epochs on a few shards (plumbing for the notebook)
SHORT_EVAL_WINDOWS = 4096
SHORT_WARMUP_STEPS = 25  # a few hundred updates in all: the LR warm-up and loss-scale window must fit inside


def guard_budget_gb(mem_gb: float, device_type: str, system_ram_gb: float | None) -> float:
    """Memory-guard budget: `mem_gb`, capped on CUDA at 85% of system RAM (the guard then measures host RSS only)."""
    if device_type != "cuda" or system_ram_gb is None:
        return mem_gb
    return min(mem_gb, round(CUDA_RAM_SHARE * system_ram_gb, 2))


class _Tee:
    """A text stream that writes to two streams (console and the run log)."""

    def __init__(self, first: TextIO, second: TextIO) -> None:
        self.first, self.second = first, second

    def write(self, text: str) -> int:
        self.first.write(text)
        self.second.write(text)
        return len(text)

    def flush(self) -> None:
        self.first.flush()
        self.second.flush()

    def isatty(self) -> bool:
        return False


class RunLog:
    """Context manager: copies logging records and everything printed to stdout / stderr into `<run_dir>/train.log`.

    Appends, so a resumed run continues the same file; the original streams and handlers are restored on exit.
    """

    FORMAT = "%(asctime)s %(levelname)s %(message)s"

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> "RunLog":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a", buffering=1)
        self.handler = logging.StreamHandler(self.file)
        self.handler.setFormatter(logging.Formatter(self.FORMAT))
        self.handler.setLevel(logging.INFO)
        root = logging.getLogger()
        self.root_level = root.level
        root.addHandler(self.handler)
        root.setLevel(min(root.level or logging.INFO, logging.INFO))
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(self.stdout, self.file), _Tee(self.stderr, self.file)
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        if error is not None:
            _LOG.error("run stopped by %s: %s", kind.__name__ if kind else "error", error)
        sys.stdout, sys.stderr = self.stdout, self.stderr
        root = logging.getLogger()
        root.removeHandler(self.handler)
        root.setLevel(self.root_level)
        self.file.close()


class BatchPlan:
    """The batches of each epoch that can produce a loss (X1): the schedule horizon is their count.

    Batches are drawn exactly as the existing trainer draws them (seeded per epoch, length-bucketed if configured);
    a batch is kept if it has >= 2 windows and at least one objective applies to it:
    reconstruction needs a window with >= 2 real rows (one row can never be hidden), the physics heads need a window
    with >= `min_rows_aux` rows, NT-Xent needs two windows that keep >= `min_rows_aux` rows after the shortest crop.

    Args:
        settings: run settings.
        lengths: (N,) long, real rows per train window.
    """

    def __init__(self, settings: T32BenignConfig, lengths: torch.Tensor) -> None:
        self.s, self.lengths = settings, lengths

    def applicable(self, n: torch.Tensor) -> bool:
        """True if a batch whose windows have `n` real rows yields at least one loss."""
        s = self.s
        if len(n) < 2:
            return False
        if s.use_recon and bool((n >= 2).any()):
            return True
        if s.use_physics and bool((n >= s.min_rows_aux).any()):
            return True
        shortest_crop = torch.ceil(n.double() * s.crop[0])
        return s.use_nce and int((shortest_crop >= s.min_rows_aux).sum()) >= 2

    def epoch(self, epoch: int) -> tuple[list[torch.Tensor], int]:
        """(kept batches of `epoch`, number of batches dropped because no loss applies)."""
        s = self.s
        gen = torch.Generator().manual_seed(s.seed + epoch)
        perm = torch.randperm(len(self.lengths), generator=gen)
        if s.length_bucketed:
            order = perm[torch.argsort(self.lengths[perm], stable=True)]  # random order within each length
            batches = list(order.split(s.batch_size))
            batches = [batches[i] for i in torch.randperm(len(batches), generator=gen).tolist()]
        else:
            batches = list(perm.split(s.batch_size))
        kept = [b for b in batches if self.applicable(self.lengths[b])]
        return kept, len(batches) - len(kept)

    def counts(self, n_epochs: int) -> tuple[list[int], list[int]]:
        """(kept batches per epoch, dropped batches per epoch) for epochs 0..n_epochs-1."""
        kept, dropped = [], []
        for e in range(n_epochs):
            batches, n_drop = self.epoch(e)
            kept.append(len(batches))
            dropped.append(n_drop)
        return kept, dropped


class PinnedBatchPrefetcher(BatchPrefetcher):
    """The existing prefetcher, gathering each batch into pinned host memory (fast non_blocking copies to CUDA)."""

    def __init__(self, data: FeatureShards, batches: list[torch.Tensor], first: int, depth: int, pin: bool) -> None:
        self.pin = pin  # set before the parent starts its background thread
        super().__init__(data, batches, first, depth)

    def _gather(self, bi: int) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        bi, idx, x, mask = super()._gather(bi)
        if self.pin:
            x, mask = x.pin_memory(), mask.pin_memory()
        return bi, idx, x, mask


class AcceleratorMemory:
    """Peak / current memory of the training device and the process, for metrics.jsonl (X9)."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def snapshot(self) -> dict[str, Any]:
        """Memory since the last `reset` (CUDA peak) or now (MPS driver), plus process RSS; GB, 3 decimals."""
        out: dict[str, Any] = {}
        if self.device.type == "cuda":
            out["gpu_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated(self.device) / 1e9, 3)
            out["gpu_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved(self.device) / 1e9, 3)
        elif self.device.type == "mps":
            out["mps_driver_allocated_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 3)
        if psutil is not None:
            out["rss_gb"] = round(psutil.Process().memory_info().rss / 1e9, 3)
        scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
        out["max_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 1e9, 3)
        return out

    def reset(self) -> None:
        """Starts a new CUDA peak window (no-op elsewhere)."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)


class RunProvenance:
    """Builds env.json: code, environment, device, data and cache provenance (RULE 4, X9)."""

    def __init__(self, settings: T32BenignConfig, device: torch.device, guard: ResourceGuard) -> None:
        self.s, self.device, self.guard = settings, device, guard

    def build(self, train: FeatureShards, val: FeatureShards) -> dict[str, Any]:
        """The env.json record (without preflight, added by the trainer)."""
        s = self.s
        metadata = Path(s.data_dir) / "metadata.json"
        return {
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "config_class": CONFIG_CLASS,
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": str(self.device),
            **self._accelerator(),
            "mps_memory_fraction": self.guard.mps_fraction,
            "requirements_sha256": {
                p: file_sha256(Path(p))
                for p in ("requirements-train.txt", "requirements/linux.txt")
                if Path(p).is_file()
            },
            "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
            "dataset": s.dataset,
            "data_selection": DATA_SELECTION_NOTE,
            "encoder_input_metadata": str(metadata),
            "encoder_input_metadata_sha256": file_sha256(metadata) if metadata.is_file() else None,
            "data_source": {"train": train.reader, val.split: val.reader},
            "cache_manifest_sha256": self._manifests(),
            "normalisation": "metadata.json['normalisation'] (benign train real rows), robust features re-fit (D13)",
            "rows": {"train": len(train), val.split: len(val)},
            "check_set": f"on-disk '{val.split}' split (benign, vehicle-disjoint from train); test never opened",
            "splits_opened": sorted({"train", val.split}),
            "determinism": "seeded; GPU kernels are not bit-deterministic (same seed = same result within tolerance)",
        }

    def _accelerator(self) -> dict[str, Any]:
        if self.device.type != "cuda":
            return {"cuda_available": torch.cuda.is_available()}
        props = torch.cuda.get_device_properties(self.device)
        return {
            "cuda_available": True,
            "cuda_device_name": torch.cuda.get_device_name(self.device),
            "cuda_total_memory_gb": round(props.total_memory / 1e9, 3),
            "cuda_capability": f"{props.major}.{props.minor}",
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_device_count": torch.cuda.device_count(),
        }

    def _manifests(self) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        for split in ("train", self.s.check_split):
            files = SplitFiles(Path(self.s.cache_dir) / split)
            out[split] = file_sha256(files.manifest) if files.manifest.is_file() else None
        return out


class PauseControl:
    """Graceful pause: a `PAUSE` file in the run folder, Ctrl+C or SIGTERM stop the run after the current update.

    The trainer then takes its budget-stop path: one evaluation, the epoch written as partial, `last.pt` with the
    partial epoch state, so `--resume` continues at the next batch. A second Ctrl+C interrupts immediately.

    Args:
        run_dir: run folder that holds the `PAUSE` file.
    """

    FILE = "PAUSE"

    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / self.FILE
        self.signalled: str | None = None
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> "PauseControl":
        if self.path.exists():  # a pause file left by the previous session must not stop the resumed run
            self.path.unlink()
            _LOG.info("removed the old %s file; training continues", self.FILE)
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._previous[sig] = signal.signal(sig, self._handle)
        _LOG.info("pause: `touch %s`, Ctrl+C or SIGTERM -> checkpoint after the current update", self.path)
        return self

    def __exit__(self, *exc: object) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        self._previous.clear()

    def _handle(self, signum: int, frame: Any) -> None:
        if self.signalled:  # second signal: interrupt now
            raise KeyboardInterrupt
        self.signalled = signal.Signals(signum).name
        print(f"\n{self.signalled}: pausing after the current update (again to interrupt now)", flush=True)

    def requested(self) -> bool:
        """True once a pause was asked for (file or signal)."""
        return self.signalled is not None or self.path.exists()


class T32Trainer(Trainer):
    """The base `Trainer` (`base_train.py`) with cached T32_benign data, the on-disk val check set and the X1 / X9 fixes.

    Args:
        settings: validated run settings.
        run_dir: run folder (created for a new run, read for a resume).
        resume: continue from `run_dir/last.pt`.
    """

    def __init__(self, settings: T32BenignConfig, run_dir: Path, resume: bool) -> None:  # replaces the parent's
        self.s, self.run_dir = settings, run_dir
        torch.manual_seed(settings.seed)
        np.random.seed(settings.seed)
        self.device = pick_device(settings.device)
        if settings.cpu_threads > 0:
            torch.set_num_threads(settings.cpu_threads)
        ram_gb = psutil.virtual_memory().total / 1e9 if psutil is not None else None
        self.guard_mem_gb = guard_budget_gb(settings.mem_gb, self.device.type, ram_gb)
        if self.guard_mem_gb != settings.mem_gb:
            _LOG.info(
                "CUDA: memory guard budget %.2f GB (85%% of %.1f GB system RAM; mem_gb %.1f exceeds it); "
                "it watches host RSS only, GPU memory is logged per eval",
                self.guard_mem_gb,
                ram_gb,
                settings.mem_gb,
            )
        self.guard = ResourceGuard(self.guard_mem_gb, self.device, settings.resource_guard)
        self.memory = AcceleratorMemory(self.device)
        metadata = json.loads((Path(settings.data_dir) / "metadata.json").read_text())
        t0 = time.time()
        self._load_data()
        self.load_seconds = round(time.time() - t0, 1)
        self.train_links = self.train.link_groups()
        self.train_groups = self.train.broadcast_groups()
        self.train_lengths = self.train.mask.sum(1)
        self.space = self._fit_space(metadata)
        _LOG.info(
            "loaded train %d / %s %d windows in %.1f s; device %s",
            len(self.train),
            self.val.split,
            len(self.val),
            self.load_seconds,
            self.device,
        )
        self.model = PretrainingModel(settings).to(self.device)
        self.objective = Objective(settings, self.space)
        self.evaluator = Evaluator(settings, self.val, self.objective, self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
        self.plan = BatchPlan(settings, self.train_lengths)
        self.epoch_batches, self.epoch_dropped = self.plan.counts(settings.max_epochs)
        self.steps_per_epoch = self.epoch_batches[0] if self.epoch_batches else 0
        planned = sum(self.epoch_batches)
        self.total_steps = min(planned, settings.max_steps) if settings.max_steps else planned
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, self._lr_factor)
        self.scales = LossScales(settings.norm_warmup_steps)
        self.step, self.epoch, self.best, self.bad_epochs = 0, 0, float("inf"), 0
        self.skipped = 0  # planned batches that still produced no loss (expected 0; the LR advances anyway)
        self.best_epoch: int | None = None
        self.wall_before = 0.0  # wall seconds of earlier invocations of this run (resume)
        self.run_start = time.time()
        self.metrics = run_dir / "metrics.jsonl"
        self.history = TrainingHistory(run_dir)
        self.epoch_stats = EpochStats(settings.n_features, self.device)
        if resume:
            self._load(run_dir / "last.pt")
        else:
            self._start()

    def _load_data(self) -> None:
        s = self.s
        self.firewall: FeatureFirewall = PretrainFirewall()
        firewall = self.firewall
        self.train = load_split(s.cache_dir, s.data_dir, "train", firewall, s.shards_per_split, s.workers, s.seq_len)
        val_shards = s.val_shards_per_split if s.val_shards_per_split is not None else s.shards_per_split
        self.val = load_split(s.cache_dir, s.data_dir, s.check_split, firewall, val_shards, s.workers, s.seq_len)
        self.train_rows_filter = {"min_train_rows": s.min_train_rows, "windows_before": len(self.train)}
        if s.min_train_rows > 1:
            keep = torch.nonzero(self.train.mask.sum(1) >= s.min_train_rows).squeeze(1)
            removed = len(self.train) - len(keep)
            self.train = self.train.subset(keep, "train")
            _LOG.info(
                "min_train_rows %d: removed %d of %d train windows (%.2f%%)",
                s.min_train_rows,
                removed,
                self.train_rows_filter["windows_before"],
                100.0 * removed / max(1, self.train_rows_filter["windows_before"]),
            )
        removed = self.train_rows_filter["windows_before"] - len(self.train)
        self.train_rows_filter.update(
            {"windows_removed": removed, "share_removed": removed / max(1, self.train_rows_filter["windows_before"])}
        )

    def _fit_space(self, metadata: dict[str, Any]) -> FeatureSpace:
        """Stored files hold train z-scores; robust features are re-expressed (median / IQR from train only, D13)."""
        s = self.s
        space = FeatureSpace(metadata)
        if s.robust_features:
            stored = FeatureSpace(metadata)
            space.fit_robust(self.train, tuple(s.robust_features), s.soft_clip)
            FeatureSpace.rescale(self.train, stored, space)
            FeatureSpace.rescale(self.val, stored, space)
        return space.to(self.device)

    def _start(self) -> None:
        s = self.s
        self.run_dir.mkdir(parents=True, exist_ok=True)  # the command may have created it for train.log
        if (self.run_dir / "config.json").exists():
            raise FileExistsError(f"{self.run_dir} already holds a run; resume it instead")
        config = {**dataclasses.asdict(s), "config_class": CONFIG_CLASS}
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2))
        env = RunProvenance(s, self.device, self.guard).build(self.train, self.val)
        env["scaling"] = self.space.report
        env["load_seconds"] = self.load_seconds
        env["train_rows_filter"] = self.train_rows_filter
        env["cpu_threads"] = torch.get_num_threads()
        env["prefetch_batches"] = s.prefetch_batches
        env["pinned_memory"] = self.device.type == "cuda"
        env["label_files_opened"] = self.firewall.label_files_opened
        env["label_paths_refused"] = self.firewall.label_paths_refused
        env["files_opened"] = len(self.firewall.passed)
        env["resource_guard"] = {
            "enabled": s.resource_guard,
            "budget_gb": self.guard_mem_gb,
            "measures": "process RSS + MPS driver memory" if self.device.type == "mps" else "process RSS (host RAM)",
        }
        env["batch_plan"] = {
            "planned_updates": self.total_steps,
            "kept_batches_per_epoch": self.epoch_batches,
            "dropped_no_loss_batches_per_epoch": self.epoch_dropped,
            "lr_horizon": "planned updates (X1): cosine reaches its floor at the last planned update",
        }
        env["preflight"] = preflight(s, self.model, self.objective, self.train, self.device)
        (self.run_dir / "env.json").write_text(json.dumps(env, indent=2))
        _LOG.info("preflight ok: %s", json.dumps(env["preflight"]))

    def _log(self, record: dict[str, Any]) -> None:
        if record.get("kind") == "eval":
            record.update(self.memory.snapshot())
            self.memory.reset()
        super()._log(record)

    def _save(self, name: str) -> None:
        """Writes the checkpoint once, atomically: the parent trainer's keys plus space, config class and plan."""
        state = {
            "model": self.model.state_dict(),
            "encoder": self.model.encoder.state_dict(),
            "opt": self.opt.state_dict(),
            "sched": self.sched.state_dict(),
            "scales": self.scales.state(),
            "step": self.step,
            "epoch": self.epoch,
            "best": self.best,
            "bad_epochs": self.bad_epochs,
            "settings": dataclasses.asdict(self.s),
            "torch_rng": torch.get_rng_state(),
            "space": space_state(self.space),
            "config_class": CONFIG_CLASS,
            "skipped": self.skipped,
            "total_steps": self.total_steps,
            "best_epoch": self.best_epoch,
            "wall_seconds": self.wall_seconds(),
            "epoch_stats": self.epoch_stats.state(),
        }
        path = self.run_dir / name
        tmp = path.with_name(f"{name}.tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)

    def _load(self, path: Path) -> None:
        super()._load(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.skipped = int(state.get("skipped", 0))
        self.best_epoch = state.get("best_epoch")
        self.wall_before = float(state.get("wall_seconds", 0.0))
        if state.get("epoch_stats") is not None:  # the updates of an epoch a budget stop interrupted
            self.epoch_stats.load(state["epoch_stats"])
        if state.get("total_steps") not in (None, self.total_steps):
            raise RuntimeError(
                f"{path}: planned updates {state['total_steps']} != {self.total_steps} (settings changed?)"
            )

    def wall_seconds(self) -> float:
        """Wall time of the run so far, over every invocation (a resume adds to the stored value)."""
        return round(self.wall_before + time.time() - self.run_start, 1)

    def params(self) -> tuple[int, int]:
        """(encoder parameters, all parameters incl. heads and the mask token)."""
        return (
            sum(p.numel() for p in self.model.encoder.parameters()),
            sum(p.numel() for p in self.model.parameters()),
        )

    def _train_step(self, idx: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> None:
        """One optimiser step (the existing trainer's step, same metrics.jsonl records) plus the epoch / step logs.

        Reproduces `Trainer._train_step` so the per-update values (gradient norm, LR, every raw loss, the
        reconstruction prediction of the hidden rows) reach `EpochStats` and steps.jsonl without an extra forward
        pass. A planned batch that yields no loss still advances the schedule (X1).
        """
        s = self.s
        x, mask = x.to(self.device, non_blocking=True), mask.to(self.device, non_blocking=True)
        gen = torch.Generator().manual_seed(s.seed * 1_000_003 + self.step)
        t0 = time.time()
        losses, extra = self.objective(self.model, x, mask, gen, self.train_groups[idx])
        raw = {k: float(v.detach()) for k, v in losses.items()}
        if not losses:  # no loss despite the plan: advance the schedule anyway so it stays on the horizon
            self.skipped += 1
            self.step += 1
            self.sched.step()
            self.epoch_stats.add_skip()
            return
        self.scales.update(raw, self.step)
        loss = self.objective.joint(losses, self.scales)
        if not torch.isfinite(loss):
            self._save("last.pt")
            raise FloatingPointError(f"non-finite loss at step {self.step}: {raw}")
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), s.grad_clip))  # pre-clip global norm
        self.opt.step()
        self.sched.step()
        self.step += 1
        joint, lr, seconds = float(loss.detach()), self.sched.get_last_lr()[0], time.time() - t0
        self.epoch_stats.add_update(joint, raw, grad_norm)
        if "pred" in extra:
            self.epoch_stats.add_reconstruction(extra["pred"], extra["target"], extra["hidden"])
        self.history.step(self.step, self.epoch, lr, grad_norm, joint, raw, seconds)
        if self.step % s.log_every == 0 or self.step - self.skipped == 1:  # always log the first real update
            rec = {
                "kind": "train",
                "step": self.step,
                "epoch": self.epoch,
                "joint": joint,
                **raw,
                "grad_norm": grad_norm,
                "lr": lr,
                "step_seconds": round(seconds, 3),
                "mem_gb": round(self.guard.used_bytes() / 1e9, 2),
                "same_link_pairs": same_link_pairs(self.train_links[idx].tolist()),
                "nce_masked_pairs": extra.get("nce_masked_pairs", 0),
                "losses_present": sorted(losses),
                "skipped_steps": self.skipped,
            }
            self._log(rec)
            print(
                f"step {self.step}/{self.total_steps} ep {self.epoch} joint {rec['joint']:.4f} "
                + " ".join(f"{k} {v:.4f}" for k, v in raw.items())
                + f" | {rec['step_seconds']:.2f}s/step mem {rec['mem_gb']} GB",
                flush=True,
            )

    def _budget_stop(self, start: float) -> str | None:
        if self.s.max_steps and self.step >= self.s.max_steps:  # planned batches = the LR horizon
            return "max_steps"
        pause = getattr(self, "pause", None)
        if pause is not None and pause.requested():
            return "paused"
        return super()._budget_stop(start)

    def finished_reason(self) -> str | None:
        """Why a resumed run has nothing left to do (early stopping, all planned updates, all epochs), or None."""
        if self.step == 0:
            return None
        if self.bad_epochs >= self.s.patience:
            return "early_stopping"
        if self.step >= self.total_steps:
            return "all planned updates done"
        if self.epoch >= self.s.max_epochs:
            return "max_epochs"
        return None

    def _epoch_line(self, out: dict[str, Any], is_best: bool, partial: bool = False) -> None:
        """history.jsonl + feature_history.jsonl lines for the evaluation `out` (train sums of the epoch, then reset)."""
        init = self.step == 0 and self.epoch == 0
        self.history.epoch_end(
            epoch=self.epoch + 1 if partial else self.epoch,
            step=self.step,
            lr=self.sched.get_last_lr()[0],
            evaluation=out,
            stats=None if init else self.epoch_stats,
            scales=self.scales,
            is_best=is_best,
            bad_epochs=self.bad_epochs,
            partial=partial,
        )
        if not init and not partial:
            self.epoch_stats.reset()

    def run(self) -> None:
        """Trains until max_epochs, early stopping or a step / time / memory budget; writes the summaries.

        Writes a summary record to metrics.jsonl and summary.json; history.jsonl and feature_history.jsonl get one
        line per evaluation (epoch 0 = random init; a budget stop inside an epoch writes that epoch with
        `partial: true`, which a resume replaces).
        """
        with PauseControl(self.run_dir) as self.pause:
            self._run()
        if self.pause.path.exists():
            self.pause.path.unlink()

    def _run(self) -> None:
        s = self.s
        start = time.time()
        self.run_start = start
        finished = self.finished_reason()
        if finished:
            _LOG.info("nothing to do: the run already finished (%s); best val_joint %.4f", finished, self.best)
            return
        if self.step == 0:
            self.history.resume_from(-1, 0)  # a run that stopped before its first update starts its logs again
            out = self._evaluate("init")  # random-init reference
            self._epoch_line(out, is_best=False)
        else:
            dropped = self.history.resume_from(self.epoch, self.step)
            _LOG.info("resume: dropped log lines written after the checkpoint: %s", json.dumps(dropped))
        self.history.touch()
        stop_reason = "max_epochs"
        while self.epoch < s.max_epochs:
            batches, _ = self.plan.epoch(self.epoch)
            first = self.step - sum(self.epoch_batches[: self.epoch])
            feed = PinnedBatchPrefetcher(self.train, batches, first, s.prefetch_batches, self.device.type == "cuda")
            budget_hit = None
            t_epoch = time.time()
            for bi in range(first, len(batches)):
                bi_ready, idx, x_cpu, mask_cpu = next(feed)
                if bi_ready != bi:
                    raise RuntimeError(f"prefetcher out of order: {bi_ready} != {bi}")
                self._train_step(idx, x_cpu, mask_cpu)
                budget_hit = self._budget_stop(start)
                if budget_hit:
                    break
            feed.close()
            self.epoch_stats.add_seconds(time.time() - t_epoch)
            if budget_hit:
                stop_reason = budget_hit
                break
            self.epoch += 1
            out = self._evaluate(f"epoch{self.epoch}")
            improved = out["val_joint"] < self.best - 1e-4
            if improved:
                self.best, self.bad_epochs, self.best_epoch = out["val_joint"], 0, self.epoch
            else:
                self.bad_epochs += 1
            self._epoch_line(out, is_best=improved)  # resets the epoch sums before the checkpoints store them
            if improved:
                self._save("best.pt")
            self._save("last.pt")
            if self.bad_epochs >= s.patience:
                stop_reason = "early_stopping"
                break
        if stop_reason not in ("early_stopping", "max_epochs"):
            out = self._evaluate(f"stop_{stop_reason}")
            improved = out["val_joint"] < self.best
            if improved:
                self.best, self.best_epoch = out["val_joint"], self.epoch + 1
            self._epoch_line(out, is_best=improved, partial=True)  # the epoch sums stay for a resume
            if improved:
                self._save("best.pt")
        self._save("last.pt")
        summary = {
            "kind": "summary",
            "stop_reason": stop_reason,
            "step": self.step,
            "real_updates": self.step - self.skipped,
            "planned_updates": self.total_steps,
            "skipped_steps": self.skipped,
            "final_lr": self.sched.get_last_lr()[0],
            "epoch": self.epoch,
            "best_val_joint": self.best,
            "hours": round((time.time() - start) / 3600, 3),
            "steps_per_second": round(self.step / max(time.time() - start, 1e-9), 3),
        }
        self._log(summary)
        self._write_summary(stop_reason)
        _LOG.info("done: %s", json.dumps(summary))

    def _write_summary(self, stop_reason: str) -> None:
        """summary.json of the run (RULE 4 facts next to the curves)."""
        params_encoder, params_total = self.params()
        self.history.write_summary(
            {
                "run_id": self.run_dir.name,
                "seed": self.s.seed,
                "params_encoder": params_encoder,
                "params_total": params_total,
                "best_epoch": self.best_epoch,
                "best_val_joint": self.best if np.isfinite(self.best) else None,
                "epochs_run": self.epoch,
                "steps": self.step,
                "real_updates": self.step - self.skipped,
                "planned_updates": self.total_steps,
                "stop_reason": stop_reason,
                "wall_seconds": self.wall_seconds(),
                "device": str(self.device),
                "train_windows": len(self.train),
                "val_windows": len(self.val),
                "eval_windows": len(self.evaluator.x),
                "min_train_rows": self.s.min_train_rows,
                "short": self.s.short,
                "final_lr": self.sched.get_last_lr()[0],
            }
        )


class TrainCommand:
    """Command line: settings and run folder from the options, then check, smoke, train or resume."""

    OVERRIDES = (
        "max_epochs",
        "max_steps",
        "max_hours",
        "batch_size",
        "mem_gb",
        "resource_guard",
        "device",
        "seed",
        "min_train_rows",
        "workers",
    )

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def settings(self) -> tuple[T32BenignConfig, Path, bool]:
        """(settings, run folder, resume flag); a resume takes its settings from the run's config.json."""
        a = self.args
        if a.resume:
            run_dir = Path(a.resume).expanduser()
            settings = T32BenignConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
            for name in ("max_hours", "resource_guard", "device"):
                if getattr(a, name) is not None:
                    setattr(settings, name, getattr(a, name))
            return settings, run_dir, True
        settings = T32BenignConfig.preset(a.preset)
        for name in self.OVERRIDES:
            if getattr(a, name) is not None:
                setattr(settings, name, getattr(a, name))
        for name in ("runs_dir", "cache_dir", "data_dir"):
            if getattr(a, name) is not None:
                setattr(settings, name, str(Path(getattr(a, name)).expanduser()))
        if a.short and (a.smoke or a.check):
            raise SystemExit("--short cannot be combined with --smoke or --check")
        if a.short:
            if a.short_shards < 1:
                raise SystemExit(f"--short-shards must be >= 1, got {a.short_shards}")
            settings.short = True
            settings.shards_per_split, settings.val_shards_per_split = a.short_shards, 1
            settings.max_epochs = a.max_epochs if a.max_epochs is not None else SHORT_EPOCHS
            settings.eval_windows = min(settings.eval_windows, SHORT_EVAL_WINDOWS)
            settings.warmup_steps = min(settings.warmup_steps, SHORT_WARMUP_STEPS)
            settings.norm_warmup_steps = min(settings.norm_warmup_steps, SHORT_WARMUP_STEPS)
        if a.smoke or a.check:
            settings.shards_per_split, settings.eval_windows, settings.log_every = 1, 512, 5
            settings.max_steps = settings.max_steps or 30
            settings.norm_warmup_steps = min(settings.norm_warmup_steps, 10)
            settings.warmup_steps = min(settings.warmup_steps, 10)
        settings.validate()
        return settings, Path(settings.runs_dir) / (a.run_id or self.default_run_id(settings)), False

    def default_run_id(self, settings: T32BenignConfig) -> str:
        """`model-t32b-seed<seed>` (`-recon_only` for that preset), `model-t32b-short-seed<seed>` for --short;
        check / smoke runs get a timestamped name."""
        name = f"model-{settings.dataset}-{'short-' if settings.short else ''}seed{settings.seed}"
        if self.args.check or self.args.smoke:
            stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            return f"model-{settings.dataset}-{'check' if self.args.check else 'smoke'}-{stamp}"
        return name if self.args.preset == "joint" else f"{name}-{self.args.preset}"

    def run(self) -> None:
        """Builds the trainer and runs it (the check exits after loading and the pre-flight checks)."""
        settings, run_dir, resume = self.settings()
        if not resume and run_dir.exists():
            raise SystemExit(
                f"run folder {run_dir} already exists; nothing was overwritten. Resume it with --resume {run_dir} "
                "or choose another --run-id"
            )
        _LOG.info("run folder %s (dataset %s, T = %d)", run_dir, settings.dataset, settings.seq_len)
        if self.args.check and not resume:  # no train.log for the pre-flight check
            T32Trainer(settings, run_dir, resume=False)
            _LOG.info("checks passed; report in %s", run_dir / "env.json")
            return
        with RunLog(run_dir / "train.log"):
            _LOG.info("run folder %s (dataset %s, T = %d)", run_dir, settings.dataset, settings.seq_len)
            T32Trainer(settings, run_dir, resume=resume).run()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line options (unset options keep the preset's value)."""
    p = argparse.ArgumentParser(description="TimesNet pretraining on all/T32_benign (benign windows, T = 32)")
    p.add_argument("--preset", default="joint", choices=["joint", "recon_only"])
    p.add_argument("--run-id", default=None, help="run folder name (default model-t32b-seed<seed>)")
    p.add_argument(
        "--runs-dir", default=None, help="parent of run folders (default src/runs/pretraining/all/T32_benign)"
    )
    p.add_argument("--cache-dir", default=None, help="numpy feature cache (default src/data/cache/all/T32_benign)")
    p.add_argument(
        "--data-dir", default=None, help="encoder input folder (default src/data/encoder_input/all/T32_benign)"
    )
    p.add_argument("--resume", default=None, help="run folder to resume (its config.json and last.pt)")
    p.add_argument("--check", action="store_true", help="load one shard per split, run the pre-flight checks, exit")
    p.add_argument("--smoke", action="store_true", help="one shard per split, 30 updates, small eval")
    p.add_argument(
        "--short",
        action="store_true",
        help=f"quick complete run: first --short-shards train shards, 1 val shard, {SHORT_EPOCHS} whole epochs "
        f"(--max-epochs overrides), eval <= {SHORT_EVAL_WINDOWS} windows, every artefact; plumbing, never a result",
    )
    p.add_argument("--short-shards", type=int, default=2, help="train shards for --short (10,000 windows each)")
    p.add_argument("--seed", type=int)
    p.add_argument("--max-epochs", type=int)
    p.add_argument("--max-steps", type=int, help="cap on planned updates (the LR horizon shrinks to it)")
    p.add_argument("--max-hours", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--device", help="auto (MPS, CUDA, CPU), cuda, mps or cpu")
    p.add_argument("--mem-gb", type=float)
    p.add_argument(
        "--no-resource-guard",
        dest="resource_guard",
        action="store_false",
        default=None,
        help="no MPS memory cap and no memory-budget stop (memory is still logged)",
    )
    p.add_argument("--min-train-rows", type=int, help="drop train windows with fewer real rows (X3; default 1 = off)")
    p.add_argument("--workers", type=int, help="JSON fallback parsers when the cache is missing")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point: resume, check, smoke or a full run (from the repository root)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not (Path("CLAUDE.md").is_file() and Path("src/data").is_dir()):
        raise SystemExit("run from the repository root")
    TrainCommand(parse_args(argv)).run()


if __name__ == "__main__":
    main()
