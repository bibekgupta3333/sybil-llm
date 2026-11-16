"""Base pieces of the pretraining loop: resources, pre-flight checks, batch prefetching and the base `Trainer`.

`ResourceGuard` caps MPS memory and stops a run before it passes its memory budget; `preflight` proves that padding
never matters and that the views touch real rows only; `BatchPrefetcher` gathers the next batches in a background
thread; `Trainer` holds the shared parts of the seeded, resumable training loop (learning-rate factor, metrics log,
checkpoint restore, label-free evaluation, budget stop). `train.T32Trainer` builds on it and supplies `__init__`,
`run`, `_save` and `_train_step`.

This package's own copy of `src/model/benign_gridsybil/timesnet/train.py`
(copied 2026-11-15 so that `src/model/all_t32_benign` imports nothing from it): the kept code is unchanged (same
math and random-number order). Removed because `T32Trainer` replaces or never calls them: `Trainer.__init__`,
`Trainer.run`, `Trainer._save`, `Trainer._train_step`, `Trainer._ram_estimate`, `Trainer._epoch_batches`
(`train.BatchPlan` draws the batches the same way), `environment` (`train.RunProvenance` writes env.json) and the old
command line (`--dataset grid | all`).
"""

from __future__ import annotations

import json
import math
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import torch

from .base_config import PretrainConfig
from .base_data import FeatureShards
from .heads import PretrainingModel
from .losses import Objective
from .monitors import Evaluator

try:
    import psutil
except ImportError:  # pragma: no cover - psutil ships with the pinned training env
    psutil = None


class ResourceGuard:
    """Caps MPS memory and stops the run cleanly before the process passes `mem_gb`.

    With `enabled = False` nothing is capped and the run never stops for memory (it is still measured and logged).
    """

    def __init__(self, mem_gb: float, device: torch.device, enabled: bool = True) -> None:
        self.limit = mem_gb * 1e9
        self.device = device
        self.enabled = enabled
        self.mps_fraction = None
        if enabled and device.type == "mps":
            recommended = torch.mps.recommended_max_memory()
            self.mps_fraction = float(min(1.0, max(0.1, (mem_gb - 4.0) * 1e9 / recommended)))
            torch.mps.set_per_process_memory_fraction(self.mps_fraction)

    def used_bytes(self) -> float:
        """Process RSS plus MPS driver memory."""
        rss = psutil.Process().memory_info().rss if psutil else 0.0
        gpu = torch.mps.driver_allocated_memory() if self.device.type == "mps" else 0.0
        return float(rss + gpu)

    def over_budget(self) -> bool:
        """True once usage passes 95% of the budget."""
        return self.enabled and self.used_bytes() > 0.95 * self.limit


def pick_device(name: str) -> torch.device:
    """`auto` = MPS, then CUDA, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def git(*args: str) -> str:
    """Output of a git command, or "unknown"."""
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def same_link_pairs(keys: list) -> int:
    """Pairs of windows in a batch that come from the same link (false negatives for NT-Xent)."""
    counts: dict[str, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    return sum(c * (c - 1) // 2 for c in counts.values())


@torch.no_grad()
def preflight(
    settings: PretrainConfig, model: PretrainingModel, objective: Objective, data: FeatureShards, device: torch.device
) -> dict[str, Any]:
    """Padding must not matter, views must touch real rows only, shapes must match. Raises on failure."""
    model.eval()
    x, mask = data.x[:64].to(device), data.mask[:64].to(device)
    report: dict[str, Any] = {}
    seed = settings.seed + 12345
    groups = data.broadcast_groups()[:64]
    losses_a, extra_a = objective(model, x, mask, torch.Generator().manual_seed(seed), groups)
    garbage = x + torch.randn_like(x) * 100.0 * (~mask).unsqueeze(-1)
    losses_b, _ = objective(model, garbage, mask, torch.Generator().manual_seed(seed), groups)
    for k in losses_a:
        diff = abs(float(losses_a[k]) - float(losses_b[k]))
        report[f"padding_invariance_{k}"] = diff
        if diff > 1e-3:
            raise AssertionError(f"padding changes loss {k} by {diff}")
    report["nce_masked_pairs_first64"] = extra_a.get("nce_masked_pairs", 0)
    space = objective.augment.space
    roundtrip = float((space.norm(space.raw(x), mask) - x).abs().max())
    report["scaling_roundtrip_max_abs"] = roundtrip
    if roundtrip > 1e-3:
        raise AssertionError(f"scaling is not invertible on real rows ({roundtrip})")
    _, za = model.encoder(x, mask)
    _, zb = model.encoder(garbage, mask)
    report["padding_invariance_z"] = float((za - zb).abs().max())
    if report["padding_invariance_z"] > 1e-4:
        raise AssertionError("padding changes z")
    hidden = objective.masker(mask, torch.Generator().manual_seed(seed))
    if bool((hidden & ~mask).any()):
        raise AssertionError("reconstruction masking hid a padding row")
    xi, targets, valid = objective.inject(x, mask, torch.Generator().manual_seed(seed))
    if float((xi * (~mask).unsqueeze(-1)).abs().max()) != 0.0:
        raise AssertionError("injection touched a padding row")
    if bool((targets.sum(1) > 0)[~valid].any()):
        raise AssertionError("a window with too few real rows was injected")
    report["inject_share_of_valid"] = float((targets.sum(1) > 0)[valid].float().mean()) if valid.any() else None
    v, m = objective.augment(x, mask, torch.Generator().manual_seed(seed))
    if float((v * (~m).unsqueeze(-1)).abs().max()) != 0.0 or bool((m.sum(1) > mask.sum(1)).any()):
        raise AssertionError("augmentation leaked into padding or grew a window")
    report["params_encoder"] = sum(p.numel() for p in model.encoder.parameters())
    report["params_total"] = sum(p.numel() for p in model.parameters())
    model.train()
    return report


class BatchPrefetcher:
    """Gathers the next batches (index -> x, mask on the CPU) in a background thread while the GPU trains.

    Yields (batch number, idx, x, mask) in order. With `depth = 0` it gathers in the main thread.
    """

    def __init__(self, data: FeatureShards, batches: list[torch.Tensor], first: int, depth: int) -> None:
        self.data, self.batches, self.next_bi, self.depth = data, batches, first, depth
        self.queue: queue.Queue = queue.Queue(maxsize=max(1, depth))
        self.stop = threading.Event()
        if depth > 0:
            self.thread = threading.Thread(target=self._fill, args=(first,), daemon=True)
            self.thread.start()

    def _gather(self, bi: int) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.batches[bi]
        return bi, idx, self.data.x[idx], self.data.mask[idx]

    def _fill(self, first: int) -> None:
        for bi in range(first, len(self.batches)):
            if self.stop.is_set():
                return
            self.queue.put(self._gather(bi))

    def __iter__(self) -> "BatchPrefetcher":
        return self

    def __next__(self) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.depth > 0:
            return self.queue.get()
        item = self._gather(self.next_bi)
        self.next_bi += 1
        return item

    def close(self) -> None:
        """Stops the background thread (call when leaving an epoch early)."""
        self.stop.set()
        while not self.queue.empty():
            self.queue.get_nowait()


class Trainer:
    """Seeded, resumable, budgeted training loop with label-free checkpoint selection (shared parts).

    The attributes below are set by the subclass's `__init__` (`train.T32Trainer`): `s` (settings), `run_dir`,
    `device`, `guard`, `model`, `objective`, `evaluator`, `opt`, `sched`, `scales`, `total_steps`, `metrics`,
    `step`, `epoch`, `best`, `bad_epochs`, `skipped`.
    """

    s: PretrainConfig
    model: PretrainingModel
    objective: Objective
    evaluator: Evaluator
    guard: ResourceGuard
    device: torch.device
    metrics: Path

    def _lr_factor(self, step: int) -> float:
        if step < self.s.warmup_steps:
            return (step + 1) / self.s.warmup_steps
        progress = (step - self.s.warmup_steps) / max(1, self.total_steps - self.s.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    def _log(self, record: dict[str, Any]) -> None:
        with self.metrics.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def _load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["opt"])
        self.sched.load_state_dict(state["sched"])
        self.scales.load(state["scales"])
        self.step, self.epoch, self.best, self.bad_epochs = (
            state["step"],
            state["epoch"],
            state["best"],
            state["bad_epochs"],
        )
        torch.set_rng_state(state["torch_rng"].cpu())
        print(f"resumed at epoch {self.epoch}, step {self.step}, best val_joint {self.best:.4f}", flush=True)

    def _evaluate(self, tag: str) -> dict[str, Any]:
        t0 = time.time()
        out = self.evaluator(self.model, self.scales)
        out.update(
            {
                "kind": "eval",
                "tag": tag,
                "step": self.step,
                "epoch": self.epoch,
                "eval_seconds": round(time.time() - t0, 1),
            }
        )
        self._log(out)
        keys = [
            "val_joint",
            "val_recon",
            "val_recon_interp_baseline",
            "val_nce",
            "auroc_p1",
            "auroc_p2",
            "auroc_p3",
            "z_effective_rank",
            "length_probe_r2",
        ]
        print(
            f"[eval {tag}] " + "  ".join(f"{k}={out[k]:.4f}" for k in keys if isinstance(out.get(k), float)), flush=True
        )
        return out

    def _budget_stop(self, start: float) -> str | None:
        """Reason to stop after this step, or None."""
        s = self.s
        if self.guard.over_budget():
            return "memory_budget"
        if (time.time() - start) / 3600 > s.max_hours:
            return "time_budget"
        if s.max_steps and self.step - self.skipped >= s.max_steps:  # count real updates, not skipped batches
            return "max_steps"
        return None
