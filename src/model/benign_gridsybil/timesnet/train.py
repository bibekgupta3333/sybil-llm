"""The pretraining command: resources, provenance, pre-flight checks, the training loop and the command line.

`Trainer` is seeded, resumable and budgeted (time and a 16 GB memory guard) and picks checkpoints on label-free
check-set losses. `preflight` proves that padding never matters and that the views touch real rows only. Outputs go
to `src/runs/pretraining/benign_gridsybil/T64/<run_id>/` (or `--runs-dir <dir>/<run_id>/`, e.g. on Google Drive):
config.json, env.json, metrics.jsonl, best.pt, last.pt.

    .venv-train/bin/python -m src.model.benign_gridsybil.timesnet.train [--check | --smoke | --preset recon_only]
        [--runs-dir DIR] [--run-id NAME] [--resume RUN_DIR]
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import platform
import queue
import threading
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .config import PretrainConfig
from .data import FeatureShards, FeatureSpace, LabelFirewall, file_sha256
from .heads import PretrainingModel
from .losses import LossScales, Objective
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


def environment(
    settings: PretrainConfig, train: FeatureShards, val: FeatureShards, device: torch.device, guard: ResourceGuard
) -> dict[str, Any]:
    """Everything needed to reproduce or audit the run (RULE 4)."""
    root = Path(settings.data_dir)
    req = Path("requirements-train.txt")
    return {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "mps_memory_fraction": guard.mps_fraction,
        "requirements_train_sha256": file_sha256(req) if req.exists() else None,
        "encoder_input_metadata": str(root / "metadata.json"),
        "encoder_input_metadata_sha256": file_sha256(root / "metadata.json"),
        "normalisation": "metadata.json['normalisation'] (train real rows)",
        "rows": {"train": len(train), val.split: len(val)},
        "check_set": (
            "pretrain_val folder"
            if val.split == "pretrain_val"
            else f"{settings.holdout_frac:.0%} of train sender vehicles (seed {settings.seed}); test never opened"
        ),
        "splits_opened": sorted({"train", "pretrain_val" if val.split == "pretrain_val" else "train"}),
        "label_files_opened": 0,
        "determinism": "seeded; MPS kernels are not bit-deterministic (same seed = same result within tolerance)",
    }


def same_link_pairs(keys: list[str]) -> int:
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
    """Seeded, resumable, budgeted training loop with label-free checkpoint selection."""

    def __init__(self, settings: PretrainConfig, run_dir: Path, resume: bool) -> None:
        self.s, self.run_dir = settings, run_dir
        torch.manual_seed(settings.seed)
        np.random.seed(settings.seed)
        self.device = pick_device(settings.device)
        if settings.cpu_threads > 0:
            torch.set_num_threads(settings.cpu_threads)
        self.guard = ResourceGuard(settings.mem_gb, self.device, settings.resource_guard)
        if not settings.resource_guard:
            print("resource guard OFF: no MPS memory cap, no memory-budget stop", flush=True)
        root = Path(settings.data_dir)
        firewall = LabelFirewall()
        metadata = json.loads((root / "metadata.json").read_text())
        t0 = time.time()
        self.train = FeatureShards(root, "train", firewall, settings.shards_per_split, settings.workers)
        if (root / "pretrain_val").is_dir():
            self.val = FeatureShards(root, "pretrain_val", firewall, settings.shards_per_split, settings.workers)
        else:  # train / test layout: the label-free check set is carved out of train by vehicle (never test)
            self.train, self.val = self.train.hold_out(settings.holdout_frac, settings.seed)
        self.train_links = self.train.link_keys
        self.train_groups = self.train.broadcast_groups()
        self.train_lengths = self.train.mask.sum(1)
        # scaling: stored files hold train z-scores; robust features are re-expressed (median / IQR from train only)
        self.space = FeatureSpace(metadata)
        if settings.robust_features:
            stored = FeatureSpace(metadata)
            self.space.fit_robust(self.train, tuple(settings.robust_features), settings.soft_clip)
            FeatureSpace.rescale(self.train, stored, self.space)
            FeatureSpace.rescale(self.val, stored, self.space)
        self.space.to(self.device)
        print(
            f"loaded train {len(self.train):,} / {self.val.split} {len(self.val):,} windows "
            f"in {time.time() - t0:.0f} s; device {self.device}",
            flush=True,
        )
        self.model = PretrainingModel(settings).to(self.device)
        self.objective = Objective(settings, self.space)
        self.evaluator = Evaluator(settings, self.val, self.objective, self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
        self.steps_per_epoch = math.ceil(len(self.train) / settings.batch_size)
        self.total_steps = settings.max_steps or settings.max_epochs * self.steps_per_epoch
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, self._lr_factor)
        self.scales = LossScales(settings.norm_warmup_steps)
        self.step, self.epoch, self.best, self.bad_epochs = 0, 0, float("inf"), 0
        self.skipped = 0  # steps whose batch had no applicable loss
        self.metrics = run_dir / "metrics.jsonl"
        if resume:
            self._load(run_dir / "last.pt")
        else:
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(settings), indent=2))
            env = environment(settings, self.train, self.val, self.device, self.guard)
            env["scaling"] = self.space.report
            env["cpu_threads"] = torch.get_num_threads()
            env["prefetch_batches"] = settings.prefetch_batches
            env["preflight"] = preflight(settings, self.model, self.objective, self.train, self.device)
            (run_dir / "env.json").write_text(json.dumps(env, indent=2))
            print("preflight ok:", json.dumps(env["preflight"]), flush=True)

    def _lr_factor(self, step: int) -> float:
        if step < self.s.warmup_steps:
            return (step + 1) / self.s.warmup_steps
        progress = (step - self.s.warmup_steps) / max(1, self.total_steps - self.s.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    def _log(self, record: dict[str, Any]) -> None:
        with self.metrics.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def _save(self, name: str) -> None:
        torch.save(
            {
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
            },
            self.run_dir / name,
        )

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

    def _train_step(self, idx: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> None:
        """One optimiser step on the windows `idx` (x, mask already gathered on the CPU); logs every `log_every` steps."""
        s = self.s
        x, mask = x.to(self.device, non_blocking=True), mask.to(self.device, non_blocking=True)
        gen = torch.Generator().manual_seed(s.seed * 1_000_003 + self.step)
        t0 = time.time()
        losses, extra = self.objective(self.model, x, mask, gen, self.train_groups[idx])
        raw = {k: float(v.detach()) for k, v in losses.items()}
        if not losses:  # e.g. a batch of 1-row windows: no loss applies, so no update (the step still counts)
            self.skipped += 1
            self.step += 1
            return
        self.scales.update(raw, self.step)
        loss = self.objective.joint(losses, self.scales)
        if not torch.isfinite(loss):
            self._save("last.pt")
            raise FloatingPointError(f"non-finite loss at step {self.step}: {raw}")
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), s.grad_clip))
        self.opt.step()
        self.sched.step()
        self.step += 1
        if self.step % s.log_every == 0 or self.step - self.skipped == 1:  # always log the first real update
            rec = {
                "kind": "train",
                "step": self.step,
                "epoch": self.epoch,
                "joint": float(loss.detach()),
                **raw,
                "grad_norm": grad_norm,
                "lr": self.sched.get_last_lr()[0],
                "step_seconds": round(time.time() - t0, 3),
                "mem_gb": round(self.guard.used_bytes() / 1e9, 2),
                "same_link_pairs": same_link_pairs([self.train_links[i] for i in idx.tolist()]),
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
        """Reason to stop after this step, or None."""
        s = self.s
        if self.guard.over_budget():
            return "memory_budget"
        if (time.time() - start) / 3600 > s.max_hours:
            return "time_budget"
        if s.max_steps and self.step - self.skipped >= s.max_steps:  # count real updates, not skipped batches
            return "max_steps"
        return None

    def _epoch_batches(self, epoch: int) -> list[torch.Tensor]:
        """The batches of one epoch (deterministic per seed and epoch, so a resume continues the same order)."""
        s = self.s
        gen = torch.Generator().manual_seed(s.seed + epoch)
        perm = torch.randperm(len(self.train), generator=gen)
        if not s.length_bucketed:
            return list(perm.split(s.batch_size))
        order = perm[torch.argsort(self.train_lengths[perm], stable=True)]  # random order within each length
        batches = list(order.split(s.batch_size))
        return [batches[i] for i in torch.randperm(len(batches), generator=gen).tolist()]

    def run(self) -> None:
        """Trains until max_epochs, early stopping, or a step / time / memory budget; writes a summary record."""
        s = self.s
        start = time.time()
        if self.step == 0:
            self._evaluate("init")  # random-init reference
        stop_reason = "max_epochs"
        while self.epoch < s.max_epochs:
            batches = self._epoch_batches(self.epoch)
            first = max(0, (self.step - self.epoch * self.steps_per_epoch) if self.step else 0)
            feed = BatchPrefetcher(self.train, batches, first, s.prefetch_batches)
            first_batch = (self.step - self.epoch * self.steps_per_epoch) if self.step else 0
            budget_hit = None
            for bi in range(max(0, first_batch), self.steps_per_epoch):
                bi_ready, idx, x_cpu, mask_cpu = next(feed)
                assert bi_ready == bi
                if len(idx) < 2:
                    continue
                self._train_step(idx, x_cpu, mask_cpu)
                budget_hit = self._budget_stop(start)
                if budget_hit:
                    break
            feed.close()
            if budget_hit:
                stop_reason = budget_hit
                break
            self.epoch += 1
            out = self._evaluate(f"epoch{self.epoch}")
            if out["val_joint"] < self.best - 1e-4:
                self.best, self.bad_epochs = out["val_joint"], 0
                self._save("best.pt")
            else:
                self.bad_epochs += 1
            self._save("last.pt")
            if self.bad_epochs >= s.patience:
                stop_reason = "early_stopping"
                break
        if stop_reason not in ("early_stopping", "max_epochs"):
            out = self._evaluate(f"stop_{stop_reason}")
            if out["val_joint"] < self.best:
                self.best = out["val_joint"]
                self._save("best.pt")
        self._save("last.pt")
        summary = {
            "kind": "summary",
            "stop_reason": stop_reason,
            "step": self.step,
            "epoch": self.epoch,
            "best_val_joint": self.best,
            "hours": round((time.time() - start) / 3600, 3),
        }
        self._log(summary)
        print("done:", json.dumps(summary), flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options (unset options keep the preset's value)."""
    p = argparse.ArgumentParser(description="Self-supervised TimesNet pretraining (benign + GridSybil, T = 64)")
    p.add_argument("--preset", default="joint", choices=["joint", "recon_only"])
    p.add_argument("--run-id", default=None, help="folder name under runs_dir (default: timestamp + preset)")
    p.add_argument(
        "--runs-dir",
        default=None,
        help="parent folder of new run folders, may be outside the repo (e.g. Google Drive); default: config runs_dir",
    )
    p.add_argument("--resume", default=None, help="run folder to resume from (uses its config.json and last.pt)")
    p.add_argument("--check", action="store_true", help="load one shard per split, run the pre-flight checks, exit")
    p.add_argument("--smoke", action="store_true", help="one shard per split, 30 steps, small eval")
    p.add_argument("--max-epochs", type=int)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-hours", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lambda-nce", type=float)
    p.add_argument("--mem-gb", type=float)
    p.add_argument(
        "--no-resource-guard",
        dest="resource_guard",
        action="store_false",
        default=None,
        help="do not cap MPS memory and never stop for the memory budget (memory is still logged)",
    )
    p.add_argument("--device")
    p.add_argument("--seed", type=int)
    return p.parse_args(argv)


def settings_from_args(args: argparse.Namespace) -> tuple[PretrainConfig, Path, bool]:
    """Settings, run folder and resume flag from the parsed options (unset options keep the preset's value).

    A resumed run takes its settings from its own config.json, so the run folder may live anywhere (e.g. on Google
    Drive); only the time budget and the resource guard can be changed on resume.
    """
    if args.resume:
        run_dir = Path(args.resume).expanduser()
        settings = PretrainConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
        if args.max_hours is not None:
            settings.max_hours = args.max_hours
        if args.resource_guard is not None:
            settings.resource_guard = args.resource_guard
        return settings, run_dir, True
    settings = PretrainConfig.preset(args.preset)
    for name in (
        "max_epochs",
        "max_steps",
        "max_hours",
        "batch_size",
        "lambda_nce",
        "mem_gb",
        "resource_guard",
        "device",
        "seed",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(settings, name, value)
    if args.runs_dir is not None:
        settings.runs_dir = str(Path(args.runs_dir).expanduser())  # recorded in config.json
    if args.smoke or args.check:
        settings.shards_per_split, settings.eval_windows, settings.log_every = 1, 512, 5
        settings.max_steps = settings.max_steps or 30
        settings.norm_warmup_steps = min(settings.norm_warmup_steps, 10)
        settings.warmup_steps = min(settings.warmup_steps, 10)
    settings.validate()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    kind = "check" if args.check else "smoke" if args.smoke else args.preset
    return settings, Path(settings.runs_dir) / (args.run_id or f"{stamp}-{kind}"), False


def main(argv: list[str] | None = None) -> None:
    """Entry point: resume, check, smoke or a full run."""
    args = parse_args(argv)
    if not (Path("CLAUDE.md").is_file() and Path("src/data").is_dir()):
        raise SystemExit("run from the repository root")
    settings, run_dir, resume = settings_from_args(args)
    trainer = Trainer(settings, run_dir, resume=resume)
    if args.check and not resume:
        print(f"checks passed; report in {run_dir / 'env.json'}")
        return
    trainer.run()


if __name__ == "__main__":
    main()
