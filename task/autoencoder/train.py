"""Trains the T48 autoencoder on all train windows (benign and GridSybil alike; labels are never read).

The on-disk `val` split (about 10% of the sender vehicles, disjoint from train and test) drives early stopping and the
LR schedule. Writes `task/autoencoder/runs/<run_id>/`: config.json, env.json, history.jsonl, train.log,
loss_curve.png, best.pt, summary.json.

Usage (repo root):
    .venv-train/bin/python -m task.autoencoder.train --seed 0
    .venv-train/bin/python -m task.autoencoder.train --seed 1 --epochs 20 --device cpu
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from task.autoencoder.config import AutoencoderConfig
from task.autoencoder.data import WindowDataset
from task.autoencoder.model import ConvAutoencoder, count_params
from task.autoencoder.trainer import RunLog, Trainer, pick_device


def environment(device: torch.device) -> dict:
    """Versions, host and git state for env.json (RULE 4)."""

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "argv": sys.argv,
    }


class Experiment:
    """One run: load the train and val windows, fit, write the run folder."""

    def __init__(self, settings: AutoencoderConfig, max_shards: int = 0) -> None:
        self.settings = settings
        self.max_shards = max_shards
        self.run_dir = Path(settings.runs_dir) / settings.run_id

    def run(self) -> dict:
        s = self.settings
        if (self.run_dir / "summary.json").exists():
            raise FileExistsError(f"{self.run_dir} already holds a finished run; remove it or pick another seed")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("history.jsonl", "train.log"):
            (self.run_dir / name).unlink(missing_ok=True)
        log = RunLog(self.run_dir / "train.log")
        torch.manual_seed(s.seed)
        np.random.seed(s.seed)
        device = pick_device(s.device)
        (self.run_dir / "config.json").write_text(json.dumps(s.to_dict(), indent=2))
        (self.run_dir / "env.json").write_text(json.dumps(environment(device), indent=2))

        t0 = time.time()
        train = WindowDataset.load(s.data_dir, "train", workers=s.workers, max_shards=self.max_shards)
        val = WindowDataset.load(s.data_dir, "val", workers=s.workers, max_shards=self.max_shards)
        shared = set(train.vehicles) & set(val.vehicles)
        if shared:
            raise ValueError(f"{len(shared)} sender vehicles are in both train and val")
        log(
            f"train {len(train):,} / val {len(val):,} windows ({time.time() - t0:.0f}s to load), "
            f"shape {train.x.shape[1:]} + mask; vehicles {len(set(train.vehicles)):,} / {len(set(val.vehicles)):,}"
        )
        log(
            f"padding share train {1 - train.mask.mean():.3f} / val {1 - val.mask.mean():.3f}, "
            f"real messages per window median {np.median(train.n_messages):.0f}"
        )
        model = ConvAutoencoder(s)
        params = count_params(model)
        log(f"ConvAutoencoder: {params:,} trainable parameters, input 14 x {s.seq_len}, z {s.latent_dim}")
        trainer = Trainer(model, s, device)
        result = trainer.fit((train.x, train.mask), (val.x, val.mask), self.run_dir)
        summary = {
            "run_id": s.run_id,
            "params": params,
            **result,
            "train_windows": len(train),
            "val_windows": len(val),
            "wall_seconds": round(time.time() - t0, 1),
            "device": str(device),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        log(f"done: {json.dumps(summary)}")
        return summary


def parse_args(argv: list[str] | None = None) -> tuple[AutoencoderConfig, int]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None, help="max epochs (default from the config)")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--max-shards", type=int, default=0, help="read only the first N shards (quick checks)")
    a = p.parse_args(argv)
    changes = {"seed": a.seed, "device": a.device}
    if a.epochs is not None:
        changes["max_epochs"] = a.epochs
    if a.data_dir:
        changes["data_dir"] = a.data_dir
    if a.runs_dir:
        changes["runs_dir"] = a.runs_dir
    return dataclasses.replace(AutoencoderConfig(), **changes), a.max_shards


def main(argv: list[str] | None = None) -> None:
    settings, max_shards = parse_args(argv)
    Experiment(settings, max_shards).run()


if __name__ == "__main__":
    main()
