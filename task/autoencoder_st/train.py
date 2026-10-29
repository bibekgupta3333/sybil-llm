"""Trains the supervised T48 autoencoder (classifier head on z) on the benign + GridSybil train windows.

Data: `src/data/encoder_input/benign_gridsybil/T48_tvt/` (no overlap, train / val / test = 80 / 10 / 10 by sender
vehicle, fixed on disk). Inputs are `x` and `mask` only; the target is `is_attack` (Benign 0, GridSybil 1). The BCE
class weight comes from the train counts. The val split drives the LR schedule, early stopping, best-checkpoint
selection and the two alarm thresholds (max F1 on val; benign FPR 5% on val). The test split is never read here.

Writes `task/autoencoder_st/runs/<run_id>/`: config.json, env.json, history.jsonl, train.log, curves.png, best.pt,
last.pt, summary.json.

Usage (repo root):
    .venv-train/bin/python -m task.autoencoder_st.train --seed 0
    .venv-train/bin/python -m task.autoencoder_st.train --seed 0 --resume
    .venv-train/bin/python -m task.autoencoder_st.train --seed 1 --epochs 2 --max-shards 1 --device cpu
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
import sklearn
import torch

from task.autoencoder_st.config import TOTAL_PARAMS, SupervisedAEConfig
from task.autoencoder_st.data import WindowDataset
from task.autoencoder_st.metrics import TrackScores, val_thresholds
from task.autoencoder_st.model import SupervisedAutoencoder, count_params
from task.autoencoder_st.trainer import RunLog, Trainer, pick_device, seed_everything


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
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "argv": sys.argv,
    }


class Experiment:
    """One run: load train and val, check labels and splits, fit, set val thresholds, write the run folder."""

    def __init__(self, settings: SupervisedAEConfig, max_shards: int = 0, resume: bool = False) -> None:
        self.settings = settings
        self.max_shards = max_shards
        self.resume = resume
        self.run_dir = Path(settings.runs_dir) / settings.run_id

    def run(self) -> dict:
        s = self.settings
        if (self.run_dir / "summary.json").exists():
            raise FileExistsError(f"{self.run_dir} already holds a finished run; remove it or pick another seed")
        if self.resume:
            if not (self.run_dir / "last.pt").exists():
                raise FileNotFoundError(f"--resume: no last.pt in {self.run_dir}")
            saved = SupervisedAEConfig.from_dict(json.loads((self.run_dir / "config.json").read_text()))
            if dataclasses.replace(saved, device=s.device, workers=s.workers) != s:
                raise ValueError("--resume: settings differ from the run's config.json")
        else:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            for name in ("history.jsonl", "train.log", "best.pt", "last.pt"):
                (self.run_dir / name).unlink(missing_ok=True)
        log = RunLog(self.run_dir / "train.log")
        seed_everything(s.seed)
        device = pick_device(s.device)
        (self.run_dir / "config.json").write_text(json.dumps(s.to_dict(), indent=2))
        (self.run_dir / "env.json").write_text(json.dumps(environment(device), indent=2))

        t0 = time.time()
        train = WindowDataset.load(s.data_dir, "train", workers=s.workers, max_shards=self.max_shards)
        val = WindowDataset.load(s.data_dir, "val", workers=s.workers, max_shards=self.max_shards)
        counts = self.check_inputs(train, val, log)
        pos_weight = counts["train"]["benign"] / counts["train"]["gridsybil"]
        train_tracks, val_tracks = TrackScores(train.tracks), TrackScores(val.tracks)
        log(
            f"train {len(train):,} / val {len(val):,} windows ({time.time() - t0:.0f}s to load), "
            f"shape {train.x.shape[1:]} + mask; vehicles {len(set(train.vehicles)):,} / {len(set(val.vehicles)):,}; "
            f"tracks {len(train_tracks):,} / {len(val_tracks):,}; classes {json.dumps(counts)}; "
            f"pos_weight = n_benign / n_gridsybil (train) = {pos_weight:.4f}"
        )
        log(f"padding share train {1 - train.mask.mean():.3f} / val {1 - val.mask.mean():.3f}")
        model = SupervisedAutoencoder(s)
        params = count_params(model)
        default_shape = (s.seq_len, s.n_features, s.latent_dim, s.head_hidden) == (48, 13, 16, 32)
        if default_shape and params != TOTAL_PARAMS:
            raise AssertionError(f"expected {TOTAL_PARAMS:,} parameters, got {params:,}")
        log(
            f"SupervisedAutoencoder: {params:,} trainable parameters (autoencoder {count_params(model.autoencoder):,}"
            f" + head {count_params(model.head):,}), input 14 x {s.seq_len}, z {s.latent_dim}"
        )
        trainer = Trainer(model, s, device, pos_weight)
        result = trainer.fit((train.x, train.mask, train.y), (val.x, val.mask, val.y), self.run_dir, self.resume)
        thresholds = self.fit_thresholds(trainer, val, val_tracks, log)
        summary = {
            "run_id": s.run_id,
            "params": params,
            **result,
            "train_windows": len(train),
            "val_windows": len(val),
            "train_tracks": len(train_tracks),
            "val_tracks": len(val_tracks),
            "class_counts": counts,
            "pos_weight": pos_weight,
            **thresholds,
            "wall_seconds": round(time.time() - t0, 1),
            "device": str(device),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        log(f"done: {json.dumps(summary)}")
        return summary

    @staticmethod
    def check_inputs(train: WindowDataset, val: WindowDataset, log: RunLog) -> dict:
        """Labels 0 / 1 with both classes, no sender vehicle in both splits, one label per track."""
        counts = {"train": train.check_labels(), "val": val.check_labels()}
        shared = set(train.vehicles) & set(val.vehicles)
        if shared:
            raise ValueError(f"{len(shared)} sender vehicles are in both train and val")
        for data in (train, val):
            _, mixed = TrackScores(data.tracks).track_labels(data.y)
            if mixed:
                raise ValueError(f"{data.split}: {mixed} tracks hold windows with different labels")
        log("checks: labels 0 / 1 with both classes in train and val; no shared sender vehicle; one label per track")
        return counts

    def fit_thresholds(self, trainer: Trainer, val: WindowDataset, val_tracks: TrackScores, log: RunLog) -> dict:
        """Alarm thresholds of the best model from the val split only: per window and per track (max / mean)."""
        prob = trainer.predict(val.x, val.mask)["prob"]
        window = val_thresholds(val.y, prob, self.settings.target_fpr)
        track_y, _ = val_tracks.track_labels(val.y)
        track = {
            how: val_thresholds(track_y, agg, self.settings.target_fpr)
            for how, agg in val_tracks.aggregate(prob).items()
        }
        log(f"val thresholds (window): {json.dumps(window)}")
        return {"thresholds": {"window": window, "track": track}, "threshold_source": "val split"}


def parse_args(argv: list[str] | None = None) -> tuple[SupervisedAEConfig, int, bool]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None, help="max epochs (default from the config)")
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true", help="continue the run from its last.pt")
    p.add_argument("--recon-weight", type=float, default=None, help="lambda of the reconstruction loss")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--max-shards", type=int, default=0, help="read only the first N shards per split (quick checks)")
    a = p.parse_args(argv)
    changes = {"seed": a.seed, "device": a.device}
    for key, value in (
        ("max_epochs", a.epochs),
        ("recon_weight", a.recon_weight),
        ("data_dir", a.data_dir),
        ("runs_dir", a.runs_dir),
    ):
        if value is not None:
            changes[key] = value
    return dataclasses.replace(SupervisedAEConfig(), **changes), a.max_shards, a.resume


def main(argv: list[str] | None = None) -> None:
    settings, max_shards, resume = parse_args(argv)
    Experiment(settings, max_shards, resume).run()


if __name__ == "__main__":
    main()
