"""Trains the supervised T = 64 autoencoder (classifier head on z) on the benign + GridSybil train windows.

Data: `src/data/encoder_input/benign_gridsybil/T64/` (on-disk 90 / 10 train / test by sender vehicle, no val split),
read through `task/encoder`'s npy cache. Validation = 10% of the train sender vehicles drawn with
`task.encoder.data.VehicleSplit(0.1, seed 0)` over all train windows (the val vehicles of `task/encoder` supervised
mode). Inputs are `x` and `mask` only; the target is GridSybil (1) vs benign (0). The BCE class weight comes from the
train counts. Val drives the LR schedule, early stopping, the best checkpoint and the two alarm thresholds (max F1 on
val; benign FPR 5% on val). The test split is never opened here (`task.autoencoder_st_t64.evaluate` reads it).

Writes `task/autoencoder_st_t64/runs/<run_id>/`: config.json, env.json, history.jsonl, train.log, curves.png, best.pt,
last.pt, val_vehicles.npy, summary.json.

Usage (repo root):
    .venv-train/bin/python -m task.autoencoder_st_t64.train --seed 0
    .venv-train/bin/python -m task.autoencoder_st_t64.train --seed 0 --resume
    .venv-train/bin/python -m task.autoencoder_st_t64.train --seed 9 --epochs 1 --max-windows 20000 --runs-dir /tmp/x
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from task.autoencoder_st.metrics import TrackScores, val_thresholds
from task.autoencoder_st.train import environment
from task.autoencoder_st_t64.config import TOTAL_PARAMS, SupervisedAET64Config
from task.autoencoder_st_t64.data import Split, load_train_val
from task.autoencoder_st_t64.model import SupervisedAutoencoder, count_params
from task.autoencoder_st_t64.trainer import RunLog, Trainer, pick_device, seed_everything

SPLITS_OPENED = ["train"]  # the on-disk splits this script opens; test stays for evaluate.py / the notebook


class Experiment:
    """One run: load train, hold out val vehicles, check, fit, set val thresholds, write the run folder."""

    def __init__(self, settings: SupervisedAET64Config, max_windows: int = 0, resume: bool = False) -> None:
        self.settings = settings
        self.max_windows = max_windows
        self.resume = resume
        self.run_dir = Path(settings.runs_dir) / settings.run_id

    def run(self) -> dict:
        s = self.settings
        if (self.run_dir / "summary.json").exists():
            raise FileExistsError(f"{self.run_dir} already holds a finished run; remove it or pick another seed")
        if self.resume:
            if not (self.run_dir / "last.pt").exists():
                raise FileNotFoundError(f"--resume: no last.pt in {self.run_dir}")
            saved = SupervisedAET64Config.from_dict(json.loads((self.run_dir / "config.json").read_text()))
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
        train, val, label_reads = load_train_val(s, self.max_windows)
        np.save(self.run_dir / "val_vehicles.npy", np.unique(val.vehicles).astype(str))
        counts = self.check_inputs(train, val, log)
        pos_weight = counts["train"]["benign"] / counts["train"]["gridsybil"]
        train_tracks, val_tracks = TrackScores(train.tracks), TrackScores(val.tracks)
        log(
            f"train {len(train):,} / val {len(val):,} windows ({time.time() - t0:.0f}s to load), "
            f"shape {train.x.shape[1:]} + mask; vehicles {len(set(train.vehicles)):,} / {len(set(val.vehicles)):,}; "
            f"tracks {len(train_tracks):,} / {len(val_tracks):,}; classes {json.dumps(counts)}; "
            f"pos_weight = n_benign / n_gridsybil (train) = {pos_weight:.4f}; val = {s.val_frac:g} of the train "
            f"vehicles (VehicleSplit, split seed {s.split_seed})"
        )
        log(f"padding share train {1 - train.mask.mean():.3f} / val {1 - val.mask.mean():.3f}")
        model = SupervisedAutoencoder(s)
        params = count_params(model)
        if (s.latent_dim, s.head_hidden, tuple(s.channels)) == (16, 32, (32, 64, 64)) and params != TOTAL_PARAMS:
            raise AssertionError(f"expected {TOTAL_PARAMS:,} parameters, got {params:,}")
        log(
            f"SupervisedAutoencoder: {params:,} trainable parameters (task/encoder ConvAutoencoder "
            f"{count_params(model.autoencoder):,} + head {count_params(model.head):,}), input 14 x {s.seq_len}, "
            f"z {s.latent_dim}"
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
            "train_vehicles": len(set(train.vehicles)),
            "val_vehicles": len(set(val.vehicles)),
            "train_tracks": len(train_tracks),
            "val_tracks": len(val_tracks),
            "class_counts": counts,
            "pos_weight": pos_weight,
            **thresholds,
            "splits_opened": SPLITS_OPENED,
            "label_reads": label_reads,
            "wall_seconds": round(time.time() - t0, 1),
            "device": str(device),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        log(f"done: {json.dumps({k: v for k, v in summary.items() if k != 'thresholds'})}")
        return summary

    @staticmethod
    def check_inputs(train: Split, val: Split, log: RunLog) -> dict:
        """Labels 0 / 1 with both classes, no sender vehicle in both splits, one label per track."""
        counts = {"train": train.counts(), "val": val.counts()}
        if set(train.vehicles) & set(val.vehicles):
            raise ValueError("sender vehicles in both train and val")
        for data in (train, val):
            _, mixed = TrackScores(data.tracks).track_labels(data.y)
            if mixed:
                raise ValueError(f"{data.name}: {mixed} tracks hold windows with different labels")
        log("checks: labels 0 / 1 with both classes in train and val; no shared sender vehicle; one label per track")
        return counts

    def fit_thresholds(self, trainer: Trainer, val: Split, val_tracks: TrackScores, log: RunLog) -> dict:
        """Alarm thresholds of the best model from val only: per window and per track (max / mean)."""
        prob = trainer.predict(val.x, val.mask)["prob"]
        window = val_thresholds(val.y, prob, self.settings.target_fpr)
        track_y, _ = val_tracks.track_labels(val.y)
        track = {
            how: val_thresholds(track_y, agg, self.settings.target_fpr)
            for how, agg in val_tracks.aggregate(prob).items()
        }
        log(f"val thresholds (window): {json.dumps(window)}")
        return {"thresholds": {"window": window, "track": track}, "threshold_source": "val (in-code vehicle draw)"}


def parse_args(argv: list[str] | None = None) -> tuple[SupervisedAET64Config, int, bool]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None, help="max epochs (default from the config)")
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true", help="continue the run from its last.pt")
    p.add_argument("--recon-weight", type=float, default=None, help="lambda of the reconstruction loss")
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--max-windows", type=int, default=0, help="evenly spaced sample of train windows (quick checks)")
    a = p.parse_args(argv)
    changes = {"seed": a.seed, "device": a.device}
    for key, value in (("max_epochs", a.epochs), ("recon_weight", a.recon_weight), ("runs_dir", a.runs_dir)):
        if value is not None:
            changes[key] = value
    return dataclasses.replace(SupervisedAET64Config(), **changes), a.max_windows, a.resume


def main(argv: list[str] | None = None) -> None:
    settings, max_windows, resume = parse_args(argv)
    Experiment(settings, max_windows, resume).run()


if __name__ == "__main__":
    main()
