"""Trains the unlabelled T = 20 autoencoder on every train window of all/T20_benign_grid (benign + GridSybil pooled).

Data: `src/data/encoder_input/all/T20_benign_grid/` (benign of all 8 scenario folders + GridSybil attack windows of the
2 GridSybil folders; train / val / test = 85 / 7.5 / 7.5 by vehicle key, fixed on disk, the all/T32_benign_grid
split). No label is read: the feature shards hold id, x, mask only, the `_info.json` files are never opened (the
command line installs `LabelFileGuard` in this process and in every decoding worker), and the cache holds x / mask /
ids. Train fits the weights; val (mixed, unlabelled) drives early stopping, the LR schedule and the p95 / p99
thresholds; the Mahalanobis model is fitted on train z. Test is never opened here. The protocol and every setting are
`task.autoencoder_t32.train`'s; only the window length and the model depth (`n_down`) differ.

Writes `task/autoencoder_t20/runs/<run_id>/`: config.json, env.json, history.jsonl, train.log, curves.png, best.pt,
last.pt, mahalanobis.npz, `scores/val_*.npy`, summary.json. Decoded feature shards are cached under
`task/autoencoder_t20/cache/`. The test evaluation is `task.autoencoder_t20.evaluate`. `Experiment`, `parse_args`
and `main` are shared with the T = 10 / 5 packages (class attributes select the package).

Usage (repo root):
    .venv-train/bin/python -m task.autoencoder_t20.train --seed 0
    .venv-train/bin/python -m task.autoencoder_t20.train --seed 0 --resume
    .venv-train/bin/python -m task.autoencoder_t20.train --seed 1 --epochs 2 --max-shards 1 --device cpu
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from task.autoencoder_benign_t20.model import ConvAutoencoder, count_params
from task.autoencoder_t20.config import N_DOWN, N_FEATURES, SEQ_LEN, TOTAL_PARAMS, AutoencoderT20Config
from task.autoencoder_t20.data import LabelFileGuard, UnlabelledWindows
from task.autoencoder_t20.scoring import Mahalanobis, ScoreFiles, TrackScores, quantile_thresholds
from task.autoencoder_t20.trainer import RunLog, Trainer, pick_device, seed_everything
from task.autoencoder_t32.train import environment

TRAINING_SPLITS = ("train", "val")


class Experiment:
    """One run: load unlabelled train and val, check them, fit, fit the Mahalanobis model, set val thresholds.

    Attributes:
        settings: run settings.
        max_shards: read only the first N shards per split (quick checks); 0 = all.
        resume: continue from the run's last.pt.
        guard: decoding workers refuse to open label files (`main` also installs the guard in this process).
        run_dir: the run folder.

    Class attributes (overridden by the T = 10 / 5 packages): config_cls, trainer_cls, default_shape
    ((seq_len, n_down, n_features, latent_dim) of the package's model) and total_params (asserted for that shape).
    """

    config_cls = AutoencoderT20Config
    trainer_cls = Trainer
    default_shape = (SEQ_LEN, N_DOWN, N_FEATURES, 16)
    total_params = TOTAL_PARAMS

    def __init__(
        self, settings: AutoencoderT20Config, max_shards: int = 0, resume: bool = False, guard: bool = False
    ) -> None:
        self.settings = settings
        self.max_shards = max_shards
        self.resume = resume
        self.guard = guard
        self.run_dir = Path(settings.runs_dir) / settings.run_id

    def load(self, split: str) -> UnlabelledWindows:
        if split not in TRAINING_SPLITS:
            raise ValueError(f"train.py reads only {TRAINING_SPLITS}, not {split!r}")
        s = self.settings
        return UnlabelledWindows.load(s.data_dir, split, s.workers, self.max_shards, s.cache_dir, self.guard)

    def run(self) -> dict:
        s = self.settings
        if (self.run_dir / "summary.json").exists():
            raise FileExistsError(f"{self.run_dir} already holds a finished run; remove it or pick another seed")
        if self.resume:
            if not (self.run_dir / "last.pt").exists():
                raise FileNotFoundError(f"--resume: no last.pt in {self.run_dir}")
            saved = self.config_cls.from_dict(json.loads((self.run_dir / "config.json").read_text()))
            if dataclasses.replace(saved, device=s.device, workers=s.workers, max_epochs=s.max_epochs) != s:
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
        log(f"label-file guard: process {'on' if LabelFileGuard.installed else 'off'}, workers {self.guard}")

        t0 = time.time()
        train, val = self.load("train"), self.load("val")
        for data in (train, val):
            if data.x.shape[1] != s.seq_len:
                raise ValueError(f"{data.split}: windows have {data.x.shape[1]} rows, the config says {s.seq_len}")
        shared = train.vehicles() & val.vehicles()
        if shared:
            raise ValueError(f"{len(shared)} sender vehicles are in both train and val")
        train_tracks, val_tracks = TrackScores(train.tracks()), TrackScores(val.tracks())
        log(
            f"train {len(train):,} / val {len(val):,} unlabelled windows ({time.time() - t0:.0f}s to load), "
            f"shape {train.x.shape[1:]} + mask; vehicles {len(train.vehicles()):,} / {len(val.vehicles()):,} "
            f"(none shared); tracks {len(train_tracks):,} / {len(val_tracks):,}"
        )
        log(
            f"padding share train {1 - train.mask.mean():.3f} / val {1 - val.mask.mean():.3f}, "
            f"real messages per window median {np.median(train.n_messages):.0f}, "
            f"1-3-row windows train {np.mean(train.n_messages <= 3):.3f} / val {np.mean(val.n_messages <= 3):.3f}"
        )
        model = ConvAutoencoder(s)
        params = count_params(model)
        if (s.seq_len, s.n_down, s.n_features, s.latent_dim) == self.default_shape and params != self.total_params:
            raise AssertionError(f"expected {self.total_params:,} parameters, got {params:,}")
        log(
            f"ConvAutoencoder: {params:,} trainable parameters, input 14 x {s.seq_len}, "
            f"{s.n_down} stride-2 stages (latent length {model.short_len}), z {s.latent_dim}"
        )
        trainer = self.trainer_cls(model, s, device)
        result = trainer.fit((train.x, train.mask), (val.x, val.mask), self.run_dir, self.resume)
        scoring = self.fit_scoring(trainer, train, val, val_tracks, log)
        summary = {
            "run_id": s.run_id,
            "params": params,
            **result,
            "train_windows": len(train),
            "val_windows": len(val),
            "train_tracks": len(train_tracks),
            "val_tracks": len(val_tracks),
            "train_vehicles": len(train.vehicles()),
            "val_vehicles": len(val.vehicles()),
            **scoring,
            "labels_read": False,
            "label_file_guard": {"process": LabelFileGuard.installed, "workers": self.guard},
            "wall_seconds": round(time.time() - t0, 1),
            "device": str(device),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        log(f"done: {json.dumps(summary)}")
        return summary

    def fit_scoring(
        self, trainer: Trainer, train: UnlabelledWindows, val: UnlabelledWindows, val_tracks: TrackScores, log: RunLog
    ) -> dict:
        """Mahalanobis on train z; val p95 / p99 thresholds per window and per track; val score files."""
        s = self.settings
        maha = Mahalanobis(s.mahalanobis_max_cond, s.mahalanobis_shrinkage).fit(trainer.score(train.x, train.mask)["z"])
        np.savez(self.run_dir / "mahalanobis.npz", **maha.state())
        log(f"mahalanobis on train z: condition number {maha.cond:.3g}, shrinkage {maha.shrinkage:g}")
        val_scores = trainer.score(val.x, val.mask)
        window = {"reconstruction": val_scores["error"], "mahalanobis": maha.score(val_scores["z"])}
        ScoreFiles(self.run_dir).save(
            "val", val.ids, window["reconstruction"], window["mahalanobis"], val.n_messages, val_scores["z"]
        )
        thresholds = {"window": {}, "track": {}}
        for name, values in window.items():
            thresholds["window"][name] = quantile_thresholds(values, s.threshold_quantiles)
            for how, agg in val_tracks.aggregate(values).items():
                thresholds["track"][f"{name}_{how}"] = quantile_thresholds(agg, s.threshold_quantiles)
        log(f"val thresholds: {json.dumps(thresholds)}")
        return {
            "mahalanobis": {"cond": maha.cond, "shrinkage": maha.shrinkage},
            "thresholds": thresholds,
            "threshold_source": "val split (benign + GridSybil mixed, unlabelled)",
        }


def parse_args(
    argv: list[str] | None = None, config_cls: type = AutoencoderT20Config, description: str = ""
) -> tuple[AutoencoderT20Config, int, bool]:
    p = argparse.ArgumentParser(description=description or __doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None, help="max epochs (default from the config)")
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true", help="continue the run from its last.pt")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--cache-dir", default=None, help='decoded-shard cache ("" disables it)')
    p.add_argument("--workers", type=int, default=None, help="shard-decoding processes")
    p.add_argument("--max-shards", type=int, default=0, help="read only the first N shards per split (quick checks)")
    a = p.parse_args(argv)
    changes = {"seed": a.seed, "device": a.device}
    for key, value in (
        ("max_epochs", a.epochs),
        ("data_dir", a.data_dir),
        ("runs_dir", a.runs_dir),
        ("cache_dir", a.cache_dir),
        ("workers", a.workers),
    ):
        if value is not None:
            changes[key] = value
    return dataclasses.replace(config_cls(), **changes), a.max_shards, a.resume


def main(argv: list[str] | None = None, experiment_cls: type = Experiment, description: str = "") -> None:
    """Command line entry: installs the label-file guard for the whole process, then runs."""
    settings, max_shards, resume = parse_args(argv, experiment_cls.config_cls, description)
    LabelFileGuard.install()
    experiment_cls(settings, max_shards, resume, guard=True).run()


if __name__ == "__main__":
    main()
