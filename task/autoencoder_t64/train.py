"""Trains the unlabelled T = 64 autoencoder on all train windows (benign and GridSybil pooled; labels never read).

10% of the train sender vehicles are held out as validation with `task.encoder.data.VehicleSplit` (the rule and
seed of `task/encoder`; on all train windows it gives the validation vehicles of its supervised mode). Validation
drives early stopping, the LR schedule and the p95 / p99 alarm thresholds; a Gaussian is fitted on the z of the fit
windows for the Mahalanobis score. This module opens the train split only: the test split is read by
`task.autoencoder_t64.evaluate`.

Writes `task/autoencoder_t64/runs/<run_id>/`: config.json, env.json, history.jsonl, train.log, loss_curve.png,
best.pt, mahalanobis.npz, val_vehicles.npy, val_<name>.npy (ids, n_messages, recon_error, recon_error_feat,
mahalanobis; one row per validation window) and summary.json.

Usage (repo root):
    .venv-train/bin/python -m task.autoencoder_t64.train --seed 0
    .venv-train/bin/python -m task.autoencoder_t64.train --seed 1 --epochs 20 --device cpu
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from task.autoencoder_t64.config import AutoencoderT64Config
from task.autoencoder_t64.scoring import Mahalanobis, ScoreFiles, TrackScores, quantile_thresholds, track_ids
from task.autoencoder_t64.trainer import Trainer
from task.encoder.data import VehicleSplit, WindowSet, WindowStore
from task.encoder.model import ConvAutoencoder, count_params
from task.encoder.train import RunLog, environment, pick_device

TRAIN_SPLIT = "train"  # the only split this module opens


class ValidationThresholds:
    """p95 / p99 (or other quantiles) of the unlabelled validation scores, per window and per track."""

    def __init__(self, quantiles: tuple[float, ...]) -> None:
        self.quantiles = quantiles

    def compute(self, window_ids: np.ndarray, scores: dict[str, np.ndarray]) -> dict:
        """{"window": {score: {p95, p99}}, "track": {score_max: ..., score_mean: ...}}."""
        tracks = TrackScores(track_ids(window_ids))
        out: dict = {"window": {}, "track": {}}
        for name, values in scores.items():
            out["window"][name] = quantile_thresholds(values, self.quantiles)
            for how, agg in tracks.aggregate(values).items():
                out["track"][f"{name}_{how}"] = quantile_thresholds(agg, self.quantiles)
        return out


class Experiment:
    """One run: open the train windows, hold out validation vehicles, fit, score fit + val, write the run folder."""

    def __init__(self, settings: AutoencoderT64Config) -> None:
        self.settings = settings
        self.run_dir = Path(settings.runs_dir) / settings.run_id
        self.store = WindowStore(settings.data_dir, settings.cache_dir)

    def fit_val(self, train: WindowSet) -> tuple[WindowSet, WindowSet]:
        """(fit, val) window sets: `VehicleSplit(val_frac, split_seed)` over the sender vehicles of all windows."""
        s = self.settings
        fit_rows, val_rows = VehicleSplit(s.val_frac, s.split_seed).split(train.vehicles)
        fit, val = train.take(fit_rows), train.take(val_rows)
        shared = set(fit.vehicles) & set(val.vehicles)
        if shared:
            raise ValueError(f"{len(shared)} sender vehicles are in both fit and val")
        return fit, val

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
        train = self.store.open(TRAIN_SPLIT)
        fit, val = self.fit_val(train)
        np.save(self.run_dir / "val_vehicles.npy", np.unique(val.vehicles).astype(str))
        log(
            f"train windows {len(train):,} ({time.time() - t0:.0f}s to open), shape {fit.x.shape[1:]} + mask; "
            f"fit {len(fit):,} / val {len(val):,} windows; vehicles {len(set(fit.vehicles)):,} / "
            f"{len(set(val.vehicles)):,}; val = {s.val_frac:g} of the train vehicles (split seed {s.split_seed})"
        )
        log(
            f"padding share fit {1 - fit.mask.mean():.3f} / val {1 - val.mask.mean():.3f}, "
            f"real messages per window median {np.median(fit.n_messages):.0f}"
        )
        model = ConvAutoencoder(s.architecture())
        params = count_params(model)
        log(f"ConvAutoencoder (task/encoder): {params:,} trainable parameters, input 14 x 64, z {s.latent_dim}")
        trainer = Trainer(model, s, device)
        result = trainer.fit((fit.x, fit.mask), (val.x, val.mask), self.run_dir)

        maha = Mahalanobis(s.maha_ridge).fit(trainer.score(fit.x, fit.mask)["z"])
        np.savez(self.run_dir / "mahalanobis.npz", mean=maha.mean, precision=maha.precision)
        val_out = trainer.score(val.x, val.mask)
        val_scores = {"reconstruction": val_out["error"], "mahalanobis": maha.score(val_out["z"]).astype(np.float32)}
        ScoreFiles(self.run_dir, "val").save(
            ids=val.ids,
            n_messages=val.n_messages,
            recon_error=val_scores["reconstruction"],
            recon_error_feat=val_out["error_feature"],
            mahalanobis=val_scores["mahalanobis"],
        )
        label_reads = train.label_reads + fit.label_reads + val.label_reads
        if label_reads:
            raise RuntimeError(f"training read labels: {label_reads}")
        summary = {
            "run_id": s.run_id,
            "params": params,
            **result,
            "train_windows": len(train),
            "fit_windows": len(fit),
            "val_windows": len(val),
            "fit_vehicles": len(set(fit.vehicles)),
            "val_vehicles": len(set(val.vehicles)),
            "mahalanobis_fitted_on": "z of the fit windows (train minus the validation vehicles), both classes",
            "thresholds": ValidationThresholds(s.threshold_quantiles).compute(val.ids, val_scores),
            "splits_opened": [TRAIN_SPLIT],
            "label_reads": label_reads,
            "wall_seconds": round(time.time() - t0, 1),
            "device": str(device),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        log(f"done: {json.dumps({k: v for k, v in summary.items() if k != 'lr_changes'})}")
        return summary


def parse_args(argv: list[str] | None = None) -> AutoencoderT64Config:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None, help="max epochs (default from the config)")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--cache-dir", default=None)
    a = p.parse_args(argv)
    changes = {"seed": a.seed, "device": a.device}
    for field, value in (
        ("max_epochs", a.epochs),
        ("data_dir", a.data_dir),
        ("runs_dir", a.runs_dir),
        ("cache_dir", a.cache_dir),
    ):
        if value is not None:
            changes[field] = value
    return dataclasses.replace(AutoencoderT64Config(), **changes)


def main(argv: list[str] | None = None) -> None:
    Experiment(parse_args(argv)).run()


if __name__ == "__main__":
    main()
