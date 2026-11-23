"""Trains the one-class masked autoencoder (widedeep encoder-decoder pretraining, TabMlp or TabResnet) at T = 64 / 48 / 32.

Only benign training windows fit the weights; benign validation windows drive early stopping, the LR schedule and the
alarm thresholds; no test split is opened. After training, a Mahalanobis model of z is fitted on the train windows and
the benign val p95 / p99 thresholds of the three scores (masked_recon, full_recon, mahalanobis) are stored per window
and per track (max / mean of a link's windows).

Writes `task/masked_autoencoder/runs/<encoder>/T<w>-seed<k>/` (`config.run_path`): config.json, env.json, history.jsonl, train.log, best.pt,
mahalanobis.npz, summary.json, loss_curve.png (and, with --evaluate, the evaluate.py outputs).

Usage (repo root):
    .venv-train/bin/python -m task.masked_autoencoder.train --window 32 --encoder mlp --seed 0 --evaluate
    .venv-train/bin/python -m task.masked_autoencoder.train --window 64 --encoder resnet --seed 0
    .venv-train/bin/python -m task.masked_autoencoder.train --window 48 --seed 1 --epochs 2 --max-windows 4000
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from task.autoencoder_benign_t32.train import environment
from task.masked_autoencoder.config import ENCODERS, PROTOCOLS, MAEConfig
from task.masked_autoencoder.data import OneClassSource, Windows, make_source
from task.masked_autoencoder.model import MaskedTabularAutoencoder, count_params
from task.masked_autoencoder.plots import LossCurvePlot
from task.masked_autoencoder.scoring import SCORE_NAMES, Mahalanobis, TrackScores, quantile_thresholds
from task.masked_autoencoder.trainer import MaskedTrainer, RunLog, pick_device, seed_everything


class Experiment:
    """One run: benign train / val, fit, Mahalanobis on train z, val thresholds."""

    def __init__(self, settings: MAEConfig, source: OneClassSource | None = None) -> None:
        self.settings = settings
        self.source = source or make_source(settings)
        self.run_dir = settings.run_dir

    def run(self) -> dict:
        s = self.settings
        if (self.run_dir / "summary.json").exists():
            raise FileExistsError(f"{self.run_dir} already holds a finished run; remove it or pick another seed")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("history.jsonl", "train.log", "best.pt"):
            (self.run_dir / name).unlink(missing_ok=True)
        log = RunLog(self.run_dir / "train.log")
        seed_everything(s.seed)
        device = pick_device(s.device)
        (self.run_dir / "config.json").write_text(json.dumps(s.to_dict(), indent=2))
        (self.run_dir / "env.json").write_text(json.dumps(environment(device), indent=2))

        t0 = time.time()
        train, val = self.source.fit_val()
        self.check_inputs(train, val)
        val_tracks = TrackScores(val.tracks)
        log(
            f"T = {s.window} ({s.protocol.sibling} protocol): train {len(train):,} / val {len(val):,} benign windows "
            f"({time.time() - t0:.0f}s to load), vehicles {len(set(train.vehicles)):,} / {len(set(val.vehicles)):,}, "
            f"padding share {1 - train.mask.mean():.3f} / {1 - val.mask.mean():.3f}"
        )
        seed_everything(s.seed)  # weights do not depend on how much RNG the loading used
        model = MaskedTabularAutoencoder(s)
        params = count_params(model)
        log(
            f"{s.encoder} masked autoencoder: {params:,} trainable parameters, input {s.seq_len} x {s.n_features} "
            f"+ {s.seq_len} mask columns, z {model.latent_dim}"
        )
        trainer = MaskedTrainer(model, s, device)
        result = trainer.fit((train.x, train.mask), (val.x, val.mask), self.run_dir)
        scoring = self.fit_scoring(trainer, train, val, val_tracks, log)
        summary = {
            "run_id": s.run_id,
            "window": s.window,
            "encoder": s.encoder,
            "seed": s.seed,
            "params": params,
            "latent_dim": model.latent_dim,
            **result,
            "train_windows": len(train),
            "val_windows": len(val),
            "train_vehicles": len(set(train.vehicles)),
            "val_vehicles": len(set(val.vehicles)),
            "val_tracks": len(val_tracks),
            **scoring,
            "label_reads": list(self.source.label_reads),
            "wall_seconds": round(time.time() - t0, 1),
            "device": str(device),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        log(f"loss curve: {LossCurvePlot(self.run_dir).save()}")
        log(
            f"done: best epoch {result['best_epoch']} val masked MSE {result['best_val_loss']:.5f}, "
            f"{result['epochs_run']} epochs, {result['stop_reason']}, {summary['wall_seconds']:.0f}s"
        )
        return summary

    @staticmethod
    def check_inputs(train: Windows, val: Windows) -> None:
        """No sender vehicle in both train and val (labels are not read here)."""
        shared = set(train.vehicles) & set(val.vehicles)
        if shared:
            raise ValueError(f"{len(shared)} sender vehicles are in both train and val")

    def fit_scoring(
        self, trainer: MaskedTrainer, train: Windows, val: Windows, val_tracks: TrackScores, log: RunLog
    ) -> dict:
        """Mahalanobis on train z; val p95 / p99 thresholds of each score per window and per track."""
        s = self.settings
        z_train = trainer.scorer(train.x, train.mask)["z"]
        maha = Mahalanobis(s.mahalanobis_max_cond, s.mahalanobis_shrinkage).fit(z_train)
        np.savez(self.run_dir / "mahalanobis.npz", **maha.state())
        log(f"mahalanobis on train z: condition number {maha.cond:.3g}, shrinkage {maha.shrinkage:g}")
        out = trainer.scorer(val.x, val.mask)
        window = {
            "masked_recon": out["masked_recon"],
            "full_recon": out["full_recon"],
            "mahalanobis": maha.score(out["z"]),
        }
        thresholds = {"window": {}, "track": {}}
        for name in SCORE_NAMES:
            thresholds["window"][name] = quantile_thresholds(window[name], s.threshold_quantiles)
            for how, agg in val_tracks.aggregate(window[name]).items():
                thresholds["track"][f"{name}_{how}"] = quantile_thresholds(agg, s.threshold_quantiles)
        log(f"val thresholds: {json.dumps(thresholds)}")
        return {
            "mahalanobis": {"cond": maha.cond, "shrinkage": maha.shrinkage},
            "thresholds": thresholds,
            "threshold_source": "benign val split",
            "z_dead_units_train": int((z_train.std(axis=0) == 0).sum()),
        }


def parse_args(argv: list[str] | None = None) -> tuple[MAEConfig, bool]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--window", type=int, choices=sorted(PROTOCOLS), required=True)
    p.add_argument("--encoder", choices=ENCODERS, default="mlp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None, help="max epochs (default from the config)")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--runs-dir", default=None, help="root of the run tree (default runs/; the run goes to <encoder>/T<w>-seed<k>)"
    )
    p.add_argument("--max-windows", type=int, default=0, help="cap per split (quick checks; run folder gets -check)")
    p.add_argument("--evaluate", action="store_true", help="run evaluate.py on the finished run")
    a = p.parse_args(argv)
    changes = {"encoder": a.encoder, "seed": a.seed, "device": a.device, "max_windows": a.max_windows}
    if a.epochs is not None:
        changes["max_epochs"] = a.epochs
    if a.runs_dir:
        changes["runs_dir"] = a.runs_dir
    return MAEConfig.for_window(a.window, **changes), a.evaluate


def main(argv: list[str] | None = None) -> None:
    settings, evaluate = parse_args(argv)
    experiment = Experiment(settings)
    experiment.run()
    if evaluate:
        from task.masked_autoencoder.evaluate import Evaluation

        Evaluation(experiment.run_dir, settings.device, source=experiment.source).run()


if __name__ == "__main__":
    main()
