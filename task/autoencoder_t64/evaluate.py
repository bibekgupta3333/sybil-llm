"""Final evaluation of a finished run on the test split; writes `test_<name>.npy` and `results.json`.

Order: every test window is scored from x + mask (reconstruction error, Mahalanobis distance of z; thresholds and
the Gaussian come from the run folder, fitted on train / validation without labels) and the score files are written;
only then are the test labels read. GridSybil = positive; an alarm is `score >= threshold` (the `task/encoder`
convention). The train labels are read here once, to describe the training set (GridSybil share); the training
code never reads them.

Length alone uses the fixed rule score = -n_messages (shorter = more anomalous), decided before looking at any
label; `task/encoder` reports the same direction but picked it on the test labels, and inside length strata it
reports the better of the two directions.

Usage (repo root):
    .venv-train/bin/python -m task.autoencoder_t64.evaluate --run task/autoencoder_t64/runs/seed0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from task.autoencoder_t64.config import AutoencoderT64Config
from task.autoencoder_t64.scoring import Mahalanobis, ScoreFiles, TrackScores, track_ids
from task.autoencoder_t64.trainer import Trainer
from task.encoder.data import VehicleSplit, WindowStore
from task.encoder.evaluate import FEATURES

SHORT_LONG_STRATA = (("1-3 rows", 1, 3), (">=4 rows", 4, 64))
ENCODER_STRATA = (("n<=3", 1, 3), ("4-16", 4, 16), ("17-48", 17, 48), ("49-64", 49, 64))
ENCODER_RESULTS = Path("task/encoder/runs/evaluation/results.json")
SCORES = {"reconstruction": "recon_error", "mahalanobis": "mahalanobis"}


class Metrics:
    """Binary detection metrics with GridSybil (1) as the positive class."""

    @staticmethod
    def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
        """AUROC, or None when only one class is present."""
        return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None

    @staticmethod
    def pr_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
        """Average precision, or None when only one class is present."""
        return float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else None

    @staticmethod
    def at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
        """Precision / recall / F1 / benign FPR + confusion counts for `score >= threshold` = alarm."""
        pred = scores >= threshold
        tp, fp = int((pred & (labels == 1)).sum()), int((pred & (labels == 0)).sum())
        fn, tn = int((~pred & (labels == 1)).sum()), int((~pred & (labels == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "benign_fpr": fp / (fp + tn) if fp + tn else 0.0,
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }

    @classmethod
    def full(cls, labels: np.ndarray, scores: np.ndarray, thresholds: dict[str, float]) -> dict:
        """AUROC, PR-AUC and the threshold metrics for every named threshold (e.g. val p95 / p99)."""
        return {
            "auroc": cls.auroc(labels, scores),
            "pr_auc": cls.pr_auc(labels, scores),
            "thresholds": {f"val_{q}": cls.at_threshold(labels, scores, t) for q, t in thresholds.items()},
        }


class Evaluation:
    """Scores the test split with a finished run, then reads the labels and writes `results.json`.

    Attributes:
        run_dir: the run folder written by `task.autoencoder_t64.train`.
        settings: the run's config.
    """

    def __init__(self, run_dir: str | Path, device: str = "auto") -> None:
        self.run_dir = Path(run_dir)
        self.summary = json.loads((self.run_dir / "summary.json").read_text())
        self.trainer = Trainer.from_checkpoint(self.run_dir / "best.pt", device)
        self.settings: AutoencoderT64Config = self.trainer.settings
        self.store = WindowStore(self.settings.data_dir, self.settings.cache_dir)

    def score_test(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Writes the test score files from x + mask, then returns them with the labels (read last)."""
        test = self.store.open("test")
        state = np.load(self.run_dir / "mahalanobis.npz")
        maha = Mahalanobis(self.settings.maha_ridge)
        maha.mean, maha.precision = state["mean"], state["precision"]
        out = self.trainer.score(test.x, test.mask)
        files = ScoreFiles(self.run_dir, "test")
        files.save(
            ids=test.ids,
            n_messages=test.n_messages,
            recon_error=out["error"],
            recon_error_feat=out["error_feature"],
            mahalanobis=maha.score(out["z"]).astype(np.float32),
        )
        if test.label_reads:
            raise RuntimeError("test labels were read before the scores were written")
        labels = np.asarray(test.labels("test: evaluation labels, read after every test score is on disk"))
        return files.load(), labels.astype(np.int64)

    def training_set(self) -> dict:
        """Class counts of the train / fit / val windows (labels read for description only) + a val alignment check."""
        s = self.settings
        train = self.store.open("train")
        labels = np.asarray(train.labels("train: class shares for the report (not used by training)")).astype(int)
        fit_rows, val_rows = VehicleSplit(s.val_frac, s.split_seed).split(train.vehicles)
        val_ids = np.load(ScoreFiles(self.run_dir, "val").path("ids"))
        if not np.array_equal(np.asarray(train.ids[np.sort(val_rows)]).astype(str), val_ids):
            raise ValueError("validation window ids of the run differ from the split rule")

        def counts(rows: np.ndarray) -> dict:
            y = labels[rows]
            return {
                "windows": int(len(y)),
                "benign": int((y == 0).sum()),
                "gridsybil": int(y.sum()),
                "gridsybil_share": float(y.mean()),
            }

        return {"train": counts(np.arange(len(labels))), "fit": counts(fit_rows), "val": counts(val_rows)}

    @staticmethod
    def strata(labels: np.ndarray, scores: dict[str, np.ndarray], n: np.ndarray, bins: tuple) -> dict:
        """AUROC of each score (and of -n_messages) inside each length stratum."""
        out = {}
        for name, lo, hi in bins:
            sel = (n >= lo) & (n <= hi)
            y = labels[sel]
            out[name] = {
                "windows": int(sel.sum()),
                "benign": int((y == 0).sum()),
                "gridsybil": int(y.sum()),
                **{f"auroc_{k}": Metrics.auroc(y, v[sel]) for k, v in scores.items()},
                "auroc_length_minus_n": Metrics.auroc(y, -n[sel].astype(np.float64)),
            }
        return out

    def tracks(self, ids: np.ndarray, labels: np.ndarray, scores: dict[str, np.ndarray]) -> dict:
        """Per-track (one link) metrics for the max and mean of each window score; thresholds from the val tracks."""
        tracks = TrackScores(track_ids(ids))
        y, mixed = tracks.track_labels(labels)
        out = {
            "tracks": len(tracks),
            "benign": int((y == 0).sum()),
            "gridsybil": int(y.sum()),
            "mixed_label_tracks": mixed,
            "share_single_window": float((tracks.n_windows == 1).mean()),
        }
        for name, values in scores.items():
            for how, agg in tracks.aggregate(values).items():
                out[f"{name}_{how}"] = Metrics.full(y, agg, self.summary["thresholds"]["track"][f"{name}_{how}"])
        return out

    @staticmethod
    def encoder_context() -> dict | None:
        """The T = 64 benign-only autoencoder of `task/encoder` (its stored results; context only)."""
        if not ENCODER_RESULTS.is_file():
            return None
        stored = json.loads(ENCODER_RESULTS.read_text())
        out = {"source": str(ENCODER_RESULTS), "length_only": stored["baseline_summary"]["length"]}
        for key in ("ae:recon_error", "ae:mahalanobis"):
            method = stored["methods"][key]
            out[key] = {
                "seed0": next(r for r in method["per_seed"] if r["seed"] == 0),
                "mean_over_seeds": {
                    "auroc": method["summary"]["auroc"],
                    "pr_auc": method["summary"]["pr_auc"],
                    "strata_auroc": method["summary"]["strata_auroc"],
                },
            }
        out["strata_length_only"] = stored["strata"]
        return out

    def run(self) -> dict:
        """Scores test, computes every metric, writes `results.json` and returns it."""
        test, labels = self.score_test()
        n = test["n_messages"].astype(np.int64)
        scores = {name: test[key].astype(np.float64) for name, key in SCORES.items()}
        thresholds = self.summary["thresholds"]["window"]
        val = ScoreFiles(self.run_dir, "val").load()
        length = -n.astype(np.float64)
        feat = test["recon_error_feat"]
        results = {
            "run": self.run_dir.name,
            "note": "one seed, side experiment; GridSybil = positive; alarm = score >= threshold",
            "training": {
                k: self.summary[k]
                for k in ("params", "best_epoch", "best_val_loss", "epochs_run", "stop_reason", "wall_seconds")
            }
            | {"lr_halvings": len(self.summary["lr_changes"])},
            "training_set": self.training_set(),
            "test": {
                "windows": int(len(labels)),
                "benign": int((labels == 0).sum()),
                "gridsybil": int(labels.sum()),
                "gridsybil_share": float(labels.mean()),
            },
            "window": {name: Metrics.full(labels, s, thresholds[name]) for name, s in scores.items()},
            "val_alarm_rate": {
                name: {q: float((val[key] >= t).mean()) for q, t in thresholds[name].items()}
                for name, key in SCORES.items()
            },
            "length_only": {
                "rule": "score = -n_messages (shorter = more anomalous), fixed before reading labels",
                "auroc": Metrics.auroc(labels, length),
                "pr_auc": Metrics.pr_auc(labels, length),
            },
            "strata_short_long": self.strata(labels, scores, n, SHORT_LONG_STRATA),
            "strata_encoder_bins": self.strata(labels, scores, n, ENCODER_STRATA),
            "track": self.tracks(test["ids"], labels, scores),
            "per_feature": {
                f: {
                    "benign_mean": float(feat[labels == 0, j].mean()),
                    "gridsybil_mean": float(feat[labels == 1, j].mean()),
                    "auroc": Metrics.auroc(labels, feat[:, j]),
                }
                for j, f in enumerate(FEATURES)
            },
            "context_task_encoder_benign_only_T64": self.encoder_context(),
        }
        (self.run_dir / "results.json").write_text(json.dumps(results, indent=2))
        return results


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", default=f"{AutoencoderT64Config().runs_dir}/seed0")
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    r = Evaluation(a.run, a.device).run()
    print(json.dumps({k: r[k] for k in ("training", "test", "window", "length_only", "strata_short_long")}, indent=2))
    print(f"wrote {Path(a.run) / 'results.json'}")


if __name__ == "__main__":
    main()
