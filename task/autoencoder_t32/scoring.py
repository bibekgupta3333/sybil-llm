"""Scores and their evaluation.

Label-free part (used by `train.py`): the Mahalanobis model of z, per-track aggregation and quantile thresholds of
`task/autoencoder_benign_t32/scoring.py`, plus `ScoreFiles` (per-window score arrays on disk, aligned with the ids).

Labelled part (used by `evaluate.py` and the notebook after every score exists): `Evaluation`, which turns window
scores + labels into the result tables. Every score is "higher = more anomalous"; an alarm is score > threshold; the
positive class is GridSybil. Thresholds always come from the unlabelled val split and are passed in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from task.autoencoder_benign_t32.scoring import Mahalanobis, TrackScores, quantile_thresholds  # noqa: F401

SCORES = ("reconstruction", "mahalanobis")
SHORT_MAX_ROWS = 3  # windows of 1-3 real rows = "short" (F11)


class ScoreFiles:
    """Per-window arrays of one split under `<run_dir>/scores/`: `<split>_<name>.npy`, all in the order of
    `<split>_ids.npy`."""

    NAMES = ("ids", "reconstruction", "mahalanobis", "n_messages", "z")

    def __init__(self, run_dir: str | Path) -> None:
        self.folder = Path(run_dir) / "scores"

    def save(self, split: str, ids: np.ndarray, error: np.ndarray, maha: np.ndarray, n: np.ndarray, z: np.ndarray):
        self.folder.mkdir(parents=True, exist_ok=True)
        arrays = (ids, error.astype(np.float32), maha.astype(np.float32), n.astype(np.int32), z.astype(np.float32))
        if len({len(a) for a in arrays}) != 1:
            raise ValueError(f"{split}: score arrays differ in length")
        for name, array in zip(self.NAMES, arrays):
            np.save(self.folder / f"{split}_{name}.npy", array)

    def load(self, split: str) -> dict[str, np.ndarray]:
        return {name: np.load(self.folder / f"{split}_{name}.npy") for name in self.NAMES}


def ranking(y: np.ndarray, scores: np.ndarray) -> dict:
    """AUROC, PR-AUC, the positive share (PR-AUC of a random score) and the class counts; NaN with one class."""
    y = np.asarray(y).astype(int)
    out = {"n_benign": int((y == 0).sum()), "n_gridsybil": int(y.sum()), "chance": float(y.mean()) if len(y) else 0.0}
    both = out["n_benign"] > 0 and out["n_gridsybil"] > 0
    out["auroc"] = float(roc_auc_score(y, scores)) if both else float("nan")
    out["pr_auc"] = float(average_precision_score(y, scores)) if both else float("nan")
    return out


def at_threshold(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Recall (GridSybil), benign FPR and precision of the rule score > threshold."""
    y = np.asarray(y).astype(int)
    alarm = np.asarray(scores) > threshold
    tp, fp = int((alarm & (y == 1)).sum()), int((alarm & (y == 0)).sum())
    return {
        "threshold": float(threshold),
        "recall": tp / max(int(y.sum()), 1),
        "benign_fpr": fp / max(int((y == 0).sum()), 1),
        "precision": tp / max(tp + fp, 1),
        "alarm_share": float(alarm.mean()) if len(y) else 0.0,
    }


def detection(y: np.ndarray, scores: np.ndarray, thresholds: dict[str, float]) -> dict:
    """`ranking` + `at_threshold` for each named threshold (`p95`, `p99`)."""
    return {**ranking(y, scores), **{q: at_threshold(y, scores, t) for q, t in thresholds.items()}}


class Evaluation:
    """Result tables of one scored, labelled split.

    Attributes:
        scores: {"reconstruction": (N,), "mahalanobis": (N,)} window scores.
        y: (N,) 1 = GridSybil, 0 = benign.
        n_messages: (N,) real rows per window.
        tracks: (N,) track id per window.
        group: (N,) "0709" / "1416".
        scenario: (N,) source folder, e.g. "DoSRandomSybil_0709".
        thresholds: {"window": {score: {p95, p99}}, "track": {"<score>_<max|mean>": {p95, p99}}} from val.
    """

    def __init__(
        self,
        scores: dict[str, np.ndarray],
        y: np.ndarray,
        n_messages: np.ndarray,
        tracks: np.ndarray,
        group: np.ndarray,
        scenario: np.ndarray,
        thresholds: dict,
    ) -> None:
        self.scores = {k: np.asarray(v, dtype=np.float64) for k, v in scores.items()}
        self.y = np.asarray(y).astype(int)
        self.n_messages, self.tracks = np.asarray(n_messages), np.asarray(tracks)
        self.group, self.scenario = np.asarray(group), np.asarray(scenario)
        self.thresholds = thresholds

    def length_only(self, rows: np.ndarray) -> dict:
        """Window length alone. Fixed rule: score = -n_messages (shorter = more anomalous), set before any test label
        was read. `auroc_best_direction` = max(a, 1 - a) is the convention of `task/autoencoder_st_t32` (direction
        picked on the evaluated labels); reported for comparison only."""
        y, n = self.y[rows], self.n_messages[rows].astype(np.float64)
        fixed = ranking(y, -n)
        return {
            "rule": "score = -n_messages",
            "auroc": fixed["auroc"],
            "pr_auc": fixed["pr_auc"],
            "auroc_best_direction": max(fixed["auroc"], 1.0 - fixed["auroc"]),
            "best_direction": "shorter = anomalous" if fixed["auroc"] >= 0.5 else "longer = anomalous",
        }

    def window(self, name: str, rows: np.ndarray) -> dict:
        """One score per window: all rows, short (1-3 rows) and long (>= 4 rows) windows, per group."""
        y, s, n = self.y[rows], self.scores[name][rows], self.n_messages[rows]
        out = detection(y, s, self.thresholds["window"][name])
        short = n <= SHORT_MAX_ROWS
        out["rows_1_3"] = ranking(y[short], s[short])
        out["rows_ge_4"] = ranking(y[~short], s[~short])
        out["by_group"] = {
            g: detection(y[grp], s[grp], self.thresholds["window"][name])
            for g in sorted(set(self.group[rows].tolist()))
            for grp in [self.group[rows] == g]
        }
        return out

    def track(self, name: str, rows: np.ndarray) -> dict:
        """Per-track max / mean of the window scores, with the val track thresholds."""
        agg = TrackScores(self.tracks[rows])
        y, mixed = agg.track_labels(self.y[rows])
        out = {"tracks": len(agg), "mixed_label_tracks": mixed}
        for how, values in agg.aggregate(self.scores[name][rows]).items():
            out[how] = detection(y, values, self.thresholds["track"][f"{name}_{how}"])
        return out

    def benign_fpr_by_folder(self, name: str, rows: np.ndarray) -> dict:
        """Share of benign windows above the val p95 / p99 window thresholds, per source folder."""
        benign = rows & (self.y == 0)
        out = {}
        for folder in sorted(set(self.scenario[benign].tolist())):
            s = self.scores[name][benign & (self.scenario == folder)]
            out[folder] = {"n_benign": len(s)}
            out[folder].update({q: float((s > t).mean()) for q, t in self.thresholds["window"][name].items()})
        return out

    def report(self, rows: np.ndarray | None = None) -> dict:
        """Everything for the windows selected by the boolean `rows` (all windows when None)."""
        rows = np.ones(len(self.y), dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
        y = self.y[rows]
        return {
            "windows": int(rows.sum()),
            "benign": int((y == 0).sum()),
            "gridsybil": int(y.sum()),
            "gridsybil_share": float(y.mean()),
            "share_rows_1_3": float((self.n_messages[rows] <= SHORT_MAX_ROWS).mean()),
            "window": {name: self.window(name, rows) for name in self.scores},
            "track": {name: self.track(name, rows) for name in self.scores},
            "benign_fpr_by_folder": {name: self.benign_fpr_by_folder(name, rows) for name in self.scores},
            "length_only": self.length_only(rows),
        }
