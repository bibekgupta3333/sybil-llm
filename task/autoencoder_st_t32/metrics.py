"""Binary detection metrics, val-only alarm thresholds and per-track aggregation.

Scores are GridSybil probabilities (positive class = 1). An alarm is raised when score >= threshold.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ranking_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Threshold-free AUROC and PR-AUC (average precision)."""
    return {"auroc": float(roc_auc_score(y, scores)), "pr_auc": float(average_precision_score(y, scores))}


def threshold_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Precision, recall, F1, balanced accuracy, benign FPR and confusion counts at one threshold."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "benign_fpr": float(fp / max(fp + tn, 1)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def epoch_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """The per-epoch tracking set: AUROC, PR-AUC, F1 and balanced accuracy at 0.5."""
    at_half = threshold_metrics(y, scores, 0.5)
    return {**ranking_metrics(y, scores), "f1": at_half["f1"], "balanced_accuracy": at_half["balanced_accuracy"]}


def max_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Score threshold with the highest F1 (alarm when score >= threshold)."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    p, r = precision[:-1], recall[:-1]
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return float(thresholds[int(np.argmax(f1))])


def fpr_threshold(benign_scores: np.ndarray, target_fpr: float) -> float:
    """(1 - target_fpr) quantile of the benign scores: about `target_fpr` of benign windows raise an alarm."""
    return float(np.quantile(np.asarray(benign_scores, dtype=np.float64), 1.0 - target_fpr))


def val_thresholds(y: np.ndarray, scores: np.ndarray, target_fpr: float) -> dict[str, dict]:
    """{"max_f1": ..., "benign_fpr_<p>": ...}: each threshold with its metrics on the same (val) scores."""
    y = np.asarray(y).astype(int)
    picks = {
        "max_f1": max_f1_threshold(y, scores),
        f"benign_fpr_{round(target_fpr * 100):d}pct": fpr_threshold(np.asarray(scores)[y == 0], target_fpr),
    }
    return {name: threshold_metrics(y, scores, t) for name, t in picks.items()}


class TrackScores:
    """Window scores aggregated per track (one link): max and mean of its windows.

    Attributes:
        tracks: (K,) sorted unique track ids.
        index: (N,) position of each window's track in `tracks`.
        n_windows: (K,) windows per track.
    """

    def __init__(self, track_ids: np.ndarray) -> None:
        self.tracks, self.index, self.n_windows = np.unique(
            np.asarray(track_ids), return_inverse=True, return_counts=True
        )

    def __len__(self) -> int:
        return len(self.tracks)

    def aggregate(self, scores: np.ndarray) -> dict[str, np.ndarray]:
        """{"max": (K,), "mean": (K,)} of the window scores of each track."""
        scores = np.asarray(scores, dtype=np.float64)
        mx = np.full(len(self.tracks), -np.inf)
        np.maximum.at(mx, self.index, scores)
        mean = np.bincount(self.index, weights=scores, minlength=len(self.tracks)) / self.n_windows
        return {"max": mx, "mean": mean}

    def track_labels(self, labels: np.ndarray) -> tuple[np.ndarray, int]:
        """Majority label per track and the number of tracks whose windows disagree (expected 0)."""
        labels = np.asarray(labels)
        values, codes = np.unique(labels, return_inverse=True)
        counts = np.zeros((len(self.tracks), len(values)), dtype=np.int64)
        np.add.at(counts, (self.index, codes), 1)
        mixed = int(((counts > 0).sum(axis=1) > 1).sum())
        return values[counts.argmax(axis=1)], mixed
