"""Label-free anomaly scores on top of the autoencoder: Mahalanobis distance on z, per-track aggregation, thresholds.

Everything here is fitted on unlabeled windows (Mahalanobis on train z, thresholds on validation scores). Labels enter
only through `track_labels`, which the analysis calls after every score has been computed.
"""

from __future__ import annotations

import numpy as np


class Mahalanobis:
    """Gaussian model of z: score = sqrt((z - mean)^T C^-1 (z - mean)).

    The covariance is shrunk towards a scaled identity, C = (1 - a) S + a (tr S / d) I, when the condition number of
    the sample covariance S exceeds `max_cond`; the inverse is a Hermitian pseudo-inverse either way.

    Attributes:
        mean: (d,) mean of the fitted z.
        precision: (d, d) inverse covariance.
        cond: condition number of the sample covariance.
        shrinkage: shrinkage weight actually applied (0 if none).
    """

    def __init__(self, max_cond: float = 1e6, shrinkage: float = 0.01) -> None:
        self.max_cond = max_cond
        self.default_shrinkage = shrinkage
        self.mean: np.ndarray | None = None
        self.precision: np.ndarray | None = None
        self.cond = float("nan")
        self.shrinkage = 0.0

    def fit(self, z: np.ndarray) -> "Mahalanobis":
        z = np.asarray(z, dtype=np.float64)
        self.mean = z.mean(axis=0)
        cov = np.cov(z - self.mean, rowvar=False)
        self.cond = float(np.linalg.cond(cov))
        if not np.isfinite(self.cond) or self.cond > self.max_cond:
            self.shrinkage = self.default_shrinkage
            d = cov.shape[0]
            cov = (1.0 - self.shrinkage) * cov + self.shrinkage * np.trace(cov) / d * np.eye(d)
        self.precision = np.linalg.pinv(cov, hermitian=True)
        return self

    def score(self, z: np.ndarray) -> np.ndarray:
        """(N,) Mahalanobis distances."""
        if self.mean is None:
            raise RuntimeError("fit() first")
        dz = np.asarray(z, dtype=np.float64) - self.mean
        return np.sqrt(np.maximum(np.einsum("nd,de,ne->n", dz, self.precision, dz), 0.0))

    def state(self) -> dict:
        """Arrays for `np.savez`."""
        return {
            "mean": self.mean,
            "precision": self.precision,
            "cond": np.float64(self.cond),
            "shrinkage": np.float64(self.shrinkage),
        }

    @classmethod
    def from_state(cls, state) -> "Mahalanobis":
        m = cls()
        m.mean, m.precision = np.asarray(state["mean"]), np.asarray(state["precision"])
        m.cond, m.shrinkage = float(state["cond"]), float(state["shrinkage"])
        return m


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
        """Majority label per track and the number of tracks whose windows disagree (expected 0; analysis only)."""
        labels = np.asarray(labels)
        values, codes = np.unique(labels, return_inverse=True)
        counts = np.zeros((len(self.tracks), len(values)), dtype=np.int64)
        np.add.at(counts, (self.index, codes), 1)
        mixed = int(((counts > 0).sum(axis=1) > 1).sum())
        return values[counts.argmax(axis=1)], mixed


def quantile_thresholds(scores: np.ndarray, quantiles: tuple[float, ...]) -> dict[str, float]:
    """{"p95": ..., "p99": ...} of unlabeled scores."""
    return {f"p{round(q * 100):d}": float(np.quantile(scores, q)) for q in quantiles}
