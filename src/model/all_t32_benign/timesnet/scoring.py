"""Label-free anomaly scores on frozen TimesNet embeddings and the one-class detection metrics (WBS S1.4).

Implements the scoring half of the one-class evaluation in `src/model/all_t32_benign/PLAN.md`. Everything that is
fitted here is fitted on unlabeled windows only: `Mahalanobis` and `KNNScorer` on benign train z, thresholds on benign
val scores. Labels enter only through `DetectionReport`, which `evaluate.py` calls after every score file is written.

Protocol and metric definitions mirror `task/autoencoder_benign_t32/` (same split, same test windows), so the numbers
are directly comparable: Mahalanobis with the same shrinkage rule, per-track max / mean (track = window id without
the crop index), thresholds at benign val p95 / p99 with alarms at `score > threshold`, and a length-only reference
whose direction is fixed in advance (score = -n_real_rows: shorter windows are more suspicious, F11), never chosen on
test labels.

kNN (fixed before any result): cosine distance on L2-normalised z, score = 1 - mean cosine similarity of the K
nearest bank windows, K in {5, 10, 20} (the adopted plan, `docs/plan/clarification.md` section 5). Cosine rather than
L2 because z is a masked mean over the real rows, so its norm moves with the number of rows averaged; L2 would mix
window length into the score, which the length-only reference is there to measure separately.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

SHORT_MAX_ROWS = 3  # windows with <= 3 real rows (F11: DataReplay / DoS are 95-100% such windows)
MIN_RECON_ROWS = 2  # fewer real rows: nothing can be hidden, the reconstruction error is filled (not measured)


class Mahalanobis:
    """Gaussian model of z: score = sqrt((z - mean)^T C^-1 (z - mean)) (as `task/autoencoder_benign_t32`).

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
        """Fits mean and (shrunk if ill-conditioned) precision on z (N, d) of benign train windows."""
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

    def score(self, z: np.ndarray, chunk: int = 65536) -> np.ndarray:
        """(N, d) -> (N,) Mahalanobis distances (float64)."""
        if self.mean is None:
            raise RuntimeError("Mahalanobis.fit() first")
        out = np.empty(len(z), dtype=np.float64)
        for start in range(0, len(z), chunk):
            dz = np.asarray(z[start : start + chunk], dtype=np.float64) - self.mean
            out[start : start + chunk] = np.sqrt(np.maximum(np.einsum("nd,de,ne->n", dz, self.precision, dz), 0.0))
        return out

    def describe(self) -> dict[str, float]:
        """Fit diagnostics for the summary."""
        return {"cond": self.cond, "shrinkage": self.shrinkage, "max_cond": self.max_cond}


class KNNScorer:
    """Memory bank of benign train z; score = 1 - mean cosine similarity of the K nearest bank windows.

    The bank is a deterministic subsample (seeded) of at most `max_bank` train windows. Brute force in chunks of
    queries, so memory stays at `chunk` x bank floats.

    Attributes:
        k_values: the K reported (sorted).
        bank: (M, d) float32 L2-normalised bank.
        bank_rows: (M,) train rows in the bank (sorted).
    """

    METRIC = "cosine"

    def __init__(self, k_values: Sequence[int] = (5, 10, 20), max_bank: int = 200_000, seed: int = 0) -> None:
        if not k_values or min(k_values) < 1:
            raise ValueError(f"k_values must be positive, got {k_values}")
        if max_bank < max(k_values):
            raise ValueError(f"max_bank {max_bank} is smaller than the largest K {max(k_values)}")
        self.k_values = tuple(sorted(int(k) for k in k_values))
        self.max_bank, self.seed = max_bank, seed
        self.bank: np.ndarray | None = None
        self.bank_rows: np.ndarray | None = None

    @staticmethod
    def _unit(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float32)
        return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)

    def fit(self, z_train: np.ndarray) -> "KNNScorer":
        """Builds the bank from (N, d) train z; subsampled without replacement by `seed` if N > max_bank."""
        n = len(z_train)
        if n < max(self.k_values):
            raise ValueError(f"{n} train windows, fewer than the largest K {max(self.k_values)}")
        rows = np.arange(n)
        if n > self.max_bank:
            rows = np.sort(np.random.default_rng(self.seed).choice(n, size=self.max_bank, replace=False))
        self.bank_rows = rows
        self.bank = self._unit(np.asarray(z_train)[rows])
        return self

    def score(self, z: np.ndarray, chunk: int = 1024) -> dict[int, np.ndarray]:
        """(N, d) -> {K: (N,) float64 score} with score = 1 - mean cosine similarity of the K nearest."""
        if self.bank is None:
            raise RuntimeError("KNNScorer.fit() first")
        k_max = self.k_values[-1]
        out = {k: np.empty(len(z), dtype=np.float64) for k in self.k_values}
        for start in range(0, len(z), chunk):
            sim = self._unit(z[start : start + chunk]) @ self.bank.T  # (c, M)
            top = np.partition(sim, sim.shape[1] - k_max, axis=1)[:, -k_max:]
            top = -np.sort(-top, axis=1).astype(np.float64)  # most similar first
            for k in self.k_values:
                out[k][start : start + chunk] = 1.0 - top[:, :k].mean(axis=1)
        return out

    def describe(self) -> dict[str, Any]:
        """Bank diagnostics for the summary."""
        return {
            "metric": self.METRIC,
            "score": "1 - mean cosine similarity of the K nearest bank windows",
            "k_values": list(self.k_values),
            "bank_size": None if self.bank is None else int(len(self.bank)),
            "max_bank": self.max_bank,
            "bank_seed": self.seed,
        }


class TrackScores:
    """Window scores aggregated per track (one link): max and mean of its windows.

    Attributes:
        tracks: (K,) sorted unique track ids.
        index: (N,) position of each window's track in `tracks`.
        n_windows: (K,) windows per track.
    """

    def __init__(self, window_ids: Sequence[str]) -> None:
        self.tracks, self.index, self.n_windows = np.unique(
            np.asarray([track_of(i) for i in window_ids]), return_inverse=True, return_counts=True
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


def track_of(window_id: str) -> str:
    """`<receiver_file>#<pseudo>-<sender>#<k>` -> `<receiver_file>#<pseudo>-<sender>` (the link)."""
    return window_id.rsplit("#", 1)[0]


def quantile_thresholds(scores: np.ndarray, quantiles: Sequence[float]) -> dict[str, float]:
    """{"p95": ..., "p99": ...} of unlabeled (benign val) scores."""
    return {f"p{round(q * 100):d}": float(np.quantile(np.asarray(scores, dtype=np.float64), q)) for q in quantiles}


def length_score(n_real_rows: np.ndarray) -> np.ndarray:
    """Length-only reference score, direction fixed in advance: -n_real_rows (shorter = more suspicious)."""
    return -np.asarray(n_real_rows, dtype=np.float64)


def _share(flags: np.ndarray) -> float | None:
    """Share of True in `flags` (None if empty)."""
    return float(np.mean(flags)) if len(flags) else None


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


@dataclasses.dataclass(frozen=True)
class BinaryMetrics:
    """Detection metrics of one score on benign (label 0) vs attack (label 1) windows or tracks.

    Alarms are `score > threshold` (as the baseline). AUROC / PR-AUC are None when a class is empty.

    Attributes:
        n_benign: benign items.
        n_attack: attack items.
        chance: attack share (= PR-AUC of a random score).
        auroc: area under the ROC curve.
        pr_auc: average precision.
        at_threshold: {"p95": {"threshold", "recall", "benign_fpr"}, ...}.
    """

    n_benign: int
    n_attack: int
    chance: float | None
    auroc: float | None
    pr_auc: float | None
    at_threshold: dict[str, dict[str, float | None]]

    @classmethod
    def compute(
        cls, benign: np.ndarray, attack: np.ndarray, thresholds: Mapping[str, float] | None = None
    ) -> "BinaryMetrics":
        """Metrics of `benign` (Nb,) vs `attack` (Na,) scores; higher = more anomalous."""
        benign, attack = np.asarray(benign, dtype=np.float64), np.asarray(attack, dtype=np.float64)
        y = np.r_[np.zeros(len(benign), dtype=int), np.ones(len(attack), dtype=int)]
        s = np.r_[benign, attack]
        both = len(benign) > 0 and len(attack) > 0
        auroc = float(roc_auc_score(y, s)) if both else None
        pr_auc = float(average_precision_score(y, s)) if both else None
        at = {}
        for name, th in (thresholds or {}).items():
            at[name] = {
                "threshold": float(th),
                "recall": _finite_or_none((attack > th).mean()) if len(attack) else None,
                "benign_fpr": _finite_or_none((benign > th).mean()) if len(benign) else None,
            }
        chance = float(y.mean()) if len(y) else None
        return cls(len(benign), len(attack), chance, auroc, pr_auc, at)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready dict."""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ScoredSplit:
    """The saved window scores of one split (no labels).

    Attributes:
        ids: (N,) window ids.
        n_real_rows: (N,) real rows per window.
        scores: {score name: (N,) float64}, higher = more anomalous.
    """

    ids: np.ndarray
    n_real_rows: np.ndarray
    scores: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.ids)


class DetectionReport:
    """Every metric of every score for one attack split vs the benign test split.

    Window level, every score (recon, Mahalanobis, kNN, length-only) on the same subsets with the same val thresholds:
    all windows, n >= 2 real rows (the windows whose reconstruction error is measured, not filled), n <= 3 and
    n >= 4. Track level: per-track max / mean with thresholds from the benign val tracks. The "all" reconstruction
    numbers depend on the fill value of the < 2-row windows; `counts["recon_filled_share"]` says how many.
    """

    LEVELS = ("max", "mean")

    def __init__(self, val: ScoredSplit, quantiles: Sequence[float]) -> None:
        """Fits window and track thresholds on the benign val scores (label-free)."""
        self.quantiles = tuple(quantiles)
        self.val_tracks = TrackScores(val.ids)
        self.thresholds: dict[str, dict[str, dict[str, float]]] = {"window": {}, "track_max": {}, "track_mean": {}}
        for name, s in val.scores.items():
            self.thresholds["window"][name] = quantile_thresholds(s, self.quantiles)
            agg = self.val_tracks.aggregate(s)
            for level in self.LEVELS:
                self.thresholds[f"track_{level}"][name] = quantile_thresholds(agg[level], self.quantiles)
        self.thresholds["window"]["length_only"] = quantile_thresholds(length_score(val.n_real_rows), self.quantiles)

    def compare(
        self, benign: ScoredSplit, attack: ScoredSplit, benign_labels: np.ndarray, attack_labels: np.ndarray
    ) -> dict[str, Any]:
        """Metrics of `attack` vs `benign` (labels only to aggregate tracks and count mixed tracks)."""
        out: dict[str, Any] = {"window": {}, "track": {}, "counts": {}}
        subsets = {
            "all": (np.ones(len(benign), bool), np.ones(len(attack), bool)),
            f"n_ge_{MIN_RECON_ROWS}": (benign.n_real_rows >= MIN_RECON_ROWS, attack.n_real_rows >= MIN_RECON_ROWS),
            f"n_le_{SHORT_MAX_ROWS}": (benign.n_real_rows <= SHORT_MAX_ROWS, attack.n_real_rows <= SHORT_MAX_ROWS),
            f"n_ge_{SHORT_MAX_ROWS + 1}": (benign.n_real_rows > SHORT_MAX_ROWS, attack.n_real_rows > SHORT_MAX_ROWS),
        }
        named = dict(benign.scores)
        named["length_only"] = length_score(benign.n_real_rows)
        attack_named = dict(attack.scores)
        attack_named["length_only"] = length_score(attack.n_real_rows)
        for name in named:
            th = self.thresholds["window"][name]
            out["window"][name] = {
                sub: BinaryMetrics.compute(named[name][b], attack_named[name][a], th).as_dict()
                for sub, (b, a) in subsets.items()
            }
        for sub, (b, a) in subsets.items():
            out["counts"][sub] = {"benign_windows": int(b.sum()), "attack_windows": int(a.sum())}
        out["counts"]["recon_filled_share"] = {
            "benign": _share(benign.n_real_rows < MIN_RECON_ROWS),
            "attack": _share(attack.n_real_rows < MIN_RECON_ROWS),
        }
        b_tracks, a_tracks = TrackScores(benign.ids), TrackScores(attack.ids)
        _, b_mixed = b_tracks.track_labels(benign_labels)
        _, a_mixed = a_tracks.track_labels(attack_labels)
        out["counts"]["tracks"] = {
            "benign": len(b_tracks),
            "attack": len(a_tracks),
            "mixed_label_tracks": b_mixed + a_mixed,
            "shared_track_ids": len(set(b_tracks.tracks) & set(a_tracks.tracks)),
        }
        for name in benign.scores:
            b_agg, a_agg = b_tracks.aggregate(benign.scores[name]), a_tracks.aggregate(attack.scores[name])
            out["track"][name] = {
                level: BinaryMetrics.compute(
                    b_agg[level], a_agg[level], self.thresholds[f"track_{level}"][name]
                ).as_dict()
                for level in self.LEVELS
            }
        return out
