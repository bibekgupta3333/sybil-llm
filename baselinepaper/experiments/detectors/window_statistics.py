"""Window statistics and the base class of the classical detectors that are fitted on them.

A window (32 rows x 13 features + prefix mask) is reduced to 65 numbers: per feature over its real rows, the mean,
the population standard deviation, the minimum, the maximum and last - first. Padding rows never enter. A one-row
window has std 0 and last - first 0, so the window length stays visible to every detector (protocol section 4;
length is controlled in the evaluation).

`StatisticsDetector` does what every classical detector shares: `self.standardise(x, mask)`, the statistics, a
`StandardScaler` fitted on the statistics of the whole training pool, and the seeded training subsample of the
methods that do not scale. A subclass implements `fit_model` and `score_model` on the scaled statistics.

Every value is fixed in `baselinepaper/05-protocol/hyperparameters-classical.md`.
"""

from __future__ import annotations

import abc
from typing import ClassVar

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

from baselinepaper.experiments.data import FeatureStatistics, LabelledWindowSet, WindowSet
from baselinepaper.experiments.detectors.base import Detector

STATISTIC_NAMES = ("mean", "std", "min", "max", "last_minus_first")
SUBSAMPLE_SIZE = 20_000
CHUNK_WINDOWS = 32_768


class WindowStatistics:
    """Reduces windows to per-feature statistics over their real rows.

    Attributes:
        chunk_windows: windows reduced at a time (bounds memory; the result does not depend on it).
    """

    def __init__(self, chunk_windows: int = CHUNK_WINDOWS) -> None:
        self.chunk_windows = chunk_windows

    @staticmethod
    def n_outputs(n_features: int) -> int:
        """Number of statistics per window."""
        return len(STATISTIC_NAMES) * n_features

    @staticmethod
    def reduce(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Statistics of one batch.

        Args:
            x: (N, T, F) windows; the values of padding rows are ignored.
            mask: (N, T) bool, True = real row. Every window needs at least one real row.

        Returns:
            (N, 5 F) float64: means, population stds, mins, maxs, last - first (F columns each).
        """
        x = np.asarray(x, dtype=np.float64)
        real = np.asarray(mask, dtype=bool)
        n = real.sum(axis=1)
        if (n == 0).any():
            raise ValueError("a window has no real row")
        rows = real[:, :, None]
        mean = np.where(rows, x, 0.0).sum(axis=1) / n[:, None]
        variance = np.where(rows, (x - mean[:, None, :]) ** 2, 0.0).sum(axis=1) / n[:, None]
        minimum = np.where(rows, x, np.inf).min(axis=1)
        maximum = np.where(rows, x, -np.inf).max(axis=1)
        window = np.arange(len(x))
        first = x[window, real.argmax(axis=1)]
        last = x[window, real.shape[1] - 1 - real[:, ::-1].argmax(axis=1)]
        return np.concatenate([mean, np.sqrt(variance), minimum, maximum, last - first], axis=1)

    def __call__(self, x: np.ndarray, mask: np.ndarray, transform=None) -> np.ndarray:
        """Statistics of many windows, chunk by chunk.

        Args:
            x: (N, T, F) windows (may be a read-only memory map).
            mask: (N, T) bool.
            transform: optional `f(x_chunk, mask_chunk) -> x_chunk` applied before the reduction (the detector's
                `standardise`).

        Returns:
            (N, 5 F) float64.
        """
        out = np.empty((len(x), self.n_outputs(x.shape[2])), dtype=np.float64)
        for lo in range(0, len(x), self.chunk_windows):
            xc = np.asarray(x[lo : lo + self.chunk_windows])
            mc = np.asarray(mask[lo : lo + self.chunk_windows], dtype=bool)
            out[lo : lo + len(xc)] = self.reduce(transform(xc, mc) if transform is not None else xc, mc)
        return out


class StatisticsDetector(Detector):
    """A detector fitted on scaled window statistics.

    Class attributes (set by a subclass, besides those of `Detector`):
        uses_subsample: True if the model is fitted on (or keeps as reference) the seeded subsample of
            `SUBSAMPLE_SIZE` training windows instead of the whole training pool.
        score_batch: scaled statistic rows handed to `score_model` at a time.
    """

    uses_subsample: ClassVar[bool] = False
    score_batch: ClassVar[int] = 65_536

    def __init__(self, regime: str, device: str = "cpu", statistics: FeatureStatistics | None = None) -> None:
        super().__init__(regime, device, statistics)
        self._reducer = WindowStatistics()
        self._scaler: StandardScaler | None = None
        self._info: dict = {}

    def hyperparameters(self) -> dict:
        return {
            "input": "window statistics over real rows, after standardise(x, mask)",
            "statistics": list(STATISTIC_NAMES),
            "std_ddof": 0,
            "scaler": "StandardScaler(with_mean=True, with_std=True) fitted on the whole training pool",
            "training_windows": f"seeded subsample of {SUBSAMPLE_SIZE}" if self.uses_subsample else "all",
            "subsample_size": SUBSAMPLE_SIZE if self.uses_subsample else None,
            "validation_data_used": False,
            "scikit_learn": sklearn.__version__,
            "estimator": self.estimator_parameters(),
        }

    @abc.abstractmethod
    def estimator_parameters(self) -> dict:
        """The estimator's fixed parameters (JSON-ready)."""

    @abc.abstractmethod
    def fit_model(self, features: np.ndarray, labels: np.ndarray | None, seed: int) -> dict:
        """Fits the model.

        Args:
            features: (N, 65) scaled statistics of the training windows (the subsample if `uses_subsample`).
            labels: (N,) 0 / 1 in the supervised regime, else None.
            seed: seeds every random choice of the estimator.

        Returns:
            Facts about the fitted model for summary.json.
        """

    @abc.abstractmethod
    def score_model(self, features: np.ndarray) -> np.ndarray:
        """(N,) scores of scaled statistics, higher = more anomalous."""

    def scaled_statistics(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """(N, 65) statistics of the standardised windows, scaled with the fitted scaler."""
        if self._scaler is None:
            raise RuntimeError(f"{self.name}: fit before scoring")
        return self._scaler.transform(self._reducer(x, mask, transform=self.standardise))

    @staticmethod
    def subsample_rows(n: int, seed: int) -> np.ndarray:
        """Sorted rows of the seeded training subsample (every row if the pool is not larger)."""
        if n <= SUBSAMPLE_SIZE:
            return np.arange(n)
        return np.sort(np.random.default_rng(seed).choice(n, size=SUBSAMPLE_SIZE, replace=False))

    def fit(self, train: WindowSet | LabelledWindowSet, val: WindowSet | LabelledWindowSet | None, seed: int) -> None:
        """Fits the scaler on the whole training pool, then the model. `val` is not used."""
        del val
        raw = self._reducer(train.x, train.mask, transform=self.standardise)
        self._scaler = StandardScaler().fit(raw)
        features = self._scaler.transform(raw)
        del raw
        labels = np.asarray(train.labels) if isinstance(train, LabelledWindowSet) else None
        rows = self.subsample_rows(len(features), seed) if self.uses_subsample else None
        if rows is not None:
            features = features[rows]
            labels = labels[rows] if labels is not None else None
        self._info = {
            "training_windows": len(train),
            "fitted_windows": len(features),
            "subsample_size": len(features) if self.uses_subsample else None,
            "n_statistics": int(features.shape[1]),
            "zero_variance_statistics": int((self._scaler.var_ == 0).sum()),
        }
        self._info.update(self.fit_model(features, labels, seed))

    def score(self, x: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        features = self.scaled_statistics(x, mask)
        scores = np.empty(len(features), dtype=np.float64)
        for lo in range(0, len(features), self.score_batch):
            scores[lo : lo + self.score_batch] = self.score_model(features[lo : lo + self.score_batch])
        return {self.score_names[0]: scores}

    def fit_info(self) -> dict:
        return dict(self._info)
