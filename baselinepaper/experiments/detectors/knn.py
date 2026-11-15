"""kNN distance on window statistics.

Reference set = the seeded subsample of 20,000 training windows (`window_statistics.SUBSAMPLE_SIZE`), exact
(brute-force) Euclidean search; score = mean distance to the k = 5 nearest reference windows. The subsample makes
the detector stochastic.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from baselinepaper.experiments import config as cfg
from baselinepaper.experiments.detectors.base import register
from baselinepaper.experiments.detectors.window_statistics import StatisticsDetector

N_NEIGHBORS = 5


@register
class KnnDistance(StatisticsDetector):
    """score = mean Euclidean distance to the k nearest reference windows (scaled window statistics)."""

    name = "knn"
    regimes = cfg.UNLABELLED_REGIMES
    is_stochastic = True
    score_names = ("distance",)
    uses_subsample = True
    score_batch = 8_192

    PARAMETERS = {"n_neighbors": N_NEIGHBORS, "algorithm": "brute", "metric": "minkowski", "p": 2, "n_jobs": 1}

    def estimator_parameters(self) -> dict:
        return {"class": "sklearn.neighbors.NearestNeighbors", **self.PARAMETERS, "aggregation": "mean of k distances"}

    def fit_model(self, features: np.ndarray, labels: np.ndarray | None, seed: int) -> dict:
        del labels, seed
        if len(features) < N_NEIGHBORS:
            raise ValueError(f"knn needs at least {N_NEIGHBORS} reference windows, got {len(features)}")
        self._index = NearestNeighbors(**self.PARAMETERS).fit(features)
        return {"reference_windows": int(len(features)), "k": N_NEIGHBORS}

    def score_model(self, features: np.ndarray) -> np.ndarray:
        distances, _ = self._index.kneighbors(features, return_distance=True)
        return distances.mean(axis=1)
