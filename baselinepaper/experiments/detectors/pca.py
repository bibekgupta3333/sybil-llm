"""PCA reconstruction error on window statistics.

The components that explain 95% of the training variance of the scaled statistics are kept (rule fixed a priori,
`hyperparameters-classical.md`); a window's score is the mean squared difference between its scaled statistics and
their projection onto those components. Full SVD: deterministic, one run.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from baselinepaper.experiments import config as cfg
from baselinepaper.experiments.detectors.base import register
from baselinepaper.experiments.detectors.window_statistics import StatisticsDetector

EXPLAINED_VARIANCE = 0.95


@register
class PcaReconstruction(StatisticsDetector):
    """score = mean squared PCA reconstruction error of the scaled window statistics."""

    name = "pca"
    regimes = cfg.UNLABELLED_REGIMES
    is_stochastic = False
    score_names = ("reconstruction",)

    def estimator_parameters(self) -> dict:
        return {
            "class": "sklearn.decomposition.PCA",
            "n_components": EXPLAINED_VARIANCE,
            "component_rule": "smallest number of components explaining >= 95% of the training variance",
            "svd_solver": "full",
            "whiten": False,
            "tol": 0.0,
        }

    def fit_model(self, features: np.ndarray, labels: np.ndarray | None, seed: int) -> dict:
        del labels, seed
        self._pca = PCA(n_components=EXPLAINED_VARIANCE, svd_solver="full", whiten=False).fit(features)
        return {
            "n_components": int(self._pca.n_components_),
            "explained_variance_ratio_sum": float(self._pca.explained_variance_ratio_.sum()),
        }

    def score_model(self, features: np.ndarray) -> np.ndarray:
        residual = features - self._pca.inverse_transform(self._pca.transform(features))
        return (residual**2).mean(axis=1)
