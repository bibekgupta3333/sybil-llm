"""Isolation Forest on window statistics, library defaults, fitted on every training window.

Score = minus the decision function (a window that is isolated in few splits scores high). Each tree draws 256
training windows with the run's seed, so the detector is stochastic.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from baselinepaper.experiments import config as cfg
from baselinepaper.experiments.detectors.base import register
from baselinepaper.experiments.detectors.window_statistics import StatisticsDetector


@register
class IsolationForestDetector(StatisticsDetector):
    """score = -IsolationForest.decision_function of the scaled window statistics."""

    name = "iforest"
    regimes = cfg.UNLABELLED_REGIMES
    is_stochastic = True
    score_names = ("isolation",)

    PARAMETERS = {
        "n_estimators": 100,
        "max_samples": "auto",
        "contamination": "auto",
        "max_features": 1.0,
        "bootstrap": False,
        "n_jobs": 1,
    }

    def estimator_parameters(self) -> dict:
        return {"class": "sklearn.ensemble.IsolationForest", **self.PARAMETERS, "random_state": "the run's seed"}

    def fit_model(self, features: np.ndarray, labels: np.ndarray | None, seed: int) -> dict:
        del labels
        self._forest = IsolationForest(**self.PARAMETERS, random_state=seed).fit(features)
        return {"max_samples_per_tree": int(self._forest.max_samples_), "offset": float(self._forest.offset_)}

    def score_model(self, features: np.ndarray) -> np.ndarray:
        return -self._forest.decision_function(features)
