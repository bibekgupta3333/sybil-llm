"""One-class SVM (RBF kernel) on window statistics, library defaults.

Training is quadratic in the number of windows, so the model is fitted on the seeded subsample of 20,000 training
windows (`window_statistics.SUBSAMPLE_SIZE`); the subsample makes the detector stochastic. Score = minus the
decision function.
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM

from baselinepaper.experiments import config as cfg
from baselinepaper.experiments.detectors.base import register
from baselinepaper.experiments.detectors.window_statistics import StatisticsDetector


@register
class OneClassSvmDetector(StatisticsDetector):
    """score = -OneClassSVM.decision_function of the scaled window statistics."""

    name = "ocsvm"
    regimes = cfg.UNLABELLED_REGIMES
    is_stochastic = True
    score_names = ("decision",)
    uses_subsample = True
    score_batch = 8_192

    PARAMETERS = {
        "kernel": "rbf",
        "gamma": "scale",
        "nu": 0.5,
        "tol": 1e-3,
        "shrinking": True,
        "max_iter": -1,
        "cache_size": 1000,
    }

    def estimator_parameters(self) -> dict:
        return {"class": "sklearn.svm.OneClassSVM", **self.PARAMETERS}

    def fit_model(self, features: np.ndarray, labels: np.ndarray | None, seed: int) -> dict:
        del labels, seed
        self._svm = OneClassSVM(**self.PARAMETERS).fit(features)
        return {"support_vectors": int(len(self._svm.support_)), "gamma": float(self._svm._gamma)}

    def score_model(self, features: np.ndarray) -> np.ndarray:
        return -self._svm.decision_function(features)
