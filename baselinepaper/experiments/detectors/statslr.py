"""Logistic regression on window statistics (supervised regime: benign vs GridSybil).

Library defaults (L2, C = 1, lbfgs, no class weights) with the iteration cap raised to 1,000 so the solver can
converge; fitted on every training window with its label. Score = P(GridSybil). lbfgs from a zero start is
deterministic: one run. Scoring takes x and mask only.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from baselinepaper.experiments import config as cfg
from baselinepaper.experiments.detectors.base import register
from baselinepaper.experiments.detectors.window_statistics import StatisticsDetector


@register
class StatisticsLogisticRegression(StatisticsDetector):
    """score = P(attack) of a logistic regression on the scaled window statistics."""

    name = "statslr"
    regimes = (cfg.SUPERVISED,)
    is_stochastic = False
    score_names = ("probability",)

    PARAMETERS = {
        "C": 1.0,
        "l1_ratio": 0.0,
        "solver": "lbfgs",
        "tol": 1e-4,
        "fit_intercept": True,
        "class_weight": None,
        "max_iter": 1000,
    }

    def estimator_parameters(self) -> dict:
        return {"class": "sklearn.linear_model.LogisticRegression", **self.PARAMETERS}

    def fit_model(self, features: np.ndarray, labels: np.ndarray | None, seed: int) -> dict:
        del seed
        if labels is None:
            raise ValueError("statslr needs a labelled training set (supervised regime)")
        self._model = LogisticRegression(**self.PARAMETERS).fit(features, labels)
        if list(self._model.classes_) != [0, 1]:
            raise ValueError(f"statslr expects labels 0 / 1, got classes {self._model.classes_.tolist()}")
        iterations = int(self._model.n_iter_[0])
        return {
            "iterations": iterations,
            "converged": iterations < self.PARAMETERS["max_iter"],
            "training_attack_share": float(labels.mean()),
        }

    def score_model(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(features)[:, 1]
