"""Length-only control: the score is minus the number of real rows of the window.

DataReplay / DoS attackers change pseudonym every 1-2 messages, so their links - and windows - are much shorter than
benign ones (F11). This control shows how much of a detector's AUROC that shortcut alone explains.

Direction: shorter = more anomalous, score = -n_messages. It is the rule the earlier studies report as "length
only" (`task/autoencoder_benign_t32/autoencoder_benign_T32.ipynb`: `s = -n_messages`), fixed here before any score
of this protocol exists; it is never chosen or flipped on labels. Nothing is fitted.
"""

from __future__ import annotations

import numpy as np

from baselinepaper.experiments import config as cfg
from baselinepaper.experiments.detectors.base import Detector, register

LENGTH_ONLY = "length_only"
LENGTH_SCORE = "length"


@register
class LengthOnly(Detector):
    """score = -(number of real rows); deterministic, untrained."""

    name = LENGTH_ONLY
    regimes = (cfg.CONTROL_REGIME,)
    is_stochastic = False
    score_names = (LENGTH_SCORE,)

    def hyperparameters(self) -> dict:
        return {"rule": "score = -n_messages (shorter window = more anomalous)", "fitted": False}

    def fit(self, train, val, seed: int) -> None:
        """Nothing to fit."""

    def score(self, x: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        return {LENGTH_SCORE: -np.asarray(mask, dtype=bool).sum(axis=1).astype(np.float64)}
