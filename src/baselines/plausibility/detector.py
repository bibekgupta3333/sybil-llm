"""Threshold decision per BSM over the plausibility checks (F2MD ThresholdApp over LegacyChecks or CaTChChecks).

Implements tracker 4.5. F2MD `mdApplications/ThresholdApp.cc` (lines 30-82): a BSM is flagged misbehaving when any
decision check has a value <= θ (0.28125); an intersection fires when any of its values is <= θ. Checks that do
not apply keep the value 1. Sudden appearance is computed as a diagnostic and never decides (spec §4).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from src.baselines.plausibility.catch_checks import CATCH_CHECK_CLASSES
from src.baselines.plausibility.checks import CHECK_CLASSES, CheckSuite
from src.baselines.plausibility.config import PlausibilityConfig
from src.baselines.plausibility.parser import ReceiverStream


@dataclasses.dataclass
class Decision:
    """Detector output for one receiver stream.

    Attributes:
        flagged: (n,) bool, BSM flagged misbehaving.
        fired: check name -> (n,) bool, value <= θ (decision and diagnostic checks).
        evaluated: check name -> (n,) bool, the check applied.
    """

    flagged: np.ndarray
    fired: dict[str, np.ndarray]
    evaluated: dict[str, np.ndarray]


class ThresholdDetector:
    """Runs the configured checks (`checks_version`) on a receiver stream and applies ThresholdApp's rule."""

    def __init__(self, config: PlausibilityConfig) -> None:
        self._config = config
        classes = CATCH_CHECK_CLASSES if config.checks_version == "catch" else CHECK_CLASSES
        self._suite = CheckSuite(config, classes)

    def detect(self, stream: ReceiverStream) -> Decision:
        """Scores and flags every BSM of the stream."""
        results = self._suite.run(stream)
        flagged = np.zeros(len(stream), dtype=bool)
        fired: dict[str, np.ndarray] = {}
        for name in self._suite.names:
            fired[name] = results.evaluated[name] & (results.value[name] <= self._config.threshold)
            if name in self._config.decision_checks:
                flagged |= fired[name]
        return Decision(flagged, fired, results.evaluated)
