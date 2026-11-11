"""Label-free anomaly scores on top of the autoencoder, shared with `task/autoencoder_benign_t32` (imported).

Mahalanobis distance on z (fitted on benign train z), per-track max / mean aggregation and quantile thresholds
(benign val). Labels enter only through `TrackScores.track_labels`, which the evaluation calls after scoring.
"""

from __future__ import annotations

from task.autoencoder_benign_t32.scoring import Mahalanobis, TrackScores, quantile_thresholds

__all__ = ["Mahalanobis", "TrackScores", "quantile_thresholds"]
