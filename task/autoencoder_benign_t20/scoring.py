"""Label-free anomaly scores, shared with `task/autoencoder_benign_t32` (imported): Mahalanobis distance on z (fitted
on benign train z), per-track max / mean aggregation and quantile thresholds (benign val)."""

from __future__ import annotations

from task.autoencoder_benign_t32.scoring import Mahalanobis, TrackScores, quantile_thresholds

__all__ = ["Mahalanobis", "TrackScores", "quantile_thresholds"]
