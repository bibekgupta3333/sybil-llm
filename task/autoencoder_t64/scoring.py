"""Label-free scores on top of the autoencoder: track ids, per-track aggregation, quantile thresholds.

The Mahalanobis model is `task.encoder.train.Mahalanobis` (same ridge as the T = 64 benign-only runs) and the
per-track aggregation is `task.autoencoder_overlap.scoring.TrackScores`; both are re-exported here so the rest of
the package has one import point. Nothing in this module reads a label.
"""

from __future__ import annotations

import numpy as np

from task.autoencoder_overlap.scoring import TrackScores, quantile_thresholds
from task.encoder.train import Mahalanobis

__all__ = ["Mahalanobis", "TrackScores", "quantile_thresholds", "track_id", "track_ids", "ScoreFiles"]


def track_id(window_id: str) -> str:
    """`<receiver_file>#<pseudo>-<sender>#<k>` -> `<receiver_file>#<pseudo>-<sender>`: one link (receiver, sender).

    The id holds the true sender next to the pseudonym, so a track is a privileged grouping (F13).
    """
    return window_id.rsplit("#", 1)[0]


def track_ids(window_ids: np.ndarray) -> np.ndarray:
    """(N,) track id of every window id."""
    return np.asarray([track_id(str(i)) for i in window_ids])


class ScoreFiles:
    """Per-window score files of one split in a run folder: `<split>_<name>.npy`, all aligned with `<split>_ids.npy`.

    Attributes:
        run_dir: the run folder.
        split: "val" or "test".
    """

    NAMES = ("ids", "n_messages", "recon_error", "recon_error_feat", "mahalanobis")

    def __init__(self, run_dir, split: str) -> None:
        self.run_dir, self.split = run_dir, split

    def path(self, name: str):
        return self.run_dir / f"{self.split}_{name}.npy"

    def save(self, **arrays: np.ndarray) -> None:
        """Writes one .npy per array; every array must have one row per window id."""
        if set(arrays) != set(self.NAMES):
            raise ValueError(f"expected arrays {self.NAMES}, got {sorted(arrays)}")
        n = len(arrays["ids"])
        for name, values in arrays.items():
            if len(values) != n:
                raise ValueError(f"{self.split}_{name}: {len(values)} rows for {n} window ids")
            np.save(self.path(name), np.asarray(values).astype(str) if name == "ids" else np.asarray(values))

    def load(self) -> dict[str, np.ndarray]:
        """{name: array} for every score file of the split."""
        return {name: np.load(self.path(name)) for name in self.NAMES}
