"""T64 windows for the supervised autoencoder, read through `task/encoder`'s npy cache.

Train = on-disk train minus the validation vehicles; val = `VehicleSplit(val_frac, split_seed)` over the sender
vehicles of all train windows (the `task/encoder` supervised-mode draw); test = on-disk test, opened only by
`task.autoencoder_st_t64.evaluate`. Inputs are `x` and `mask`; the target `y` (0 benign / 1 GridSybil) is read from
the cache's label file through `WindowSet.labels`, which logs every read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from task.autoencoder_st_t64.config import SupervisedAET64Config
from task.autoencoder_t64.scoring import track_ids
from task.encoder.data import VehicleSplit, WindowSet, WindowStore


@dataclass
class Split:
    """One split in RAM: x (N, 64, 13) float32, mask (N, 64) bool, y (N,) float32, bookkeeping."""

    name: str
    x: np.ndarray
    mask: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    vehicles: np.ndarray
    n_messages: np.ndarray

    def __post_init__(self) -> None:
        self.tracks = track_ids(self.ids)
        self.period = np.asarray([str(v).split(":", 1)[0] for v in self.vehicles])

    def __len__(self) -> int:
        return len(self.ids)

    def counts(self) -> dict[str, int]:
        """Class counts; raises unless labels are 0 / 1 with both classes present."""
        if not set(np.unique(self.y).tolist()) <= {0.0, 1.0}:
            raise ValueError(f"{self.name}: labels outside 0 / 1")
        c = {"benign": int((self.y == 0).sum()), "gridsybil": int((self.y == 1).sum())}
        if min(c.values()) == 0:
            raise ValueError(f"{self.name}: only one class present {c}")
        return c


def _split(name: str, ws: WindowSet, y: np.ndarray) -> Split:
    return Split(name, ws.x, ws.mask, np.asarray(y, dtype=np.float32), ws.ids, ws.vehicles, ws.n_messages)


def load_train_val(s: SupervisedAET64Config, max_windows: int = 0) -> tuple[Split, Split, list[str]]:
    """(train, val, label reads) from the on-disk train split only."""
    store = WindowStore(s.data_dir, s.cache_dir)
    full = store.open("train", max_windows)
    labels = np.asarray(full.labels("train: supervised targets"))
    fit_rows, val_rows = VehicleSplit(s.val_frac, s.split_seed).split(full.vehicles)
    fit_rows, val_rows = np.sort(fit_rows), np.sort(val_rows)
    fit, val = full.take(fit_rows), full.take(val_rows)
    if set(fit.vehicles) & set(val.vehicles):
        raise ValueError("a sender vehicle is in both train and val")
    return _split("train", fit, labels[fit_rows]), _split("val", val, labels[val_rows]), full.label_reads


def load_test(s: SupervisedAET64Config, max_windows: int = 0) -> Split:
    """The on-disk test split (call only after training; labels are read here, use them after scoring)."""
    ws = WindowStore(s.data_dir, s.cache_dir).open("test", max_windows)
    rows = np.arange(len(ws))
    ws = ws.take(rows)
    return _split("test", ws, ws.labels("test: evaluation after scoring"))
