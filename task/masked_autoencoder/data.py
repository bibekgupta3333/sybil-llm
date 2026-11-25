"""One-class data of each window size, read through the sibling ConvAE packages (imported, not copied).

Every source returns the same three things:
  * `fit_val()`: benign train and benign val windows as `Windows` (x, mask, ids, tracks, n_messages, vehicles), the
    exact windows the sibling one-class ConvAE was trained and early-stopped on;
  * `eval_splits()` / `load(split)`: the windows to score (val + test splits), without labels;
  * `labels(split, reason)`: the attack code of each window of a split, logged in `label_reads`. Training never calls it
    for T = 48 / 32 (the on-disk train / val splits are benign by construction); for T = 64 it is called once to select
    the benign train windows, exactly as `task/encoder` does in mode ae.

T = 64: `task.encoder.data.WindowStore` (cache `task/encoder/runs/cache`) + `VehicleSplit(0.1, seed 0)` over the benign
train pool. T = 48 / 32: `task.autoencoder_benign_t48.data.WindowDataset` / `task.autoencoder_benign_t32.data.WindowDataset`
with the siblings' caches. T = 20 / 10 / 5: `task.autoencoder_benign_t32.data.WindowDataset` (the all/T<w>_benign
layout is the all/T32_benign one) with this package's cache `task/masked_autoencoder/cache/T<w>`.
"""

from __future__ import annotations

import abc
import dataclasses

import numpy as np

from task.autoencoder_benign_t32.data import WindowDataset as WindowDataset32
from task.autoencoder_benign_t32.data import track_id
from task.autoencoder_benign_t48.data import WindowDataset as WindowDataset48
from task.encoder.data import VehicleSplit, WindowStore
from task.masked_autoencoder.config import MAEConfig

GRIDSYBIL_CODE = 16
T64_VAL_FRAC, T64_SPLIT_SEED = 0.1, 0  # task/encoder EncoderTaskConfig defaults (val_frac, split_seed)


@dataclasses.dataclass
class Windows:
    """Label-free windows of one split.

    Attributes:
        split: split name ("train", "val", "test", "test_<attack>").
        x: (N, T, 13) float32, padding rows zero.
        mask: (N, T) bool, True = real row.
        ids: (N,) window ids.
        n_messages: (N,) real rows per window.
        vehicles: (N,) sender vehicle keys (split checks only; never an input).
    """

    split: str
    x: np.ndarray
    mask: np.ndarray
    ids: np.ndarray
    n_messages: np.ndarray
    vehicles: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def tracks(self) -> np.ndarray:
        """(N,) link of each window (`track_id`)."""
        return np.asarray([track_id(str(i)) for i in self.ids])

    def take(self, rows: np.ndarray) -> "Windows":
        rows = np.asarray(rows)
        return Windows(
            self.split,
            np.ascontiguousarray(self.x[rows]),
            np.ascontiguousarray(self.mask[rows]),
            self.ids[rows],
            self.n_messages[rows],
            self.vehicles[rows],
        )

    def head(self, n: int) -> "Windows":
        """Evenly spaced subset of `n` windows (quick checks); all windows when n is 0."""
        if not n or n >= len(self):
            return self
        return self.take(np.unique(np.linspace(0, len(self) - 1, n).astype(np.int64)))


class OneClassSource(abc.ABC):
    """Benign fit / val windows, evaluation splits and gated labels of one window size."""

    def __init__(self, settings: MAEConfig) -> None:
        self.settings = settings
        self.label_reads: list[str] = []

    @abc.abstractmethod
    def fit_val(self) -> tuple[Windows, Windows]:
        """Benign training and benign validation windows."""

    @abc.abstractmethod
    def eval_splits(self) -> tuple[str, ...]:
        """Test splits to score (benign and attack windows; "val" is scored separately)."""

    @abc.abstractmethod
    def load(self, split: str) -> Windows:
        """Windows of a split (also "val"), without labels."""

    @abc.abstractmethod
    def _labels(self, split: str) -> np.ndarray:
        """Attack code (0 benign, 16 GridSybil, ...) per window of `split`."""

    def labels(self, split: str, reason: str) -> np.ndarray:
        """Attack codes of `split`; every call is recorded in `label_reads`."""
        self.label_reads.append(f"{split}: {reason}")
        return self._labels(split)


class T64Source(OneClassSource):
    """task/encoder mode ae: benign train windows, val = 10% of their sender vehicles (split seed 0); test mixed."""

    def __init__(self, settings: MAEConfig) -> None:
        super().__init__(settings)
        p = settings.protocol
        self.store = WindowStore(p.data_dir, p.cache_dir, settings.workers)
        self._val: Windows | None = None
        self._sets: dict = {}

    def _open(self, split: str):
        if split not in self._sets:
            self._sets[split] = self.store.open(split)
        return self._sets[split]

    @staticmethod
    def _windows(split: str, ws) -> Windows:
        return Windows(split, ws.x, ws.mask, np.asarray(ws.ids), np.asarray(ws.n_messages), np.asarray(ws.vehicles))

    def fit_val(self) -> tuple[Windows, Windows]:
        train = self._open("train")
        codes = self.labels("train", "select benign train windows (one-class), as task/encoder mode ae")
        pool = train.take(np.flatnonzero(codes == 0))
        fit_rows, val_rows = VehicleSplit(T64_VAL_FRAC, T64_SPLIT_SEED).split(pool.vehicles)
        fit, val = self._windows("train", pool.take(fit_rows)), self._windows("val", pool.take(val_rows))
        self._val = val
        n = self.settings.max_windows
        return fit.head(n), val.head(n)

    def eval_splits(self) -> tuple[str, ...]:
        return ("test",)

    def load(self, split: str) -> Windows:
        if split == "val":
            if self._val is None:
                # the val windows need the benign selection of train; recorded like in training
                self.fit_val()
            return self._val.head(self.settings.max_windows)
        return self._windows(split, self._open(split)).head(self.settings.max_windows)

    def _labels(self, split: str) -> np.ndarray:
        if split == "val":
            return np.zeros(len(self.load("val")), dtype=np.int64)
        ws = self._open(split)
        binary = np.asarray(ws.labels("masked_autoencoder evaluation"), dtype=np.int64)
        codes = np.where(binary == 1, GRIDSYBIL_CODE, 0)
        if split != "train" and self.settings.max_windows:
            rows = np.unique(np.linspace(0, len(codes) - 1, self.settings.max_windows).astype(np.int64))
            codes = codes[rows] if self.settings.max_windows < len(codes) else codes
        return codes


class OnDiskSource(OneClassSource):
    """T = 48 / 32: on-disk benign train / val / test and attack test splits (sibling caches)."""

    dataset_cls = WindowDataset32

    def __init__(self, settings: MAEConfig) -> None:
        super().__init__(settings)
        self._sets: dict = {}

    def _open(self, split: str):
        if split not in self._sets:
            p, s = self.settings.protocol, self.settings
            self._sets[split] = self.dataset_cls.load(p.data_dir, split, s.workers, 0, p.cache_dir)
        return self._sets[split]

    def load(self, split: str) -> Windows:
        d = self._open(split)
        out = Windows(split, d.x, d.mask, np.asarray(d.ids), np.asarray(d.n_messages), np.asarray(d.vehicles))
        return out.head(self.settings.max_windows)

    def fit_val(self) -> tuple[Windows, Windows]:
        return self.load("train"), self.load("val")

    def eval_splits(self) -> tuple[str, ...]:
        return ("test", *self.settings.protocol.attack_splits)

    def _labels(self, split: str) -> np.ndarray:
        codes = np.asarray(self._open(split).info["label"]).astype(np.int64)
        n = self.settings.max_windows
        if n and n < len(codes):
            codes = codes[np.unique(np.linspace(0, len(codes) - 1, n).astype(np.int64))]
        return codes


class T48Source(OnDiskSource):
    dataset_cls = WindowDataset48


class T32Source(OnDiskSource):
    """all/T<w>_benign layout: T = 32 and the short windows T = 20 / 10 / 5."""

    dataset_cls = WindowDataset32


def make_source(settings: MAEConfig) -> OneClassSource:
    """The source of `settings.window`."""
    if settings.window == 64:
        return T64Source(settings)
    if settings.window == 48:
        return T48Source(settings)
    return T32Source(settings)  # T = 32, 20, 10, 5
