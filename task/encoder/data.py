"""Reader for the benign + GridSybil T64 encoder input, a label-gated window set and the vehicle fit / val split.

The JSON shards (`<split>/b64/part-XXXXX.json` + `part-XXXXX_info.json`) are decoded once into `.npy` files in the
cache folder and then memory-mapped, so a run never holds two copies of the windows. Labels are kept in a separate
file and are only returned by `WindowSet.labels(reason=...)`, so the scoring code cannot see them by accident.
`sender` is used only to build vehicle keys for the fit / validation split; it is never an input.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from task.encoder.config import N_FEATURES, SEQ_LEN

ATTACK_CODE = 16  # GridSybil
CACHE_VERSION = 1


def vehicle_key(scenario: str, sender: int | str) -> str:
    """`("GridSybil_0709", 10551)` -> `"0709:10551"`: physical vehicle within a time window (D8')."""
    return f"{scenario.rsplit('_', 1)[1]}:{sender}"


@dataclass
class DecodedShard:
    """One shard: windows (W, T, 13), mask (W, T), and per-window bookkeeping."""

    x: np.ndarray
    mask: np.ndarray
    ids: list[str]
    n_messages: np.ndarray
    vehicles: list[str]
    labels: np.ndarray


class ShardDecoder:
    """Decodes a feature shard and its `_info.json` (same window order, checked)."""

    @staticmethod
    def info_path(path: Path) -> Path:
        return path.with_name(path.stem + "_info.json")

    def count(self, path: Path) -> int:
        """Windows in a shard (from the smaller info file)."""
        return len(json.loads(self.info_path(path).read_text())["windows"])

    def decode(self, path: str | Path) -> DecodedShard:
        path = Path(path)
        data = json.loads(path.read_text())["windows"]
        info = json.loads(self.info_path(path).read_text())["windows"]
        if [w["id"] for w in data] != [w["id"] for w in info]:
            raise ValueError(f"{path}: window order differs from its info file")
        x = np.asarray([w["x"] for w in data], dtype=np.float32)
        mask = np.asarray([w["mask"] for w in data], dtype=bool)
        if x.shape[1:] != (SEQ_LEN, N_FEATURES):
            raise ValueError(f"{path}: windows have shape {x.shape[1:]}, expected {(SEQ_LEN, N_FEATURES)}")
        x[~mask] = 0.0  # padding rows are zero in the input
        labels = np.asarray([1 if int(w["label"]) == ATTACK_CODE else 0 for w in info], dtype=np.int8)
        codes = {int(w["label"]) for w in info}
        if not codes <= {0, ATTACK_CODE}:
            raise ValueError(f"{path}: unexpected label codes {sorted(codes)}")
        return DecodedShard(
            x=x,
            mask=mask,
            ids=[w["id"] for w in info],
            n_messages=np.asarray([int(w["n_messages"]) for w in info], dtype=np.int32),
            vehicles=[vehicle_key(w["scenario"], w["sender"]) for w in info],
            labels=labels,
        )


def _decode(path: str) -> DecodedShard:
    return ShardDecoder().decode(path)


class WindowSet:
    """Windows of one split (memory-mapped) with the label file kept behind `labels()`.

    Attributes:
        split: "train" or "test".
        x: (N, T, 13) float32, padding rows zero.
        mask: (N, T) bool, True = real row.
        ids, n_messages, vehicles: per-window bookkeeping (vehicles: never an input).
    """

    def __init__(
        self,
        split: str,
        x: np.ndarray,
        mask: np.ndarray,
        ids: np.ndarray,
        n_messages: np.ndarray,
        vehicles: np.ndarray,
        label_source: np.ndarray | Path,
        label_rows: np.ndarray | None = None,
    ) -> None:
        self.split = split
        self.x, self.mask = x, mask
        self.ids, self.n_messages, self.vehicles = ids, n_messages, vehicles
        self._label_source = label_source
        self._label_rows = label_rows
        self.label_reads: list[str] = []

    def __len__(self) -> int:
        return len(self.ids)

    def labels(self, reason: str) -> np.ndarray:
        """0 benign / 1 GridSybil. `reason` is logged ("select benign train", "evaluation after scoring", ...)."""
        self.label_reads.append(reason)
        src = self._label_source
        labels = np.load(src) if isinstance(src, Path) else src
        return labels if self._label_rows is None else labels[self._label_rows]

    def take(self, rows: np.ndarray) -> "WindowSet":
        """A subset (rows sorted, arrays copied into RAM); its labels stay unread until `labels()` is called."""
        rows = np.sort(np.asarray(rows))
        label_rows = rows if self._label_rows is None else self._label_rows[rows]
        return WindowSet(
            self.split,
            np.ascontiguousarray(self.x[rows]),
            np.ascontiguousarray(self.mask[rows]),
            self.ids[rows],
            self.n_messages[rows],
            self.vehicles[rows],
            self._label_source,
            label_rows,
        )


class WindowStore:
    """Builds (once) and opens the `.npy` cache of one split of the T64 encoder input."""

    def __init__(self, data_dir: str | Path, cache_dir: str | Path, workers: int = 8) -> None:
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.workers = workers
        self.decoder = ShardDecoder()

    def shards(self, split: str) -> list[Path]:
        paths = sorted((self.data_dir / split).glob("b*/part-?????.json"))
        if not paths:
            raise FileNotFoundError(f"no shards for split {split!r} under {self.data_dir}")
        return paths

    def _folder(self, split: str) -> Path:
        return self.cache_dir / split

    def build(self, split: str) -> Path:
        """Decodes every shard of `split` into the cache (skipped when a complete cache exists)."""
        folder = self._folder(split)
        manifest = folder / "manifest.json"
        paths = self.shards(split)
        source = {"data_dir": str(self.data_dir), "shards": [p.name for p in paths], "version": CACHE_VERSION}
        if manifest.exists() and json.loads(manifest.read_text()).get("source") == source:
            return folder
        folder.mkdir(parents=True, exist_ok=True)
        counts = [self.decoder.count(p) for p in paths]
        total = int(sum(counts))
        x = np.lib.format.open_memmap(folder / "x.npy", "w+", np.float32, (total, SEQ_LEN, N_FEATURES))
        mask = np.lib.format.open_memmap(folder / "mask.npy", "w+", bool, (total, SEQ_LEN))
        ids, vehicles, n_msg, labels = [], [], [], []
        start = 0
        ctx = mp.get_context("fork")
        with ctx.Pool(min(self.workers, len(paths))) as pool:
            for shard in pool.imap(_decode, [str(p) for p in paths]):
                stop = start + len(shard.ids)
                x[start:stop], mask[start:stop] = shard.x, shard.mask
                ids += shard.ids
                vehicles += shard.vehicles
                n_msg.append(shard.n_messages)
                labels.append(shard.labels)
                start = stop
        if start != total:
            raise ValueError(f"{split}: decoded {start} windows, info files list {total}")
        x.flush()
        mask.flush()
        del x, mask
        np.savez(
            folder / "meta.npz", ids=np.asarray(ids), vehicles=np.asarray(vehicles), n_messages=np.concatenate(n_msg)
        )
        np.save(folder / "labels.npy", np.concatenate(labels))
        manifest.write_text(json.dumps({"source": source, "windows": total}, indent=2))
        return folder

    def open(self, split: str, max_windows: int = 0) -> WindowSet:
        """Memory-mapped window set (first `max_windows` windows when > 0; then copied into RAM)."""
        folder = self.build(split)
        meta = np.load(folder / "meta.npz")
        x = np.load(folder / "x.npy", mmap_mode="r")
        mask = np.load(folder / "mask.npy", mmap_mode="r")
        ids, vehicles, n_msg = meta["ids"], meta["vehicles"], meta["n_messages"]
        labels: np.ndarray | Path = folder / "labels.npy"
        if max_windows:
            # an evenly spaced sample keeps both classes and both time windows in a quick check
            rows = np.linspace(0, len(ids) - 1, min(max_windows, len(ids))).astype(np.int64)
            rows = np.unique(rows)
            x, mask = np.ascontiguousarray(x[rows]), np.ascontiguousarray(mask[rows])
            ids, vehicles, n_msg = ids[rows], vehicles[rows], n_msg[rows]
            labels = np.load(labels)[rows]
        return WindowSet(split, x, mask, ids, n_msg, vehicles, labels)


class VehicleSplit:
    """Holds out a share of sender vehicles (each vehicle entirely on one side)."""

    def __init__(self, frac: float, seed: int) -> None:
        self.frac, self.seed = frac, seed

    def split(self, vehicles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Row indices (fit, val) for the given per-row vehicle keys."""
        unique = np.unique(vehicles)
        rng = np.random.default_rng(self.seed)
        chosen = rng.permutation(unique)[: max(1, int(round(self.frac * len(unique))))]
        is_val = np.isin(vehicles, chosen)
        return np.flatnonzero(~is_val), np.flatnonzero(is_val)
