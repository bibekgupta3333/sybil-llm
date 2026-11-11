"""Label-free reader for the all/T32_benign_grid encoder input (train / val / test fixed on disk).

Each feature shard `<split>/b32/part-XXXXX.json` holds {windows: [{id, x (32 x 13), mask (32)}]} and no label. The
labels and the rest of the bookkeeping live in the sibling files `part-XXXXX_info.json`; this module never opens them
(the glob below cannot match them, and nothing here builds their path). They are read only by `labels.py`, which the
evaluation calls after every score has been written.

Window id: `<Family>_<group>/<run>/<receiver_file>#<sender_pseudo>-<sender>#<window index in the link>`. The track (one
link) is the id without its last `#<index>` part; the vehicle key `<group>:<sender>` (the unit of the on-disk split) is
parsed from the id as well, for the split-disjointness check.

Decoding the JSON takes minutes, so each split is cached once as .npy (x, mask, ids) under `cache_dir/<split>/` with a
manifest of the feature-shard names, sizes and modification times; a changed shard set rebuilds the cache.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

SHARD_GLOB = "b*/part-?????.json"  # feature shards only: `part-00000_info.json` does not match
SHARD_FIELDS = {"id", "x", "mask"}
INFO_SUFFIX = "_info.json"  # the label files; never opened by this module


class LabelFileGuard:
    """Process-wide audit hook that makes any attempt to open a `*_info.json` file raise PermissionError.

    `train.py` installs it in its own process and in every shard-decoding worker before the first file is read, so a
    finished training run is proof that no label file was opened. An audit hook cannot be removed: install it only
    in a process that must never read labels.
    """

    installed = False

    @classmethod
    def install(cls) -> None:
        if cls.installed:
            return

        def hook(event: str, args: tuple) -> None:
            if event == "open" and args and str(args[0]).endswith(INFO_SUFFIX):
                raise PermissionError(f"label file opened in a label-free process: {args[0]}")

        sys.addaudithook(hook)
        cls.installed = True


def track_id(window_id: str) -> str:
    """`"<file>#1094892-9489#2"` -> `"<file>#1094892-9489"`: the link (receiver, pseudonym, sender) of a window."""
    head, sep, tail = window_id.rpartition("#")
    if not sep or not tail.isdigit() or "#" not in head:
        raise ValueError(f"window id without a `#<pseudo>-<sender>#<index>` suffix: {window_id!r}")
    return head


def vehicle_key(window_id: str) -> str:
    """`"GridSybil_0709/<run>/<file>#1094892-9489#2"` -> `"0709:9489"`: the sender vehicle within its time window."""
    link = track_id(window_id)
    scenario = link.split("/", 1)[0]
    return f"{scenario.rsplit('_', 1)[-1]}:{link.rpartition('-')[2]}"


def read_feature_shard(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """One feature shard -> x (W, T, F) float32 with padding rows zeroed, mask (W, T) bool, ids."""
    windows = json.loads(Path(path).read_text())["windows"]
    extra = set().union(*(set(w) for w in windows)) - SHARD_FIELDS if windows else set()
    if extra:
        raise ValueError(f"{path}: feature shard holds fields other than id, x, mask: {sorted(extra)}")
    x = np.asarray([w["x"] for w in windows], dtype=np.float32)
    mask = np.asarray([w["mask"] for w in windows], dtype=bool)
    x[~mask] = 0.0
    return x, mask, [w["id"] for w in windows]


class FeatureCache:
    """Decoded x, mask, ids of one split under `<cache_dir>/<split>/`, valid while the shard manifest is unchanged."""

    FILES = ("x.npy", "mask.npy", "ids.npy")

    def __init__(self, cache_dir: str | Path, split: str, paths: list[Path]) -> None:
        self.folder = Path(cache_dir) / split
        self.manifest = [
            {"name": str(p.relative_to(p.parents[2])), "size": p.stat().st_size, "mtime": int(p.stat().st_mtime)}
            for p in paths
        ]

    def valid(self) -> bool:
        path = self.folder / "manifest.json"
        return path.exists() and json.loads(path.read_text()) == self.manifest

    def load(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, mask, ids = (np.load(self.folder / name) for name in self.FILES)
        return x, mask, ids

    def save(self, x: np.ndarray, mask: np.ndarray, ids: np.ndarray) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        for name, array in zip(self.FILES, (x, mask, ids)):
            np.save(self.folder / name, array)
        (self.folder / "manifest.json").write_text(json.dumps(self.manifest))


class UnlabelledWindows:
    """All windows of one split, in RAM, without any label or info column.

    Attributes:
        split: "train", "val" or "test".
        x: (N, T, F) float32, padding rows zero.
        mask: (N, T) bool, True = real message.
        ids: (N,) window ids.
        n_messages: (N,) real rows per window (from the mask).
    """

    __slots__ = ("split", "x", "mask", "ids", "n_messages")

    def __init__(self, split: str, x: np.ndarray, mask: np.ndarray, ids: np.ndarray) -> None:
        self.split = split
        self.x, self.mask, self.ids = x, mask, ids
        self.n_messages = mask.sum(axis=1).astype(np.int32)

    @classmethod
    def load(
        cls,
        data_dir: str | Path,
        split: str,
        workers: int = 8,
        max_shards: int = 0,
        cache_dir: str | Path = "",
        guard: bool = False,
    ) -> "UnlabelledWindows":
        """Decodes every feature shard of `split` (the first `max_shards` when > 0), or reads the cache.

        With `guard`, the decoding workers install `LabelFileGuard` before they read anything.
        """
        paths = sorted((Path(data_dir) / split).glob(SHARD_GLOB))
        if max_shards:
            paths = paths[:max_shards]
        if not paths:
            raise FileNotFoundError(f"no shards for split {split!r} under {data_dir}")
        cache = FeatureCache(cache_dir, split, paths) if cache_dir and not max_shards else None
        if cache is not None and cache.valid():
            x, mask, ids = cache.load()
        else:
            names = [str(p) for p in paths]
            if workers <= 1:
                parts = [read_feature_shard(n) for n in names]
            else:
                with ProcessPoolExecutor(
                    min(workers, len(names)),
                    mp_context=get_context("spawn"),
                    initializer=LabelFileGuard.install if guard else None,
                ) as pool:
                    parts = list(pool.map(read_feature_shard, names))
            x, mask = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
            ids = np.asarray(sum((p[2] for p in parts), []))
            if cache is not None:
                cache.save(x, mask, ids)
        data = cls(split, x, mask, ids)
        if not np.all(data.mask[:, 0]) or np.any(np.diff(data.mask.astype(np.int8), axis=1) > 0):
            raise ValueError(f"{split}: masks are not non-empty prefix masks")
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"{split}: window ids are not unique")
        return data

    def __len__(self) -> int:
        return len(self.ids)

    def tracks(self) -> np.ndarray:
        """(N,) track id of each window."""
        return np.asarray([track_id(i) for i in self.ids])

    def vehicles(self) -> set[str]:
        """Vehicle keys `<group>:<sender>` of the split (split checks only)."""
        return {vehicle_key(i) for i in self.ids}
