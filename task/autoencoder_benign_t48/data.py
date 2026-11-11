"""Reader for the T48_benign encoder input (benign train / val / test + `test_gridsybil`, fixed on disk).

Each shard `<split>/b48/part-XXXXX.json` holds {windows: [{id, x (48 x 13), mask (48)}]}; its `part-XXXXX_info.json`
lists the same windows in the same order with their bookkeeping (scenario, run, sender, n_messages, label, ...).
Links are cut into consecutive non-overlapping crops of at most 48 messages. Training uses only `x` and `mask`;
labels stay in `info` and are read only by the evaluation after scoring.

Window id: `<receiver_file>#<sender_pseudo>-<sender>#<window index in the link>`. The track (one link) is the id
without its last `#<index>` part.

Each split is cached once as .npy under `cache_dir/<split>/` together with a manifest of the shard names, sizes and
modification times; a changed shard set rebuilds the cache.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from task.autoencoder_benign_t32.data import SplitCache, track_id, vehicle_key

__all__ = ["INFO_FIELDS", "SplitCache", "WindowDataset", "read_shard", "track_id", "vehicle_key"]

INFO_FIELDS = (
    "scenario",
    "run",
    "receiver",
    "sender",
    "sender_pseudo",
    "link_window_index",
    "n_messages",
    "label",
    "label_name",
    "is_attack",
)


def read_shard(path: str) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, list]]:
    """One shard -> x (W, T, F) float32 with padding rows zeroed, mask (W, T) bool, ids, info columns."""
    p = Path(path)
    data = json.loads(p.read_text())["windows"]
    info = json.loads(p.with_name(p.stem + "_info.json").read_text())["windows"]
    ids = [w["id"] for w in data]
    if ids != [w["id"] for w in info]:
        raise ValueError(f"{p}: window order differs from its info file")
    x = np.asarray([w["x"] for w in data], dtype=np.float32)
    mask = np.asarray([w["mask"] for w in data], dtype=bool)
    x[~mask] = 0.0
    columns = {k: [w[k] for w in info] for k in INFO_FIELDS}
    return x, mask, ids, columns


class WindowDataset:
    """All windows of one split, in RAM.

    Attributes:
        split: "train", "val", "test" or "test_gridsybil".
        x: (N, T, F) float32, padding rows zero.
        mask: (N, T) bool, True = real message.
        ids: (N,) window ids.
        tracks: (N,) track id of each window (`track_id`).
        n_messages: (N,) real rows per window.
        vehicles: (N,) `vehicle_key` of the sender (split checks only).
        info: the remaining info columns (labels among them); never read by training.
    """

    def __init__(self, split: str, x: np.ndarray, mask: np.ndarray, ids: np.ndarray, info: dict[str, np.ndarray]):
        self.split = split
        self.x, self.mask, self.ids, self.info = x, mask, ids, info
        self.tracks = np.asarray([track_id(i) for i in ids])
        self.n_messages = mask.sum(axis=1).astype(np.int32)
        self.vehicles = np.asarray([vehicle_key(s, v) for s, v in zip(info["scenario"], info["sender"])])

    @classmethod
    def load(
        cls, data_dir: str | Path, split: str, workers: int = 8, max_shards: int = 0, cache_dir: str | Path = ""
    ) -> "WindowDataset":
        """Decodes every shard of `split` (the first `max_shards` when > 0) in parallel, or reads the cache."""
        paths = sorted((Path(data_dir) / split).glob("b*/part-?????.json"))
        if max_shards:
            paths = paths[:max_shards]
        if not paths:
            raise FileNotFoundError(f"no shards for split {split!r} under {data_dir}")
        cache = SplitCache(cache_dir, split, paths) if cache_dir and not max_shards else None
        if cache is not None and cache.valid():
            x, mask, ids, info = cache.load()
        else:
            with ProcessPoolExecutor(min(workers, len(paths)), mp_context=get_context("spawn")) as pool:
                parts = list(pool.map(read_shard, [str(p) for p in paths]))
            info = {k: np.asarray(sum((c[k] for *_, c in parts), [])) for k in INFO_FIELDS}
            ids = np.asarray(sum((p[2] for p in parts), []))
            x, mask = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
            if cache is not None:
                cache.save(x, mask, ids, info)
        data = cls(split, x, mask, ids, info)
        if not np.array_equal(data.n_messages, info["n_messages"].astype(np.int32)):
            raise ValueError(f"{split}: mask row counts differ from n_messages in the info files")
        if not np.all(data.mask[:, 0]) or np.any(np.diff(data.mask.astype(np.int8), axis=1) > 0):
            raise ValueError(f"{split}: masks are not non-empty prefix masks")
        return data

    def __len__(self) -> int:
        return len(self.ids)

    def take(self, rows: np.ndarray) -> "WindowDataset":
        """Subset of windows (rows in the given order)."""
        return WindowDataset(
            self.split, self.x[rows], self.mask[rows], self.ids[rows], {k: v[rows] for k, v in self.info.items()}
        )
