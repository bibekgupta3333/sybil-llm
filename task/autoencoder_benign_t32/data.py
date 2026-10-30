"""Reader for the all/T32_benign encoder input (benign train / val / test + attack test splits, fixed on disk).

Each shard `<split>/b32/part-XXXXX.json` holds {windows: [{id, x (32 x 13), mask (32)}]}; its `part-XXXXX_info.json`
lists the same windows in the same order with their bookkeeping (family, scenario, sender, n_messages, label, ...).
Links are cut into consecutive non-overlapping crops of at most 32 messages. Training uses only `x` and `mask`;
labels stay in `info` and are read only by the analysis after scoring.

Window id: `<receiver_file>#<sender_pseudo>-<sender>#<window index in the link>`. The track (one link) is the id
without its last `#<index>` part.

Decoding the 3.3 GB of JSON takes minutes, so each split is cached once as .npy under `cache_dir/<split>/` together
with a manifest of the shard names, sizes and modification times; a changed shard set rebuilds the cache.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

INFO_FIELDS = (
    "family",
    "scenario",
    "receiver",
    "sender",
    "sender_pseudo",
    "link_window_index",
    "n_messages",
    "label",
    "label_name",
    "is_attack",
)


def vehicle_key(scenario: str, sender: int | str) -> str:
    """`("GridSybil_0709", 10551)` -> `"0709:10551"`: one physical vehicle within one time window (the split key)."""
    return f"{scenario.rsplit('_', 1)[-1]}:{sender}"


def track_id(window_id: str) -> str:
    """`"<file>#1094892-9489#2"` -> `"<file>#1094892-9489"`: the link (receiver, pseudonym, sender) of a window."""
    head, sep, tail = window_id.rpartition("#")
    if not sep or not tail.isdigit() or "#" not in head:
        raise ValueError(f"window id without a `#<pseudo>-<sender>#<index>` suffix: {window_id!r}")
    return head


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
    columns = {k: [w.get(k) for w in info] for k in INFO_FIELDS}
    return x, mask, ids, columns


class SplitCache:
    """Decoded arrays of one split under `<cache_dir>/<split>/`, valid while the shard manifest is unchanged."""

    def __init__(self, cache_dir: str | Path, split: str, paths: list[Path]) -> None:
        self.folder = Path(cache_dir) / split
        self.manifest = [{"name": str(p.relative_to(p.parents[2])), "size": p.stat().st_size} for p in paths]
        for entry, p in zip(self.manifest, paths):
            entry["mtime"] = int(p.stat().st_mtime)
            info = p.with_name(p.stem + "_info.json")
            entry["info_size"] = info.stat().st_size

    def valid(self) -> bool:
        path = self.folder / "manifest.json"
        return path.exists() and json.loads(path.read_text()) == self.manifest

    def load(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        f = self.folder
        info = dict(np.load(f / "info.npz"))
        return np.load(f / "x.npy"), np.load(f / "mask.npy"), np.load(f / "ids.npy"), info

    def save(self, x: np.ndarray, mask: np.ndarray, ids: np.ndarray, info: dict[str, np.ndarray]) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        np.save(self.folder / "x.npy", x)
        np.save(self.folder / "mask.npy", mask)
        np.save(self.folder / "ids.npy", ids)
        np.savez(self.folder / "info.npz", **info)
        (self.folder / "manifest.json").write_text(json.dumps(self.manifest))


class WindowDataset:
    """All windows of one split, in RAM.

    Attributes:
        split: "train", "val", "test" or "test_<attack>".
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
