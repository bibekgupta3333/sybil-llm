"""Reader for the all/T32_benign_grid encoder input (train / val / test = 85 / 7.5 / 7.5 by vehicle key, fixed on disk).

Content: benign windows (sender code 0, never an attacker) from all 8 scenario folders + GridSybil attack windows
(code 16) from the 2 GridSybil folders. Each shard `<split>/b32/part-XXXXX.json` holds
{windows: [{id, x (32 x 13), mask (32)}]}; its `part-XXXXX_info.json` lists the same windows in the same order with
their bookkeeping (family, scenario, sender, n_messages, label, is_attack, ...). Links are cut into consecutive
non-overlapping crops of at most 32 messages. The model sees only `x` and `mask`. The target `y` = `is_attack`
(Benign 0, GridSybil 1) comes from the info files and is kept apart from the inputs; family, sender and ids are used
only for split checks, track aggregation and the analysis.

Window id: `<receiver_file>#<sender_pseudo>-<sender>#<window index in the link>`. The track (one link) is the id up to
its last `#`.

Decoding the JSON takes minutes, so each split is cached once as .npy under `cache_dir/<split>/` together with a
manifest of the shard names, sizes and modification times; a changed shard set rebuilds the cache.
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
    if any(set(w) - {"id", "x", "mask"} for w in data):
        raise ValueError(f"{p}: feature shard holds fields other than id, x, mask")
    missing = [k for k in INFO_FIELDS if info and k not in info[0]]
    if missing:
        raise ValueError(f"{p}: info file lacks fields {missing}")
    x = np.asarray([w["x"] for w in data], dtype=np.float32)
    mask = np.asarray([w["mask"] for w in data], dtype=bool)
    x[~mask] = 0.0
    columns = {k: [w[k] for w in info] for k in INFO_FIELDS}
    return x, mask, ids, columns


class SplitCache:
    """Decoded arrays of one split under `<cache_dir>/<split>/`, valid while the shard manifest is unchanged."""

    def __init__(self, cache_dir: str | Path, split: str, paths: list[Path]) -> None:
        self.folder = Path(cache_dir) / split
        self.manifest = []
        for p in paths:
            info = p.with_name(p.stem + "_info.json")
            self.manifest.append(
                {
                    "name": str(p.relative_to(p.parents[2])),
                    "size": p.stat().st_size,
                    "mtime": int(p.stat().st_mtime),
                    "info_size": info.stat().st_size,
                    "info_mtime": int(info.stat().st_mtime),
                }
            )

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
        split: "train", "val" or "test".
        x: (N, T, F) float32, padding rows zero.
        mask: (N, T) bool, True = real message.
        y: (N,) float32 target, 1 = GridSybil (`is_attack`), 0 = benign.
        ids: (N,) window ids.
        tracks: (N,) track id of each window (`track_id`).
        n_messages: (N,) real rows per window.
        vehicles: (N,) `vehicle_key` of the sender (split checks only).
        info: the info columns.
    """

    def __init__(self, split: str, x: np.ndarray, mask: np.ndarray, ids: np.ndarray, info: dict[str, np.ndarray]):
        self.split = split
        self.x, self.mask, self.ids, self.info = x, mask, ids, info
        self.y = np.asarray(info["is_attack"]).astype(np.float32)
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

    def check_labels(self) -> dict[str, int]:
        """Raises unless labels are 0 / 1 with both classes present and agree with `label_name` / the trace code;
        returns the class counts."""
        values = set(np.unique(self.info["is_attack"]).tolist())
        if not values <= {0, 1}:
            raise ValueError(f"{self.split}: is_attack takes values {sorted(values)}, expected 0 / 1")
        names = np.asarray(self.info["label_name"])
        expected = np.where(self.y == 1, "GridSybil", "Benign")
        if not np.array_equal(names, expected):
            raise ValueError(f"{self.split}: label_name disagrees with is_attack ({int((names != expected).sum())})")
        families = np.asarray(self.info["family"])
        if np.any(families[self.y == 1] != "GridSybil"):
            raise ValueError(f"{self.split}: GridSybil windows outside the GridSybil folders")
        counts = {"benign": int((self.y == 0).sum()), "gridsybil": int((self.y == 1).sum())}
        if min(counts.values()) == 0:
            raise ValueError(f"{self.split}: only one class present {counts}")
        return counts

    def take(self, rows: np.ndarray) -> "WindowDataset":
        """Subset of windows (rows in the given order)."""
        return WindowDataset(
            self.split, self.x[rows], self.mask[rows], self.ids[rows], {k: v[rows] for k, v in self.info.items()}
        )
