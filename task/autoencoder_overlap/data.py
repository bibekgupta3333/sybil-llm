"""Reader for the T48_tvt_s24 encoder input (overlapping windows; train / val / test by sender vehicle, fixed on disk).

Each shard `<split>/b48/part-XXXXX.json` holds {windows: [{id, x (48 x 13), mask (48)}]}; its `part-XXXXX_info.json`
lists the same windows in the same order with their bookkeeping (scenario, sender, start, n_messages, label, ...).
Links longer than 48 messages are cut into windows at starts 0, 24, ... plus one end-aligned window, so consecutive
windows of a link share up to 24 messages. Training uses only `x` and `mask`; labels stay in `info` and are read only
by the analysis after scoring.

Window id: `<receiver_file>#<pseudo>-<sender>#<link_window_index>@<start>`. The track (one link) is the id without
its last `#<link_window_index>@<start>` part.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

INFO_FIELDS = (
    "scenario",
    "sender",
    "receiver",
    "link_window_index",
    "start",
    "n_messages",
    "label",
    "label_name",
    "is_attack",
)


def vehicle_key(scenario: str, sender: int | str) -> str:
    """`("GridSybil_0709", 10551)` -> `"0709:10551"`: one physical vehicle within one time window."""
    return f"{scenario.rsplit('_', 1)[-1]}:{sender}"


def track_id(window_id: str) -> str:
    """`"<file>#1-10635#2@48"` -> `"<file>#1-10635"`: the link (receiver, pseudonym, sender) a window was cut from."""
    head, sep, tail = window_id.rpartition("#")
    if not sep or "@" not in tail:
        raise ValueError(f"window id without a `#<index>@<start>` suffix: {window_id!r}")
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


class WindowDataset:
    """All windows of one split, in RAM.

    Attributes:
        split: "train", "val" or "test".
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
    def load(cls, data_dir: str | Path, split: str, workers: int = 8, max_shards: int = 0) -> "WindowDataset":
        """Decodes every shard of `split` (the first `max_shards` when > 0) in parallel."""
        paths = sorted((Path(data_dir) / split).glob("b*/part-?????.json"))
        if max_shards:
            paths = paths[:max_shards]
        if not paths:
            raise FileNotFoundError(f"no shards for split {split!r} under {data_dir}")
        with ProcessPoolExecutor(min(workers, len(paths)), mp_context=get_context("spawn")) as pool:
            parts = list(pool.map(read_shard, [str(p) for p in paths]))
        info = {k: np.asarray(sum((c[k] for *_, c in parts), [])) for k in INFO_FIELDS}
        ids = np.asarray(sum((p[2] for p in parts), []))
        data = cls(split, np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), ids, info)
        if not np.array_equal(data.n_messages, info["n_messages"].astype(np.int32)):
            raise ValueError(f"{split}: mask row counts differ from n_messages in the info files")
        return data

    def __len__(self) -> int:
        return len(self.ids)

    def take(self, rows: np.ndarray) -> "WindowDataset":
        """Subset of windows (rows in the given order)."""
        return WindowDataset(
            self.split, self.x[rows], self.mask[rows], self.ids[rows], {k: v[rows] for k, v in self.info.items()}
        )
