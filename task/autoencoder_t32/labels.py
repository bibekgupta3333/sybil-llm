"""Labels and bookkeeping of the windows, from the `part-XXXXX_info.json` files. Evaluation only.

Nothing in `data.py`, `trainer.py` or `train.py` imports this module: the training side reads the feature shards
alone. `evaluate.py` calls `WindowLabels.load` after the scores of a split have been computed and written.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

INFO_FIELDS = ("family", "scenario", "sender", "n_messages", "label_name", "attack_code", "is_attack")
INFO_GLOB = "b*/part-?????_info.json"


def read_info_shard(path: str) -> tuple[list[str], dict[str, list]]:
    """One info file -> ids and the `INFO_FIELDS` columns, in file order."""
    windows = json.loads(Path(path).read_text())["windows"]
    return [w["id"] for w in windows], {k: [w[k] for w in windows] for k in INFO_FIELDS}


class WindowLabels:
    """Info columns of one split, aligned with a given id order.

    Attributes:
        split: "train", "val" or "test".
        ids: (N,) window ids.
        y: (N,) int, 1 = GridSybil (`is_attack`), 0 = benign.
        family, scenario: (N,) source folder family (`GridSybil`) and folder (`GridSybil_0709`).
        group: (N,) time window of the folder, "0709" or "1416".
        n_messages: (N,) real rows per window as the info files state it.
    """

    def __init__(self, split: str, ids: np.ndarray, columns: dict[str, np.ndarray]) -> None:
        self.split, self.ids = split, ids
        self.y = columns["is_attack"].astype(int)
        self.family, self.scenario = columns["family"], columns["scenario"]
        self.group = np.asarray([s.rsplit("_", 1)[-1] for s in self.scenario])
        self.n_messages = columns["n_messages"].astype(np.int32)
        self.attack_code, self.label_name = columns["attack_code"].astype(int), columns["label_name"]

    @classmethod
    def load(cls, data_dir: str | Path, split: str, workers: int = 8) -> "WindowLabels":
        paths = [str(p) for p in sorted((Path(data_dir) / split).glob(INFO_GLOB))]
        if not paths:
            raise FileNotFoundError(f"no info files for split {split!r} under {data_dir}")
        if workers <= 1:
            parts = [read_info_shard(p) for p in paths]
        else:
            with ProcessPoolExecutor(min(workers, len(paths)), mp_context=get_context("spawn")) as pool:
                parts = list(pool.map(read_info_shard, paths))
        ids = np.asarray(sum((p[0] for p in parts), []))
        columns = {k: np.asarray(sum((p[1][k] for p in parts), [])) for k in INFO_FIELDS}
        return cls(split, ids, columns)

    def __len__(self) -> int:
        return len(self.ids)

    def check(self, ids: np.ndarray, n_messages: np.ndarray) -> None:
        """Raises unless these labels describe exactly the scored windows, in the same order, with clean codes."""
        if not np.array_equal(self.ids, np.asarray(ids)):
            raise ValueError(f"{self.split}: info ids differ from the scored window ids")
        if not np.array_equal(self.n_messages, np.asarray(n_messages)):
            raise ValueError(f"{self.split}: n_messages in the info files differ from the mask row counts")
        if set(np.unique(self.y).tolist()) != {0, 1}:
            raise ValueError(f"{self.split}: is_attack must take both values 0 and 1")
        if not np.array_equal(self.attack_code, np.where(self.y == 1, 16, 0)):
            raise ValueError(f"{self.split}: attack codes other than 0 (benign) / 16 (GridSybil)")
        if np.any(self.family[self.y == 1] != "GridSybil"):
            raise ValueError(f"{self.split}: GridSybil windows outside the GridSybil folders")

    def class_counts(self) -> dict[str, int]:
        return {"benign": int((self.y == 0).sum()), "gridsybil": int((self.y == 1).sum())}
