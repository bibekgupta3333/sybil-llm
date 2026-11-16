"""Shared synthetic fixtures: a tiny encoder input folder in the all/T32_benign layout (no real data is read)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from src.model.all_t32_benign.timesnet.base_data import FEATURES

SEQ_LEN = 32
SYNTHETIC_MEAN = [480.0, 650.0, 480.0, 650.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.0, 170.0, -0.25, -0.1]
SYNTHETIC_STD = [480.0, 430.0, 390.0, 300.0, 5.0, 6.0, 7.0, 8.0, 1.0, 1.3, 420.0, 1.7, 0.65]
SCENARIOS = ("GridSybil_1416", "DoSRandomSybil_0709", "DataReplaySybil_1416")


class SyntheticEncoderInput:
    """Writes `<root>/<split>/b32/part-XXXXX.json` feature shards, matching `_info` label files and metadata.json.

    Label files hold garbage and are made unreadable, so any attempt to open one fails loudly.
    """

    def __init__(self, root: Path, seed: int = 0, shard_size: int = 40) -> None:
        self.root, self.rng, self.shard_size = root, np.random.default_rng(seed), shard_size
        self.counts: dict[str, int] = {}

    def split(self, name: str, lengths: list[int], first_sender: int) -> None:
        """Writes one split whose windows have the given real-row counts (senders numbered from first_sender)."""
        windows = [self._window(i, n, first_sender) for i, n in enumerate(lengths)]
        folder = self.root / name / "b32"
        folder.mkdir(parents=True, exist_ok=True)
        for k, start in enumerate(range(0, len(windows), self.shard_size)):
            part = {"seq_len": SEQ_LEN, "features": FEATURES, "windows": windows[start : start + self.shard_size]}
            (folder / f"part-{k:05d}.json").write_text(json.dumps(part))
            info = folder / f"part-{k:05d}_info.json"
            info.write_text("LABELS MUST NEVER BE READ")
            info.chmod(0)
        self.counts[name] = len(windows)

    def metadata(self) -> dict:
        """Writes and returns metadata.json (features, normalisation, counts per split)."""
        meta = {
            "features": list(FEATURES),
            "normalisation": {"features": list(FEATURES), "mean": SYNTHETIC_MEAN, "std": SYNTHETIC_STD},
            "counts_by_split": {k: {"windows": v} for k, v in self.counts.items()},
        }
        (self.root / "metadata.json").write_text(json.dumps(meta))
        return meta

    def _window(self, i: int, n: int, first_sender: int) -> dict:
        x = np.zeros((SEQ_LEN, len(FEATURES)))
        x[:n] = np.round(self.rng.normal(size=(n, len(FEATURES))), 4)
        sender = first_sender + i % 7
        scenario = SCENARIOS[i % len(SCENARIOS)]
        wid = f"{scenario}/run{i % 2}/traceJSON-{100 + i % 5}-{99 + i % 5}-A0-0-7.json#{5000 + sender}-{sender}#{i}"
        return {"id": wid, "x": x.tolist(), "mask": [r < n for r in range(SEQ_LEN)]}


def default_lengths(rng: np.random.Generator, n: int, share_one_row: float) -> list[int]:
    """Real-row counts: `share_one_row` of the windows have 1 row, the rest 2..32."""
    ones = int(round(share_one_row * n))
    return [1] * ones + rng.integers(2, SEQ_LEN + 1, size=n - ones).tolist()


@pytest.fixture
def encoder_input(tmp_path: Path) -> Iterator[Path]:
    """A synthetic all/T32_benign-like folder: train 200 (40% 1-row), val 60, test 40, test_gridsybil 30 windows."""
    rng = np.random.default_rng(1)
    root = tmp_path / "encoder_input"
    data = SyntheticEncoderInput(root)
    data.split("train", default_lengths(rng, 200, 0.4), 1)
    data.split("val", default_lengths(rng, 60, 0.1), 101)
    data.split("test", default_lengths(rng, 40, 0.1), 201)
    data.split("test_gridsybil", default_lengths(rng, 30, 0.2), 301)
    data.metadata()
    yield root
    for info in root.glob("*/b32/*_info.json"):
        info.chmod(0o600)  # let tmp_path clean up
