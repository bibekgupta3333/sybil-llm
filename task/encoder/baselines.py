"""Shortcut baselines for benign vs GridSybil on the T64 encoder input.

Two baselines that use no learned encoder, so the autoencoder and the supervised model can be compared against
what the window length and simple per-window statistics already give:

* length only: score = n_messages (both directions are reported; the better one is named),
* simple statistics: per-window mean / std / min / max of each of the 13 features over real rows + n_messages,
  standardised, logistic regression fitted on the train split, scored on the test split.

The T64 shards are read directly (read-only). ``sender`` is kept only as the cluster key for the evaluation's
vehicle bootstrap; it is never a model input. Test labels are attached to the score file after every test score
exists.

Run from the repo root::

    .venv-train/bin/python -m task.encoder.baselines
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "src" / "data" / "encoder_input" / "benign_gridsybil" / "T64"
DEFAULT_OUT_DIR = REPO_ROOT / "task" / "encoder" / "runs" / "baselines"
STAT_NAMES = ("mean", "std", "min", "max")


@dataclasses.dataclass(frozen=True)
class BaselineConfig:
    """Settings of the shortcut baselines.

    Attributes:
        data_dir: T64 encoder-input folder (``metadata.json`` + ``<split>/b64/part-*.json``).
        out_dir: Where the score file, config, env and summary are written.
        seed: Random seed of the logistic regression solver.
        c: Inverse L2 strength of the logistic regression.
        max_iter: Solver iteration cap.
        class_weight: ``"balanced"`` or ``None``.
        max_shards: Read only the first shards of each split (``None`` = all; for quick checks).
    """

    data_dir: str = str(DEFAULT_DATA_DIR)
    out_dir: str = str(DEFAULT_OUT_DIR)
    seed: int = 0
    c: float = 1.0
    max_iter: int = 2000
    class_weight: str | None = "balanced"
    max_shards: int | None = None


def window_stats(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-window mean / std / min / max of every feature over real rows, plus the number of real rows.

    Args:
        x: Windows, shape ``(N, T, F)``.
        mask: 1 for real rows, 0 for padding, shape ``(N, T)``.

    Returns:
        Array ``(N, 4 * F + 1)``: means (F), stds (F), mins (F), maxs (F), n_real.
    """
    m = mask.astype(bool)
    n = m.sum(axis=1).astype(np.float64)
    safe_n = np.maximum(n, 1.0)[:, None]
    w = m[:, :, None]
    mean = np.where(w, x, 0.0).sum(axis=1) / safe_n
    var = np.where(w, (x - mean[:, None, :]) ** 2, 0.0).sum(axis=1) / safe_n
    xmin = np.where(w, x, np.inf).min(axis=1)
    xmax = np.where(w, x, -np.inf).max(axis=1)
    empty = n == 0
    xmin[empty] = 0.0
    xmax[empty] = 0.0
    return np.concatenate([mean, np.sqrt(var), xmin, xmax, n[:, None]], axis=1)


@dataclasses.dataclass
class WindowTable:
    """Per-window features of one split, in shard order."""

    ids: np.ndarray
    labels: np.ndarray
    n_messages: np.ndarray
    senders: np.ndarray
    stats: np.ndarray

    @classmethod
    def concat(cls, parts: list["WindowTable"]) -> "WindowTable":
        """Joins shard tables into one."""
        return cls(*(np.concatenate([getattr(p, f.name) for p in parts]) for f in dataclasses.fields(cls)))


class ShardReader:
    """Reads the T64 shards of one split and reduces every window to its statistics row."""

    def __init__(self, data_dir: Path, max_shards: int | None = None) -> None:
        self._data_dir = data_dir
        self._max_shards = max_shards
        self.metadata = json.loads((data_dir / "metadata.json").read_text())
        self.features = list(self.metadata["features"])

    def shard_entries(self, split: str) -> list[dict]:
        """Shard entries of ``metadata.json`` for one split."""
        entries = [s for s in self.metadata["shards"] if s["split"] == split]
        return entries[: self._max_shards] if self._max_shards else entries

    def iter_tables(self, split: str) -> Iterator[WindowTable]:
        """Yields one table per shard; the raw windows are dropped after reduction."""
        for entry in self.shard_entries(split):
            data = json.loads((self._data_dir / entry["file"]).read_text())
            info = json.loads((self._data_dir / entry["info"]).read_text())
            windows, meta = data["windows"], info["windows"]
            if len(windows) != len(meta):
                raise ValueError(f"{entry['file']}: {len(windows)} windows but {len(meta)} info rows")
            for w, i in zip(windows, meta):
                if w["id"] != i["id"]:
                    raise ValueError(f"{entry['file']}: id mismatch {w['id']} != {i['id']}")
            x = np.asarray([w["x"] for w in windows], dtype=np.float64)
            mask = np.asarray([w["mask"] for w in windows], dtype=np.float64)
            stats = window_stats(x, mask)
            n_messages = np.asarray([i["n_messages"] for i in meta], dtype=np.int64)
            if not np.array_equal(n_messages, stats[:, -1].astype(np.int64)):
                raise ValueError(f"{entry['file']}: n_messages differs from the mask sum")
            yield WindowTable(
                ids=np.asarray([i["id"] for i in meta]),
                labels=np.asarray([int(i["is_attack"]) for i in meta], dtype=np.int64),
                n_messages=n_messages,
                senders=np.asarray([int(i["sender"]) for i in meta], dtype=np.int64),
                stats=stats,
            )

    def read(self, split: str) -> WindowTable:
        """All windows of one split."""
        return WindowTable.concat(list(self.iter_tables(split)))


class LengthBaseline:
    """Score = n_messages, in the direction that separates the classes better."""

    def __init__(self) -> None:
        self.direction = 1

    def scores(self, n_messages: np.ndarray) -> np.ndarray:
        """Score of every window in the chosen direction."""
        return self.direction * n_messages.astype(np.float64)

    def choose_direction(self, n_messages: np.ndarray, labels: np.ndarray) -> dict:
        """Picks the better direction and returns both AUROCs.

        The direction is chosen on the labels passed in (the test split in this task, which is generous to the
        baseline: it can only make the shortcut look stronger).
        """
        auc_pos = float(roc_auc_score(labels, n_messages))
        self.direction = 1 if auc_pos >= 0.5 else -1
        return {
            "auroc_plus_n": auc_pos,
            "auroc_minus_n": 1.0 - auc_pos,
            "direction": (
                "+n_messages (longer = GridSybil)" if self.direction == 1 else "-n_messages (shorter = GridSybil)"
            ),
        }


class StatsLogisticBaseline:
    """Standardised per-window statistics -> logistic regression."""

    def __init__(self, config: BaselineConfig) -> None:
        self._model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=config.c, max_iter=config.max_iter, class_weight=config.class_weight, random_state=config.seed
            ),
        )

    def fit(self, stats: np.ndarray, labels: np.ndarray) -> "StatsLogisticBaseline":
        """Fits on train windows."""
        self._model.fit(stats, labels)
        return self

    def predict_proba(self, stats: np.ndarray) -> np.ndarray:
        """P(GridSybil) per window."""
        return self._model.predict_proba(stats)[:, 1]


def environment() -> dict:
    """Interpreter, library versions and git commit of this run."""
    import sklearn

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "executable": sys.executable,
    }
    try:
        import torch

        env["torch"] = torch.__version__
    except ImportError:  # pragma: no cover - torch is in .venv-train
        env["torch"] = None
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        env["git_commit"] = None
    return env


class BaselineRunner:
    """Reads both splits, fits the baselines on train, scores test, writes the run folder."""

    def __init__(self, config: BaselineConfig) -> None:
        self._config = config
        self._reader = ShardReader(Path(config.data_dir), config.max_shards)

    def feature_names(self) -> list[str]:
        """Column names of the statistics matrix."""
        names = [f"{s}_{f}" for s in STAT_NAMES for f in self._reader.features]
        return names + ["n_messages"]

    def run(self) -> dict:
        """Runs both baselines and returns the summary that is also written to ``summary.json``."""
        out = Path(self._config.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        start = time.time()
        train = self._reader.read("train")
        test = self._reader.read("test")
        read_s = time.time() - start

        stats_model = StatsLogisticBaseline(self._config).fit(train.stats, train.labels)
        stats_prob = stats_model.predict_proba(test.stats)
        train_auc = float(roc_auc_score(train.labels, stats_model.predict_proba(train.stats)))

        # Every test score exists from here on; test labels are used only below.
        length = LengthBaseline()
        length_info = length.choose_direction(test.n_messages, test.labels)
        np.savez(
            out / "scores_test.npz",
            id=test.ids,
            label=test.labels,
            n_messages=test.n_messages,
            sender=test.senders,
            length_score=length.scores(test.n_messages),
            stats_prob=stats_prob,
        )
        summary = {
            "n_train": int(len(train.ids)),
            "n_test": int(len(test.ids)),
            "train_counts": {"benign": int((train.labels == 0).sum()), "gridsybil": int((train.labels == 1).sum())},
            "test_counts": {"benign": int((test.labels == 0).sum()), "gridsybil": int((test.labels == 1).sum())},
            "length": length_info,
            "stats_lr": {
                "train_auroc": train_auc,
                "test_auroc": float(roc_auc_score(test.labels, stats_prob)),
                "n_features": int(train.stats.shape[1]),
            },
            "read_seconds": round(read_s, 1),
            "total_seconds": round(time.time() - start, 1),
        }
        config = dataclasses.asdict(self._config) | {"feature_names": self.feature_names()}
        (out / "config.json").write_text(json.dumps(config, indent=2))
        (out / "env.json").write_text(json.dumps(environment(), indent=2))
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary


def parse_args(argv: list[str] | None = None) -> BaselineConfig:
    """CLI -> config."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=None, help="read only the first shards per split")
    args = parser.parse_args(argv)
    return BaselineConfig(data_dir=args.data_dir, out_dir=args.out_dir, seed=args.seed, max_shards=args.max_shards)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    summary = BaselineRunner(parse_args(argv)).run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
