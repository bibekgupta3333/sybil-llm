"""Reproducible data-preparation pipeline for VeReMi-Extension (plan task 2.6).

Replaces the windowing/feature/split cells of ``notebooks/eda_veremi.ipynb`` with a
class-based, testable script, and fixes the GridSybil_0709 windowing defect documented in
``docs/research-notes/data_understanding/gridsybil-windowing-defect.md``:

The notebook grouped ground-truth messages by ``(family, group, subfolder, senderPseudo)``
to build per-identity sequences before windowing. In ``GridSybil_0709``, the ground-truth
value ``senderPseudo == 1`` is a shared sentinel across 653 distinct physical vehicles, so
that grouping silently spliced 653 unrelated vehicles' messages together by timestamp,
producing windows with physically meaningless ``dt``/``dpos``/``dspd`` values.

The fix: the sequence key is ``(family, group, subfolder, sender, senderPseudo)`` --
``sender`` (the physical vehicle, resolved 100%-reliably from the trace filename, see
``VehicleLabelResolver``) always disambiguates the collision, and every previously-correct
identity is completely unaffected (its ``sender`` was already implied by its unique
``senderPseudo``). ``WindowBuilder`` additionally asserts, defensively, that no sequence
group ever mixes more than one physical sender.

Read-only against ``data/VeReMi-Dataset/``. Writes only under ``--out`` (never
``data/prepared_data/`` unless explicitly requested -- see CLAUDE.md rule 2).

Usage:
    .venv/bin/python scripts/prepare_data.py --base data/VeReMi-Dataset --out data/prepared_data_v2
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import re
import sys
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

_LOG = logging.getLogger("prepare_data")

_TRACE_FILENAME = re.compile(r"traceJSON-(\d+)-(\d+)-A(\d+)-(\d+)-(\d+)\.json")
_ATTACK_MAP: dict[int, str] = {
    0: "Benign",
    16: "GridSybil",
    17: "DataReplaySybil",
    18: "DoSRandomSybil",
    19: "DoSDisruptiveSybil",
}
# The sequence key: adding `sender` over the notebook's (family, group, subfolder, senderPseudo)
# is the fix for the GridSybil_0709 splice (see module docstring).
_SEQUENCE_KEYS = ["family", "group", "subfolder", "sender", "senderPseudo"]
_FEATURE_COLS = [
    "pos_x",
    "pos_y",
    "spd_x",
    "spd_y",
    "acl_x",
    "acl_y",
    "hed_x",
    "hed_y",
    "dt",
    "dpos_x",
    "dpos_y",
    "dspd_x",
    "dspd_y",
]
_TELEPORT_SPEED_MPS = 60.0  # implied speed no honest vehicle reaches; a residual splice would


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    """Frozen knobs for one prepared-data build, recorded verbatim in the output manifest.

    Attributes:
        base_dir: root of the raw ``VeReMi-Dataset`` folder (read-only).
        out_dir: destination for the prepared artifacts.
        window_size: timesteps per window (T).
        stride: sliding-window stride.
        min_seq_len: minimum messages in a sequence to emit at least one window.
        test_size: fraction held out for test in the first ``GroupShuffleSplit``.
        val_size: fraction of the remainder held out for validation in the second split.
        seed: random state for both splits.
    """

    base_dir: pathlib.Path = pathlib.Path("data/VeReMi-Dataset")
    out_dir: pathlib.Path = pathlib.Path("data/prepared_data_v2")
    window_size: int = 20
    stride: int = 10
    min_seq_len: int = 20
    test_size: float = 0.2
    val_size: float = 0.2
    seed: int = 42


@dataclasses.dataclass(frozen=True)
class ParsedTraceFilename:
    """Fields decoded from a ``traceJSON-{node}-{pseudo}-A{code}-{start}-{sim}.json`` name."""

    node_id: int
    pseudo_id: int
    attack_code: int
    start_time: int
    sim_num: int

    @staticmethod
    def parse(name: str) -> "ParsedTraceFilename | None":
        match = _TRACE_FILENAME.match(name)
        if not match:
            return None
        g = match.groups()
        return ParsedTraceFilename(int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4]))


@dataclasses.dataclass(frozen=True)
class ScenarioRun:
    """One (family, group, subfolder) simulation run directory."""

    family: str
    group: str
    subfolder: str
    path: pathlib.Path

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.family, self.group, self.subfolder)


class ScenarioIndex:
    """Enumerates every simulation run under the raw dataset root."""

    def __init__(self, base_dir: pathlib.Path) -> None:
        self._base_dir = base_dir

    def runs(self) -> list[ScenarioRun]:
        runs = []
        for family_dir in sorted(p for p in self._base_dir.iterdir() if p.is_dir()):
            family, _, group = family_dir.name.rpartition("_")
            if not family:
                raise ValueError(f"cannot split family/group from folder name {family_dir.name!r}")
            for sub_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
                runs.append(ScenarioRun(family, group, sub_dir.name, sub_dir))
        if not runs:
            raise FileNotFoundError(f"no scenario runs found under {self._base_dir}")
        return runs


class VehicleLabelResolver:
    """Resolves an attack label for every (run, physical sender) pair from trace filenames."""

    def resolve(self, runs: Sequence[ScenarioRun]) -> pd.DataFrame:
        records = []
        for run in runs:
            for trace_file in sorted(run.path.glob("traceJSON-*.json")):
                parsed = ParsedTraceFilename.parse(trace_file.name)
                if parsed is None:
                    continue
                records.append(
                    {
                        "family": run.family,
                        "group": run.group,
                        "subfolder": run.subfolder,
                        "sender": parsed.node_id,
                        "attack_code": parsed.attack_code,
                        "attack_label": _ATTACK_MAP.get(parsed.attack_code, f"Unknown_{parsed.attack_code}"),
                        "is_attacker": parsed.attack_code != 0,
                    }
                )
        table = pd.DataFrame.from_records(records)
        if table.empty:
            raise ValueError("no vehicle trace filenames matched the expected pattern")
        return table


class GroundTruthLoader:
    """Streams every run's ground-truth ledger into one flat, labeled dataframe."""

    def __init__(self, labels: pd.DataFrame) -> None:
        self._lookup = {
            (row.family, row.group, row.subfolder, row.sender): (row.attack_code, row.attack_label, row.is_attacker)
            for row in labels.itertuples()
        }

    def load(self, runs: Sequence[ScenarioRun]) -> pd.DataFrame:
        frames = [pd.DataFrame(self._rows(run)) for run in runs]
        table = pd.concat(frames, ignore_index=True)
        if table.empty:
            raise ValueError("no ground-truth messages were loaded")
        return table

    def _rows(self, run: ScenarioRun) -> Iterator[dict[str, Any]]:
        ledgers = sorted(run.path.glob("traceGroundTruthJSON-*.json"))
        if not ledgers:
            raise FileNotFoundError(f"no ground-truth ledger under {run.path}")
        for ledger in ledgers:
            with open(ledger, encoding="utf-8") as fh:
                for line in fh:
                    obj = json.loads(line)
                    code, label, is_attacker = self._lookup.get(
                        (run.family, run.group, run.subfolder, obj["sender"]), (-1, "Unknown", False)
                    )
                    yield {
                        "sendTime": obj["sendTime"],
                        "sender": obj["sender"],
                        "senderPseudo": obj["senderPseudo"],
                        "pos_x": obj["pos"][0],
                        "pos_y": obj["pos"][1],
                        "spd_x": obj["spd"][0],
                        "spd_y": obj["spd"][1],
                        "acl_x": obj["acl"][0],
                        "acl_y": obj["acl"][1],
                        "hed_x": obj["hed"][0],
                        "hed_y": obj["hed"][1],
                        "family": run.family,
                        "group": run.group,
                        "subfolder": run.subfolder,
                        "attack_code": code,
                        "attack_label": label,
                        "is_attacker": is_attacker,
                    }


class FeatureEngineer:
    """Sorts ground truth into sequences and derives the delta features within each."""

    def __init__(self, sequence_keys: Sequence[str] = tuple(_SEQUENCE_KEYS)) -> None:
        self._keys = list(sequence_keys)

    def add_deltas(self, ground_truth: pd.DataFrame) -> pd.DataFrame:
        sorted_gt = ground_truth.sort_values([*self._keys, "sendTime"]).copy()
        grouped = sorted_gt.groupby(self._keys, sort=False)
        sorted_gt["dt"] = grouped["sendTime"].diff().fillna(0.0)
        sorted_gt["dpos_x"] = grouped["pos_x"].diff().fillna(0.0)
        sorted_gt["dpos_y"] = grouped["pos_y"].diff().fillna(0.0)
        sorted_gt["dspd_x"] = grouped["spd_x"].diff().fillna(0.0)
        sorted_gt["dspd_y"] = grouped["spd_y"].diff().fillna(0.0)
        return sorted_gt


@dataclasses.dataclass(frozen=True)
class WindowSet:
    """The prepared window tensor and its per-window metadata, before splitting."""

    features: np.ndarray  # (N, T, D) float32
    metadata: pd.DataFrame
    max_implied_speed: float


class WindowBuilder:
    """Slices each sequence into fixed-length, fixed-stride windows."""

    def __init__(self, config: PipelineConfig, sequence_keys: Sequence[str] = tuple(_SEQUENCE_KEYS)) -> None:
        self._cfg = config
        self._keys = list(sequence_keys)

    def build(self, features_df: pd.DataFrame) -> WindowSet:
        windows: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []
        max_implied_speed = 0.0
        for name, group in features_df.groupby(self._keys, sort=False):
            if len(group) < self._cfg.min_seq_len:
                continue
            self._assert_single_sender(group, name)
            values = group[_FEATURE_COLS].to_numpy(dtype=np.float32)
            max_implied_speed = max(max_implied_speed, self._max_implied_speed(values))
            for start in range(0, len(values) - self._cfg.window_size + 1, self._cfg.stride):
                window = values[start : start + self._cfg.window_size]
                windows.append(window)
                rows.append(self._window_metadata(name, group, start))
        if not windows:
            raise ValueError("no windows were produced -- check min_seq_len against the input data")
        return WindowSet(np.stack(windows), pd.DataFrame(rows), max_implied_speed)

    def _assert_single_sender(self, group: pd.DataFrame, name: tuple[Any, ...]) -> None:
        senders = group["sender"].unique()
        if len(senders) != 1:
            raise ValueError(
                f"sequence group {name} mixes {len(senders)} physical senders: {senders} "
                "-- the grouping key no longer disambiguates identities"
            )

    def _max_implied_speed(self, values: np.ndarray) -> float:
        dt = values[1:, _FEATURE_COLS.index("dt")]
        dpos_x = values[1:, _FEATURE_COLS.index("dpos_x")]
        dpos_y = values[1:, _FEATURE_COLS.index("dpos_y")]
        with np.errstate(divide="ignore", invalid="ignore"):
            implied = np.hypot(dpos_x, dpos_y) / dt
        implied = implied[np.isfinite(implied)]
        return float(implied.max()) if implied.size else 0.0

    def _window_metadata(self, name: tuple[Any, ...], group: pd.DataFrame, start: int) -> dict[str, Any]:
        family, grp, subfolder, sender, pseudo = name
        end = start + self._cfg.window_size - 1
        return {
            "family": family,
            "group": grp,
            "subfolder": subfolder,
            "sender": sender,
            "senderPseudo": pseudo,
            "attack_code": group["attack_code"].iloc[0],
            "attack_label": group["attack_label"].iloc[0],
            "is_attacker": bool(group["is_attacker"].iloc[0]),
            "window_start_idx": start,
            "start_time": float(group["sendTime"].iloc[start]),
            "end_time": float(group["sendTime"].iloc[end]),
        }


@dataclasses.dataclass(frozen=True)
class Splits:
    """Sender-disjoint train/val/test plus the orthogonal scenario-holdout partition."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    group_0709_idx: np.ndarray
    group_1416_idx: np.ndarray
    sender_uid: pd.Series


class SplitBuilder:
    """Sender-level GroupShuffleSplit train/val/test, plus the 0709/1416 scenario holdout."""

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config

    def build(self, metadata: pd.DataFrame) -> Splits:
        sender_uid = (
            metadata["family"]
            + "_"
            + metadata["group"]
            + "_"
            + metadata["subfolder"]
            + "_"
            + metadata["sender"].astype(str)
            + "_"
            + metadata["senderPseudo"].astype(str)
        )
        n = len(metadata)
        gss1 = GroupShuffleSplit(n_splits=1, test_size=self._cfg.test_size, random_state=self._cfg.seed)
        train_val_idx, test_idx = next(gss1.split(np.zeros(n), groups=sender_uid))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=self._cfg.val_size, random_state=self._cfg.seed)
        rel_train_idx, rel_val_idx = next(gss2.split(train_val_idx, groups=sender_uid.iloc[train_val_idx]))
        train_idx, val_idx = train_val_idx[rel_train_idx], train_val_idx[rel_val_idx]
        self._assert_disjoint(sender_uid, train_idx, val_idx, test_idx)

        group_0709 = np.where((metadata["group"] == "0709").to_numpy())[0]
        group_1416 = np.where((metadata["group"] == "1416").to_numpy())[0]
        return Splits(train_idx, val_idx, test_idx, group_0709, group_1416, sender_uid)

    @staticmethod
    def _assert_disjoint(
        sender_uid: pd.Series, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray
    ) -> None:
        train_s, val_s, test_s = (set(sender_uid.iloc[i]) for i in (train_idx, val_idx, test_idx))
        if train_s & val_s or train_s & test_s or val_s & test_s:
            raise AssertionError("sender-level leakage across train/val/test -- refusing to write a corrupted split")


class Normalizer:
    """Per-feature z-score statistics computed from the train split only."""

    @staticmethod
    def fit(windows: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat = windows[train_idx].reshape(-1, windows.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std[std < 1e-8] = 1.0
        return mean, std


class ArtifactWriter:
    """Serialises the prepared windows, splits, and manifest to ``out_dir``."""

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config

    def write(
        self, window_set: WindowSet, splits: Splits, norm_mean: np.ndarray, norm_std: np.ndarray
    ) -> dict[str, Any]:
        out = self._cfg.out_dir
        out.mkdir(parents=True, exist_ok=True)
        metadata = window_set.metadata.assign(sender_uid=splits.sender_uid)
        label_to_int = {label: i for i, label in enumerate(sorted(metadata["attack_label"].unique()))}
        y_multiclass = metadata["attack_label"].map(label_to_int).to_numpy(dtype=np.int64)
        y_binary = metadata["is_attacker"].astype(np.int64).to_numpy()

        np.save(out / "X_windows.npy", window_set.features)
        np.save(out / "y_multiclass.npy", y_multiclass)
        np.save(out / "y_binary.npy", y_binary)
        np.save(out / "idx_train.npy", splits.train_idx)
        np.save(out / "idx_val.npy", splits.val_idx)
        np.save(out / "idx_test.npy", splits.test_idx)
        np.save(out / "idx_group_0709.npy", splits.group_0709_idx)
        np.save(out / "idx_group_1416.npy", splits.group_1416_idx)
        np.save(out / "norm_mean.npy", norm_mean)
        np.save(out / "norm_std.npy", norm_std)
        metadata.to_parquet(out / "window_metadata.parquet", index=False)

        config = {
            "window_size": self._cfg.window_size,
            "stride": self._cfg.stride,
            "min_seq_len": self._cfg.min_seq_len,
            "sequence_keys": _SEQUENCE_KEYS,
            "feature_cols": _FEATURE_COLS,
            "feature_dim": len(_FEATURE_COLS),
            "n_windows": int(window_set.features.shape[0]),
            "n_train": int(len(splits.train_idx)),
            "n_val": int(len(splits.val_idx)),
            "n_test": int(len(splits.test_idx)),
            "n_group_0709": int(len(splits.group_0709_idx)),
            "n_group_1416": int(len(splits.group_1416_idx)),
            "label_to_int": label_to_int,
            "int_to_label": {str(v): k for k, v in label_to_int.items()},
            "norm_mean": norm_mean.tolist(),
            "norm_std": norm_std.tolist(),
            "max_implied_speed_mps": window_set.max_implied_speed,
            "seed": self._cfg.seed,
        }
        with open(out / "config.json", "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
        return config


class DataPreparationPipeline:
    """Orchestrates scenario indexing through artifact writing."""

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config

    def run(self) -> dict[str, Any]:
        runs = ScenarioIndex(self._cfg.base_dir).runs()
        _LOG.info("found %d scenario runs", len(runs))
        labels = VehicleLabelResolver().resolve(runs)
        _LOG.info("resolved labels for %d (run, sender) pairs", len(labels))
        ground_truth = GroundTruthLoader(labels).load(runs)
        _LOG.info("loaded %d ground-truth messages", len(ground_truth))
        features = FeatureEngineer().add_deltas(ground_truth)
        window_set = WindowBuilder(self._cfg).build(features)
        _LOG.info(
            "built %d windows (max implied speed in-sequence: %.1f m/s)",
            len(window_set.features),
            window_set.max_implied_speed,
        )
        if window_set.max_implied_speed > _TELEPORT_SPEED_MPS:
            _LOG.info(
                "max implied speed exceeds the %.0f m/s teleport threshold -- expected for attacker "
                "sequences (the discriminative signal itself); would indicate a residual splice bug "
                "if it appeared on a benign sequence instead",
                _TELEPORT_SPEED_MPS,
            )
        splits = SplitBuilder(self._cfg).build(window_set.metadata)
        norm_mean, norm_std = Normalizer.fit(window_set.features, splits.train_idx)
        config = ArtifactWriter(self._cfg).write(window_set, splits, norm_mean, norm_std)
        _LOG.info("wrote prepared data to %s", self._cfg.out_dir)
        return config


def _parse_args(argv: Sequence[str] | None) -> PipelineConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=pathlib.Path, default=PipelineConfig.base_dir)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=PipelineConfig.out_dir,
        help="Defaults to a NEW directory, never data/prepared_data/, per CLAUDE.md rule 2.",
    )
    parser.add_argument("--window-size", type=int, default=PipelineConfig.window_size)
    parser.add_argument("--stride", type=int, default=PipelineConfig.stride)
    parser.add_argument("--min-seq-len", type=int, default=PipelineConfig.min_seq_len)
    parser.add_argument("--seed", type=int, default=PipelineConfig.seed)
    args = parser.parse_args(argv)
    return PipelineConfig(
        base_dir=args.base,
        out_dir=args.out,
        window_size=args.window_size,
        stride=args.stride,
        min_seq_len=args.min_seq_len,
        seed=args.seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    DataPreparationPipeline(_parse_args(argv)).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
