"""Window cache of the protocol: one input, one split, one normalisation for every detector.

Sources (read only):

* `all/T32_benign`: benign train / val / test and the four attack test splits, used as they are.
* `all/T32_benign_grid`: the GridSybil windows. The same crops as T32_benign, but another split and another
  normalisation. Every GridSybil window is assigned to train / val / test by its vehicle key in the **T32_benign**
  split, un-normalised with the grid input's mean / std and re-normalised with T32_benign's. The first message of
  a link has no previous message, so both inputs store its `log_dtau` as 0 (= "missing", not a measured value);
  that cell stays 0 instead of being shifted by the difference of the two means.

The test-vehicle GridSybil windows derived this way must reproduce `T32_benign/test_gridsybil`; they are kept in
the cache as `rederived_test_gridsybil` for that check only and are never used for training or evaluation.

Cache layout, `<cache_dir>/<split>/`: `x.npy` (N, 32, 13) float32 with zero padding rows, `mask.npy` (N, 32) bool
prefix mask, `n_messages.npy`, `vehicle_key.npy`, `scenario.npy`, `link_window_index.npy`, `window_id.npy` (bytes),
all in the same row
order; `<cache_dir>/meta.json` holds counts, the normalisation and the source manifest. No label array is stored:
the split a window is in is its class, and only `training_set("supervised")` / `validation_set("supervised")`
turn that into a label vector.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from baselinepaper.experiments import config as cfg

REDERIVED_TEST_GRIDSYBIL = "rederived_test_gridsybil"
LOG_DTAU = "log_dtau"
_ARRAYS = ("x", "mask", "n_messages", "vehicle_key", "scenario", "window_id")


def vehicle_key(scenario: str, sender: int | str) -> str:
    """`("GridSybil_0709", 10551)` -> `"0709:10551"`: one physical vehicle within one time window (the split key)."""
    return f"{scenario.rsplit('_', 1)[-1]}:{sender}"


def is_prefix_mask(mask: np.ndarray) -> bool:
    """True if every row is a non-empty run of True followed only by False."""
    mask = np.asarray(mask, dtype=bool)
    return bool(mask[:, 0].all() and not (np.diff(mask.astype(np.int8), axis=1) > 0).any())


@dataclasses.dataclass(frozen=True)
class WindowSet:
    """Unlabelled windows: all a detector may see outside the supervised regime.

    Attributes:
        x: (N, 32, 13) float32, normalised with the T32_benign statistics, padding rows zero.
        mask: (N, 32) bool, True = real message (prefix mask).
    """

    x: np.ndarray
    mask: np.ndarray

    def __len__(self) -> int:
        return len(self.x)


@dataclasses.dataclass(frozen=True)
class LabelledWindowSet(WindowSet):
    """Windows with labels; returned only for the `supervised` regime.

    Attributes:
        labels: (N,) int8, 0 = benign, 1 = GridSybil.
    """

    labels: np.ndarray


@dataclasses.dataclass(frozen=True)
class Split:
    """One cached split with its bookkeeping (evaluation and checks; detectors get `WindowSet`s).

    Attributes:
        name: cache split name (`config.ALL_SPLITS`).
        x: (N, 32, 13) float32 (memory-mapped).
        mask: (N, 32) bool (memory-mapped).
        n_messages: (N,) real rows per window.
        vehicle_key: (N,) `<group>:<sender vehicle>`, the split key and the bootstrap cluster.
        scenario: (N,) source folder, e.g. `GridSybil_0709`.
        link_window_index: (N,) position of the window in its link (0 = the link's first crop).
        window_id: (N,) bytes, `<receiver_file>#<sender_pseudo>-<sender>#<window index in the link>`.
    """

    name: str
    x: np.ndarray
    mask: np.ndarray
    n_messages: np.ndarray
    vehicle_key: np.ndarray
    scenario: np.ndarray
    link_window_index: np.ndarray
    window_id: np.ndarray

    def __len__(self) -> int:
        return len(self.n_messages)

    @property
    def family(self) -> np.ndarray:
        """(N,) attack family of the source folder (`GridSybil_0709` -> `GridSybil`)."""
        values, inverse = np.unique(self.scenario, return_inverse=True)
        return np.asarray([v.rsplit("_", 1)[0] for v in values])[inverse]

    @property
    def group(self) -> np.ndarray:
        """(N,) time window of the source folder (`GridSybil_0709` -> `0709`)."""
        values, inverse = np.unique(self.scenario, return_inverse=True)
        return np.asarray([v.rsplit("_", 1)[-1] for v in values])[inverse]

    def features(self) -> WindowSet:
        """x and mask only."""
        return WindowSet(self.x, self.mask)


def read_shard(job: tuple[str, str, int | None]) -> dict:
    """Decodes one shard and its info file.

    Args:
        job: (shard path, info field that holds the attack code, attack code to keep or None for all windows).

    Returns:
        x (W, T, F) float32 as stored (padding not touched), mask, ids, scenario, sender, link_window_index,
        n_messages and code of the kept windows.
    """
    path, code_field, keep_code = job
    p = Path(path)
    data = json.loads(p.read_text())["windows"]
    info = json.loads(p.with_name(p.stem + "_info.json").read_text())["windows"]
    if [w["id"] for w in data] != [w["id"] for w in info]:
        raise ValueError(f"{p}: window order differs from its info file")
    keep = [i for i, w in enumerate(info) if keep_code is None or w[code_field] == keep_code]
    seq_len, n_features = len(data[0]["mask"]), len(data[0]["x"][0])
    return {
        "x": np.asarray([data[i]["x"] for i in keep], dtype=np.float32).reshape(len(keep), seq_len, n_features),
        "mask": np.asarray([data[i]["mask"] for i in keep], dtype=bool).reshape(len(keep), seq_len),
        "ids": [info[i]["id"] for i in keep],
        "scenario": [info[i]["scenario"] for i in keep],
        "sender": [info[i]["sender"] for i in keep],
        "link_window_index": [info[i]["link_window_index"] for i in keep],
        "n_messages": [info[i]["n_messages"] for i in keep],
        "code": [info[i][code_field] for i in keep],
    }


class EncoderInput:
    """One encoder-input folder (`metadata.json` + `<split>/b32/part-*.json`), read only."""

    def __init__(self, root: str | Path, workers: int = 8) -> None:
        self.root = Path(root)
        self.workers = workers
        self.metadata = json.loads((self.root / "metadata.json").read_text())

    @property
    def features(self) -> list[str]:
        return list(self.metadata["normalisation"]["features"])

    @property
    def mean(self) -> np.ndarray:
        return np.asarray(self.metadata["normalisation"]["mean"], dtype=np.float64)

    @property
    def std(self) -> np.ndarray:
        return np.asarray(self.metadata["normalisation"]["std"], dtype=np.float64)

    def vehicle_ids(self, split: str) -> set[str]:
        """Vehicle keys the input's own split assigns to `split`."""
        return set(self.metadata["split"]["vehicle_ids"][split])

    def shards(self, split: str) -> list[Path]:
        paths = sorted((self.root / split).glob("b*/part-?????.json"))
        if not paths:
            raise FileNotFoundError(f"no shards for split {split!r} under {self.root}")
        return paths

    def manifest(self, splits: tuple[str, ...]) -> dict:
        """What the cache was built from: metadata hash and every shard's size and modification time."""
        entries = []
        for split in splits:
            for p in self.shards(split):
                info = p.with_name(p.stem + "_info.json")
                entries.append(
                    {
                        "file": str(p.relative_to(self.root)),
                        "size": p.stat().st_size,
                        "mtime": int(p.stat().st_mtime),
                        "info_size": info.stat().st_size,
                    }
                )
        digest = hashlib.sha256((self.root / "metadata.json").read_bytes()).hexdigest()
        return {"root": str(self.root), "metadata_sha256": digest, "shards": entries}

    def read(self, split: str, code_field: str, keep_code: int | None = None) -> dict:
        """All (kept) windows of `split`, shard after shard, as one dict of arrays."""
        jobs = [(str(p), code_field, keep_code) for p in self.shards(split)]
        if self.workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(min(self.workers, len(jobs)), mp_context=get_context("spawn")) as pool:
                parts = list(pool.map(read_shard, jobs))
        else:
            parts = [read_shard(j) for j in jobs]
        out = {
            "x": np.concatenate([p["x"] for p in parts]),
            "mask": np.concatenate([p["mask"] for p in parts]),
            "ids": np.asarray(sum((p["ids"] for p in parts), []), dtype=np.bytes_),
            "scenario": np.asarray(sum((p["scenario"] for p in parts), []), dtype=np.str_),
            "link_window_index": np.asarray(sum((p["link_window_index"] for p in parts), []), dtype=np.int64),
            "n_messages": np.asarray(sum((p["n_messages"] for p in parts), []), dtype=np.int64),
            "code": np.asarray(sum((p["code"] for p in parts), []), dtype=np.int64),
        }
        senders = sum((p["sender"] for p in parts), [])
        out["vehicle_key"] = np.asarray([vehicle_key(s, v) for s, v in zip(out["scenario"], senders)], dtype=np.str_)
        return out


class Renormaliser:
    """Maps windows normalised with one input's statistics to another input's normalisation.

    x' = (x * std_from + mean_from - mean_to) / std_to on real rows; padding rows stay zero; the `log_dtau` of the
    first message of a link (stored as 0 = missing in both inputs) stays 0.
    """

    def __init__(self, source: EncoderInput, target: EncoderInput) -> None:
        if source.features != target.features:
            raise ValueError("the two inputs list different features")
        self.scale = source.std / target.std
        self.shift = (source.mean - target.mean) / target.std
        self.log_dtau = source.features.index(LOG_DTAU)

    def __call__(self, x: np.ndarray, mask: np.ndarray, link_window_index: np.ndarray) -> np.ndarray:
        out = x.astype(np.float64) * self.scale + self.shift
        out[np.asarray(link_window_index) == 0, 0, self.log_dtau] = 0.0
        out[~mask] = 0.0
        return out.astype(np.float32)


@dataclasses.dataclass(frozen=True)
class FeatureStatistics:
    """Per-feature mean / std of real rows, expressed in the cached (T32_benign-normalised) units.

    Attributes:
        mean: (13,) float64.
        std: (13,) float64.
        n_values: (13,) values each statistic was computed from.
    """

    mean: np.ndarray
    std: np.ndarray
    n_values: np.ndarray

    def apply(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """(x - mean) / std on real rows; padding rows stay zero. Returns a new float32 array."""
        out = ((np.asarray(x, dtype=np.float64) - self.mean) / self.std).astype(np.float32)
        out[~np.asarray(mask, dtype=bool)] = 0.0
        return out

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "n_values": self.n_values.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureStatistics":
        return cls(
            np.asarray(d["mean"], dtype=np.float64), np.asarray(d["std"], dtype=np.float64), np.asarray(d["n_values"])
        )


class FeatureMoments:
    """Running per-feature sums over the valid cells of batches of windows."""

    def __init__(self, n_features: int) -> None:
        self.n = np.zeros(n_features, dtype=np.int64)
        self.s1 = np.zeros(n_features, dtype=np.float64)
        self.s2 = np.zeros(n_features, dtype=np.float64)

    def add(self, x: np.ndarray, valid: np.ndarray) -> None:
        """x (W, T, F), valid (W, T, F) bool: cells that count."""
        v = np.where(valid, x.astype(np.float64), 0.0)
        self.n += valid.sum(axis=(0, 1))
        self.s1 += v.sum(axis=(0, 1))
        self.s2 += (v * v).sum(axis=(0, 1))

    def statistics(self) -> FeatureStatistics:
        """Mean and population standard deviation (the encoder inputs' convention)."""
        mean = self.s1 / self.n
        return FeatureStatistics(mean, np.sqrt(np.maximum(self.s2 / self.n - mean * mean, 0.0)), self.n.copy())


class WindowCache:
    """Builds and serves the cached splits and the training / validation set of each regime.

    Attributes:
        config: the protocol settings (paths, GridSybil folders, pooling seed).
        root: cache folder.
    """

    def __init__(self, config: cfg.ProtocolConfig | None = None) -> None:
        self.config = config or cfg.ProtocolConfig()
        self.root = Path(self.config.cache_dir)
        self._splits: dict[str, Split] = {}

    # ------------------------------------------------------------------ reading

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    def exists(self) -> bool:
        """True once a build has finished (meta.json is written last)."""
        return self.meta_path.exists()

    @property
    def meta(self) -> dict:
        """Counts, normalisation and source manifest written by `build`."""
        if not self.exists():
            raise FileNotFoundError(
                f"no cache under {self.root}; run `python -m baselinepaper.experiments.data` from the repo root"
            )
        return json.loads(self.meta_path.read_text())

    def fingerprint(self) -> str:
        """sha256 of meta.json: recorded by every run, so score files can be tied to the cache they came from."""
        return hashlib.sha256(self.meta_path.read_bytes()).hexdigest()

    def split(self, name: str) -> Split:
        """One cached split; x, mask and window ids are memory-mapped."""
        if name not in self._splits:
            if name not in cfg.ALL_SPLITS and name != REDERIVED_TEST_GRIDSYBIL:
                raise KeyError(f"unknown split {name!r}; known: {cfg.ALL_SPLITS}")
            folder = self.root / name
            if not self.exists() or not folder.is_dir():
                raise FileNotFoundError(f"split {name!r} is not cached under {self.root}; build the cache first")
            big = {k: np.load(folder / f"{k}.npy", mmap_mode="r") for k in ("x", "mask", "window_id")}
            small = {
                k: np.load(folder / f"{k}.npy") for k in ("n_messages", "vehicle_key", "scenario", "link_window_index")
            }
            self._splits[name] = Split(name=name, **big, **small)
        return self._splits[name]

    def features(self, name: str) -> WindowSet:
        """x and mask of one split: what the runner hands to `Detector.score`."""
        return self.split(name).features()

    # ------------------------------------------------------------------ regimes

    def training_set(self, regime: str) -> WindowSet | LabelledWindowSet:
        """The training windows of a regime (protocol section 3).

        `benign_only`: benign train of all 8 folders. `benign_gridfolders`: benign train of the two GridSybil
        folders. `mix_all`: benign train of all 8 folders + GridSybil train, pooled. `mix_gridfolders`: benign +
        GridSybil train windows whose source folder is a GridSybil folder, pooled. `supervised`: as `mix_all`,
        with labels. Pooled sets are in one fixed random order (`pool_seed`), so the position of a window says
        nothing about its class. Only `supervised` returns a `LabelledWindowSet`; for every other regime the
        returned object has no label attribute at all.
        """
        return self._regime_set(regime, cfg.TRAIN_BENIGN, cfg.TRAIN_GRIDSYBIL)

    def validation_set(self, regime: str) -> WindowSet | LabelledWindowSet:
        """The validation windows of a regime: the same composition as its training set, from the val vehicles."""
        return self._regime_set(regime, cfg.VAL_BENIGN, cfg.VAL_GRIDSYBIL)

    def regime_rows(self, regime: str, benign: str, gridsybil: str) -> tuple[np.ndarray, np.ndarray]:
        """Rows of the splits `benign` and `gridsybil` that belong to a regime's pool (GridSybil rows may be empty).

        Evaluation uses it with the two validation splits to rebuild the scores of a regime's own validation pool.
        """
        if regime not in cfg.REGIMES:
            raise ValueError(f"unknown regime {regime!r}; known: {cfg.REGIMES}")
        b, g = self.split(benign), self.split(gridsybil)
        in_folders = regime in cfg.GRIDFOLDER_REGIMES
        rows_b = (
            np.flatnonzero(np.isin(b.scenario, self.config.gridsybil_scenarios)) if in_folders else np.arange(len(b))
        )
        if regime in (cfg.BENIGN_ONLY, cfg.BENIGN_GRIDFOLDERS):
            rows_g = np.arange(0)
        elif in_folders:
            rows_g = np.flatnonzero(np.isin(g.scenario, self.config.gridsybil_scenarios))
        else:
            rows_g = np.arange(len(g))
        return rows_b, rows_g

    def _regime_set(self, regime: str, benign: str, gridsybil: str) -> WindowSet | LabelledWindowSet:
        rows_b, rows_g = self.regime_rows(regime, benign, gridsybil)
        b, g = self.split(benign), self.split(gridsybil)
        if regime == cfg.BENIGN_ONLY:
            return b.features()
        if len(rows_g) == 0:
            return WindowSet(np.asarray(b.x[rows_b]), np.asarray(b.mask[rows_b]))
        n = len(rows_b) + len(rows_g)
        # Row i of the pooled set comes from position `order[i]` of [benign rows, GridSybil rows].
        order = np.random.default_rng(self.config.pool_seed).permutation(n)
        place = np.empty(n, dtype=np.int64)
        place[order] = np.arange(n)
        x = np.empty((n, *b.x.shape[1:]), dtype=np.float32)
        mask = np.empty((n, b.mask.shape[1]), dtype=bool)
        x[place[: len(rows_b)]], mask[place[: len(rows_b)]] = b.x[rows_b], b.mask[rows_b]
        x[place[len(rows_b) :]], mask[place[len(rows_b) :]] = g.x[rows_g], g.mask[rows_g]
        if regime != cfg.SUPERVISED:
            return WindowSet(x, mask)
        labels = np.zeros(n, dtype=np.int8)
        labels[place[len(rows_b) :]] = 1
        return LabelledWindowSet(x, mask, labels)

    def regime_composition(self, regime: str, part: str = "train") -> dict:
        """Window counts and attack share of a regime's training (`part="train"`) or validation (`"val"`) pool."""
        rows_b, rows_g = self.regime_rows(regime, f"{part}_benign", f"{part}_gridsybil")
        n = len(rows_b) + len(rows_g)
        return {"benign": len(rows_b), "gridsybil": len(rows_g), "windows": n, "attack_share": len(rows_g) / n}

    def regime_statistics(self, regime: str) -> "FeatureStatistics":
        """Per-feature mean / std of the real rows of a regime's training pool, in the cached units.

        Label-free: computed over the pooled training windows as they are. As in the encoder inputs, the
        `log_dtau` of the first message of a link (no previous message, stored as 0) is not a measured value and
        is left out of that feature's statistics. Detectors of the regimes in `config.RESTANDARDISED_REGIMES` use
        the result to re-standardise their inputs (`FeatureStatistics.apply`); `benign_only` and `supervised` keep
        the cached normalisation. Computed once per regime and stored in `<cache_dir>/regime_statistics.json`.
        """
        path = self.root / "regime_statistics.json"
        stored = json.loads(path.read_text()) if path.exists() else {}
        if stored.get("cache") != self.fingerprint():
            stored = {"cache": self.fingerprint(), "regimes": {}}
        if regime not in stored["regimes"]:
            rows_b, rows_g = self.regime_rows(regime, cfg.TRAIN_BENIGN, cfg.TRAIN_GRIDSYBIL)
            log_dtau = self.meta["features"].index(LOG_DTAU)
            total = FeatureMoments(self.config.n_features)
            for split, rows in ((self.split(cfg.TRAIN_BENIGN), rows_b), (self.split(cfg.TRAIN_GRIDSYBIL), rows_g)):
                for lo in range(0, len(rows), 65536):
                    r = rows[lo : lo + 65536]
                    valid = np.repeat(np.asarray(split.mask[r])[:, :, None], self.config.n_features, axis=2)
                    valid[split.link_window_index[r] == 0, 0, log_dtau] = False
                    total.add(np.asarray(split.x[r]), valid)
            stored["regimes"][regime] = total.statistics().to_dict()
            path.write_text(json.dumps(stored, indent=1))
        return FeatureStatistics.from_dict(stored["regimes"][regime])

    # ------------------------------------------------------------------ building

    def build(self, force: bool = False, log=print) -> dict:
        """Decodes both inputs into the cache and returns meta.json's content.

        Raises:
            RuntimeError: too little free disk space, or the re-normalised GridSybil test windows do not
                reproduce `T32_benign/test_gridsybil` within `max_rederive_abs_diff`.
        """
        if self.exists() and not force:
            log(f"cache present under {self.root} (use --force to rebuild)")
            return self.meta
        self.root.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(self.root).free / 1e9
        if free_gb < self.config.min_free_gb:
            raise RuntimeError(f"{free_gb:.1f} GB free, {self.config.min_free_gb} GB required to build the cache")
        if self.meta_path.exists():
            self.meta_path.unlink()
        self._splits.clear()
        benign_input = EncoderInput(self.config.benign_input_dir, self.config.workers)
        grid_input = EncoderInput(self.config.grid_input_dir, self.config.workers)
        counts, source_padding = {}, {}

        for name, folder in cfg.BENIGN_INPUT_SPLITS.items():
            part = benign_input.read(folder, code_field="label")
            codes = set(np.unique(part["code"]).tolist())
            if codes != {cfg.EXPECTED_ATTACK_CODE[name]}:
                raise RuntimeError(f"{name}: attack codes {codes}, expected {cfg.EXPECTED_ATTACK_CODE[name]}")
            source_padding[name] = self._save(name, part)
            counts[name] = len(part["ids"])
            log(f"{name}: {counts[name]:,} windows")

        grid_parts = [
            grid_input.read(s, code_field="attack_code", keep_code=cfg.GRIDSYBIL_CODE) for s in grid_input_splits()
        ]
        grid = {k: np.concatenate([p[k] for p in grid_parts]) for k in grid_parts[0]}
        source_padding["gridsybil_source"] = int(np.count_nonzero(grid["x"][~grid["mask"]]))
        grid["x"] = Renormaliser(grid_input, benign_input)(grid["x"], grid["mask"], grid["link_window_index"])
        assigned = np.zeros(len(grid["ids"]), dtype=bool)
        for name, source_split in (
            (cfg.TRAIN_GRIDSYBIL, "train"),
            (cfg.VAL_GRIDSYBIL, "val"),
            (REDERIVED_TEST_GRIDSYBIL, "test"),
        ):
            rows = np.isin(grid["vehicle_key"], sorted(benign_input.vehicle_ids(source_split)))
            if (assigned & rows).any():
                raise RuntimeError(f"a GridSybil vehicle key is in two T32_benign splits ({name})")
            assigned |= rows
            self._save(name, {k: v[rows] for k, v in grid.items()})
            counts[name] = int(rows.sum())
            log(f"{name}: {counts[name]:,} windows (from {grid_input.root.name}, re-normalised)")
        if not assigned.all():
            raise RuntimeError(f"{int((~assigned).sum())} GridSybil windows have a vehicle key in no T32_benign split")

        meta = {
            "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "config": self.config.to_dict(),
            "features": benign_input.features,
            "normalisation": {
                "applied": {"mean": benign_input.mean.tolist(), "std": benign_input.std.tolist()},
                "grid_input": {"mean": grid_input.mean.tolist(), "std": grid_input.std.tolist()},
            },
            "windows": counts,
            "source_nonzero_padding_values": source_padding,
            "sources": {
                "benign_input": benign_input.manifest(tuple(cfg.BENIGN_INPUT_SPLITS.values())),
                "grid_input": grid_input.manifest(grid_input_splits()),
            },
        }
        self.meta_path.write_text(json.dumps(meta, indent=1))
        check = self.rederivation_check()
        log(f"re-derived GridSybil test windows vs T32_benign/test_gridsybil: {check}")
        if not check["passed"]:
            self.meta_path.unlink()
            raise RuntimeError(f"re-derived GridSybil test windows differ from T32_benign/test_gridsybil: {check}")
        return meta

    def _save(self, name: str, part: dict) -> int:
        """Writes one split; returns how many padding values were non-zero in the source."""
        x, mask = part["x"], part["mask"]
        if x.shape[1:] != (self.config.seq_len, self.config.n_features):
            raise RuntimeError(f"{name}: windows of shape {x.shape[1:]}")
        if not is_prefix_mask(mask):
            raise RuntimeError(f"{name}: masks are not non-empty prefix masks")
        if not np.array_equal(mask.sum(axis=1), part["n_messages"]):
            raise RuntimeError(f"{name}: mask row counts differ from n_messages in the info files")
        if len(np.unique(part["ids"])) != len(part["ids"]):
            raise RuntimeError(f"{name}: duplicate window ids")
        nonzero_padding = int(np.count_nonzero(x[~mask]))
        x[~mask] = 0.0
        folder = self.root / name
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "x.npy", x)
        np.save(folder / "mask.npy", mask)
        np.save(folder / "n_messages.npy", part["n_messages"].astype(np.int16))
        np.save(folder / "vehicle_key.npy", part["vehicle_key"])
        np.save(folder / "scenario.npy", part["scenario"])
        np.save(folder / "link_window_index.npy", part["link_window_index"].astype(np.int16))
        np.save(folder / "window_id.npy", part["ids"])
        return nonzero_padding

    def rederivation_check(self) -> dict:
        """Compares `rederived_test_gridsybil` with `test_gridsybil` window by window (matched by id)."""
        ref, new = self.split("test_gridsybil"), self.split(REDERIVED_TEST_GRIDSYBIL)
        ref_ids, new_ids = np.asarray(ref.window_id), np.asarray(new.window_id)
        same_ids = len(ref_ids) == len(new_ids) and np.array_equal(np.sort(ref_ids), np.sort(new_ids))
        out = {"windows_reference": len(ref_ids), "windows_rederived": len(new_ids), "same_id_set": bool(same_ids)}
        if same_ids:
            a, b = np.argsort(ref_ids), np.argsort(new_ids)
            diff = np.abs(np.asarray(ref.x)[a].astype(np.float64) - np.asarray(new.x)[b])
            out["masks_identical"] = bool(np.array_equal(np.asarray(ref.mask)[a], np.asarray(new.mask)[b]))
            out["max_abs_diff"] = float(diff.max())
            out["max_abs_diff_per_feature"] = diff.max(axis=(0, 1)).tolist()
            out["mean_abs_diff_real_rows"] = float(diff[np.asarray(ref.mask)[a]].mean())
        out["tolerance"] = self.config.max_rederive_abs_diff
        out["passed"] = bool(
            same_ids and out["masks_identical"] and out["max_abs_diff"] <= self.config.max_rederive_abs_diff
        )
        return out

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())


def grid_input_splits() -> tuple[str, ...]:
    """Split folders of the T32_benign_grid input (all are read; its own split is not used)."""
    return ("train", "val", "test")


def main() -> None:
    """`python -m baselinepaper.experiments.data [--force]`: builds the cache."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild an existing cache")
    args = parser.parse_args()
    cache = WindowCache()
    meta = cache.build(force=args.force)
    print(json.dumps(meta["windows"], indent=1))
    print(f"cache size: {cache.size_bytes() / 1e9:.2f} GB under {cache.root}")


if __name__ == "__main__":
    main()
