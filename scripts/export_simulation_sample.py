"""Export a stratified sample of prepared trajectory windows for the browser simulator.

Reads ``data/prepared_data/`` (read-only), stitches each sampled identity's stride-10 windows
back into its unique step sequence, resolves broadcast pseudonyms to physical vehicles from the
raw ground-truth ledgers, and writes a compact bundle to ``simulation/public/data/``:

Also exports, for up to ``forged_pairs_per_cell`` sampled attackers per (family, group) cell, a
"forged broadcast" companion identity (tag ``forged_broadcast``, ``data_source: "raw_veremi"``): the
ground-truth ledger -- and therefore every "prepared" identity above -- only ever records a vehicle's
own *true* physical position, under whichever pseudonym it is legitimately using. A Sybil attacker's
*fabricated* position only appears in a neighboring vehicle's received-message log (a type-3 entry
whose ``senderPseudo`` never appears in ground truth for that sender). Those are read directly from
the raw per-vehicle trace files and exported as extra identities sharing the real one's physical
``sender``; each prepared attacker identity that got a companion records the companion's id in
``paired_identity_id`` (and vice versa), so the simulator can show -- and let a researcher jump
between -- the true path (prepared data) and the lie (raw VeReMi, never in ``X_windows.npy``) for the
same physical vehicle.

Each forged identity is also matched with the Benign identity from the same run whose time span
overlaps it longest (``benign_match_id`` plus the shared ``overlap: [t0, t1]``), requiring at least one
full window of steps on both sides inside the overlap. Benign vehicles not already in the sample are
added and tagged ``overlap_benign``. The simulator's "Benign vs attack" view clips both to ``overlap``.

* ``manifest.json`` - feature names, normalization stats, class palette, and one record per identity
  (metadata, provenance, byte offsets into the binaries).
* ``steps.f32``     - row-major ``float32`` matrix of shape ``(total_steps, 13)``.
* ``times.f64``     - absolute simulation time (seconds) per step, ``float64``.

Every stitching step is asserted against the source windows so the exported sequences are
bit-identical to what the model sees. Nothing under ``data/`` is ever written.

Usage:
    .venv/bin/python scripts/export_simulation_sample.py --out simulation/public/data --seed 42
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import pathlib
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

_LOG = logging.getLogger("export_simulation_sample")

_WINDOW = 20
_STRIDE = 10
_BENIGN = "Benign"
_FORGED_TAG = "forged_broadcast"
# How each family's attack shows up in neighbors' received messages (verified on the raw 1416 runs):
# GridSybil broadcasts ghosts under pseudonyms ground truth never records, so one fake pseudonym is the
# attack. The other three rotate pseudonyms every 1-2 messages (all recorded in ground truth) while
# broadcasting positions far from the truth, so the attack is everything heard from that vehicle.
_FAKE_PSEUDONYM, _ALL_PSEUDONYMS = "fake_pseudonym", "all_pseudonyms"
_ATTACK_TRACE_STRATEGY: dict[str, str] = {"GridSybil": _FAKE_PSEUDONYM}


def attack_trace_strategy(family: str) -> str:
    """`fake_pseudonym` for GridSybil, `all_pseudonyms` for DataReplay / DoSRandom / DoSDisruptive."""
    return _ATTACK_TRACE_STRATEGY.get(family, _ALL_PSEUDONYMS)


_MIN_FORGED_STEPS = _WINDOW  # need at least one full window for the heatmap/consistency views
_CLASS_PALETTE: dict[str, str] = {
    "Benign": "#16A34A",
    "GridSybil": "#DC2626",
    "DoSDisruptiveSybil": "#F97316",
    "DoSRandomSybil": "#2563EB",
    "DataReplaySybil": "#8B5CF6",
}
# Ledger lines are written with `sender` immediately before `senderPseudo`.
_LEDGER_IDS = re.compile(r'"sender":\s*(\d+),\s*"senderPseudo":\s*(\d+)')


class StitchError(ValueError):
    """Raised when an identity's windows cannot be stitched into one consistent sequence."""


@dataclasses.dataclass(frozen=True)
class SampleConfig:
    """Knobs for the stratified sample.

    Attributes:
        seed: RNG seed, recorded in the manifest.
        pseudonyms_per_cell: target surviving identities per (class, group) cell.
        max_pseudonyms_per_sender: cap so one prolific attacker cannot fill a cell.
        stress_per_class: longest normal identities to add per class, tagged ``stress``.
        max_normal_windows: identities above this are excluded from normal draws (the two
            spliced GridSybil_0709 sentinels are the only ones that exceed it).
        defect_demo_windows: how many leading windows of each spliced identity to export.
        forged_pairs_per_cell: how many sampled attackers per (family, group) cell to attempt a
            forged-broadcast pairing for (see module docstring). Not every attempt succeeds.
        prepared_dir: ``data/prepared_data``.
        raw_dir: ``data/VeReMi-Dataset`` (ground-truth ledgers, read-only).
        out_dir: destination for the bundle.
        cache_dir: destination for the pseudonym-map cache (gitignored).
    """

    seed: int = 42
    pseudonyms_per_cell: int = 35
    max_pseudonyms_per_sender: int = 8
    stress_per_class: int = 2
    max_normal_windows: int = 500
    defect_demo_windows: int = 40
    forged_pairs_per_cell: int = 6
    prepared_dir: pathlib.Path = pathlib.Path("data/prepared_data")
    raw_dir: pathlib.Path = pathlib.Path("data/VeReMi-Dataset")
    out_dir: pathlib.Path = pathlib.Path("simulation/public/data")
    cache_dir: pathlib.Path = pathlib.Path("simulation/.cache")


@dataclasses.dataclass(frozen=True)
class StitchedIdentity:
    """One identity's unique step sequence reconstructed from its windows.

    Attributes:
        steps: ``(n_steps, 13)`` float32, bit-identical to the source windows.
        times: ``(n_steps,)`` float64 absolute simulation seconds.
        window_rows: row indices into ``X_windows.npy`` in ``window_start_idx`` order.
    """

    steps: np.ndarray
    times: np.ndarray
    window_rows: np.ndarray

    @property
    def n_windows(self) -> int:
        return int(len(self.window_rows))


class WindowStore:
    """Read-only access to the prepared windows, metadata, and prep manifest."""

    def __init__(self, prepared_dir: pathlib.Path) -> None:
        self._dir = prepared_dir
        self.x = np.load(prepared_dir / "X_windows.npy", mmap_mode="r")
        self.meta = pd.read_parquet(prepared_dir / "window_metadata.parquet")
        with open(prepared_dir / "config.json", encoding="utf-8") as fh:
            self.config: dict[str, Any] = json.load(fh)
        if self.x.shape[0] != len(self.meta):
            raise ValueError("X_windows.npy and window_metadata.parquet disagree on row count")
        self._by_uid = self.meta.groupby("sender_uid", sort=False).indices

    @property
    def feature_cols(self) -> list[str]:
        return list(self.config["feature_cols"])

    def feature_index(self, name: str) -> int:
        return self.feature_cols.index(name)

    def uids(self) -> Iterable[str]:
        return self._by_uid.keys()

    def rows_for(self, uid: str) -> np.ndarray:
        """Row indices (in file order) of all windows belonging to ``uid``."""
        return np.asarray(self._by_uid[uid])

    def window_counts(self) -> pd.Series:
        return self.meta.groupby("sender_uid", sort=False).size()

    def identity_table(self) -> pd.DataFrame:
        """One row per sender_uid with its static metadata and window count."""
        first = self.meta.drop_duplicates("sender_uid").set_index("sender_uid")
        first = first[["family", "group", "subfolder", "senderPseudo", "attack_code", "attack_label", "is_attacker"]]
        return first.join(self.window_counts().rename("n_windows"))


class IdentityStitcher:
    """Rebuilds an identity's step sequence from its overlapping windows, with proofs."""

    def __init__(self, store: WindowStore, max_windows: int | None = None) -> None:
        self._store = store
        self._dt = store.feature_index("dt")
        self._max_windows = max_windows

    def stitch(self, uid: str) -> StitchedIdentity:
        rows = self._store.rows_for(uid)
        sub = self._store.meta.iloc[rows]
        order = np.argsort(sub["window_start_idx"].to_numpy(), kind="stable")
        rows, sub = rows[order], sub.iloc[order]
        if self._max_windows is not None:
            rows, sub = rows[: self._max_windows], sub.iloc[: self._max_windows]

        starts = sub["window_start_idx"].to_numpy()
        expected = np.arange(len(starts)) * _STRIDE
        if not np.array_equal(starts, expected):
            raise StitchError(f"{uid}: window_start_idx has gaps or a non-{_STRIDE} stride")

        windows = np.asarray(self._store.x[rows])  # (K, 20, 13), sorted rows -> real read
        if np.isnan(windows).any():
            raise StitchError(f"{uid}: NaN in source windows")
        for k in range(len(windows) - 1):
            if not np.array_equal(windows[k, _STRIDE:], windows[k + 1, :_STRIDE]):
                raise StitchError(f"{uid}: windows {k} and {k + 1} disagree on their overlap")

        head = windows[:-1, :_STRIDE].reshape(-1, windows.shape[2])
        steps = np.concatenate([head, windows[-1]], axis=0)
        self._verify_reslice(steps, windows, uid)
        times = self._anchor_times(steps, sub, uid)
        return StitchedIdentity(steps=steps, times=times, window_rows=rows)

    @staticmethod
    def _verify_reslice(steps: np.ndarray, windows: np.ndarray, uid: str) -> None:
        for k in range(len(windows)):
            if not np.array_equal(steps[k * _STRIDE : k * _STRIDE + _WINDOW], windows[k]):
                raise StitchError(f"{uid}: re-slicing window {k} does not reproduce the source")

    def _anchor_times(self, steps: np.ndarray, sub: pd.DataFrame, uid: str) -> np.ndarray:
        """Anchor each window at its float64 start_time; fill within-window via cumsum(dt)."""
        dt = steps[:, self._dt].astype(np.float64)
        times = np.empty(len(steps), dtype=np.float64)
        start_t = sub["start_time"].to_numpy(dtype=np.float64)
        end_t = sub["end_time"].to_numpy(dtype=np.float64)
        for k, t0 in enumerate(start_t):
            lo, hi = k * _STRIDE, (k * _STRIDE + _WINDOW if k == len(start_t) - 1 else (k + 1) * _STRIDE)
            times[lo:hi] = t0 + np.cumsum(dt[lo:hi]) - dt[lo]
            anchored_end = t0 + np.sum(dt[lo + 1 : k * _STRIDE + _WINDOW])
            if abs(anchored_end - end_t[k]) >= 1e-2:
                raise StitchError(f"{uid}: window {k} end_time drifts by {anchored_end - end_t[k]:.4f}s")
        if np.any(np.diff(times) < 0):
            raise StitchError(f"{uid}: non-monotonic time within a pseudonym")
        return times


class PseudonymResolver:
    """Maps (subfolder, senderPseudo) -> physical sender from the ground-truth ledgers.

    Streams each ledger once with a regex (no per-line JSON parse) and caches the result.
    Read-only on ``data/``; the cache lives under the simulation folder.
    """

    def __init__(self, raw_dir: pathlib.Path, cache_path: pathlib.Path) -> None:
        self._raw_dir = raw_dir
        self._cache_path = cache_path
        self._map: dict[str, dict[str, int]] = {}  # subfolder_key -> {pseudo: sender}

    def load_or_build(self, subfolder_keys: Iterable[tuple[str, str, str]]) -> None:
        """Ensure every (family, group, subfolder) key is resolved, building missing ones."""
        if self._cache_path.exists():
            with open(self._cache_path, encoding="utf-8") as fh:
                self._map = json.load(fh)
        missing = [k for k in set(subfolder_keys) if self._key(*k) not in self._map]
        for family, group, subfolder in sorted(missing):
            self._map[self._key(family, group, subfolder)] = self._scan(family, group, subfolder)
        if missing:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w", encoding="utf-8") as fh:
                json.dump(self._map, fh)

    def sender_of(self, family: str, group: str, subfolder: str, pseudo: int) -> int | None:
        return self._map.get(self._key(family, group, subfolder), {}).get(str(pseudo))

    def pseudonym_total(self, family: str, group: str, subfolder: str, sender: int) -> int:
        table = self._map.get(self._key(family, group, subfolder), {})
        return sum(1 for s in table.values() if s == sender)

    def pseudonyms_of(self, family: str, group: str, subfolder: str, sender: int) -> set[int]:
        """Every senderPseudo ground truth has ever recorded as this sender's *legitimate* identity."""
        table = self._map.get(self._key(family, group, subfolder), {})
        return {int(pseudo) for pseudo, s in table.items() if s == sender}

    @staticmethod
    def _key(family: str, group: str, subfolder: str) -> str:
        return f"{family}_{group}/{subfolder}"

    def _scan(self, family: str, group: str, subfolder: str) -> dict[str, int]:
        folder = self._raw_dir / f"{family}_{group}" / subfolder
        ledgers = sorted(folder.glob("traceGroundTruthJSON-*.json"))
        if not ledgers:
            raise FileNotFoundError(f"no ground-truth ledger under {folder}")
        table: dict[str, int] = {}
        for ledger in ledgers:
            _LOG.info("scanning %s", ledger)
            with open(ledger, encoding="utf-8") as fh:
                for line in fh:
                    match = _LEDGER_IDS.search(line)
                    if match:
                        table.setdefault(match.group(2), int(match.group(1)))
        return table


def _rows_to_forged_trajectory(
    rows: Sequence[Mapping[str, Any]], feature_cols: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Sorts broadcast rows by ``sendTime`` and derives dt/dpos/dspd the same way FeatureEngineer does.

    Pure function of already-parsed rows (each a decoded type-3 JSON object) -- no file I/O, so it's
    directly unit-testable against synthetic rows.
    """
    ordered = sorted(rows, key=lambda r: r["sendTime"])
    times = np.array([r["sendTime"] for r in ordered], dtype=np.float64)
    pos_x = np.array([r["pos"][0] for r in ordered], dtype=np.float64)
    pos_y = np.array([r["pos"][1] for r in ordered], dtype=np.float64)
    spd_x = np.array([r["spd"][0] for r in ordered], dtype=np.float64)
    spd_y = np.array([r["spd"][1] for r in ordered], dtype=np.float64)
    acl_x = np.array([r["acl"][0] for r in ordered], dtype=np.float64)
    acl_y = np.array([r["acl"][1] for r in ordered], dtype=np.float64)
    hed_x = np.array([r["hed"][0] for r in ordered], dtype=np.float64)
    hed_y = np.array([r["hed"][1] for r in ordered], dtype=np.float64)

    def _diff(values: np.ndarray) -> np.ndarray:
        d = np.diff(values, prepend=values[0])
        d[0] = 0.0  # this identity's first step: no previous row, same convention as FeatureEngineer
        return d

    columns = {
        "pos_x": pos_x,
        "pos_y": pos_y,
        "spd_x": spd_x,
        "spd_y": spd_y,
        "acl_x": acl_x,
        "acl_y": acl_y,
        "hed_x": hed_x,
        "hed_y": hed_y,
        "dt": _diff(times),
        "dpos_x": _diff(pos_x),
        "dpos_y": _diff(pos_y),
        "dspd_x": _diff(spd_x),
        "dspd_y": _diff(spd_y),
    }
    steps = np.zeros((len(ordered), len(feature_cols)), dtype=np.float32)
    for name, values in columns.items():
        steps[:, feature_cols.index(name)] = values
    return steps, times


class BroadcastForger:
    """Finds a physical attacker's fabricated broadcast pseudonym in neighbors' received-message logs.

    Read-only on ``data/``. ``scan`` streams every trace file in one subfolder exactly once, so
    trying several candidate senders in the same run costs one pass, not one per candidate.
    """

    def __init__(self, raw_dir: pathlib.Path) -> None:
        self._raw_dir = raw_dir

    def scan(
        self, family: str, group: str, subfolder: str, target_senders: set[int]
    ) -> dict[int, dict[int, dict[int, dict[str, Any]]]]:
        """``sender -> fake senderPseudo -> messageID -> row``, deduplicated across receivers."""
        if not target_senders:
            return {}
        folder = self._raw_dir / f"{family}_{group}" / subfolder
        needles = [f'"sender":{s},' for s in target_senders]
        out: dict[int, dict[int, dict[int, dict[str, Any]]]] = {}
        for trace_file in sorted(folder.glob("traceJSON-*.json")):
            with open(trace_file, encoding="utf-8") as fh:
                for line in fh:
                    if not any(needle in line for needle in needles):
                        continue
                    obj = json.loads(line)
                    if obj.get("type") != 3 or obj.get("sender") not in target_senders:
                        continue
                    out.setdefault(obj["sender"], {}).setdefault(obj["senderPseudo"], {})[obj["messageID"]] = obj
        return out

    @staticmethod
    def pick_fake_pseudonym(
        scanned: Mapping[int, Mapping[int, Mapping[int, Mapping[str, Any]]]],
        sender: int,
        real_pseudonyms: set[int],
        min_steps: int = _MIN_FORGED_STEPS,
    ) -> tuple[int, list[dict[str, Any]]] | None:
        """The busiest pseudonym `sender` broadcasts that ground truth never recorded, if one has enough rows."""
        fake = {p: rows for p, rows in scanned.get(sender, {}).items() if p not in real_pseudonyms}
        eligible = {p: rows for p, rows in fake.items() if len(rows) >= min_steps}
        if not eligible:
            return None
        pseudo, rows = max(eligible.items(), key=lambda kv: len(kv[1]))
        return pseudo, list(rows.values())

    @staticmethod
    def pick_all_pseudonyms(
        scanned: Mapping[int, Mapping[int, Mapping[int, Mapping[str, Any]]]],
        sender: int,
        min_steps: int = _MIN_FORGED_STEPS,
    ) -> tuple[int, list[dict[str, Any]], int] | None:
        """Every message heard from `sender` under any pseudonym, sorted by sendTime.

        Returns `(most_used_pseudonym, rows, n_pseudonyms)`, or None under `min_steps` messages.
        """
        by_pseudo = scanned.get(sender, {})
        rows = [row for msgs in by_pseudo.values() for row in msgs.values()]
        if len(rows) < min_steps:
            return None
        top = max(by_pseudo, key=lambda p: len(by_pseudo[p]))
        return top, sorted(rows, key=lambda r: r["sendTime"]), len(by_pseudo)

    @classmethod
    def pick_attack_trace(
        cls,
        scanned: Mapping[int, Mapping[int, Mapping[int, Mapping[str, Any]]]],
        sender: int,
        family: str,
        real_pseudonyms: set[int],
        min_steps: int = _MIN_FORGED_STEPS,
    ) -> tuple[str, int, list[dict[str, Any]], int] | None:
        """`(strategy, pseudonym, rows, n_pseudonyms)` using the family's strategy, or None."""
        if attack_trace_strategy(family) == _FAKE_PSEUDONYM:
            found = cls.pick_fake_pseudonym(scanned, sender, real_pseudonyms, min_steps)
            return None if found is None else (_FAKE_PSEUDONYM, found[0], found[1], 1)
        found_all = cls.pick_all_pseudonyms(scanned, sender, min_steps)
        return None if found_all is None else (_ALL_PSEUDONYMS, found_all[0], found_all[1], found_all[2])


@dataclasses.dataclass(frozen=True)
class ForgedDemo:
    """One physical attacker's fabricated broadcast pseudonym, paired by `sender` with its real identity."""

    family: str
    group: str
    subfolder: str
    label: str
    sender: int
    fake_pseudonym: int
    steps: np.ndarray
    times: np.ndarray
    attack_trace: str = _FAKE_PSEUDONYM
    n_pseudonyms: int = 1


class BenignMatcher:
    """Ranks the Benign identities of one run by how long they overlap a given time window.

    Pure logic over `window_metadata.parquet` columns; no file I/O. Used to pair each fabricated
    broadcast with an honest vehicle driving in the same run at the same time.
    """

    def __init__(self, meta: pd.DataFrame) -> None:
        benign = meta[meta["attack_label"] == _BENIGN]
        self._spans = benign.groupby("sender_uid").agg(
            family=("family", "first"),
            group=("group", "first"),
            subfolder=("subfolder", "first"),
            t0=("start_time", "min"),
            t1=("end_time", "max"),
        )

    def candidates(
        self, family: str, group: str, subfolder: str, t0: float, t1: float
    ) -> list[tuple[str, float, float]]:
        """`(uid, overlap_start, overlap_end)` for every benign uid overlapping `[t0, t1]`, longest first."""
        s = self._spans
        cell = s[(s["family"] == family) & (s["group"] == group) & (s["subfolder"] == subfolder)]
        lo = np.maximum(cell["t0"].to_numpy(), t0)
        hi = np.minimum(cell["t1"].to_numpy(), t1)
        order = np.argsort(-(hi - lo), kind="stable")
        return [(str(cell.index[i]), float(lo[i]), float(hi[i])) for i in order if hi[i] > lo[i]]


def _steps_within(times: np.ndarray, t0: float, t1: float) -> int:
    return int(np.count_nonzero((times >= t0) & (times <= t1)))


@dataclasses.dataclass(frozen=True)
class SampledIdentity:
    """A sampling decision: which uid to export and how to tag it."""

    uid: str
    tags: tuple[str, ...] = ()
    max_windows: int | None = None


class StratifiedSampler:
    """Chooses identities per (class, group) cell. Pure logic; no I/O."""

    def __init__(self, identities: pd.DataFrame, config: SampleConfig, sender_lookup: Mapping[str, int | None]) -> None:
        self._ids = identities
        self._cfg = config
        self._sender = sender_lookup
        self._rng = np.random.default_rng(config.seed)

    def sample(self) -> list[SampledIdentity]:
        normal = self._ids[self._ids["n_windows"] <= self._cfg.max_normal_windows]
        chosen: list[SampledIdentity] = []
        for label in sorted(normal["attack_label"].unique()):
            for group in sorted(normal["group"].unique()):
                cell = normal[(normal["attack_label"] == label) & (normal["group"] == group)]
                picks = self._sample_benign(cell) if label == _BENIGN else self._sample_by_sender(cell)
                chosen.extend(SampledIdentity(uid) for uid in picks)
            longest = normal[normal["attack_label"] == label].sort_values("n_windows", ascending=False)
            chosen.extend(SampledIdentity(uid, tags=("stress",)) for uid in longest.index[: self._cfg.stress_per_class])
        spliced = self._ids[self._ids["n_windows"] > self._cfg.max_normal_windows]
        chosen.extend(
            SampledIdentity(uid, tags=("defect_demo",), max_windows=self._cfg.defect_demo_windows)
            for uid in spliced.index
        )
        return self._dedupe(chosen)

    def _sample_benign(self, cell: pd.DataFrame) -> list[str]:
        n = min(self._cfg.pseudonyms_per_cell, len(cell))
        return list(self._rng.choice(cell.index.to_numpy(), size=n, replace=False))

    def _sample_by_sender(self, cell: pd.DataFrame) -> list[str]:
        """Draw whole physical senders so the multi-identity view has complete groups."""
        by_sender: dict[tuple[str, int | None], list[str]] = {}
        for uid, row in cell.iterrows():
            by_sender.setdefault((row["subfolder"], self._sender.get(uid)), []).append(uid)
        keys = list(by_sender)
        self._rng.shuffle(keys)
        picks: list[str] = []
        for key in keys:
            if len(picks) >= self._cfg.pseudonyms_per_cell:
                break
            uids = sorted(by_sender[key])
            picks.extend(uids[: self._cfg.max_pseudonyms_per_sender])
        return picks

    @staticmethod
    def _dedupe(chosen: Sequence[SampledIdentity]) -> list[SampledIdentity]:
        merged: dict[str, SampledIdentity] = {}
        for item in chosen:
            prev = merged.get(item.uid)
            tags = tuple(sorted(set((prev.tags if prev else ()) + item.tags)))
            merged[item.uid] = SampledIdentity(item.uid, tags, item.max_windows)
        return list(merged.values())


class SampleWriter:
    """Serialises stitched identities into the manifest + binary bundle."""

    def __init__(self, store: WindowStore, resolver: PseudonymResolver, config: SampleConfig) -> None:
        self._store = store
        self._resolver = resolver
        self._cfg = config

    def write(self, records: list[dict[str, Any]], steps: np.ndarray, times: np.ndarray) -> dict[str, Any]:
        out = self._cfg.out_dir
        out.mkdir(parents=True, exist_ok=True)
        steps_bytes = np.ascontiguousarray(steps, dtype=np.float32).tobytes()
        times_bytes = np.ascontiguousarray(times, dtype=np.float64).tobytes()
        (out / "steps.f32").write_bytes(steps_bytes)
        (out / "times.f64").write_bytes(times_bytes)
        benign_rows = np.concatenate(
            [np.arange(r["row_offset"], r["row_offset"] + r["n_steps"]) for r in records if r["label"] == _BENIGN]
        )
        px, py = self._store.feature_index("pos_x"), self._store.feature_index("pos_y")
        manifest = {
            "version": 1,
            "seed": self._cfg.seed,
            "window_size": _WINDOW,
            "stride": _STRIDE,
            "feature_cols": self._store.feature_cols,
            "norm_mean": self._store.config["norm_mean"],
            "norm_std": self._store.config["norm_std"],
            "label_to_int": self._store.config["label_to_int"],
            "class_palette": _CLASS_PALETTE,
            "road_bbox": [
                float(steps[benign_rows, px].min()),
                float(steps[benign_rows, py].min()),
                float(steps[benign_rows, px].max()),
                float(steps[benign_rows, py].max()),
            ],
            "total_steps": int(len(steps)),
            "binaries": {
                "steps": {
                    "file": "steps.f32",
                    "dtype": "float32",
                    "shape": [int(len(steps)), steps.shape[1]],
                    "sha256": hashlib.sha256(steps_bytes).hexdigest(),
                },
                "times": {
                    "file": "times.f64",
                    "dtype": "float64",
                    "shape": [int(len(times))],
                    "sha256": hashlib.sha256(times_bytes).hexdigest(),
                },
            },
            "identities": records,
        }
        with open(out / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1)
        return manifest


class Exporter:
    """Orchestrates store -> resolver -> sampler -> stitcher -> writer."""

    def __init__(self, config: SampleConfig) -> None:
        self._cfg = config
        self._store = WindowStore(config.prepared_dir)
        self._resolver = PseudonymResolver(config.raw_dir, config.cache_dir / "pseudonym_map.json")

    def run(self) -> dict[str, Any]:
        ids = self._store.identity_table()
        self._resolver.load_or_build(
            (r["family"], r["group"], r["subfolder"]) for _, r in ids.drop_duplicates("subfolder").iterrows()
        )
        sender_lookup = {
            uid: self._resolver.sender_of(r["family"], r["group"], r["subfolder"], int(r["senderPseudo"]))
            for uid, r in ids.iterrows()
        }
        surviving = self._surviving_counts(ids, sender_lookup)
        chosen = StratifiedSampler(ids, self._cfg, sender_lookup).sample()
        _LOG.info("sampled %d identities", len(chosen))

        records: list[dict[str, Any]] = []
        step_blocks: list[np.ndarray] = []
        time_blocks: list[np.ndarray] = []
        offset = 0
        uid_to_id: dict[str, int] = {}

        def add_prepared(uid: str, tags: Sequence[str], stitched: StitchedIdentity) -> int:
            nonlocal offset
            rid = len(records)
            uid_to_id[uid] = rid
            records.append(self._prepared_record(rid, uid, tags, stitched, offset, ids, sender_lookup, surviving))
            step_blocks.append(stitched.steps)
            time_blocks.append(stitched.times)
            offset += len(stitched.steps)
            return rid

        for item in sorted(chosen, key=lambda s: s.uid):
            add_prepared(item.uid, item.tags, IdentityStitcher(self._store, item.max_windows).stitch(item.uid))

        matcher = BenignMatcher(self._store.meta)
        pairs = self._find_forged_demos(ids, sender_lookup, chosen)
        _LOG.info("found %d forged-broadcast pairs", len(pairs))
        matched = 0
        for anchor_uid, demo in pairs:
            forged_id = len(records)
            records.append(
                {
                    "id": forged_id,
                    "sender_uid": f"{demo.family}_{demo.group}/{demo.subfolder}/broadcast_{demo.sender}_{demo.fake_pseudonym}",
                    "label": demo.label,
                    "label_int": int(self._store.config["label_to_int"][demo.label]),
                    "family": demo.family,
                    "group": demo.group,
                    "subfolder": demo.subfolder,
                    "sender_pseudo": demo.fake_pseudonym,
                    "sender": demo.sender,
                    "pseudonyms_total": None,
                    "pseudonyms_surviving": 1,
                    "tags": [_FORGED_TAG],
                    "row_offset": offset,
                    "n_steps": int(len(demo.steps)),
                    "n_windows": max(0, (len(demo.steps) - _WINDOW) // _STRIDE + 1),
                    "x_windows_rows": [-1],  # never part of X_windows.npy -- observed broadcast, not prepared data
                    "t_start": float(demo.times[0]),
                    "t_end": float(demo.times[-1]),
                    "data_source": "raw_veremi",
                    "paired_identity_id": uid_to_id[anchor_uid],
                    "benign_match_id": None,
                    "overlap": None,
                    "attack_trace": demo.attack_trace,
                    "broadcast_pseudonyms": demo.n_pseudonyms,
                }
            )
            records[uid_to_id[anchor_uid]]["paired_identity_id"] = forged_id
            step_blocks.append(demo.steps)
            time_blocks.append(demo.times)
            offset += len(demo.steps)

            match = self._match_benign(matcher, demo)
            if match is None:
                _LOG.warning(
                    "no time-overlapping benign vehicle for forged pseudonym %d in %s",
                    demo.fake_pseudonym,
                    demo.subfolder,
                )
                continue
            benign_uid, stitched, lo, hi = match
            benign_id = uid_to_id.get(benign_uid)
            if benign_id is None:
                benign_id = add_prepared(benign_uid, ("overlap_benign",), stitched)
            records[forged_id]["benign_match_id"] = benign_id
            records[forged_id]["overlap"] = [lo, hi]
            matched += 1
        _LOG.info("matched %d of %d forged broadcasts with a time-overlapping benign vehicle", matched, len(pairs))

        steps = np.concatenate(step_blocks, axis=0)
        times = np.concatenate(time_blocks, axis=0)
        manifest = SampleWriter(self._store, self._resolver, self._cfg).write(records, steps, times)
        self._report(manifest)
        return manifest

    def _prepared_record(
        self,
        rid: int,
        uid: str,
        tags: Sequence[str],
        stitched: StitchedIdentity,
        offset: int,
        ids: pd.DataFrame,
        sender_lookup: Mapping[str, int | None],
        surviving: Mapping[tuple[str, int | None], int],
    ) -> dict[str, Any]:
        meta = ids.loc[uid]
        sender = sender_lookup[uid]
        return {
            "id": rid,
            "sender_uid": uid,
            "label": meta["attack_label"],
            "label_int": int(self._store.config["label_to_int"][meta["attack_label"]]),
            "family": meta["family"],
            "group": meta["group"],
            "subfolder": meta["subfolder"],
            "sender_pseudo": int(meta["senderPseudo"]),
            "sender": sender,
            "pseudonyms_total": (
                self._resolver.pseudonym_total(meta["family"], meta["group"], meta["subfolder"], sender)
                if sender is not None
                else None
            ),
            "pseudonyms_surviving": surviving.get((meta["subfolder"], sender), 1),
            "tags": list(tags),
            "row_offset": offset,
            "n_steps": int(len(stitched.steps)),
            "n_windows": stitched.n_windows,
            "x_windows_rows": stitched.window_rows.tolist(),
            "t_start": float(stitched.times[0]),
            "t_end": float(stitched.times[-1]),
            "data_source": "prepared",
            "paired_identity_id": None,
            "benign_match_id": None,
            "overlap": None,
            "attack_trace": None,
            "broadcast_pseudonyms": None,
        }

    def _match_benign(
        self,
        matcher: BenignMatcher,
        demo: ForgedDemo,
        max_tries: int = 5,
    ) -> tuple[str, StitchedIdentity, float, float] | None:
        """The benign identity overlapping `demo` longest, with >= one window of steps on both sides."""
        candidates = matcher.candidates(
            demo.family, demo.group, demo.subfolder, float(demo.times[0]), float(demo.times[-1])
        )
        for uid, lo, hi in candidates[:max_tries]:
            try:
                stitched = IdentityStitcher(self._store).stitch(uid)
            except StitchError as err:
                _LOG.warning("skipping benign candidate %s: %s", uid, err)
                continue
            if _steps_within(stitched.times, lo, hi) >= _WINDOW and _steps_within(demo.times, lo, hi) >= _WINDOW:
                return uid, stitched, lo, hi
        return None

    def _find_forged_demos(
        self,
        ids: pd.DataFrame,
        sender_lookup: Mapping[str, int | None],
        chosen: Sequence[SampledIdentity],
    ) -> list[tuple[str, ForgedDemo]]:
        """Forged-broadcast companions for up to `forged_pairs_per_cell` sampled attackers per cell.

        Returns `(anchor_uid, demo)` pairs, where `anchor_uid` is the sampled *prepared* identity the
        demo pairs with -- the longest-sampled pseudonym of that physical vehicle, so the pairing
        always points at an identity actually present in this export. Each subfolder is scanned once
        regardless of how many of its attackers end up paired.
        """
        forger = BroadcastForger(self._cfg.raw_dir)
        chosen_uids = {item.uid for item in chosen}
        attackers = ids[(ids["attack_label"] != _BENIGN) & (ids.index.isin(chosen_uids))]

        anchor_uid: dict[tuple[str, int], str] = {}
        for uid, row in attackers.sort_values("n_windows", ascending=False).iterrows():
            sender = sender_lookup.get(uid)
            if sender is not None:
                anchor_uid.setdefault((row["subfolder"], sender), uid)

        pairs: list[tuple[str, ForgedDemo]] = []
        for (family, group, subfolder), cell in attackers.groupby(["family", "group", "subfolder"]):
            candidate_senders = sorted({sender_lookup[uid] for uid in cell.index if sender_lookup[uid] is not None})
            if not candidate_senders:
                _LOG.warning("no resolved physical senders for %s/%s", family, subfolder)
                continue
            scanned = forger.scan(family, group, subfolder, set(candidate_senders))
            paired_in_cell = 0
            for sender in candidate_senders:
                if paired_in_cell >= self._cfg.forged_pairs_per_cell:
                    break
                anchor = anchor_uid.get((subfolder, sender))
                if anchor is None:
                    continue
                real_pseudonyms = self._resolver.pseudonyms_of(family, group, subfolder, sender)
                found = BroadcastForger.pick_attack_trace(scanned, sender, family, real_pseudonyms)
                if found is None:
                    continue
                strategy, fake_pseudonym, rows, n_pseudonyms = found
                steps, times = _rows_to_forged_trajectory(rows, self._store.feature_cols)
                demo = ForgedDemo(
                    family, group, subfolder, family, sender, fake_pseudonym, steps, times, strategy, n_pseudonyms
                )
                pairs.append((anchor, demo))
                paired_in_cell += 1
                _LOG.info(
                    "attack trace: %s sender %d, %s, %d pseudonym(s), %d messages",
                    family,
                    sender,
                    strategy,
                    n_pseudonyms,
                    len(steps),
                )
            if paired_in_cell == 0:
                _LOG.warning("no forged broadcast pseudonym found for any sampled attacker in %s/%s", family, subfolder)
        return pairs

    @staticmethod
    def _surviving_counts(
        ids: pd.DataFrame, sender_lookup: Mapping[str, int | None]
    ) -> dict[tuple[str, int | None], int]:
        counts: dict[tuple[str, int | None], int] = {}
        for uid, row in ids.iterrows():
            key = (row["subfolder"], sender_lookup[uid])
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _report(self, manifest: dict[str, Any]) -> None:
        out = self._cfg.out_dir
        total = sum(p.stat().st_size for p in out.iterdir())
        by_cell: dict[tuple[str, str], int] = {}
        for r in manifest["identities"]:
            by_cell[(r["label"], r["group"])] = by_cell.get((r["label"], r["group"]), 0) + 1
        _LOG.info(
            "wrote %d identities, %d steps, %.2f MB to %s",
            len(manifest["identities"]),
            manifest["total_steps"],
            total / 1e6,
            out,
        )
        for (label, group), n in sorted(by_cell.items()):
            _LOG.info("  %-20s %s  %3d identities", label, group, n)
        paired = sum(1 for r in manifest["identities"] if r["data_source"] == "raw_veremi")
        _LOG.info("  %d prepared-vs-raw-VeReMi pairs available for map compare", paired)


def _parse_args(argv: Sequence[str] | None) -> SampleConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, default=SampleConfig.out_dir)
    parser.add_argument("--seed", type=int, default=SampleConfig.seed)
    parser.add_argument("--per-cell", type=int, default=SampleConfig.pseudonyms_per_cell)
    parser.add_argument("--prepared-dir", type=pathlib.Path, default=SampleConfig.prepared_dir)
    parser.add_argument("--raw-dir", type=pathlib.Path, default=SampleConfig.raw_dir)
    parser.add_argument("--cache-dir", type=pathlib.Path, default=SampleConfig.cache_dir)
    args = parser.parse_args(argv)
    return SampleConfig(
        seed=args.seed,
        pseudonyms_per_cell=args.per_cell,
        prepared_dir=args.prepared_dir,
        raw_dir=args.raw_dir,
        out_dir=args.out,
        cache_dir=args.cache_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    Exporter(_parse_args(argv)).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
