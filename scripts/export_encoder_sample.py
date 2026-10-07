"""Export a stratified sample of the T = 64 encoder input for the simulator's "Encoder input" tab.

Reads ``src/data/encoder_input/benign_gridsybil/T64/`` (read-only: ``metadata.json`` plus every
``<split>/b64/part-*.json`` shard and its ``_info.json`` companion) and writes a compact bundle to
``simulation/public/data/encoder/``:

* ``manifest.json``         - features, units, train normalisation, full-dataset summary (computed
  over EVERY window, not the sample), the sampling rule, curated presenter windows with the rule that
  picked each one, and one record per sampled window (provenance, link position, label, row offset).
* ``x.f32``                 - little-endian float32 real rows (``n x 13`` per window, normalised, exactly
  the shard values), concatenated in manifest order. Padding is not stored.
* ``predictions/index.json`` - ``{"models": []}`` (written only when absent; the encoder is untrained).

Sampling: for every split x scenario x class stratum, sender vehicles (``<group>:<sender>``, the split
key) are shuffled with a seeded RNG and added with ALL their windows in that stratum until the stratum
holds at least the target number of windows; senders with more than ``max_sender_windows`` windows in
the stratum are skipped (counted). Curated windows are then picked by explicit rules over the full
dataset and every crop of their links is added. If ``x.f32 + manifest.json`` exceeds the budget, the
per-stratum target is lowered and the sample redrawn.

Labels are carried for evaluation display only; nothing here feeds a model.
Nothing under ``data/`` or ``src/data/`` is ever written.

Usage:
    .venv/bin/python scripts/export_encoder_sample.py
    .venv/bin/python scripts/export_encoder_sample.py --per-stratum 300 --seed 0 --workers 10
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import multiprocessing
import pathlib
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

_LOG = logging.getLogger("export_encoder_sample")

SPLITS: tuple[str, ...] = ("train", "pretrain_val", "val", "test")
CLASSES: tuple[str, ...] = ("Benign", "GridSybil")
N_FEATURES = 13
UNITS: dict[str, str] = {
    "claimed_pos_x": "m", "claimed_pos_y": "m", "rx_pos_x": "m", "rx_pos_y": "m",
    "claimed_vel_x": "m/s", "claimed_vel_y": "m/s", "rx_vel_x": "m/s", "rx_vel_y": "m/s",
    "claimed_acl_x": "m/s²", "claimed_acl_y": "m/s²", "range": "m", "bearing": "rad", "log_dtau": "log s",
}
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class ExportError(ValueError):
    """Raised when the source data or the bundle fails a consistency check."""


@dataclasses.dataclass(frozen=True)
class ExportConfig:
    """Knobs for the export.

    Attributes:
        source_dir: encoder-input folder holding ``metadata.json`` and the split shards.
        out_dir: bundle destination; must lie under ``allowed_out_root``.
        allowed_out_root: the only directory the writer may write under.
        seed: RNG seed for sender shuffling, recorded in the manifest.
        per_stratum: initial target windows per split x scenario x class stratum.
        max_sender_windows: senders with more windows than this in a stratum are skipped.
        budget_bytes: limit on ``x.f32 + manifest.json``.
        target_step: amount the per-stratum target is lowered by when over budget.
        min_per_stratum: the target is never lowered below this (the export fails instead).
        workers: parallel shard readers (1 = in-process).
        length_only_auc_val: length-only AUC on val, quoted from the EDA notebook.
        length_only_auc_source: the notebook that measured it.
    """

    source_dir: pathlib.Path = pathlib.Path("src/data/encoder_input/benign_gridsybil/T64")
    out_dir: pathlib.Path = pathlib.Path("simulation/public/data/encoder")
    allowed_out_root: pathlib.Path = pathlib.Path("simulation/public/data/encoder")
    seed: int = 0
    per_stratum: int = 300
    max_sender_windows: int = 400
    budget_bytes: int = 8_000_000
    target_step: int = 25
    min_per_stratum: int = 25
    workers: int = 10
    length_only_auc_val: float = 0.531
    length_only_auc_source: str = "src/eda/benign_gridsybil/eda_encoder_input_T64.ipynb"


@dataclasses.dataclass(frozen=True)
class WindowRecord:
    """Metadata of one encoder window (from its ``_info.json`` entry plus the shard's real rows).

    Attributes:
        id: window id as written by the encoder-input notebook.
        split: train / pretrain_val / val / test.
        scenario: e.g. ``GridSybil_1416``.
        run: ``<scenario>/<run folder>``.
        receiver_file: ``<scenario>/<run folder>/traceJSON-....json`` (the listening vehicle's log).
        receiver: receiver vehicle id.
        sender: true sender vehicle id (privileged for GridSybil ghosts, F13).
        sender_pseudo: pseudonym heard on the air.
        k: crop index of this window within its link.
        n: real (unpadded) rows.
        label: attack code (0 benign, 16 GridSybil); evaluation only.
        label_name: class name; evaluation only.
        shard: shard path relative to ``source_dir``.
        max_range_m: largest real-row ``range`` in metres (de-normalised with the train stats).
    """

    id: str
    split: str
    scenario: str
    run: str
    receiver_file: str
    receiver: int
    sender: int
    sender_pseudo: int
    k: int
    n: int
    label: int
    label_name: str
    shard: str
    max_range_m: float

    @property
    def group(self) -> str:
        """Scenario time window (``0709`` / ``1416``)."""
        return self.scenario.rsplit("_", 1)[1]

    @property
    def sender_key(self) -> str:
        """The split key ``<group>:<sender vehicle>``."""
        return f"{self.group}:{self.sender}"

    @property
    def link(self) -> str:
        """Link key ``scenario|run|receiver_file|sender|sender_pseudo``."""
        return f"{self.scenario}|{self.run}|{self.receiver_file}|{self.sender}|{self.sender_pseudo}"

    def sort_key(self) -> tuple[Any, ...]:
        """Manifest order: split, scenario, run, receiver file, sender, pseudonym, crop index."""
        return (SPLITS.index(self.split), self.scenario, self.run, self.receiver_file, self.sender,
                self.sender_pseudo, self.k)


def _load_json(path: pathlib.Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class ShardReader:
    """Reads one shard + its info file and checks that ids, masks and padding agree."""

    def __init__(self, source_dir: pathlib.Path, range_mean: float, range_std: float, range_index: int) -> None:
        self._dir = source_dir
        self._mean = range_mean
        self._std = range_std
        self._range_index = range_index

    def _pairs(self, shard: str) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
        data = _load_json(self._dir / shard)
        info = _load_json(self._dir / shard.replace(".json", "_info.json"))
        if len(data["windows"]) != len(info["windows"]):
            raise ExportError(f"{shard}: shard and info disagree on window count")
        for win, meta in zip(data["windows"], info["windows"]):
            if win["id"] != meta["id"]:
                raise ExportError(f"{shard}: id mismatch {win['id']} != {meta['id']}")
            yield win, meta

    @staticmethod
    def real_rows(win: Mapping[str, Any], n: int) -> np.ndarray:
        """Returns the ``n x 13`` float32 real rows after checking mask and zero padding."""
        x = np.asarray(win["x"], dtype=np.float32)
        mask = np.asarray(win["mask"], dtype=np.int64)
        if x.shape[1] != N_FEATURES or mask.shape[0] != x.shape[0]:
            raise ExportError(f"{win['id']}: bad shape {x.shape}")
        if int(mask.sum()) != n or not np.all(mask[:n] == 1):
            raise ExportError(f"{win['id']}: mask is not n = {n} leading ones")
        if np.any(x[n:] != 0):
            raise ExportError(f"{win['id']}: padding rows are not zero")
        return x[:n]

    def records(self, shard: str) -> list[WindowRecord]:
        """Returns every window's metadata (and max range) in shard order."""
        split = shard.split("/", 1)[0]
        out = []
        for win, meta in self._pairs(shard):
            n = int(meta["n_messages"])
            rows = self.real_rows(win, n)
            rng_m = rows[:, self._range_index].astype(np.float64) * self._std + self._mean
            out.append(WindowRecord(
                id=meta["id"], split=split, scenario=meta["scenario"], run=meta["run"],
                receiver_file=meta["receiver_file"], receiver=int(meta["receiver"]), sender=int(meta["sender"]),
                sender_pseudo=int(meta["sender_pseudo"]), k=int(meta["link_window_index"]), n=n,
                label=int(meta["label"]), label_name=meta["label_name"], shard=shard,
                max_range_m=float(rng_m.max()) if n else float("nan")))
        return out

    def rows_for(self, shard: str, ids: Iterable[str]) -> dict[str, np.ndarray]:
        """Returns ``{id: n x 13 float32}`` for the requested ids found in this shard."""
        wanted = set(ids)
        out = {}
        for win, meta in self._pairs(shard):
            if win["id"] in wanted:
                out[win["id"]] = self.real_rows(win, int(meta["n_messages"]))
        if len(out) != len(wanted):
            raise ExportError(f"{shard}: {len(wanted) - len(out)} requested windows not found")
        return out


# Worker-side state for fork-based pools (set before the pool forks).
_WORKER_READER: ShardReader | None = None


def _worker_records(shard: str) -> list[WindowRecord]:
    assert _WORKER_READER is not None
    return _WORKER_READER.records(shard)


def _worker_rows(task: tuple[str, list[str]]) -> dict[str, np.ndarray]:
    assert _WORKER_READER is not None
    return _WORKER_READER.rows_for(*task)


class EncoderInput:
    """Read-only view of the encoder-input folder (metadata + shards), optionally read in parallel."""

    def __init__(self, source_dir: pathlib.Path, workers: int = 1) -> None:
        self.dir = source_dir
        self.metadata: dict[str, Any] = _load_json(source_dir / "metadata.json")
        self.features: list[str] = list(self.metadata["features"])
        norm = self.metadata["normalisation"]
        if list(norm["features"]) != self.features or len(self.features) != N_FEATURES:
            raise ExportError("normalisation features disagree with metadata features")
        self.mean = [float(v) for v in norm["mean"]]
        self.std = [float(v) for v in norm["std"]]
        self.shards: list[str] = [s["file"] for s in self.metadata["shards"]]
        self._workers = max(1, workers)
        r = self.features.index("range")
        self.reader = ShardReader(source_dir, self.mean[r], self.std[r], r)

    def _map(self, fn: Any, tasks: list[Any]) -> list[Any]:
        global _WORKER_READER
        _WORKER_READER = self.reader
        if self._workers == 1 or len(tasks) <= 1:
            return [fn(t) for t in tasks]
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(min(self._workers, len(tasks))) as pool:
            return pool.map(fn, tasks, chunksize=1)

    def all_records(self) -> list[WindowRecord]:
        """Every window of every shard."""
        records = [r for part in self._map(_worker_records, self.shards) for r in part]
        if len(records) != int(self.metadata["counts"]["windows"]):
            raise ExportError(f"read {len(records)} windows, metadata says {self.metadata['counts']['windows']}")
        return records

    def rows(self, records: Sequence[WindowRecord]) -> dict[str, np.ndarray]:
        """Real rows of the given windows, read from their shards."""
        by_shard: dict[str, list[str]] = defaultdict(list)
        for r in records:
            by_shard[r.shard].append(r.id)
        out: dict[str, np.ndarray] = {}
        for part in self._map(_worker_rows, sorted(by_shard.items())):
            out.update(part)
        return out


class DatasetIndex:
    """Full-dataset facts: link sizes, sender split membership, per-split summary."""

    def __init__(self, records: Sequence[WindowRecord]) -> None:
        self.records = list(records)
        self.by_id = {r.id: r for r in self.records}
        if len(self.by_id) != len(self.records):
            raise ExportError("duplicate window ids")
        self.links: dict[str, list[WindowRecord]] = defaultdict(list)
        self.sender_splits: dict[str, set[str]] = defaultdict(set)
        for r in self.records:
            self.links[r.link].append(r)
            self.sender_splits[r.sender_key].add(r.split)
        for key, members in self.links.items():
            members.sort(key=lambda w: w.k)
            if [w.k for w in members] != list(range(len(members))):
                raise ExportError(f"link {key}: crop indices are not 0..K-1")

    def link_size(self, record: WindowRecord) -> int:
        """K: number of windows of the record's link in the full dataset."""
        return len(self.links[record.link])

    def splits_of_sender(self, record: WindowRecord) -> list[str]:
        """Splits the record's sender vehicle appears in over the full dataset (split order)."""
        return [s for s in SPLITS if s in self.sender_splits[record.sender_key]]

    @property
    def senders_in_multiple_splits(self) -> int:
        return sum(1 for s in self.sender_splits.values() if len(s) > 1)

    def summary(self, seq_len: int) -> dict[str, Any]:
        """Full-dataset counts per split (every window, not the sample)."""
        per_split: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            ws = [r for r in self.records if r.split == split]
            msgs = sum(r.n for r in ws)
            per_split[split] = {
                "windows": len(ws),
                "benign": sum(1 for r in ws if r.label_name == "Benign"),
                "gridsybil": sum(1 for r in ws if r.label_name == "GridSybil"),
                "messages": msgs,
                "padding_share": round(1 - msgs / (len(ws) * seq_len), 4) if ws else None,
                "sender_vehicles": len({r.sender_key for r in ws}),
            }
        msgs = sum(r.n for r in self.records)
        return {
            "windows": len(self.records),
            "messages": msgs,
            "padding_share": round(1 - msgs / (len(self.records) * seq_len), 4),
            "sender_vehicles": len(self.sender_splits),
            "links": len(self.links),
            "senders_in_multiple_splits": self.senders_in_multiple_splits,
            "per_split": per_split,
        }


class StratifiedSampler:
    """Seeded sender-vehicle sampling per split x scenario x class stratum."""

    def __init__(self, index: DatasetIndex, seed: int, max_sender_windows: int) -> None:
        self._index = index
        self._seed = seed
        self._max = max_sender_windows
        self.strata: dict[tuple[str, str, str], dict[str, list[WindowRecord]]] = defaultdict(lambda: defaultdict(list))
        for r in index.records:
            self.strata[(r.split, r.scenario, r.label_name)][r.sender_key].append(r)

    def stratum_keys(self) -> list[tuple[str, str, str]]:
        return sorted(self.strata, key=lambda s: (SPLITS.index(s[0]), s[1], CLASSES.index(s[2])))

    def sample(self, target: int) -> tuple[list[WindowRecord], dict[str, Any]]:
        """Returns the sampled windows and per-stratum stats (senders taken / skipped)."""
        chosen: list[WindowRecord] = []
        stats: dict[str, Any] = {}
        for i, key in enumerate(self.stratum_keys()):
            senders = self.strata[key]
            order = sorted(senders)
            rng = np.random.default_rng([self._seed, i])
            taken, skipped, count = 0, 0, 0
            for j in rng.permutation(len(order)):
                if count >= target:
                    break
                ws = senders[order[j]]
                if len(ws) > self._max:
                    skipped += 1
                    continue
                chosen.extend(ws)
                count += len(ws)
                taken += 1
            stats["/".join(key)] = {"windows": count, "senders": taken, "skipped_large_senders": skipped,
                                    "senders_available": len(order)}
        return chosen, stats


class CuratedSelector:
    """Presenter windows chosen by explicit, recorded rules (ties -> smallest id)."""

    def __init__(self, index: DatasetIndex) -> None:
        self._index = index
        self._sorted = sorted(index.records, key=lambda r: r.id)

    def full_dataset_rules(self) -> list[tuple[dict[str, str], WindowRecord]]:
        """Rules 1-4, evaluated over every window."""
        out = []
        valid = [r for r in self._sorted if not np.isnan(r.max_range_m)]
        far = max(valid, key=lambda r: r.max_range_m)  # max() keeps the first (smallest id) on ties
        out.append(({"name": f"Farthest claimed position ({far.max_range_m / 1000:.1f} km ghost)"
                     if far.max_range_m >= 1000 else "Farthest claimed position",
                     "rule": "argmax over all windows of the largest real-row range (m); ties -> smallest id"}, far))
        longest = max(self._sorted, key=lambda r: self._index.link_size(r))
        first = self._index.links[longest.link][0]
        out.append(({"name": "Longest link, cropped",
                     "rule": "link with the most windows K over the full dataset (ties -> smallest id); its first crop"},
                    first))
        single = next((r for r in self._sorted if r.split == "test" and r.n == 1), None)
        if single is not None:
            out.append(({"name": "Single-message window",
                         "rule": "first test window (sorted id) with n = 1"}, single))
        ghost = next((r for r in self._sorted if r.split == "test" and r.label_name == "GridSybil"
                      and r.sender_pseudo == 1 and r.n >= 20), None)
        if ghost is not None:
            out.append(({"name": "Ghost under shared pseudonym 1",
                         "rule": "first GridSybil test window (sorted id) with sender_pseudo = 1 and n >= 20"}, ghost))
        return out

    @staticmethod
    def same_receiver(sampled: Sequence[WindowRecord]) -> list[tuple[dict[str, str], WindowRecord]]:
        """Rule 5: the test receiver file with most windows of both classes among the sampled ones."""
        per_file: dict[str, dict[str, list[WindowRecord]]] = defaultdict(lambda: defaultdict(list))
        for r in sampled:
            if r.split == "test":
                per_file[r.receiver_file][r.label_name].append(r)
        both = [(f, c) for f, c in per_file.items() if c["Benign"] and c["GridSybil"]]
        if not both:
            return []
        best_file, cls = max(sorted(both), key=lambda fc: (min(len(fc[1]["Benign"]), len(fc[1]["GridSybil"])),
                                                           len(fc[1]["Benign"]) + len(fc[1]["GridSybil"])))
        rule = ("test receiver file with the most sampled windows of both classes (score = min(benign, gridsybil), "
                "then total, ties -> smallest path); its largest-n window of each class (ties -> smallest id)")
        out = []
        for name in CLASSES:
            pick = sorted(cls[name], key=lambda r: (-r.n, r.id))[0]
            out.append(({"name": f"Benign and GridSybil heard by the same receiver ({name})", "rule": rule}, pick))
        return out


class BundleWriter:
    """Writes the bundle; refuses any path outside the allowed output root."""

    def __init__(self, out_dir: pathlib.Path, allowed_root: pathlib.Path) -> None:
        out = (REPO_ROOT / out_dir).resolve() if not out_dir.is_absolute() else out_dir.resolve()
        root = (REPO_ROOT / allowed_root).resolve() if not allowed_root.is_absolute() else allowed_root.resolve()
        if out != root and root not in out.parents:
            raise ExportError(f"refusing to write {out}: outside {root}")
        self.out = out

    def _guard(self, path: pathlib.Path) -> pathlib.Path:
        path = path.resolve()
        if self.out not in path.parents:
            raise ExportError(f"refusing to write {path}")
        return path

    def write(self, manifest_bytes: bytes, x: np.ndarray) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        self._guard(self.out / "x.f32").write_bytes(x.astype("<f4", copy=False).tobytes())
        self._guard(self.out / "manifest.json").write_bytes(manifest_bytes)
        pred = self._guard(self.out / "predictions" / "index.json")
        if not pred.exists():
            pred.parent.mkdir(parents=True, exist_ok=True)
            pred.write_text(json.dumps({"models": []}, indent=2) + "\n", encoding="utf-8")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class ManifestBuilder:
    """Assembles manifest dict + row matrix for one candidate sample."""

    def __init__(self, cfg: ExportConfig, data: EncoderInput, index: DatasetIndex) -> None:
        self._cfg = cfg
        self._data = data
        self._index = index
        self._full = index.summary(int(data.metadata["max_seq_len"]))
        self._full.update({"length_only_auc_val": cfg.length_only_auc_val,
                           "length_only_auc_source": cfg.length_only_auc_source})

    def window_records(self, sample: Sequence[WindowRecord], curated: Sequence[tuple[dict[str, str], WindowRecord]]
                       ) -> list[WindowRecord]:
        """Sample plus every crop of each curated window's link, deduplicated, in manifest order."""
        ids = {r.id: r for r in sample}
        for _, rec in curated:
            for w in self._index.links[rec.link]:
                ids[w.id] = w
        return sorted(ids.values(), key=WindowRecord.sort_key)

    def build(self, windows: Sequence[WindowRecord], curated: Sequence[tuple[dict[str, str], WindowRecord]],
              target: int, stratum_stats: Mapping[str, Any], created_utc: str, git_commit: str) -> dict[str, Any]:
        per_split_class = {s: {c: 0 for c in CLASSES} for s in SPLITS}
        rows, entries = 0, []
        for i, r in enumerate(windows):
            per_split_class[r.split][r.label_name] += 1
            entries.append({
                "i": i, "id": r.id, "split": r.split, "scenario": r.scenario, "run": r.run,
                "receiver_file": r.receiver_file, "receiver": r.receiver, "sender": r.sender,
                "sender_pseudo": r.sender_pseudo, "link": r.link, "k": r.k, "K": self._index.link_size(r),
                "n": r.n, "label": r.label, "label_name": r.label_name, "row": rows,
                "sender_splits_full": self._index.splits_of_sender(r),
                "raw_trace": f"data/VeReMi-Dataset/{r.receiver_file}",
            })
            rows += r.n
        rule = (f"seed {self._cfg.seed}; per split x scenario x class stratum, sender vehicles (<group>:<sender>) "
                f"shuffled and added with ALL their windows in the stratum until it holds >= {target} windows; "
                f"senders with > {self._cfg.max_sender_windows} windows in the stratum skipped; plus every crop of "
                f"each curated window's link")
        return {
            "version": 1,
            "created_utc": created_utc,
            "git_commit": git_commit,
            "seed": self._cfg.seed,
            "source": self._cfg.source_dir.as_posix(),
            "seq_len": int(self._data.metadata["max_seq_len"]),
            "features": self._data.features,
            "units": [UNITS[f] for f in self._data.features],
            "norm": {"mean": self._data.mean, "std": self._data.std,
                     "fitted_on": self._data.metadata["normalisation"].get("fitted_on", "")},
            "full_dataset": self._full,
            "sample": {"rule": rule, "per_stratum_target": target, "n_windows": len(windows), "n_rows": rows,
                       "per_split_class": per_split_class, "strata": dict(stratum_stats)},
            "curated": [{**meta, "window_id": rec.id} for meta, rec in curated],
            "windows": entries,
        }


def encode_manifest(manifest: Mapping[str, Any]) -> bytes:
    """Compact JSON (the bundle is size-budgeted)."""
    return (json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


class BundleVerifier:
    """Re-reads the written bundle and checks it against the source shards."""

    def __init__(self, out_dir: pathlib.Path, data: EncoderInput, index: DatasetIndex, budget: int) -> None:
        self._out = out_dir
        self._data = data
        self._index = index
        self._budget = budget

    def verify(self) -> dict[str, Any]:
        manifest = _load_json(self._out / "manifest.json")
        x = np.fromfile(self._out / "x.f32", dtype="<f4")
        if x.size % N_FEATURES:
            raise ExportError("x.f32 size is not a multiple of 13")
        x = x.reshape(-1, N_FEATURES)
        ws = manifest["windows"]
        expected = 0
        for i, w in enumerate(ws):
            if w["i"] != i or w["row"] != expected:
                raise ExportError(f"window {w['id']}: offset {w['row']} != {expected}")
            expected += w["n"]
            if w["sender_splits_full"] != [w["split"]]:
                raise ExportError(f"window {w['id']}: sender in splits {w['sender_splits_full']}")
        if expected != x.shape[0] or manifest["sample"]["n_rows"] != expected:
            raise ExportError(f"rows: manifest {expected}, file {x.shape[0]}")
        if manifest["full_dataset"]["senders_in_multiple_splits"] != 0:
            raise ExportError("a sender vehicle appears in more than one split")
        ids = {w["id"] for w in ws}
        for c in manifest["curated"]:
            if c["window_id"] not in ids:
                raise ExportError(f"curated window {c['window_id']} missing")
        links: dict[str, set[int]] = defaultdict(set)
        for w in ws:
            links[w["link"]].add(w["k"])
        for link, ks in links.items():
            if ks != set(range(len(self._index.links[link]))):
                raise ExportError(f"link {link}: not every crop included")
        source = self._data.rows([self._index.by_id[w["id"]] for w in ws])
        for w in ws:
            if not np.array_equal(x[w["row"]: w["row"] + w["n"]], source[w["id"]]):
                raise ExportError(f"window {w['id']}: rows differ from the shard")
        size = (self._out / "x.f32").stat().st_size + (self._out / "manifest.json").stat().st_size
        if size > self._budget:
            raise ExportError(f"bundle {size} bytes over budget {self._budget}")
        return {"bytes": size, "windows": len(ws), "rows": expected}


class Exporter:
    """End-to-end: read everything, sample within budget, write, verify."""

    def __init__(self, cfg: ExportConfig) -> None:
        self._cfg = cfg
        src = cfg.source_dir if cfg.source_dir.is_absolute() else REPO_ROOT / cfg.source_dir
        self._data = EncoderInput(src, cfg.workers)
        self._writer = BundleWriter(cfg.out_dir, cfg.allowed_out_root)

    def select(self, index: DatasetIndex) -> tuple[dict[str, Any], np.ndarray]:
        """Draws the largest sample (target lowered stepwise) whose bundle fits the budget."""
        sampler = StratifiedSampler(index, self._cfg.seed, self._cfg.max_sender_windows)
        selector = CuratedSelector(index)
        builder = ManifestBuilder(self._cfg, self._data, index)
        fixed = selector.full_dataset_rules()
        created, commit = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), _git_commit()
        target = self._cfg.per_stratum
        while target >= self._cfg.min_per_stratum:
            sample, stats = sampler.sample(target)
            curated = fixed + selector.same_receiver(sample)
            windows = builder.window_records(sample, curated)
            manifest = builder.build(windows, curated, target, stats, created, commit)
            size = len(encode_manifest(manifest)) + manifest["sample"]["n_rows"] * N_FEATURES * 4
            _LOG.info("target %d: %d windows, %d rows, %.2f MB", target, len(windows),
                      manifest["sample"]["n_rows"], size / 1e6)
            if size <= self._cfg.budget_bytes:
                return manifest, self._matrix(windows)
            target -= self._cfg.target_step
        raise ExportError("no per-stratum target fits the budget")

    def _matrix(self, windows: Sequence[WindowRecord]) -> np.ndarray:
        rows = self._data.rows(windows)
        if not windows:
            return np.zeros((0, N_FEATURES), np.float32)
        return np.concatenate([rows[w.id] for w in windows]).astype(np.float32)

    def run(self) -> dict[str, Any]:
        index = DatasetIndex(self._data.all_records())
        if index.senders_in_multiple_splits:
            raise ExportError(f"{index.senders_in_multiple_splits} sender vehicles appear in more than one split")
        manifest, x = self.select(index)
        self._writer.write(encode_manifest(manifest), x)
        report = BundleVerifier(self._writer.out, self._data, index, self._cfg.budget_bytes).verify()
        _LOG.info("verified %s", report)
        for split, cls in manifest["sample"]["per_split_class"].items():
            _LOG.info("  %-13s %s", split, cls)
        for c in manifest["curated"]:
            w = index.by_id[c["window_id"]]
            _LOG.info("  curated %-60s %s n=%d %s", c["name"], w.id, w.n, w.label_name)
        return manifest


def _parse_args(argv: Sequence[str] | None) -> ExportConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = ExportConfig()
    parser.add_argument("--source", type=pathlib.Path, default=d.source_dir)
    parser.add_argument("--out", type=pathlib.Path, default=d.out_dir)
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--per-stratum", type=int, default=d.per_stratum)
    parser.add_argument("--max-sender-windows", type=int, default=d.max_sender_windows)
    parser.add_argument("--budget-bytes", type=int, default=d.budget_bytes)
    parser.add_argument("--workers", type=int, default=d.workers)
    a = parser.parse_args(argv)
    return ExportConfig(source_dir=a.source, out_dir=a.out, seed=a.seed, per_stratum=a.per_stratum,
                        max_sender_windows=a.max_sender_windows, budget_bytes=a.budget_bytes, workers=a.workers)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    Exporter(_parse_args(argv)).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
