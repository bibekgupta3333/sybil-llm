"""Export a label-free detection sample for the simulator's "Detection (model-all)" tab (plan task S1.4.4).

Method: kNN cosine anomaly score on the frozen pretrained encoder, no fine-tuning, no probe.

* The encoder of a pretraining run (default ``src/runs/pretraining/all/T24/model-all/best.pt``) is rebuilt from the
  run's ``config.json`` and loaded from the checkpoint's ``encoder`` weights. Inputs are scaled exactly as training
  scaled them: the shards hold train z-scores (``metadata.json``), robust features are re-expressed with the
  median / IQR + soft tail recorded in the run's ``env.json`` (``scaling``).
* z = the encoder's masked mean pool, L2-normalised.
* Memory bank: z of a seeded sample of unlabeled train windows from the vehicles the run trained on.
* Calibration: z of a seeded sample of unlabeled train windows from the run's label-free check-set vehicles
  (10% of train sender vehicles, seed 0, re-derived exactly as ``FeatureShards.hold_out`` does); disjoint from the
  bank by vehicle and never seen by the optimiser.
* Score = 1 - mean cosine similarity to the K nearest bank windows (higher = more anomalous).
  θ = the ``quantile`` of the calibration scores; no labels are used and no val split exists (D8′), so θ is not tuned.
* Test: the ``all/T24`` test split, group 1416 only (F14 affects GridSybil_0709). Every 1416 test window is scored
  first; only then does ``LabelVault`` open the ``_info`` label files. The stratified per-class display sample is
  drawn after scoring (labels pick which windows are shown, never their scores).

Writes ``simulation/public/data/detection/`` (gitignored): ``manifest.json`` and ``x.f32`` (little-endian float32
real rows in raw units, ``n x 13`` per window, concatenated in ``windows`` order). Nothing under ``data/``,
``src/data/`` or ``src/runs/`` is ever written.

Usage:
    .venv-train/bin/python scripts/export_detection_sample.py
    .venv-train/bin/python scripts/export_detection_sample.py --per-class 300 --bank 60000 --k 10 --device mps
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gzip
import hashlib
import json
import logging
import multiprocessing
import pathlib
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.benign_gridsybil.timesnet.config import PretrainConfig  # noqa: E402
from src.model.benign_gridsybil.timesnet.data import (  # noqa: E402
    FEATURES,
    CompactWindowReader,
    FeatureShards,
    FeatureSpace,
    LabelFirewall,
    ShardPart,
)
from src.model.benign_gridsybil.timesnet.encoder import TimesNetEncoder  # noqa: E402

_LOG = logging.getLogger("export_detection_sample")

CLASSES: tuple[str, ...] = ("Benign", "GridSybil", "DataReplaySybil", "DoSRandomSybil", "DoSDisruptiveSybil")
UNITS: dict[str, str] = {
    "claimed_pos_x": "m",
    "claimed_pos_y": "m",
    "rx_pos_x": "m",
    "rx_pos_y": "m",
    "claimed_vel_x": "m/s",
    "claimed_vel_y": "m/s",
    "rx_vel_x": "m/s",
    "rx_vel_y": "m/s",
    "claimed_acl_x": "m/s²",
    "claimed_acl_y": "m/s²",
    "range": "m",
    "bearing": "rad",
    "log_dtau": "log s",
}
PROTECTED = ("data", "src/data", "src/runs")


class ExportError(ValueError):
    """Raised when the inputs or the bundle fail a consistency check."""


@dataclasses.dataclass(frozen=True)
class DetectionConfig:
    """Knobs of the export (recorded in the manifest where they shape a number).

    Attributes:
        run_dir: pretraining run folder (config.json, env.json, checkpoint, SHA256SUMS).
        checkpoint: checkpoint file name inside ``run_dir``.
        data_dir: encoder input of the run (metadata.json, train/, test/).
        out_dir: bundle destination; must lie under ``allowed_out_root``.
        allowed_out_root: the only directory the writer may write under.
        group: scenario group of the test windows ("1416"; 0709 waits on F14).
        per_class: display windows per class (all if fewer).
        bank: memory-bank windows (unlabeled train, trained-on vehicles).
        calib: calibration windows for θ (unlabeled train, check-set vehicles).
        k: nearest bank neighbours per score.
        quantile: θ = this quantile of the calibration scores.
        seed: RNG seed for every sample.
        device: torch device ("auto" = MPS, then CUDA, then CPU).
        batch_size: windows per encoder / kNN batch.
        workers: parallel shard readers (1 = in-process).
        candidate_rate: share of train windows kept as sampling candidates before the final draw.
        budget_bytes: limit on ``x.f32 + manifest.json``.
    """

    run_dir: pathlib.Path = pathlib.Path("src/runs/pretraining/all/T24/model-all")
    checkpoint: str = "best.pt"
    data_dir: pathlib.Path | None = None
    out_dir: pathlib.Path = pathlib.Path("simulation/public/data/detection")
    allowed_out_root: pathlib.Path = pathlib.Path("simulation/public/data/detection")
    group: str = "1416"
    per_class: int = 300
    bank: int = 60_000
    calib: int = 20_000
    k: int = 10
    quantile: float = 0.95
    seed: int = 0
    device: str = "auto"
    batch_size: int = 2048
    workers: int = 8
    candidate_rate: float = 0.06
    budget_bytes: int = 15_000_000

    def validate(self) -> None:
        """Fails early on settings that cannot work."""
        if not 0.0 < self.quantile < 1.0:
            raise ExportError("quantile must lie in (0, 1)")
        if self.k < 1 or self.bank < self.k:
            raise ExportError("need 1 <= k <= bank")
        if self.per_class < 1 or self.calib < 1:
            raise ExportError("per_class and calib must be >= 1")


def pick_device(name: str) -> torch.device:
    """`auto` = MPS, then CUDA, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def git_short_sha() -> str:
    """Short commit of the repo, or "unknown"."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_of(path: pathlib.Path) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def auroc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """Area under the ROC curve (Mann-Whitney, ties count half); None if a side is empty."""
    if len(positive) == 0 or len(negative) == 0:
        return None
    values = np.concatenate([positive, negative]).astype(np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts  # 0-based first rank of each distinct value
    mean_rank = starts + (counts + 1) / 2.0  # average 1-based rank of the tied block
    ranks = mean_rank[inverse]
    n_pos, n_neg = len(positive), len(negative)
    u = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


@dataclasses.dataclass
class WindowSet:
    """Windows in the stored scaling (train z-scores): x (N, T, 13), mask (N, T), ids."""

    x: torch.Tensor
    mask: torch.Tensor
    ids: list[str]

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def lengths(self) -> np.ndarray:
        """Real rows per window."""
        return self.mask.sum(1).numpy().astype(np.int64)

    @staticmethod
    def from_parts(parts: Sequence[ShardPart], seq_len: int) -> "WindowSet":
        """Pads compact shard parts to seq_len and builds the mask (1 = real row)."""
        total = sum(len(p) for p in parts)
        x = np.zeros((total, seq_len, len(FEATURES)), dtype=np.float32)
        mask = np.zeros((total, seq_len), dtype=bool)
        ids: list[str] = []
        start = 0
        for part in parts:
            end = start + len(part)
            part.fill(x[start:end], mask[start:end])
            ids.extend(part.ids)
            start = end
        return WindowSet(torch.from_numpy(x), torch.from_numpy(mask), ids)

    def take(self, rows: np.ndarray) -> "WindowSet":
        """The windows at `rows`, in that order."""
        index = torch.as_tensor(rows, dtype=torch.long)
        return WindowSet(self.x[index], self.mask[index], [self.ids[i] for i in rows.tolist()])


def select_part(part: ShardPart, keep: np.ndarray) -> ShardPart:
    """A compact ShardPart reduced to the windows where `keep` is True."""
    if part.lengths is None:
        raise ExportError("expected the compact layout (all/T24)")
    ends = np.cumsum(part.lengths)
    starts = ends - part.lengths
    rows = np.concatenate([np.arange(s, e) for s, e in zip(starts[keep], ends[keep])]) if keep.any() else []
    return ShardPart(
        part.x[np.asarray(rows, dtype=np.int64)],
        None,
        part.lengths[keep],
        [i for i, k in zip(part.ids, keep) if k],
    )


class TrainCandidateReader:
    """Reads one train feature shard (through the pretraining `LabelFirewall`) and keeps a seeded random share.

    Each kept window carries a uniform key u (seeded by shard index); the final samples are the windows with the
    smallest keys, which is a uniform draw without replacement whatever the worker order.
    """

    def __init__(self, seq_len: int, seed: int, rate: float) -> None:
        self.reader = CompactWindowReader(seq_len)
        self.firewall = LabelFirewall()
        self.seed, self.rate = seed, rate

    def __call__(self, job: tuple[int, str]) -> tuple[ShardPart, np.ndarray, list[str]]:
        index, path = job
        part = self.reader.read(str(self.firewall.check(pathlib.Path(path))))
        keys = np.random.default_rng([self.seed, index]).random(len(part))
        keep = keys < self.rate
        vehicles = sorted({FeatureShards.vehicle_of(i) for i in part.ids})
        return select_part(part, keep), keys[keep], vehicles


class GroupTestReader:
    """Reads one test feature shard (never its `_info` file) and keeps the windows of one scenario group.

    The group is read from the window id (`<scenario>_<group>/...`), not from a label file.
    """

    def __init__(self, seq_len: int, group: str) -> None:
        self.reader = CompactWindowReader(seq_len)
        self.group = group

    def __call__(self, path: str) -> ShardPart:
        if "_info" in pathlib.Path(path).name:
            raise PermissionError(f"label file {path} requested before scoring")
        part = self.reader.read(path)
        keep = np.array([i.split("/", 1)[0].rsplit("_", 1)[1] == self.group for i in part.ids], dtype=bool)
        return select_part(part, keep)


class ShardPool:
    """Maps a reader over shard jobs, in a fork pool or in-process (workers = 1)."""

    def __init__(self, workers: int) -> None:
        self.workers = workers

    def map(self, fn: Any, jobs: Sequence[Any]) -> list[Any]:
        if self.workers <= 1 or len(jobs) <= 1:
            return [fn(j) for j in jobs]
        with multiprocessing.get_context("fork").Pool(min(self.workers, len(jobs))) as pool:
            return list(pool.imap(fn, jobs))


class TrainSampler:
    """Draws the memory bank (trained-on vehicles) and the calibration set (check-set vehicles) from train.

    The check-set vehicles are re-derived as `FeatureShards.hold_out(holdout_frac, seed)` derives them during
    training: sorted vehicle keys of all train windows, shuffled with `default_rng(seed)`, the first
    round(frac * n) are the check set. No label file is opened (every path passes `LabelFirewall`).
    """

    def __init__(self, data_dir: pathlib.Path, settings: PretrainConfig, cfg: DetectionConfig) -> None:
        self.data_dir, self.settings, self.cfg = data_dir, settings, cfg

    def check_vehicles(self, vehicles: set[str]) -> set[str]:
        """The run's label-free check-set vehicles."""
        ordered = sorted(vehicles)
        np.random.default_rng(self.settings.seed).shuffle(ordered)
        return set(ordered[: max(1, int(round(self.settings.holdout_frac * len(ordered))))])

    def sample(self) -> tuple[WindowSet, WindowSet, dict[str, Any]]:
        """(bank, calibration, report)."""
        cfg = self.cfg
        paths = CompactWindowReader(self.settings.seq_len).find(self.data_dir, "train")
        if not paths:
            raise ExportError(f"no train shards under {self.data_dir}")
        reader = TrainCandidateReader(self.settings.seq_len, cfg.seed, cfg.candidate_rate)
        results = ShardPool(cfg.workers).map(reader, list(enumerate(str(p) for p in paths)))
        vehicles: set[str] = set()
        for _, _, v in results:
            vehicles.update(v)
        check = self.check_vehicles(vehicles)
        pooled = WindowSet.from_parts([r[0] for r in results], self.settings.seq_len)
        keys = np.concatenate([r[1] for r in results]) if results else np.zeros(0)
        in_check = np.array([FeatureShards.vehicle_of(i) in check for i in pooled.ids], dtype=bool)
        bank_rows = self._smallest(keys, ~in_check, cfg.bank, "bank")
        calib_rows = self._smallest(keys, in_check, cfg.calib, "calibration")
        report = {
            "train_shards": len(paths),
            "train_vehicles": len(vehicles),
            "check_vehicles": len(check),
            "candidates": len(pooled),
        }
        return pooled.take(bank_rows), pooled.take(calib_rows), report

    @staticmethod
    def _smallest(keys: np.ndarray, allowed: np.ndarray, n: int, name: str) -> np.ndarray:
        rows = np.flatnonzero(allowed)
        if len(rows) < n:
            raise ExportError(f"only {len(rows)} {name} candidates for {n} windows; raise candidate_rate")
        return np.sort(rows[np.argsort(keys[rows], kind="stable")[:n]])


class TrainedRun:
    """A pretraining run: settings, verified checkpoint, frozen encoder and the scaling training used."""

    def __init__(self, run_dir: pathlib.Path, checkpoint: str, data_dir: pathlib.Path | None) -> None:
        self.run_dir = run_dir
        self.settings = PretrainConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
        self.data_dir = pathlib.Path(data_dir or self.settings.data_dir)
        if not self.data_dir.is_absolute():
            self.data_dir = REPO_ROOT / self.data_dir
        self.ckpt_path = run_dir / checkpoint
        self.sha256 = sha256_of(self.ckpt_path)
        self._verify_sum(checkpoint)
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.epoch = int(state.get("epoch", -1))
        self.best_val_joint = float(state["best"]) if state.get("best") is not None else None
        self.encoder = TimesNetEncoder(self.settings)
        self.encoder.load_state_dict(state["encoder"])
        self.encoder.eval()
        metadata = json.loads((self.data_dir / "metadata.json").read_text())
        env = json.loads((run_dir / "env.json").read_text())
        self.stored = FeatureSpace(metadata)
        self.space = self.scaling_from_report(metadata, env.get("scaling", {}))

    def _verify_sum(self, checkpoint: str) -> None:
        sums = self.run_dir / "SHA256SUMS"
        if not sums.exists():
            _LOG.warning("no SHA256SUMS in %s; checkpoint not verified", self.run_dir)
            return
        listed = dict(reversed(line.split(None, 1)) for line in sums.read_text().splitlines() if line.strip())
        expected = listed.get(checkpoint)
        if expected is not None and expected != self.sha256:
            raise ExportError(f"{checkpoint} sha256 {self.sha256} != SHA256SUMS {expected}")

    @staticmethod
    def scaling_from_report(metadata: dict[str, Any], report: dict[str, Any]) -> FeatureSpace:
        """The run's FeatureSpace: train z-scores, robust features from env.json's median / IQR + soft tail."""
        space = FeatureSpace(metadata)
        for j, name in enumerate(space.features):
            entry = report.get(name, {"method": "train z-score"})
            if entry["method"].startswith("median"):
                space.mean[j], space.std[j], space.robust[j] = entry["median"], entry["iqr"], True
                space.soft_clip = float(entry["soft_clip"])
            elif "mean" in entry and abs(entry["mean"] - float(space.mean[j])) > 1e-3 * max(1.0, abs(entry["mean"])):
                raise ExportError(f"{name}: env.json mean {entry['mean']} differs from metadata {float(space.mean[j])}")
        space.report = report
        return space


class Embedder:
    """Frozen encoder -> L2-normalised z; windows are batched by length (as training batched them)."""

    def __init__(self, run: TrainedRun, device: torch.device, batch_size: int) -> None:
        self.encoder = run.encoder.to(device)
        self.stored, self.space = run.stored, run.space
        self.device, self.batch_size = device, batch_size

    @torch.no_grad()
    def __call__(self, windows: WindowSet) -> torch.Tensor:
        lengths = torch.from_numpy(windows.lengths)
        order = torch.argsort(lengths, stable=True)
        out = torch.zeros(len(windows), self.encoder.norm.normalized_shape[0])
        for batch in order.split(self.batch_size):
            x, mask = windows.x[batch], windows.mask[batch]
            x = self.space.norm(self.stored.raw(x), mask)  # stored z-scores -> the run's scaling
            _, z = self.encoder(x.to(self.device), mask.to(self.device))
            out[batch] = torch.nn.functional.normalize(z.float(), dim=1).cpu()
        return out


class KnnScorer:
    """Score = 1 - mean cosine similarity to the K nearest bank windows (inputs already L2-normalised)."""

    def __init__(self, bank: torch.Tensor, k: int, device: torch.device, batch_size: int = 4096) -> None:
        if k > len(bank):
            raise ExportError("k exceeds the bank size")
        self.bank, self.k, self.device, self.batch_size = bank.to(device), k, device, batch_size

    @torch.no_grad()
    def __call__(self, z: torch.Tensor) -> np.ndarray:
        scores = []
        for start in range(0, len(z), self.batch_size):
            sims = z[start : start + self.batch_size].to(self.device) @ self.bank.T
            top = sims.topk(self.k, dim=1).values
            scores.append((1.0 - top.mean(dim=1)).cpu())
        return torch.cat(scores).numpy().astype(np.float64) if scores else np.zeros(0)


class Threshold:
    """θ = a quantile of label-free calibration scores; rank = share of calibration scores <= a score."""

    def __init__(self, calibration_scores: np.ndarray, quantile: float) -> None:
        self.sorted = np.sort(np.asarray(calibration_scores, dtype=np.float64))
        self.quantile = quantile
        self.theta = float(np.quantile(self.sorted, quantile))

    def flags(self, scores: np.ndarray) -> np.ndarray:
        """True where a score is above θ."""
        return np.asarray(scores) > self.theta

    def rank_pct(self, scores: np.ndarray) -> np.ndarray:
        """Percent of calibration scores <= each score."""
        return 100.0 * np.searchsorted(self.sorted, np.asarray(scores), side="right") / len(self.sorted)


class LabelVault:
    """Opens the `_info` label files only after every window has a score (firewall for "labels after scoring")."""

    def __init__(self, info_paths: Sequence[pathlib.Path]) -> None:
        self.info_paths = list(info_paths)
        self.opened = False

    def reveal(self, ids: Sequence[str], scores: np.ndarray | None) -> list[dict[str, Any]]:
        """Label records for `ids` (same order); refuses unless `scores` is complete and finite."""
        if scores is None or len(scores) != len(ids) or not np.all(np.isfinite(scores)):
            raise PermissionError("labels may be read only after every window is scored")
        wanted = set(ids)
        found: dict[str, dict[str, Any]] = {}
        for path in self.info_paths:
            with gzip.open(path, "rt") as f:
                for w in json.load(f)["windows"]:
                    if w["id"] in wanted:
                        found[w["id"]] = {
                            "class": w["label_name"],
                            "label": int(w["is_attack"]),
                            "scenario": w["scenario"],
                        }
        self.opened = True
        missing = wanted - found.keys()
        if missing:
            raise ExportError(f"{len(missing)} scored windows have no label record")
        return [found[i] for i in ids]


class MetricsTable:
    """Per-class metrics of a scored set: AUROC vs benign, flag rate at θ, length-only baseline AUROC."""

    @staticmethod
    def build(classes: np.ndarray, scores: np.ndarray, lengths: np.ndarray, flags: np.ndarray) -> dict[str, Any]:
        benign = classes == "Benign"
        table: dict[str, Any] = {}
        for name in CLASSES:
            rows = classes == name
            if not rows.any():
                table[name] = {
                    "n": 0,
                    "auroc_vs_benign": None,
                    "flag_rate": None,
                    "length_baseline_auroc": None,
                    "mean_rows": None,
                }
                continue
            attack = name != "Benign"
            table[name] = {
                "n": int(rows.sum()),
                "auroc_vs_benign": auroc(scores[rows], scores[benign]) if attack else None,
                "flag_rate": float(flags[rows].mean()),
                "length_baseline_auroc": (
                    auroc(-lengths[rows].astype(float), -lengths[benign].astype(float)) if attack else None
                ),
                "mean_rows": float(lengths[rows].mean()),
            }
        return table


class BundleWriter:
    """Writes manifest.json + x.f32 under the allowed root only."""

    def __init__(self, out_dir: pathlib.Path, allowed_root: pathlib.Path) -> None:
        out = (REPO_ROOT / out_dir).resolve() if not out_dir.is_absolute() else out_dir.resolve()
        root = (REPO_ROOT / allowed_root).resolve() if not allowed_root.is_absolute() else allowed_root.resolve()
        if out != root and root not in out.parents:
            raise ExportError(f"refusing to write {out}: outside {root}")
        for name in PROTECTED:
            guard = (REPO_ROOT / name).resolve()
            if out == guard or guard in out.parents:
                raise ExportError(f"refusing to write under protected {guard}")
        self.out = out

    def write(self, manifest: dict[str, Any], rows: np.ndarray, budget_bytes: int) -> int:
        """Writes both files; returns the bundle size in bytes."""
        payload = json.dumps(manifest, separators=(",", ":"), allow_nan=False).encode()
        blob = np.ascontiguousarray(rows, dtype="<f4").tobytes()
        if len(payload) + len(blob) > budget_bytes:
            raise ExportError(f"bundle {len(payload) + len(blob):,} B exceeds budget {budget_bytes:,} B")
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "x.f32").write_bytes(blob)
        (self.out / "manifest.json").write_bytes(payload)
        return len(payload) + len(blob)


class DetectionExport:
    """Loads the run, builds bank + θ, scores every test window of the group, then reads labels and writes the bundle."""

    def __init__(self, cfg: DetectionConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self.writer = BundleWriter(cfg.out_dir, cfg.allowed_out_root)
        self.run = TrainedRun(self._abs(cfg.run_dir), cfg.checkpoint, cfg.data_dir and self._abs(cfg.data_dir))

    @staticmethod
    def _abs(path: pathlib.Path) -> pathlib.Path:
        return path if path.is_absolute() else REPO_ROOT / path

    def read_test(self) -> tuple[WindowSet, list[pathlib.Path]]:
        """Test windows of the group (features only) and the label files, which stay closed for now."""
        reader = CompactWindowReader(self.run.settings.seq_len)
        paths = reader.find(self.run.data_dir, "test")
        if not paths:
            raise ExportError(f"no test shards under {self.run.data_dir}")
        parts = ShardPool(self.cfg.workers).map(
            GroupTestReader(self.run.settings.seq_len, self.cfg.group), [str(p) for p in paths]
        )
        infos = [p.with_name(p.name.replace(".json.gz", "_info.json.gz")) for p in paths]
        return WindowSet.from_parts(parts, self.run.settings.seq_len), infos

    def run_all(self) -> dict[str, Any]:
        """The whole export; returns a summary (also logged)."""
        cfg, t0 = self.cfg, time.time()
        bank_set, calib_set, sample_report = TrainSampler(self.run.data_dir, self.run.settings, cfg).sample()
        test, info_paths = self.read_test()
        _LOG.info("bank %d, calibration %d, test %s %d windows", len(bank_set), len(calib_set), cfg.group, len(test))
        device = pick_device(cfg.device)
        embed = Embedder(self.run, device, cfg.batch_size)
        scorer = KnnScorer(embed(bank_set), cfg.k, device)
        threshold = Threshold(scorer(embed(calib_set)), cfg.quantile)
        scores = scorer(embed(test))  # every test window is scored before any label is read
        flags = threshold.flags(scores)
        labels = LabelVault(info_paths).reveal(test.ids, scores)
        classes = np.array([r["class"] for r in labels])
        lengths = test.lengths
        full = MetricsTable.build(classes, scores, lengths, flags)
        rows = self._display_rows(classes)
        sample = MetricsTable.build(classes[rows], scores[rows], lengths[rows], flags[rows])
        manifest, x_raw = self._manifest(test, rows, labels, scores, flags, threshold, full, sample, device)
        seconds = time.time() - t0
        manifest["caveats"].append(f"exported in {seconds:.0f} s on {device}; sampling: {sample_report}")
        size = self.writer.write(manifest, x_raw, cfg.budget_bytes)
        summary = {"theta": threshold.theta, "sample": sample, "full": full, "bytes": size, "seconds": seconds}
        _LOG.info("summary %s", json.dumps(summary, indent=1))
        return summary

    def _display_rows(self, classes: np.ndarray) -> np.ndarray:
        """Seeded stratified display sample (drawn after scoring; changes only which windows are shown)."""
        rng = np.random.default_rng(self.cfg.seed)
        picked = []
        for name in CLASSES:
            rows = np.flatnonzero(classes == name)
            take = min(self.cfg.per_class, len(rows))
            picked.append(np.sort(rng.choice(rows, size=take, replace=False)) if take else rows)
        return np.concatenate(picked).astype(np.int64)

    def _manifest(
        self,
        test: WindowSet,
        rows: np.ndarray,
        labels: list[dict[str, Any]],
        scores: np.ndarray,
        flags: np.ndarray,
        threshold: Threshold,
        full: dict[str, Any],
        sample: dict[str, Any],
        device: torch.device,
    ) -> tuple[dict[str, Any], np.ndarray]:
        cfg, run = self.cfg, self.run
        shown = test.take(rows)
        raw = run.stored.raw(shown.x)  # stored z-scores -> raw units (metres, m/s, ...)
        lengths = shown.lengths
        x_rows = raw[shown.mask].numpy().astype("<f4")
        ranks = threshold.rank_pct(scores[rows])
        windows, offset = [], 0
        for j, r in enumerate(rows.tolist()):
            windows.append(
                {
                    "i": j,
                    "id": test.ids[r],
                    "scenario": labels[r]["scenario"],
                    "class": labels[r]["class"],
                    "label": labels[r]["label"],
                    "n": int(lengths[j]),
                    "offset": offset,
                    "score": round(float(scores[r]), 6),
                    "flag": bool(flags[r]),
                    "rank_pct": round(float(ranks[j]), 2),
                }
            )
            offset += int(lengths[j])
        if offset * len(FEATURES) != x_rows.size:
            raise ExportError("row offsets do not match the exported rows")
        benign = np.array([lab["class"] == "Benign" for lab in labels])
        manifest = {
            "schema": 1,
            "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_short_sha(),
            "model": {
                "run_id": run.run_dir.name,
                "checkpoint": cfg.checkpoint,
                "sha256": run.sha256,
                "epoch": run.epoch,
                "best_val_joint": run.best_val_joint,
                "dataset": run.settings.dataset,
                "seq_len": run.settings.seq_len,
                "d_model": run.settings.d_model,
            },
            "method": {
                "name": "kNN cosine anomaly score (label-free)",
                "k": cfg.k,
                "bank_size": cfg.bank,
                "theta": threshold.theta,
                "theta_rule": (
                    f"θ = p{cfg.quantile * 100:g} of the scores of {cfg.calib:,} unlabeled train windows from the "
                    f"run's check-set vehicles (not in the bank, never trained on); no labels used; no val split "
                    f"exists (D8′), so θ is not tuned. score = 1 − mean cosine similarity of z to its {cfg.k} "
                    f"nearest bank windows; bank = {cfg.bank:,} unlabeled train windows of the trained-on vehicles. "
                    f"rank_pct = % of those calibration scores ≤ the window's score."
                ),
                "seed": cfg.seed,
            },
            "scope": {
                "split": "test",
                "group": cfg.group,
                "note": "0709 excluded until F14 is decided",
                "per_class": cfg.per_class,
            },
            "features": list(FEATURES),
            "units": dict(UNITS),
            "metrics": {
                "sample": sample,
                "full_test_1416": full,
                "benign_fpr_at_theta": float(flags[benign].mean()) if benign.any() else None,
            },
            "caveats": [
                "label-free anomaly score of the pretrained encoder: no fine-tuning, no probe, θ not tuned "
                "(pretraining health, not a tuned detector)",
                "DataReplay / DoS windows are mostly 1–3 rows (F11); compare AUROC with length_baseline_auroc "
                "(score = −real rows) to see whether the model beats the length shortcut",
                "the bank is unlabeled train, which is mostly attack windows (DoS / DataReplay); a high score means "
                "'unlike the train mix', not 'unlike benign'",
                "Benign = every benign test window of the 4 families of the group, pooled; the families re-simulate "
                "the same traffic (F1), so benign windows repeat across families",
                f"benign_fpr_at_theta is computed on the full {cfg.group} test split; 'sample' = the displayed "
                f"windows (seeded, ≤ {cfg.per_class} per class, drawn after scoring)",
                "labels (_info files) were opened only after every test window was scored; they are for display "
                "and metrics only",
                "x.f32 holds raw units de-normalised from the stored 4-decimal z-scores",
            ],
            "windows": windows,
        }
        return manifest, x_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    defaults = DetectionConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--run", type=pathlib.Path, default=defaults.run_dir, help="pretraining run folder")
    p.add_argument("--checkpoint", default=defaults.checkpoint)
    p.add_argument("--per-class", type=int, default=defaults.per_class)
    p.add_argument("--bank", type=int, default=defaults.bank)
    p.add_argument("--calib", type=int, default=defaults.calib)
    p.add_argument("--k", type=int, default=defaults.k)
    p.add_argument("--quantile", type=float, default=defaults.quantile)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--device", default=defaults.device)
    p.add_argument("--workers", type=int, default=defaults.workers)
    p.add_argument("--out", type=pathlib.Path, default=defaults.out_dir)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    cfg = DetectionConfig(
        run_dir=args.run,
        checkpoint=args.checkpoint,
        per_class=args.per_class,
        bank=args.bank,
        calib=args.calib,
        k=args.k,
        quantile=args.quantile,
        seed=args.seed,
        device=args.device,
        workers=args.workers,
        out_dir=args.out,
    )
    DetectionExport(cfg).run_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
