"""Label-free data access for pretraining.

`LabelFirewall` guards every file the run opens (feature shards of `train`, or `pretrain_val` if present, never `_info`
label files, never `val` / `test`). `FeatureShards` holds one split in RAM and can carve a check set out of train by
sender vehicle. `FeatureSpace` knows the 13-feature order and the train normalisation.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import math
import multiprocessing
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

FEATURES = [
    "claimed_pos_x",
    "claimed_pos_y",
    "rx_pos_x",
    "rx_pos_y",
    "claimed_vel_x",
    "claimed_vel_y",
    "rx_vel_x",
    "rx_vel_y",
    "claimed_acl_x",
    "claimed_acl_y",
    "range",
    "bearing",
    "log_dtau",
]


class LabelFirewall:
    """Guards every file the run opens: only feature shards of the pretraining splits, never labels.

    Two layouts are accepted: `<split>/b<bucket>/part-XXXXX.json` (benign + GridSybil T64) and
    `<split>/part-XXXXX.json.gz` (all attacks, compact T24). Label files (`part-XXXXX_info.json[.gz]`) never pass.
    """

    ALLOWED_SPLITS = ("train", "pretrain_val")
    SHARD = re.compile(r"^part-\d{5}\.json(\.gz)?$")
    BUCKET = re.compile(r"^b\d+$")

    def check(self, path: Path) -> Path:
        """Returns `path` if it is a feature shard of an allowed split; raises otherwise."""
        folder = path.parent
        split = folder.parent.name if self.BUCKET.match(folder.name) else folder.name
        if split not in self.ALLOWED_SPLITS:
            raise PermissionError(f"label firewall: split {split!r} is not allowed in pretraining ({path})")
        if not self.SHARD.match(path.name) or "_info" in path.name:
            raise PermissionError(f"label firewall: {path.name} is not a feature shard")
        return path


@dataclasses.dataclass
class ShardPart:
    """The windows of one shard as read by a worker.

    Padded layout: `x` (W, T, 13) and `mask` (W, T). Compact layout: `x` (M, 13) holds only the real rows of all
    windows one after another, `lengths` (W,) the real rows per window, `mask` is None.
    """

    x: np.ndarray
    mask: np.ndarray | None
    lengths: np.ndarray | None
    ids: list[str]

    def __len__(self) -> int:
        return len(self.ids)

    def fill(self, x_out: np.ndarray, mask_out: np.ndarray) -> None:
        """Writes the windows into preallocated (W, T, 13) / (W, T) slices (padding rows stay 0 / False)."""
        if self.mask is not None:
            x_out[...] = self.x
            mask_out[...] = self.mask
            return
        offsets = np.repeat(np.cumsum(self.lengths) - self.lengths, self.lengths)
        window = np.repeat(np.arange(len(self.lengths)), self.lengths)
        row = np.arange(len(self.x)) - offsets
        x_out[window, row] = self.x
        mask_out[window, row] = True


class ShardReader:
    """Finds and reads the feature shards of one layout (subclasses); `read` runs in a fork pool worker."""

    name = "base"

    def find(self, root: Path, split: str) -> list[Path]:
        """Sorted feature shards of `split` (label files never match)."""
        raise NotImplementedError

    def read(self, path: str) -> ShardPart:
        """One shard -> ShardPart."""
        raise NotImplementedError

    @staticmethod
    def pick(root: Path, split: str, seq_len: int) -> "ShardReader":
        """The reader whose layout is present under `root / split`."""
        for reader in (CompactWindowReader(seq_len), PaddedWindowReader(seq_len)):
            if reader.find(root, split):
                return reader
        raise FileNotFoundError(f"no shards for split {split!r} under {root}")


class PaddedWindowReader(ShardReader):
    """`<split>/b<bucket>/part-XXXXX.json` = {windows: [{id, x (T x 13), mask (T)}]} (benign + GridSybil T64)."""

    name = "padded json (b<bucket>/part-XXXXX.json)"

    def __init__(self, seq_len: int) -> None:
        self.seq_len = seq_len

    def find(self, root: Path, split: str) -> list[Path]:
        return sorted((root / split).glob("b*/part-?????.json"))

    def read(self, path: str) -> ShardPart:
        data = json.loads(Path(path).read_text())
        x = np.asarray([w["x"] for w in data["windows"]], dtype=np.float32)
        mask = np.asarray([w["mask"] for w in data["windows"]], dtype=bool)
        if x.shape[1] != self.seq_len:
            raise ValueError(f"{path}: windows have {x.shape[1]} rows, settings say seq_len {self.seq_len}")
        return ShardPart(x, mask, None, [w["id"] for w in data["windows"]])


class CompactWindowReader(ShardReader):
    """`<split>/part-XXXXX.json.gz` = {seq_len, features, windows: [{id, n, x (n real rows x 13)}]} (all attacks, T24).

    Only real rows are stored; `ShardPart.fill` pads every window to seq_len and builds the mask (1 = real row).
    """

    name = "compact gzipped json (part-XXXXX.json.gz)"

    def __init__(self, seq_len: int) -> None:
        self.seq_len = seq_len

    def find(self, root: Path, split: str) -> list[Path]:
        return sorted((root / split).glob("part-?????.json.gz"))

    def read(self, path: str) -> ShardPart:
        with gzip.open(path, "rt") as f:
            data = json.load(f)
        if data["seq_len"] != self.seq_len:
            raise ValueError(f"{path}: seq_len {data['seq_len']}, settings say {self.seq_len}")
        if data["features"] != FEATURES:
            raise ValueError(f"{path}: unexpected feature order {data['features']}")
        windows = data["windows"]
        lengths = np.asarray([w["n"] for w in windows], dtype=np.int64)
        if lengths.size and (lengths.min() < 1 or lengths.max() > self.seq_len):
            raise ValueError(f"{path}: window lengths outside 1..{self.seq_len}")
        rows = np.asarray([r for w in windows for r in w["x"]], dtype=np.float32).reshape(-1, len(FEATURES))
        if len(rows) != int(lengths.sum()):
            raise ValueError(f"{path}: {len(rows)} rows but the windows say {int(lengths.sum())}")
        return ShardPart(rows, None, lengths, [w["id"] for w in windows])


class FeatureShards:
    """All windows of one split in RAM: x (N, T, 13) float32, mask (N, T) bool, ids (link bookkeeping only).

    The reader is picked by the files found (padded T64 or compact T24). x and mask are preallocated once and filled
    shard by shard, so the peak is the final arrays plus the compact rows of the shards read so far.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        firewall: LabelFirewall,
        max_shards: int | None,
        workers: int,
        seq_len: int = 64,
    ):
        reader = ShardReader.pick(root, split, seq_len)
        paths = reader.find(root, split)
        if max_shards is not None:
            paths = paths[:max_shards]
        paths = [str(firewall.check(Path(p))) for p in paths]
        with multiprocessing.get_context("fork").Pool(max(1, min(workers, len(paths)))) as pool:
            parts: list[ShardPart | None] = list(pool.imap(reader.read, paths))
        total = sum(len(p) for p in parts)
        x = np.zeros((total, seq_len, len(FEATURES)), dtype=np.float32)
        mask = np.zeros((total, seq_len), dtype=bool)
        self.ids: list[str] = []
        start = 0
        for i, part in enumerate(parts):
            end = start + len(part)
            part.fill(x[start:end], mask[start:end])
            self.ids.extend(part.ids)
            parts[i] = None  # free the shard's rows as soon as they are copied
            start = end
        self.split = split
        self.x = torch.from_numpy(x)
        self.mask = torch.from_numpy(mask)
        self.n_shards = len(paths)
        self.reader = reader.name

    @staticmethod
    def ram_estimate_gb(windows: int, seq_len: int) -> float:
        """Resident size of `windows` windows: x float32 + mask bool + ~150 bytes per id string."""
        return windows * (seq_len * len(FEATURES) * 4 + seq_len + 150) / 1e9

    def __len__(self) -> int:
        return self.x.shape[0]

    @staticmethod
    def vehicle_of(window_id: str) -> str:
        """`<scenario>/<run>/<file>#<pseudo>-<sender>#<k>` -> `"<group>:<sender>"` (bookkeeping only, never an input)."""
        scenario = window_id.split("/", 1)[0]
        sender = window_id.rsplit("#", 2)[1].split("-")[-1]
        return f"{scenario.rsplit('_', 1)[1]}:{sender}"

    def subset(self, rows: torch.Tensor, split: str) -> "FeatureShards":
        """A view of some windows under another split name."""
        part = object.__new__(FeatureShards)
        part.split, part.n_shards, part.reader = split, self.n_shards, getattr(self, "reader", "")
        part.x, part.mask = self.x[rows], self.mask[rows]
        part.ids = [self.ids[i] for i in rows.tolist()]
        return part

    def hold_out(self, frac: float, seed: int, in_place: bool = False) -> tuple["FeatureShards", "FeatureShards"]:
        """Splits off a check set by sender vehicle (each vehicle entirely in one part): (rest, check set).

        With `in_place = True` the rest is compacted inside this object's arrays (no second copy of the train data;
        this object becomes the rest), which keeps the peak RAM of the 5M-window all/T24 input near its final size.
        """
        vehicles = sorted({self.vehicle_of(i) for i in self.ids})
        rng = np.random.default_rng(seed)
        rng.shuffle(vehicles)
        chosen = set(vehicles[: max(1, int(round(frac * len(vehicles))))])
        is_check = torch.tensor([self.vehicle_of(i) in chosen for i in self.ids], dtype=torch.bool)
        check = self.subset(torch.nonzero(is_check).squeeze(1), "pretrain_check")
        keep = torch.nonzero(~is_check).squeeze(1)
        if not in_place:
            return self.subset(keep, self.split), check
        chunk = 65536
        for start in range(0, len(keep), chunk):  # destination rows never pass their source rows
            src = keep[start : start + chunk]
            self.x[start : start + len(src)] = self.x[src]
            self.mask[start : start + len(src)] = self.mask[src]
        self.x, self.mask = self.x[: len(keep)], self.mask[: len(keep)]
        self.ids = [self.ids[i] for i in keep.tolist()]
        return self, check

    @property
    def link_keys(self) -> list[str]:
        """Window id without the crop index: windows of the same link share it (diagnostics only)."""
        return [i.rsplit("#", 1)[0] for i in self.ids]

    def link_groups(self) -> torch.Tensor:
        """(N,) long: windows of the same link share a number (diagnostics only; cheaper than `link_keys`)."""
        codes: dict[str, int] = {}
        return torch.tensor([codes.setdefault(i.rsplit("#", 1)[0], len(codes)) for i in self.ids], dtype=torch.long)

    @staticmethod
    def broadcast_of(window_id: str) -> str:
        """`<scenario>/<run>/<file>#<pseudo>-<sender>#<k>` -> `"<scenario>/<run>|<pseudo>"`.

        Uses only what a receiver can observe (run and the pseudonym on the air), not the true sender. Every copy and
        every crop of one pseudonym's broadcasts in a run shares this key; ghost pseudonym 1 is one key per run.
        """
        run = "/".join(window_id.split("/", 2)[:2])
        pseudo = window_id.rsplit("#", 2)[1].split("-")[0]
        return f"{run}|{pseudo}"

    def broadcast_groups(self) -> torch.Tensor:
        """(N,) long: windows with the same broadcast key get the same number (NT-Xent never pairs them as negatives)."""
        codes: dict[str, int] = {}
        return torch.tensor([codes.setdefault(self.broadcast_of(i), len(codes)) for i in self.ids], dtype=torch.long)


class FeatureSpace:
    """Feature order and scaling; converts between scaled and raw units on tensors.

    By default every feature is a train z-score (mean / std from `metadata.json`, as stored on disk). `fit_robust`
    switches chosen features to a robust scale: (raw - median) / IQR from train real rows, then a soft tail that
    compresses |z| > `soft_clip` logarithmically (sign(z) * (c + log1p(|z| - c))). The order of values is kept, so a
    12.8 km ghost is still the most extreme value, but it no longer dominates the losses or squeezes the honest range.
    """

    CP, RP, CV, RV, CA = (0, 1), (2, 3), (4, 5), (6, 7), (8, 9)
    RANGE, BEARING, LOG_DTAU = 10, 11, 12

    def __init__(self, metadata: dict[str, Any]):
        self.features: list[str] = metadata["features"]
        if self.features != FEATURES:
            raise ValueError(f"unexpected feature order: {self.features}")
        norm = metadata["normalisation"]
        self.mean = torch.tensor(norm["mean"], dtype=torch.float32)  # centre per feature
        self.std = torch.tensor(norm["std"], dtype=torch.float32)  # scale per feature
        self.robust = torch.zeros(len(FEATURES), dtype=torch.bool)  # features with the soft tail
        self.soft_clip = 5.0
        self.report: dict[str, Any] = {name: {"method": "train z-score"} for name in self.features}

    def to(self, device: torch.device) -> "FeatureSpace":
        """Moves the scaling tensors to `device`."""
        self.mean, self.std, self.robust = self.mean.to(device), self.std.to(device), self.robust.to(device)
        return self

    def _soft(self, z: torch.Tensor) -> torch.Tensor:
        c = self.soft_clip
        tail = torch.sign(z) * (c + torch.log1p((z.abs() - c).clamp(min=0)))
        return torch.where(self.robust & (z.abs() > c), tail, z)

    def _unsoft(self, z: torch.Tensor) -> torch.Tensor:
        c = self.soft_clip
        tail = torch.sign(z) * (c + torch.expm1((z.abs() - c).clamp(min=0)))
        return torch.where(self.robust & (z.abs() > c), tail, z)

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        """Scaled -> raw units."""
        return self._unsoft(x) * self.std + self.mean

    def norm(self, raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Raw -> scaled units; padding rows set to exactly 0."""
        return self._soft((raw - self.mean) / self.std) * mask.unsqueeze(-1).to(raw.dtype)

    def fit_robust(self, data: "FeatureShards", names: tuple[str, ...], soft_clip: float, chunk: int = 20000) -> None:
        """Median / IQR of the raw real rows of `data` (train only) for `names`, plus the soft tail."""
        self.soft_clip = soft_clip
        cols = [self.features.index(n) for n in names]
        values: list[np.ndarray] = []
        for start in range(0, len(data), chunk):
            x, m = data.x[start : start + chunk], data.mask[start : start + chunk]
            values.append(self.raw(x)[m][:, cols].numpy())
        allv = np.concatenate(values)
        for j, name in zip(cols, names):
            q25, q50, q75 = np.percentile(allv[:, cols.index(j)], [25, 50, 75])
            iqr = max(float(q75 - q25), 1e-6)
            self.mean[j], self.std[j], self.robust[j] = float(q50), iqr, True
            self.report[name] = {
                "method": "median / IQR + soft tail",
                "median": float(q50),
                "iqr": iqr,
                "soft_clip": soft_clip,
            }
        for j, name in enumerate(self.features):
            if not self.robust[j]:
                self.report[name].update({"mean": float(self.mean[j]), "std": float(self.std[j])})

    @staticmethod
    def rescale(data: "FeatureShards", old: "FeatureSpace", new: "FeatureSpace", chunk: int = 20000) -> None:
        """Re-expresses `data.x` (stored train z-scores) in the `new` scaling, in place, chunk by chunk."""
        for start in range(0, len(data), chunk):
            x, m = data.x[start : start + chunk], data.mask[start : start + chunk]
            data.x[start : start + chunk] = new.norm(old.raw(x), m)

    @staticmethod
    def range_bearing(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Distance and wrapped direction from the receiver to the claimed position (as in the prep notebook)."""
        dx = raw[..., 0] - raw[..., 2]
        dy = raw[..., 1] - raw[..., 3]
        bearing = torch.atan2(dy, dx)
        bearing = torch.where(bearing <= -math.pi, bearing + 2 * math.pi, bearing)
        return torch.hypot(dx, dy), bearing


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes (provenance)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
