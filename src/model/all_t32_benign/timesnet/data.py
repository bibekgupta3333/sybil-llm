"""Label-free data access for the all/T32_benign TimesNet run (WBS S1.3 data entry, S1.4 evaluation entry).

Firewalls gate every file opened: `PretrainFirewall` allows the feature arrays of the `train` and `val` splits only,
`EvalFirewall` allows the feature arrays of every split; both refuse label files (`_info`) always. `load_split`
returns a `FeatureShards`-compatible object (the class of `base_data.py`, this package's copy of the original)
from the numpy cache written by `src.pipeline.all_t32_benign.cache` (memory-mapped, then copied), or from the JSON
shards when the cache is missing. `space_state` / `restore_space` save and rebuild the exact feature scaling of a
training run (robust median / IQR + soft tail, D13) so evaluation uses the same values.

Public API (stable; evaluation code depends on these exact names):

    PretrainFirewall()                      # train / val feature arrays only
    EvalFirewall()                          # every split's feature arrays, never labels
    load_split(cache_dir: str | Path, data_dir: str | Path, split: str, firewall: FeatureFirewall,
               max_shards: int | None = None, workers: int = 8, seq_len: int = 32) -> FeatureShards
        # attributes: x (N, 32, 13) torch.float32, mask (N, 32) torch.bool, ids list[str], split, n_shards, reader;
        # methods: link_groups(), broadcast_groups(), subset(rows, split), vehicle_of(id), __len__
    space_state(space: FeatureSpace) -> dict
        # {"mean": list[13], "std": list[13], "robust": list[bool] (13), "soft_clip": float, "report": dict}
    restore_space(metadata: dict, state: dict) -> FeatureSpace
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.model.all_t32_benign.timesnet.base_data import FeatureShards, FeatureSpace, LabelFirewall
from src.pipeline.all_t32_benign.cache import CacheError, SplitFiles, file_sha256

_LOG = logging.getLogger(__name__)


class FeatureFirewall:
    """Allows feature arrays of chosen splits (JSON shards or cache files); never a label file.

    Duck-type compatible with `LabelFirewall.check`, so it can also guard the JSON fallback reader.

    Args:
        allowed_splits: split names that may be opened; None allows every split.
    """

    SHARD = LabelFirewall.SHARD
    BUCKET = LabelFirewall.BUCKET
    CACHE_FILES = frozenset(SplitFiles.NAMES)

    LABEL_MARKERS = ("_info", "label")

    def __init__(self, allowed_splits: tuple[str, ...] | None) -> None:
        self.allowed_splits = allowed_splits
        self.passed: list[str] = []  # every path this gate let through (provenance)
        self.label_paths_refused = 0  # label paths that were asked for and refused

    @property
    def label_files_opened(self) -> int:
        """Paths let through that look like label files (0 unless the gate is broken); recorded in env.json."""
        return sum(any(m in Path(p).name for m in self.LABEL_MARKERS) for p in self.passed)

    def _refuse(self, name: str, message: str) -> None:
        if any(m in name for m in self.LABEL_MARKERS):
            self.label_paths_refused += 1
        raise PermissionError(f"{type(self).__name__}: {message}")

    def check_split(self, split: str) -> str:
        """Returns `split` if it may be opened; raises PermissionError otherwise."""
        if "_info" in split or (self.allowed_splits is not None and split not in self.allowed_splits):
            self._refuse(split, f"split {split!r} is not allowed here")
        return split

    def check(self, path: Path) -> Path:
        """Returns `path` if it is a feature shard `<split>/b<bucket>/part-XXXXX.json` of an allowed split."""
        path = Path(path)
        if "_info" in path.name or not self.SHARD.match(path.name):
            self._refuse(path.name, f"{path.name} is not a feature shard (labels never pass)")
        folder = path.parent
        self.check_split(folder.parent.name if self.BUCKET.match(folder.name) else folder.name)
        self.passed.append(str(path))
        return path

    def check_cache(self, path: Path) -> Path:
        """Returns `path` if it is a cache file `<cache_dir>/<split>/{x.npy,mask.npy,ids.json,manifest.json}`."""
        path = Path(path)
        if path.name not in self.CACHE_FILES:
            self._refuse(path.name, f"{path.name} is not a feature-cache file")
        self.check_split(path.parent.name)
        self.passed.append(str(path))
        return path


class PretrainFirewall(FeatureFirewall):
    """Pretraining gate: the feature arrays of `train` and `val` only (test splits and labels refused)."""

    ALLOWED = ("train", "val")

    def __init__(self) -> None:
        super().__init__(self.ALLOWED)


class EvalFirewall(FeatureFirewall):
    """Evaluation gate: the feature arrays of every split; label files still refused (labels are read elsewhere)."""

    def __init__(self) -> None:
        super().__init__(None)


class CachedShards(FeatureShards):
    """A `FeatureShards` built from arrays already in memory (the cache) instead of parsing JSON shards.

    Args:
        x: (N, T, 13) float32 tensor (stored normalised values).
        mask: (N, T) bool tensor, True = real row.
        ids: N window ids, same order.
        split: split name.
        n_shards: source shards covered.
        reader: description of the source (provenance).
    """

    def __init__(
        self, x: torch.Tensor, mask: torch.Tensor, ids: list[str], split: str, n_shards: int, reader: str
    ) -> None:  # the parent's __init__ parses JSON; this one only stores the arrays
        if x.shape[0] != mask.shape[0] or x.shape[0] != len(ids):
            raise CacheError(f"{split}: {x.shape[0]} windows, {mask.shape[0]} masks, {len(ids)} ids")
        self.x, self.mask, self.ids = x, mask, ids
        self.split, self.n_shards, self.reader = split, n_shards, reader


class SplitLoader:
    """Loads one split from the cache (preferred) or the JSON shards, through a firewall.

    Args:
        cache_dir: root of the numpy cache.
        data_dir: encoder input folder (metadata.json + JSON shards).
        firewall: gate for every file opened.
        workers: JSON fallback parsers.
        seq_len: window length T.
    """

    def __init__(
        self, cache_dir: Path, data_dir: Path, firewall: FeatureFirewall, workers: int = 8, seq_len: int = 32
    ) -> None:
        self.cache_dir, self.data_dir, self.firewall = Path(cache_dir), Path(data_dir), firewall
        self.workers, self.seq_len = workers, seq_len

    def load(self, split: str, max_shards: int | None = None) -> FeatureShards:
        """The windows of `split` (first `max_shards` source shards if given).

        Raises:
            PermissionError: the firewall refuses the split.
            CacheError: the cache exists but was built from another metadata.json or has the wrong shape.
        """
        self.firewall.check_split(split)
        files = SplitFiles(self.cache_dir / split)
        if files.complete():
            return self._from_cache(split, files, max_shards)
        _LOG.warning("no cache for %s under %s; parsing the JSON shards (slow)", split, self.cache_dir)
        return FeatureShards(self.data_dir, split, self.firewall, max_shards, self.workers, self.seq_len)

    def _from_cache(self, split: str, files: SplitFiles, max_shards: int | None) -> FeatureShards:
        manifest = json.loads(self.firewall.check_cache(files.manifest).read_text())
        metadata = self.data_dir / "metadata.json"
        if metadata.is_file() and manifest.get("metadata_sha256") not in (None, file_sha256(metadata)):
            raise CacheError(f"{files.root} was built from another {metadata}; rebuild the cache")
        if manifest["seq_len"] != self.seq_len:
            raise CacheError(f"{files.root}: seq_len {manifest['seq_len']}, expected {self.seq_len}")
        shards = manifest["shards"] if max_shards is None else manifest["shards"][:max_shards]
        n = sum(int(s["windows"]) for s in shards)
        x_map = np.load(self.firewall.check_cache(files.x), mmap_mode="r")
        mask_map = np.load(self.firewall.check_cache(files.mask), mmap_mode="r")
        if x_map.shape[0] != manifest["n_windows"] or x_map.shape[1:] != (self.seq_len, 13):
            raise CacheError(f"{files.x}: shape {x_map.shape} does not match the manifest")
        x = torch.from_numpy(np.array(x_map[:n], dtype=np.float32))
        mask = torch.from_numpy(np.array(mask_map[:n], dtype=bool))
        del x_map, mask_map
        ids = json.loads(self.firewall.check_cache(files.ids).read_text())[:n]
        return CachedShards(x, mask, ids, split, len(shards), f"npy cache ({files.root})")


def load_split(
    cache_dir: str | Path,
    data_dir: str | Path,
    split: str,
    firewall: FeatureFirewall,
    max_shards: int | None = None,
    workers: int = 8,
    seq_len: int = 32,
) -> FeatureShards:
    """Loads one split through `firewall` (cache first, JSON fallback); see `SplitLoader.load`."""
    return SplitLoader(Path(cache_dir), Path(data_dir), firewall, workers, seq_len).load(split, max_shards)


def space_state(space: FeatureSpace) -> dict[str, Any]:
    """Serialisable scaling of a run: {"mean", "std", "robust", "soft_clip", "report"} (13 values per list)."""
    return {
        "mean": [float(v) for v in space.mean.detach().cpu().tolist()],
        "std": [float(v) for v in space.std.detach().cpu().tolist()],
        "robust": [bool(v) for v in space.robust.detach().cpu().tolist()],
        "soft_clip": float(space.soft_clip),
        "report": json.loads(json.dumps(space.report)),
    }


def restore_space(metadata: dict[str, Any], state: dict[str, Any]) -> FeatureSpace:
    """The FeatureSpace of a run, rebuilt from its metadata.json and `space_state` (same values as at training).

    Raises:
        ValueError: if the state does not hold 13 values per list.
    """
    space = FeatureSpace(metadata)
    n = len(space.features)
    if not all(len(state[k]) == n for k in ("mean", "std", "robust")):
        raise ValueError(f"space state must hold {n} values per list")
    space.mean = torch.tensor(state["mean"], dtype=torch.float32)
    space.std = torch.tensor(state["std"], dtype=torch.float32)
    space.robust = torch.tensor(state["robust"], dtype=torch.bool)
    space.soft_clip = float(state["soft_clip"])
    space.report = json.loads(json.dumps(state.get("report", {})))
    return space
