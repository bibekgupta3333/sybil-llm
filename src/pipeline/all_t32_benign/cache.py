"""Feature-shard cache for the all/T32_benign encoder input: JSON shards -> numpy arrays (features only).

Supports the TimesNet pretraining package `src/model/all_t32_benign/timesnet/` (WBS S1.3 tooling on EC2): parsing
3.1 GB of JSON on every run start costs minutes, a memory-mapped `.npy` cache costs seconds.

Reads (read-only) `<data_dir>/<split>/b32/part-XXXXX.json` feature shards in sorted order and never opens a label file
(`part-XXXXX_info.json`). Writes, per split, `<cache_dir>/<split>/`:

    x.npy          (N, T, 13) float32, exactly the stored (normalised, 4-decimal) values
    mask.npy       (N, T) bool, True = real row
    ids.json       list of N window ids, same order as x
    manifest.json  source shard names + sha256 + window counts, metadata.json sha256, created time, N, seq_len

No label, attack code or `_info` content ever reaches the cache. A split whose manifest matches the source shards
(names and sha256) is skipped, so re-running is cheap and safe. `--verify` re-checks counts, shapes, source sha256s
and that a seeded random sample of windows equals the JSON.

Usage (from the repository root, training environment):

    .venv-train/bin/python -m src.pipeline.all_t32_benign.cache [--data-dir DIR] [--cache-dir DIR]
        [--splits train val ...] [--verify] [--workers N]
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import multiprocessing
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_LOG = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "src/data/encoder_input/all/T32_benign"
DEFAULT_CACHE_DIR = "src/data/cache/all/T32_benign"
N_FEATURES = 13
CACHE_FORMAT = "t32b-feature-cache-v1"


class CacheError(ValueError):
    """The cache is missing, stale, or does not match its source shards."""


def file_sha256(path: Path, chunk: int = 1 << 24) -> str:
    """SHA-256 of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class SplitFiles:
    """The four files of one cached split.

    Attributes:
        root: `<cache_dir>/<split>` folder.
    """

    root: Path

    NAMES = ("x.npy", "mask.npy", "ids.json", "manifest.json")

    @property
    def x(self) -> Path:
        return self.root / "x.npy"

    @property
    def mask(self) -> Path:
        return self.root / "mask.npy"

    @property
    def ids(self) -> Path:
        return self.root / "ids.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    def complete(self) -> bool:
        """True if all four files exist."""
        return all((self.root / name).is_file() for name in self.NAMES)

    def read_manifest(self) -> dict[str, Any]:
        """The parsed manifest.json."""
        return json.loads(self.manifest.read_text())


class FeatureShardFinder:
    """Finds the feature shards of a split; a label file (`_info`) can never be returned."""

    SHARD = re.compile(r"^part-\d{5}\.json$")

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def splits(self) -> list[str]:
        """Every split folder that holds feature shards, sorted."""
        return sorted(p.name for p in self.data_dir.iterdir() if p.is_dir() and self.shards(p.name))

    def shards(self, split: str) -> list[Path]:
        """Sorted feature shards `<split>/b<bucket>/part-XXXXX.json` (never `_info`)."""
        paths = sorted((self.data_dir / split).glob("b*/part-?????.json"))
        return [self.check(p) for p in paths]

    def check(self, path: Path) -> Path:
        """Returns `path` if it is a feature shard; raises PermissionError for anything else (labels included)."""
        if "_info" in path.name or not self.SHARD.match(path.name):
            raise PermissionError(f"not a feature shard (labels are never read): {path}")
        return path


@dataclasses.dataclass
class DecodedShard:
    """One feature shard decoded by a worker: x (W, T, 13) float32, mask (W, T) bool, ids, file sha256."""

    name: str
    sha256: str
    x: np.ndarray
    mask: np.ndarray
    ids: list[str]


class ShardDecoder:
    """Reads one feature shard (bytes hashed once, then parsed); runs in a fork pool worker."""

    def __init__(self, seq_len: int) -> None:
        self.seq_len = seq_len

    def __call__(self, path: str) -> DecodedShard:
        """Decodes `path`; raises CacheError on an unexpected shape."""
        raw = Path(path).read_bytes()
        data = json.loads(raw)
        windows = data["windows"]
        x = np.asarray([w["x"] for w in windows], dtype=np.float32).reshape(len(windows), -1, N_FEATURES)
        mask = np.asarray([w["mask"] for w in windows], dtype=bool).reshape(len(windows), -1)
        if x.shape[1] != self.seq_len or mask.shape[1] != self.seq_len:
            raise CacheError(f"{path}: windows have {x.shape[1]} rows, expected seq_len {self.seq_len}")
        return DecodedShard(Path(path).name, hashlib.sha256(raw).hexdigest(), x, mask, [w["id"] for w in windows])


class SplitCacheBuilder:
    """Builds (or skips, if current) the cache of one split; writes into `<split>.partial` and renames at the end."""

    def __init__(self, data_dir: Path, cache_dir: Path, seq_len: int, workers: int) -> None:
        self.data_dir, self.cache_dir, self.seq_len, self.workers = data_dir, cache_dir, seq_len, workers
        self.finder = FeatureShardFinder(data_dir)

    def source_signature(self, split: str) -> list[dict[str, str]]:
        """[{file, sha256}] of the split's feature shards, in cache order."""
        return [{"file": self._rel(p), "sha256": file_sha256(p)} for p in self.finder.shards(split)]

    def is_current(self, split: str, signature: list[dict[str, str]] | None = None) -> bool:
        """True if the cached split exists and its manifest lists exactly the current source shards."""
        files = SplitFiles(self.cache_dir / split)
        if not files.complete():
            return False
        manifest = files.read_manifest()
        signature = signature if signature is not None else self.source_signature(split)
        cached = [{"file": s["file"], "sha256": s["sha256"]} for s in manifest.get("shards", [])]
        return manifest.get("format") == CACHE_FORMAT and cached == signature

    def build(self, split: str) -> dict[str, Any]:
        """Writes the split's cache unless it is current; returns its manifest."""
        signature = self.source_signature(split)
        files = SplitFiles(self.cache_dir / split)
        if self.is_current(split, signature):
            _LOG.info("%s: cache is current (%d shards), skipped", split, len(signature))
            return files.read_manifest()
        paths = [str(self.finder.check(self.data_dir / s["file"])) for s in signature]
        if not paths:
            raise CacheError(f"no feature shards for split {split!r} under {self.data_dir}")
        with multiprocessing.get_context("fork").Pool(max(1, min(self.workers, len(paths)))) as pool:
            parts = list(pool.imap(ShardDecoder(self.seq_len), paths))
        for part, expected in zip(parts, signature):
            if part.sha256 != expected["sha256"]:
                raise CacheError(f"{expected['file']} changed while the cache was built")
        manifest = self._write(split, parts, signature)
        _LOG.info("%s: cached %d windows from %d shards", split, manifest["n_windows"], len(parts))
        return manifest

    def _write(self, split: str, parts: list[DecodedShard], signature: list[dict[str, str]]) -> dict[str, Any]:
        partial = self.cache_dir / f"{split}.partial"
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        tmp = SplitFiles(partial)
        total = sum(len(p.ids) for p in parts)
        x = np.lib.format.open_memmap(tmp.x, mode="w+", dtype=np.float32, shape=(total, self.seq_len, N_FEATURES))
        mask = np.lib.format.open_memmap(tmp.mask, mode="w+", dtype=bool, shape=(total, self.seq_len))
        ids: list[str] = []
        shards: list[dict[str, Any]] = []
        start = 0
        for i, part in enumerate(parts):
            end = start + len(part.ids)
            x[start:end], mask[start:end] = part.x, part.mask
            ids.extend(part.ids)
            shards.append({**signature[i], "windows": len(part.ids), "start": start})
            parts[i] = None  # type: ignore[call-overload]  # free each shard once copied
            start = end
        x.flush()
        mask.flush()
        del x, mask
        tmp.ids.write_text(json.dumps(ids))
        metadata = self.data_dir / "metadata.json"
        manifest = {
            "format": CACHE_FORMAT,
            "split": split,
            "source_dir": str(self.data_dir),
            "metadata_sha256": file_sha256(metadata) if metadata.is_file() else None,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "n_windows": total,
            "seq_len": self.seq_len,
            "n_features": N_FEATURES,
            "dtype": {"x": "float32", "mask": "bool"},
            "labels": "none (label files are never opened)",
            "shards": shards,
        }
        tmp.manifest.write_text(json.dumps(manifest, indent=2))
        final = self.cache_dir / split
        if final.exists():
            shutil.rmtree(final)
        partial.rename(final)
        return manifest

    def _rel(self, path: Path) -> str:
        return str(path.relative_to(self.data_dir))


class CacheVerifier:
    """Re-checks a cached split against its source shards and the encoder input metadata."""

    def __init__(self, data_dir: Path, cache_dir: Path, seq_len: int, samples: int = 64, seed: int = 0) -> None:
        self.data_dir, self.cache_dir, self.seq_len = data_dir, cache_dir, seq_len
        self.samples, self.seed = samples, seed
        self.finder = FeatureShardFinder(data_dir)

    def verify(self, split: str) -> dict[str, Any]:
        """Returns a report of every check; raises CacheError on the first failure."""
        files = SplitFiles(self.cache_dir / split)
        if not files.complete():
            raise CacheError(f"{split}: cache incomplete under {files.root}")
        manifest = files.read_manifest()
        builder = SplitCacheBuilder(self.data_dir, self.cache_dir, self.seq_len, 1)
        signature = builder.source_signature(split)
        if [{"file": s["file"], "sha256": s["sha256"]} for s in manifest["shards"]] != signature:
            raise CacheError(f"{split}: source shards differ from the manifest (rebuild the cache)")
        x = np.load(files.x, mmap_mode="r")
        mask = np.load(files.mask, mmap_mode="r")
        ids = json.loads(files.ids.read_text())
        n = manifest["n_windows"]
        self._expect(x.shape == (n, self.seq_len, N_FEATURES), f"{split}: x shape {x.shape}")
        self._expect(x.dtype == np.float32 and mask.dtype == np.bool_, f"{split}: dtypes {x.dtype} / {mask.dtype}")
        self._expect(mask.shape == (n, self.seq_len), f"{split}: mask shape {mask.shape}")
        self._expect(len(ids) == n and len(set(ids)) == n, f"{split}: {len(ids)} ids for {n} windows (or duplicates)")
        self._expect(sum(s["windows"] for s in manifest["shards"]) == n, f"{split}: shard counts do not sum to {n}")
        report: dict[str, Any] = {"split": split, "n_windows": n, "shards": len(signature)}
        report.update(self._content_checks(x, mask))
        report["metadata_windows"] = self._metadata_count(split)
        if report["metadata_windows"] is not None:
            self._expect(report["metadata_windows"] == n, f"{split}: metadata says {report['metadata_windows']}")
        report["sampled_windows_equal"] = self._sample_check(split, manifest, x, mask, ids)
        return report

    def _content_checks(self, x: np.ndarray, mask: np.ndarray, chunk: int = 50_000) -> dict[str, Any]:
        """Finite values, zero padding rows, at least one real row per window."""
        nonfinite = padding_nonzero = empty = 0
        for start in range(0, len(x), chunk):
            xb, mb = np.asarray(x[start : start + chunk]), np.asarray(mask[start : start + chunk])
            nonfinite += int((~np.isfinite(xb)).sum())
            padding_nonzero += int((xb[~mb] != 0).sum())
            empty += int((mb.sum(1) == 0).sum())
        self._expect(nonfinite == 0, f"{nonfinite} non-finite values")
        self._expect(padding_nonzero == 0, f"{padding_nonzero} non-zero padding values")
        self._expect(empty == 0, f"{empty} windows without a real row")
        return {"nonfinite": nonfinite, "padding_nonzero": padding_nonzero, "empty_windows": empty}

    def _sample_check(
        self, split: str, manifest: dict[str, Any], x: np.ndarray, mask: np.ndarray, ids: list[str]
    ) -> int:
        """Decodes up to three seeded random shards again and compares a random sample of their windows exactly."""
        rng = np.random.default_rng(self.seed)
        shards = manifest["shards"]
        picked = rng.choice(len(shards), size=min(3, len(shards)), replace=False)
        decoder = ShardDecoder(self.seq_len)
        checked = 0
        for si in sorted(int(i) for i in picked):
            shard = shards[si]
            part = decoder(str(self.finder.check(self.data_dir / shard["file"])))
            rows = rng.choice(len(part.ids), size=min(self.samples, len(part.ids)), replace=False)
            for r in rows.tolist():
                g = shard["start"] + r
                same = (
                    np.array_equal(x[g], part.x[r]) and np.array_equal(mask[g], part.mask[r]) and ids[g] == part.ids[r]
                )
                self._expect(same, f"{split}: window {g} ({part.ids[r]}) differs from {shard['file']}")
                checked += 1
        return checked

    def _metadata_count(self, split: str) -> int | None:
        path = self.data_dir / "metadata.json"
        if not path.is_file():
            return None
        counts = json.loads(path.read_text()).get("counts_by_split", {})
        entry = counts.get(split)
        return int(entry["windows"]) if isinstance(entry, dict) and "windows" in entry else None

    @staticmethod
    def _expect(ok: bool, message: str) -> None:
        if not ok:
            raise CacheError(message)


class CacheCommand:
    """Command line: build (idempotent) or verify the cache of the chosen splits."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.data_dir = Path(args.data_dir).expanduser()
        self.cache_dir = Path(args.cache_dir).expanduser()
        protected = Path("data").resolve()
        if self.cache_dir.resolve() == protected or protected in self.cache_dir.resolve().parents:
            raise SystemExit(f"refusing to write under data/ (RULE 2): {self.cache_dir}")

    def run(self) -> list[dict[str, Any]]:
        """Builds or verifies each split; returns the manifests or verification reports."""
        finder = FeatureShardFinder(self.data_dir)
        splits = self.args.splits or finder.splits()
        out = []
        for split in splits:
            if self.args.verify:
                report = CacheVerifier(self.data_dir, self.cache_dir, self.args.seq_len).verify(split)
                _LOG.info("verified %s", json.dumps(report))
                out.append(report)
            else:
                builder = SplitCacheBuilder(self.data_dir, self.cache_dir, self.args.seq_len, self.args.workers)
                out.append(builder.build(split))
        return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser(description="JSON feature shards -> .npy cache (features only, never labels)")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help=f"encoder input folder (default {DEFAULT_DATA_DIR})")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help=f"cache folder (default {DEFAULT_CACHE_DIR})")
    p.add_argument("--splits", nargs="*", default=None, help="splits to cache (default: every split folder)")
    p.add_argument("--verify", action="store_true", help="re-check counts, shapes, sha256s and sampled windows")
    p.add_argument("--workers", type=int, default=8, help="parallel shard decoders (default 8)")
    p.add_argument("--seq-len", type=int, default=32, help="window length T (default 32)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    CacheCommand(parse_args(argv)).run()


if __name__ == "__main__":
    main()
