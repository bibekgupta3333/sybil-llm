"""Tests of the JSON -> numpy feature cache (src/pipeline/all_t32_benign/cache.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.pipeline.all_t32_benign.cache import (
    CacheCommand,
    CacheError,
    CacheVerifier,
    FeatureShardFinder,
    SplitCacheBuilder,
    SplitFiles,
    parse_args,
)


def _json_windows(root: Path, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs, masks, ids = [], [], []
    for path in sorted((root / split).glob("b32/part-?????.json")):
        for w in json.loads(path.read_text())["windows"]:
            xs.append(w["x"])
            masks.append(w["mask"])
            ids.append(w["id"])
    return np.asarray(xs, dtype=np.float32), np.asarray(masks, dtype=bool), ids


def test_round_trip_equals_json(encoder_input: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    manifest = SplitCacheBuilder(encoder_input, cache, 32, workers=1).build("train")
    files = SplitFiles(cache / "train")
    x, mask, ids = np.load(files.x), np.load(files.mask), json.loads(files.ids.read_text())
    jx, jmask, jids = _json_windows(encoder_input, "train")
    assert x.dtype == np.float32 and mask.dtype == np.bool_
    assert x.shape == (200, 32, 13) and mask.shape == (200, 32)
    np.testing.assert_array_equal(x, jx)
    np.testing.assert_array_equal(mask, jmask)
    assert ids == jids
    assert manifest["n_windows"] == 200 and manifest["seq_len"] == 32
    assert [s["file"] for s in manifest["shards"]] == [f"train/b32/part-{k:05d}.json" for k in range(5)]
    assert all(len(s["sha256"]) == 64 for s in manifest["shards"])
    assert manifest["metadata_sha256"] is not None
    assert "label" not in json.dumps(manifest["shards"]).lower()


def test_build_is_idempotent_and_rebuilds_on_change(encoder_input: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    builder = SplitCacheBuilder(encoder_input, cache, 32, workers=1)
    first = builder.build("val")
    assert builder.is_current("val")
    again = builder.build("val")
    assert again["created_utc"] == first["created_utc"]  # skipped, not rewritten
    shard = encoder_input / "val" / "b32" / "part-00000.json"
    data = json.loads(shard.read_text())
    data["windows"][0]["x"][0][0] = 9.5
    shard.write_text(json.dumps(data))
    assert not builder.is_current("val")
    builder.build("val")
    assert float(np.load(SplitFiles(cache / "val").x)[0, 0, 0]) == 9.5
    assert not (cache / "val.partial").exists()


def test_verify_passes_and_detects_corruption(encoder_input: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    SplitCacheBuilder(encoder_input, cache, 32, workers=1).build("test")
    report = CacheVerifier(encoder_input, cache, 32, samples=8).verify("test")
    assert report["n_windows"] == 40 and report["metadata_windows"] == 40
    assert report["sampled_windows_equal"] > 0 and report["padding_nonzero"] == 0
    x = np.load(SplitFiles(cache / "test").x)
    x[:, 0, 0] += 1.0
    np.save(SplitFiles(cache / "test").x, x)
    with pytest.raises(CacheError):
        CacheVerifier(encoder_input, cache, 32, samples=40).verify("test")


def test_label_files_never_returned(encoder_input: Path) -> None:
    finder = FeatureShardFinder(encoder_input)
    assert all("_info" not in p.name for p in finder.shards("train"))
    with pytest.raises(PermissionError):
        finder.check(encoder_input / "train" / "b32" / "part-00000_info.json")
    assert finder.splits() == ["test", "test_gridsybil", "train", "val"]


def test_command_builds_every_split_and_refuses_data_dir(encoder_input: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    args = parse_args(["--data-dir", str(encoder_input), "--cache-dir", str(cache), "--workers", "1"])
    manifests = CacheCommand(args).run()
    assert sorted(m["split"] for m in manifests) == ["test", "test_gridsybil", "train", "val"]
    verify = parse_args(["--data-dir", str(encoder_input), "--cache-dir", str(cache), "--verify"])
    assert all(r["sampled_windows_equal"] > 0 for r in CacheCommand(verify).run())
    with pytest.raises(SystemExit):
        CacheCommand(parse_args(["--cache-dir", "data/cache"]))
