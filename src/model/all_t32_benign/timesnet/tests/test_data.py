"""Tests of the firewalls, the cached split loader and the feature-space save / restore contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.model.all_t32_benign.timesnet.data import (
    EvalFirewall,
    PretrainFirewall,
    load_split,
    restore_space,
    space_state,
)
from src.model.all_t32_benign.timesnet.base_data import FeatureShards, FeatureSpace, LabelFirewall
from src.pipeline.all_t32_benign.cache import CacheError, SplitCacheBuilder


@pytest.fixture
def cache(encoder_input: Path, tmp_path: Path) -> Path:
    root = tmp_path / "cache"
    builder = SplitCacheBuilder(encoder_input, root, 32, workers=1)
    for split in ("train", "val", "test", "test_gridsybil"):
        builder.build(split)
    return root


def test_pretrain_firewall_refuses_labels_and_test(encoder_input: Path, cache: Path) -> None:
    fw = PretrainFirewall()
    assert fw.check(encoder_input / "train" / "b32" / "part-00000.json")
    assert fw.check(encoder_input / "val" / "b32" / "part-00000.json")
    assert fw.check_cache(cache / "val" / "x.npy")
    for bad in (
        encoder_input / "train" / "b32" / "part-00000_info.json",
        encoder_input / "test" / "b32" / "part-00000.json",
        encoder_input / "test_gridsybil" / "b32" / "part-00000.json",
    ):
        with pytest.raises(PermissionError):
            fw.check(bad)
    for bad_cache in (cache / "test" / "x.npy", cache / "train" / "labels.npy"):
        with pytest.raises(PermissionError):
            fw.check_cache(bad_cache)
    for split in ("test", "test_gridsybil", "train_info"):
        with pytest.raises(PermissionError):
            load_split(cache, encoder_input, split, fw)


def test_eval_firewall_allows_every_split_never_labels(encoder_input: Path, cache: Path) -> None:
    fw = EvalFirewall()
    data = load_split(cache, encoder_input, "test_gridsybil", fw)
    assert len(data) == 30
    with pytest.raises(PermissionError):
        fw.check(encoder_input / "test_gridsybil" / "b32" / "part-00000_info.json")


def test_cache_loader_equals_json_reader(encoder_input: Path, cache: Path, tmp_path: Path) -> None:
    for split in ("train", "val"):
        cached = load_split(cache, encoder_input, split, PretrainFirewall(), workers=1)
        fallback = load_split(tmp_path / "no-cache", encoder_input, split, PretrainFirewall(), workers=1)
        reference = FeatureShards(encoder_input, split, _JsonGate(), None, 1, 32)
        for other in (fallback, reference):
            assert torch.equal(cached.x, other.x) and torch.equal(cached.mask, other.mask)
            assert cached.ids == other.ids
        assert cached.x.dtype == torch.float32 and cached.mask.dtype == torch.bool
        assert torch.equal(cached.broadcast_groups(), reference.broadcast_groups())
        assert torch.equal(cached.link_groups(), reference.link_groups())
        assert cached.split == split and cached.n_shards == reference.n_shards
        sub = cached.subset(torch.tensor([0, 2]), "x")
        assert len(sub) == 2 and sub.ids == [cached.ids[0], cached.ids[2]]
        assert FeatureShards.vehicle_of(cached.ids[0]).count(":") == 1


def test_max_shards_slices_the_cache(encoder_input: Path, cache: Path) -> None:
    one = load_split(cache, encoder_input, "train", PretrainFirewall(), max_shards=1)
    assert len(one) == 40 and one.n_shards == 1


def test_stale_cache_is_refused(encoder_input: Path, cache: Path) -> None:
    meta_path = encoder_input / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["note"] = "rebuilt"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(CacheError):
        load_split(cache, encoder_input, "train", PretrainFirewall())


def test_space_state_round_trip(encoder_input: Path, cache: Path) -> None:
    metadata = json.loads((encoder_input / "metadata.json").read_text())
    train = load_split(cache, encoder_input, "train", PretrainFirewall())
    space = FeatureSpace(metadata)
    space.fit_robust(train, ("claimed_pos_x", "claimed_pos_y", "range"), 5.0)
    state = space_state(space)
    assert set(state) == {"mean", "std", "robust", "soft_clip", "report"}
    assert len(state["mean"]) == len(state["std"]) == len(state["robust"]) == 13
    assert sum(state["robust"]) == 3
    restored = restore_space(metadata, json.loads(json.dumps(state)))
    assert torch.equal(restored.mean, space.mean) and torch.equal(restored.std, space.std)
    assert torch.equal(restored.robust, space.robust) and restored.soft_clip == space.soft_clip
    x = train.x[:8] * 7.0
    assert torch.equal(restored.raw(x), space.raw(x))
    with pytest.raises(ValueError):
        restore_space(metadata, {**state, "mean": state["mean"][:5]})


class _JsonGate(LabelFirewall):
    """The existing firewall widened to `val`, to read the JSON shards directly as a reference."""

    ALLOWED_SPLITS = ("train", "val")


def test_firewall_counts_opened_and_refused_label_paths(encoder_input: Path, cache: Path) -> None:
    fw = PretrainFirewall()
    load_split(cache, encoder_input, "train", fw)
    assert fw.label_files_opened == 0 and len(fw.passed) == 4  # manifest, x, mask, ids
    with pytest.raises(PermissionError):
        fw.check(encoder_input / "train" / "b32" / "part-00000_info.json")
    with pytest.raises(PermissionError):
        fw.check(encoder_input / "test" / "b32" / "part-00000.json")
    assert fw.label_paths_refused == 1 and fw.label_files_opened == 0
