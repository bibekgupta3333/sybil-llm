"""Label-free data helpers: scaling round-trip, window-id parsing, the vehicle-grouped check set."""

from __future__ import annotations

import torch

from src.model.benign_gridsybil.timesnet.data import FeatureShards, FeatureSpace

from .conftest import synthetic_metadata, synthetic_windows

ROBUST = ("claimed_pos_x", "claimed_pos_y", "range")


def make_shards(ids: list[str], seed: int = 0) -> FeatureShards:
    """A small in-memory FeatureShards built like `subset` does (no files)."""
    x, mask = synthetic_windows([64 - (i % 60) for i in range(len(ids))], seed=seed)
    shards = object.__new__(FeatureShards)
    shards.split, shards.n_shards = "train", 1
    shards.x, shards.mask, shards.ids = x, mask, list(ids)
    return shards


def window_id(scenario: str, run: str, pseudo: str, sender: str, k: int) -> str:
    return f"{scenario}/{run}/traceJSON-900-901-A0-25200-7.json#{pseudo}-{sender}#{k}"


def test_robust_round_trip_beyond_soft_clip() -> None:
    data = make_shards([window_id("Benign_0709", "run0", "1", str(i), 0) for i in range(32)])
    data.x[0, :10, [0, 1, 10]] = 30.0  # far beyond the soft clip, like the GridSybil ghosts
    data.x[1, :10, [0, 1, 10]] = -12.0
    old = FeatureSpace(synthetic_metadata())
    new = FeatureSpace(synthetic_metadata())
    new.fit_robust(data, ROBUST, soft_clip=5.0)
    raw = old.raw(data.x)
    scaled = new.norm(raw, data.mask)
    assert float((new.norm(new.raw(scaled), data.mask) - scaled).abs().max()) <= 1e-5
    back = new.raw(scaled)[data.mask]
    in_scale_units = (back - raw[data.mask]).abs() / new.std  # float32 error measured against each feature's scale
    assert float(in_scale_units.max()) <= 1e-5
    cols = [0, 1, 10]
    assert float(scaled[..., cols].abs().max()) < 5.0 + 5.0  # the tail is compressed ...
    assert float(scaled[0, 0, 0]) > float(scaled[data.mask][:, 0][2:].max()) - 1e-6  # ... order kept
    assert torch.all(scaled[~data.mask] == 0)


def test_vehicle_and_broadcast_parsing() -> None:
    wid = "GridSybil_0709/VeReMi_1_2/traceJSON-10545-10543-A0-25200-7.json#1-10551#3"
    assert FeatureShards.vehicle_of(wid) == "0709:10551"
    assert FeatureShards.broadcast_of(wid) == "GridSybil_0709/VeReMi_1_2|1"
    other = "Benign_1416/runB/traceJSON-7-8-A0-50400-1.json#20077-20071#0"
    assert FeatureShards.vehicle_of(other) == "1416:20071"
    assert FeatureShards.broadcast_of(other) == "Benign_1416/runB|20077"


def test_broadcast_groups_share_numbers() -> None:
    ids = [
        window_id("GridSybil_0709", "r1", "1", "10", 0),
        window_id("GridSybil_0709", "r1", "1", "11", 0),  # same ghost pseudonym, other true sender
        window_id("GridSybil_0709", "r2", "1", "10", 0),  # other run
        window_id("GridSybil_0709", "r1", "1", "10", 1),
    ]
    groups = make_shards(ids).broadcast_groups()
    assert groups.tolist() == [0, 0, 1, 0]


def test_hold_out_is_vehicle_disjoint_and_deterministic() -> None:
    ids = [
        window_id(scenario, "run", str(1000 + v), str(v), k)
        for scenario in ("Benign_0709", "Benign_1416")
        for v in range(20)
        for k in range(3)
    ]
    data = make_shards(ids)
    rest, check = data.hold_out(0.25, seed=0)
    rest_vehicles = {FeatureShards.vehicle_of(i) for i in rest.ids}
    check_vehicles = {FeatureShards.vehicle_of(i) for i in check.ids}
    assert rest_vehicles.isdisjoint(check_vehicles)
    assert len(check_vehicles) == 10
    assert sorted(rest.ids + check.ids) == sorted(ids)
    assert check.split == "pretrain_check" and rest.split == "train"
    assert len(rest) == len(rest.ids) and len(check) == len(check.ids)
    rest2, check2 = data.hold_out(0.25, seed=0)
    assert rest2.ids == rest.ids and check2.ids == check.ids
    assert torch.equal(check2.x, check.x)
    _, check_other = data.hold_out(0.25, seed=1)
    assert check_other.ids != check.ids
