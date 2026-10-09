"""Unit tests for the simulation sample exporter (synthetic data, no access to data/)."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import export_simulation_sample as exp  # noqa: E402

_FEATURES = [
    "pos_x",
    "pos_y",
    "spd_x",
    "spd_y",
    "acl_x",
    "acl_y",
    "hed_x",
    "hed_y",
    "dt",
    "dpos_x",
    "dpos_y",
    "dspd_x",
    "dspd_y",
]
_DT = _FEATURES.index("dt")


def _make_store(tmp_path: pathlib.Path, sequences: dict[str, np.ndarray], t0: float = 25200.0) -> exp.WindowStore:
    """Build a prepared_data dir from full per-identity sequences using the notebook's slicing."""
    windows, rows = [], []
    for uid, seq in sequences.items():
        starts = range(0, len(seq) - exp._WINDOW + 1, exp._STRIDE)
        times = t0 + np.cumsum(seq[:, _DT].astype(np.float64)) - seq[0, _DT]
        for s in starts:
            windows.append(seq[s : s + exp._WINDOW])
            rows.append(
                {
                    "family": "GridSybil",
                    "group": "0709",
                    "subfolder": "run",
                    "senderPseudo": 7,
                    "attack_code": 16,
                    "attack_label": "GridSybil",
                    "is_attacker": True,
                    "window_start_idx": s,
                    "start_time": times[s],
                    "end_time": times[s + exp._WINDOW - 1],
                    "sender_uid": uid,
                }
            )
    np.save(tmp_path / "X_windows.npy", np.stack(windows).astype(np.float32))
    pd.DataFrame(rows).to_parquet(tmp_path / "window_metadata.parquet")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "feature_cols": _FEATURES,
                "norm_mean": [0.0] * 13,
                "norm_std": [1.0] * 13,
                "label_to_int": {"Benign": 0, "GridSybil": 4},
            }
        )
    )
    return exp.WindowStore(tmp_path)


def _sequence(n: int, seed: int = 0, dt: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    seq = rng.normal(size=(n, 13)).astype(np.float32)
    seq[:, _DT] = dt
    seq[0, _DT] = 0.0  # groupby().diff().fillna(0) at the identity's first row
    return seq


def test_stitch_reproduces_windows(tmp_path: pathlib.Path) -> None:
    seq = _sequence(57)  # 4 windows at starts 0..30, 7-row tail unrecoverable
    store = _make_store(tmp_path, {"a": seq})
    out = exp.IdentityStitcher(store).stitch("a")
    assert out.n_windows == 4
    assert out.steps.shape == (50, 13)
    assert np.array_equal(out.steps, seq[:50])
    assert np.array_equal(out.window_rows, np.arange(4))


def test_time_anchor_matches_end_time_with_long_gap(tmp_path: pathlib.Path) -> None:
    seq = _sequence(40)
    seq[25, _DT] = 101.0  # a real 101 s gap inside the sequence
    store = _make_store(tmp_path, {"a": seq})
    out = exp.IdentityStitcher(store).stitch("a")
    meta = store.meta.sort_values("window_start_idx")
    for k, (s, e) in enumerate(zip(meta["start_time"], meta["end_time"])):
        assert out.times[k * 10] == pytest.approx(s, abs=1e-6)
        assert out.times[k * 10 + 19] == pytest.approx(e, abs=1e-6)
    assert np.all(np.diff(out.times) >= 0)


def test_stitch_rejects_gap(tmp_path: pathlib.Path) -> None:
    store = _make_store(tmp_path, {"a": _sequence(60)})
    store.meta = store.meta.drop(index=1)  # remove the window starting at 10
    store._by_uid = store.meta.groupby("sender_uid", sort=False).indices
    with pytest.raises(exp.StitchError, match="gaps"):
        exp.IdentityStitcher(store).stitch("a")


def test_stitch_rejects_overlap_mismatch(tmp_path: pathlib.Path) -> None:
    store = _make_store(tmp_path, {"a": _sequence(60)})
    x = np.array(store.x)
    x[1, 0, 0] += 1.0  # corrupt the overlap region of window 1
    store.x = x
    with pytest.raises(exp.StitchError, match="overlap"):
        exp.IdentityStitcher(store).stitch("a")


def test_max_windows_truncates_for_defect_demo(tmp_path: pathlib.Path) -> None:
    store = _make_store(tmp_path, {"a": _sequence(200)})
    out = exp.IdentityStitcher(store, max_windows=3).stitch("a")
    assert out.n_windows == 3 and out.steps.shape[0] == 40


def _broadcast_row(
    send_time: float, x: float, y: float, message_id: int, sender: int = 15, pseudo: int = 20155
) -> dict:
    return {
        "type": 3,
        "sendTime": send_time,
        "sender": sender,
        "senderPseudo": pseudo,
        "messageID": message_id,
        "pos": [x, y, 0.0],
        "spd": [1.0, 2.0, 0.0],
        "acl": [0.1, 0.2, 0.0],
        "hed": [1.0, 0.0, 0.0],
    }


def test_rows_to_forged_trajectory_sorts_and_derives_deltas() -> None:
    rows = [_broadcast_row(12.0, 20.0, 0.0, message_id=2), _broadcast_row(10.0, 10.0, 0.0, message_id=1)]
    steps, times = exp._rows_to_forged_trajectory(rows, _FEATURES)
    assert list(times) == [10.0, 12.0]  # sorted by sendTime, not input order
    assert steps[0, _FEATURES.index("dt")] == 0.0  # first row: no previous broadcast
    assert steps[1, _FEATURES.index("dt")] == pytest.approx(2.0)
    assert steps[1, _FEATURES.index("dpos_x")] == pytest.approx(10.0)
    assert steps[0, _FEATURES.index("pos_x")] == 10.0
    assert steps[1, _FEATURES.index("pos_x")] == 20.0


def test_pick_fake_pseudonym_ignores_real_pseudonyms_and_needs_enough_rows() -> None:
    scanned = {
        15: {
            10155: {i: _broadcast_row(float(i), float(i), 0.0, i, pseudo=10155) for i in range(30)},  # the real one
            1: {i: _broadcast_row(float(i), float(i), 100.0, i, pseudo=1) for i in range(5)},  # too short
            20155: {i: _broadcast_row(float(i), float(i), 200.0, i, pseudo=20155) for i in range(25)},  # eligible fake
        },
    }
    found = exp.BroadcastForger.pick_fake_pseudonym(scanned, sender=15, real_pseudonyms={10155}, min_steps=20)
    assert found is not None
    pseudo, rows = found
    assert pseudo == 20155
    assert len(rows) == 25


def test_pick_fake_pseudonym_returns_none_when_nothing_is_eligible() -> None:
    scanned = {15: {10155: {0: _broadcast_row(0.0, 0.0, 0.0, 0, pseudo=10155)}}}
    assert exp.BroadcastForger.pick_fake_pseudonym(scanned, sender=15, real_pseudonyms={10155}, min_steps=20) is None
    assert exp.BroadcastForger.pick_fake_pseudonym({}, sender=15, real_pseudonyms=set(), min_steps=20) is None


def test_attack_trace_strategy_per_family() -> None:
    assert exp.attack_trace_strategy("GridSybil") == "fake_pseudonym"
    for family in ("DataReplaySybil", "DoSRandomSybil", "DoSDisruptiveSybil"):
        assert exp.attack_trace_strategy(family) == "all_pseudonyms"


def test_pick_all_pseudonyms_merges_short_lived_pseudonyms_in_time_order() -> None:
    # 12 pseudonyms x 2 messages each: no single pseudonym reaches 20, together they do.
    scanned = {
        7: {
            p: {
                10 * p + k: _broadcast_row(float(12 - p) + 0.5 * k, 0.0, 0.0, 10 * p + k, sender=7, pseudo=p)
                for k in range(2)
            }
            for p in range(12)
        }
    }
    assert exp.BroadcastForger.pick_fake_pseudonym(scanned, 7, real_pseudonyms=set(), min_steps=20) is None
    pseudo, rows, n = exp.BroadcastForger.pick_all_pseudonyms(scanned, 7, min_steps=20)
    assert n == 12 and len(rows) == 24
    times = [r["sendTime"] for r in rows]
    assert times == sorted(times)
    assert exp.BroadcastForger.pick_all_pseudonyms(scanned, 7, min_steps=25) is None


def test_pick_attack_trace_uses_the_family_strategy() -> None:
    scanned = {
        7: {
            p: {
                10 * p + k: _broadcast_row(float(p) + 0.1 * k, 0.0, 0.0, 10 * p + k, sender=7, pseudo=p)
                for k in range(2)
            }
            for p in range(12)
        }
    }
    got = exp.BroadcastForger.pick_attack_trace(scanned, 7, "DoSRandomSybil", real_pseudonyms=set(range(12)))
    assert got is not None and got[0] == "all_pseudonyms" and got[3] == 12
    # GridSybil only accepts one fake pseudonym with enough messages -- none here.
    assert exp.BroadcastForger.pick_attack_trace(scanned, 7, "GridSybil", real_pseudonyms=set()) is None


def _dummy_exporter(
    tmp_path: pathlib.Path, forged_pairs_per_cell: int, pseudonym_table: dict[str, int]
) -> exp.Exporter:
    """An Exporter with just enough state for `_find_forged_demos` -- no real prepared_data needed."""
    exporter = exp.Exporter.__new__(exp.Exporter)
    exporter._cfg = exp.SampleConfig(forged_pairs_per_cell=forged_pairs_per_cell, raw_dir=tmp_path)
    exporter._resolver = exp.PseudonymResolver(tmp_path, tmp_path / "cache.json")
    exporter._resolver._map = {"GridSybil_0709/run": {str(p): s for p, s in pseudonym_table.items()}}
    exporter._store = type("FakeStore", (), {"feature_cols": _FEATURES})()
    return exporter


def test_find_forged_demos_scans_each_subfolder_once_and_respects_the_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    ids = pd.DataFrame(
        [
            {
                "family": "GridSybil",
                "group": "0709",
                "subfolder": "run",
                "senderPseudo": p,
                "attack_code": 16,
                "attack_label": "GridSybil",
                "is_attacker": True,
                "n_windows": 20 + p,
            }
            for p in range(1, 6)  # 5 distinct physical senders, one sampled pseudonym each
        ],
        index=[f"u{p}" for p in range(1, 6)],
    )
    sender_lookup = {f"u{p}": p for p in range(1, 6)}
    chosen = [exp.SampledIdentity(uid) for uid in ids.index]
    exporter = _dummy_exporter(tmp_path, forged_pairs_per_cell=2, pseudonym_table={p: p for p in range(1, 6)})

    scan_calls: list[tuple[str, str, str, frozenset[int]]] = []

    def fake_scan(self: exp.BroadcastForger, family: str, group: str, subfolder: str, target_senders: set[int]) -> dict:
        scan_calls.append((family, group, subfolder, frozenset(target_senders)))
        return {
            s: {
                90000
                + s: {i: _broadcast_row(float(i), float(i), 0.0, i, sender=s, pseudo=90000 + s) for i in range(25)}
            }
            for s in target_senders
        }

    monkeypatch.setattr(exp.BroadcastForger, "scan", fake_scan)
    pairs = exporter._find_forged_demos(ids, sender_lookup, chosen)

    assert len(scan_calls) == 1  # one subfolder among 5 candidate senders -> exactly one scan
    assert scan_calls[0][3] == frozenset(range(1, 6))
    assert len(pairs) == 2  # forged_pairs_per_cell caps it, even though 5 senders are eligible
    assert {anchor for anchor, _ in pairs}.issubset(set(ids.index))


def test_find_forged_demos_anchors_on_the_longest_sampled_pseudonym(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    ids = pd.DataFrame(
        [
            {
                "family": "GridSybil",
                "group": "0709",
                "subfolder": "run",
                "senderPseudo": 100,
                "attack_code": 16,
                "attack_label": "GridSybil",
                "is_attacker": True,
                "n_windows": 9,
            },
            {
                "family": "GridSybil",
                "group": "0709",
                "subfolder": "run",
                "senderPseudo": 101,
                "attack_code": 16,
                "attack_label": "GridSybil",
                "is_attacker": True,
                "n_windows": 40,
            },
        ],
        index=["short", "long"],
    )
    sender_lookup = {"short": 7, "long": 7}  # same physical vehicle, two sampled pseudonyms
    chosen = [exp.SampledIdentity("short"), exp.SampledIdentity("long")]
    exporter = _dummy_exporter(tmp_path, forged_pairs_per_cell=6, pseudonym_table={100: 7, 101: 7})

    monkeypatch.setattr(
        exp.BroadcastForger,
        "scan",
        lambda self, family, group, subfolder, target_senders: {
            7: {99999: {i: _broadcast_row(float(i), float(i), 0.0, i, sender=7, pseudo=99999) for i in range(25)}}
        },
    )
    pairs = exporter._find_forged_demos(ids, sender_lookup, chosen)

    assert len(pairs) == 1
    anchor, _demo = pairs[0]
    assert anchor == "long"


def test_find_forged_demos_never_attempts_benign_identities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    ids = pd.DataFrame(
        [
            {
                "family": "GridSybil",
                "group": "0709",
                "subfolder": "run",
                "senderPseudo": 1,
                "attack_code": 0,
                "attack_label": "Benign",
                "is_attacker": False,
                "n_windows": 50,
            },
        ],
        index=["benign_u"],
    )
    exporter = _dummy_exporter(tmp_path, forged_pairs_per_cell=6, pseudonym_table={1: 1})

    scanned = []
    monkeypatch.setattr(exp.BroadcastForger, "scan", lambda self, *a, **kw: scanned.append(a) or {})
    pairs = exporter._find_forged_demos(ids, {"benign_u": 1}, [exp.SampledIdentity("benign_u")])

    assert pairs == []
    assert scanned == []  # benign identities never trigger a raw-file scan


def test_exporter_run_sets_paired_identity_id_symmetrically(tmp_path: pathlib.Path) -> None:
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    raw_dir = tmp_path / "raw"
    run_dir = raw_dir / "GridSybil_0709" / "run"
    run_dir.mkdir(parents=True)

    # `_make_store` hardcodes every uid as a GridSybil attacker; SampleWriter needs at least one
    # Benign row to compute road_bbox, so build the store directly with one of each.
    windows, rows = [], []
    for uid, seq, label, code, is_attacker, pseudo in [
        ("a", _sequence(30, seed=1), "GridSybil", 16, True, 7),
        ("benign", _sequence(30, seed=2), "Benign", 0, False, 1),
    ]:
        starts = range(0, len(seq) - exp._WINDOW + 1, exp._STRIDE)
        times = 25200.0 + np.cumsum(seq[:, _DT].astype(np.float64)) - seq[0, _DT]
        for s in starts:
            windows.append(seq[s : s + exp._WINDOW])
            rows.append(
                {
                    "family": "GridSybil",
                    "group": "0709",
                    "subfolder": "run",
                    "senderPseudo": pseudo,
                    "attack_code": code,
                    "attack_label": label,
                    "is_attacker": is_attacker,
                    "window_start_idx": s,
                    "start_time": times[s],
                    "end_time": times[s + exp._WINDOW - 1],
                    "sender_uid": uid,
                }
            )
    np.save(prepared_dir / "X_windows.npy", np.stack(windows).astype(np.float32))
    pd.DataFrame(rows).to_parquet(prepared_dir / "window_metadata.parquet")
    (prepared_dir / "config.json").write_text(
        json.dumps(
            {
                "feature_cols": _FEATURES,
                "norm_mean": [0.0] * 13,
                "norm_std": [1.0] * 13,
                "label_to_int": {"Benign": 0, "GridSybil": 4},
            }
        )
    )

    (run_dir / "traceGroundTruthJSON-1.json").write_text(
        '{"sender": 5, "senderPseudo": 7}\n{"sender": 6, "senderPseudo": 1}\n'
    )
    with open(run_dir / "traceJSON-5-999-A16-0-1.json", "w", encoding="utf-8") as fh:
        for i in range(25):
            # same clock as the benign identity (starts at t=25200), so the two overlap in time
            fh.write(
                json.dumps(_broadcast_row(25200.0 + i, float(i), 0.0, i, sender=5, pseudo=999), separators=(",", ":"))
                + "\n"
            )

    cfg = exp.SampleConfig(
        seed=0,
        pseudonyms_per_cell=5,
        forged_pairs_per_cell=5,
        prepared_dir=prepared_dir,
        raw_dir=raw_dir,
        out_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    manifest = exp.Exporter(cfg).run()

    by_uid = {r["sender_uid"]: r for r in manifest["identities"]}
    forged = next(r for r in manifest["identities"] if r["data_source"] == "raw_veremi")
    prepared = by_uid["a"]
    assert by_uid["benign"]["data_source"] == "prepared"
    assert by_uid["benign"]["paired_identity_id"] is None  # benign vehicles are never paired
    assert prepared["paired_identity_id"] == forged["id"]
    assert forged["paired_identity_id"] == prepared["id"]
    assert forged["sender"] == prepared["sender"] == 5
    assert forged["sender_pseudo"] == 999
    assert forged["tags"] == ["forged_broadcast"]
    # the sampled benign vehicle overlaps the broadcast in time, so it is reused, not duplicated
    assert forged["benign_match_id"] == by_uid["benign"]["id"]
    assert sum(r["sender_uid"] == "benign" for r in manifest["identities"]) == 1
    lo, hi = forged["overlap"]
    assert 25200.0 <= lo < hi <= 25224.0


def _benign_meta(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """window_metadata-shaped rows: (sender_uid, subfolder, start_time, end_time), all Benign."""
    return pd.DataFrame(
        [
            {
                "sender_uid": uid,
                "family": "GridSybil",
                "group": "0709",
                "subfolder": sub,
                "attack_label": "Benign",
                "start_time": t0,
                "end_time": t1,
            }
            for uid, sub, t0, t1 in rows
        ]
    )


def test_benign_matcher_ranks_by_overlap_and_stays_in_the_run() -> None:
    meta = _benign_meta(
        [
            ("short", "run", 100.0, 120.0),  # overlaps [100, 120] -> 20 s
            ("long", "run", 90.0, 200.0),  # overlaps [100, 150] -> 50 s
            ("elsewhere", "other_run", 0.0, 1e6),  # overlaps fully, but a different run
            ("before", "run", 0.0, 50.0),  # no overlap at all
        ]
    )
    cands = exp.BenignMatcher(meta).candidates("GridSybil", "0709", "run", 100.0, 150.0)
    assert [uid for uid, _, _ in cands] == ["long", "short"]
    assert cands[0][1:] == (100.0, 150.0)


def test_match_benign_requires_a_full_window_on_both_sides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    exporter = _dummy_exporter(tmp_path, forged_pairs_per_cell=6, pseudonym_table={})
    exporter._store = type("FakeStore", (), {"feature_cols": _FEATURES, "meta": None})()
    matcher = exp.BenignMatcher(_benign_meta([("thin", "run", 0.0, 10.0), ("thick", "run", 0.0, 100.0)]))

    def fake_stitch(self: exp.IdentityStitcher, uid: str) -> exp.StitchedIdentity:
        times = np.arange(0.0, 10.0 if uid == "thin" else 100.0)
        return exp.StitchedIdentity(np.zeros((len(times), 13), np.float32), times, np.arange(1))

    monkeypatch.setattr(exp.IdentityStitcher, "__init__", lambda self, store, max_windows=None: None)
    monkeypatch.setattr(exp.IdentityStitcher, "stitch", fake_stitch)

    demo = exp.ForgedDemo(
        "GridSybil", "0709", "run", "GridSybil", 5, 999, np.zeros((60, 13), np.float32), np.arange(0.0, 60.0)
    )
    uid, _stitched, lo, hi = exporter._match_benign(matcher, demo)
    assert uid == "thick"  # "thin" overlaps only 10 steps, under the 20-step window
    assert (lo, hi) == (0.0, 59.0)

    short_demo = exp.ForgedDemo(
        "GridSybil", "0709", "run", "GridSybil", 5, 999, np.zeros((10, 13), np.float32), np.arange(0.0, 10.0)
    )
    assert exporter._match_benign(matcher, short_demo) is None  # attack side too short


def test_sampler_respects_caps_and_tags() -> None:
    rows = []
    for i in range(50):  # 50 attacker pseudonyms of 5 senders in one cell
        rows.append(
            {
                "family": "GridSybil",
                "group": "0709",
                "subfolder": "run",
                "senderPseudo": i,
                "attack_code": 16,
                "attack_label": "GridSybil",
                "is_attacker": True,
                "n_windows": 9 + i % 7,
            }
        )
    rows.append(
        {
            "family": "GridSybil",
            "group": "0709",
            "subfolder": "run",
            "senderPseudo": 1,
            "attack_code": 16,
            "attack_label": "GridSybil",
            "is_attacker": True,
            "n_windows": 7000,
        }
    )
    ids = pd.DataFrame(rows, index=[f"u{i}" for i in range(51)])
    lookup = {f"u{i}": i // 10 for i in range(50)} | {"u50": None}
    cfg = exp.SampleConfig(pseudonyms_per_cell=12, max_pseudonyms_per_sender=8, stress_per_class=2)
    picks = exp.StratifiedSampler(ids, cfg, lookup).sample()
    by_uid = {p.uid: p for p in picks}
    assert "u50" in by_uid and by_uid["u50"].tags == ("defect_demo",) and by_uid["u50"].max_windows == 40
    normal = [p for p in picks if "defect_demo" not in p.tags]
    per_sender = pd.Series([lookup[p.uid] for p in normal if "stress" not in p.tags]).value_counts()
    assert per_sender.max() <= 8
    assert sum(1 for p in picks if "stress" in p.tags) == 2
    assert len({p.uid for p in picks}) == len(picks)
