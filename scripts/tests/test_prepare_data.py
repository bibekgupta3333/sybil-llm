"""Unit tests for the reproducible data-prep pipeline, including the GridSybil_0709 splice fix."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import prepare_data as pd_mod  # noqa: E402


def _write_run(base: pathlib.Path, family: str, group: str, subfolder: str,
                vehicles: dict[int, tuple[int, list[dict]]]) -> pd_mod.ScenarioRun:
    """vehicles: sender -> (attack_code, [ground-truth message dicts, each with senderPseudo/pos/spd/acl/hed/sendTime])."""
    run_dir = base / f"{family}_{group}" / subfolder
    run_dir.mkdir(parents=True)
    ledger_lines = []
    for sender, (code, messages) in vehicles.items():
        pseudo = messages[0]["senderPseudo"]
        (run_dir / f"traceJSON-{sender}-{pseudo}-A{code}-0-1.json").write_text("{}\n")
        for i, msg in enumerate(messages):
            ledger_lines.append(json.dumps({
                "type": 4, "sendTime": msg["sendTime"], "sender": sender, "senderPseudo": msg["senderPseudo"],
                "messageID": sender * 1000 + i,
                "pos": [msg["pos_x"], msg["pos_y"], 0.0], "spd": [msg["spd_x"], msg["spd_y"], 0.0],
                "acl": [0.0, 0.0, 0.0], "hed": [1.0, 0.0, 0.0],
            }))
    (run_dir / "traceGroundTruthJSON-1.json").write_text("\n".join(ledger_lines) + "\n")
    return pd_mod.ScenarioRun(family, group, subfolder, run_dir)


def _drive(n: int, t0: float = 0.0, x0: float = 0.0, pseudo: int = 100, speed: float = 12.0) -> list[dict]:
    return [{"sendTime": t0 + i, "pos_x": x0 + speed * i, "pos_y": 0.0, "spd_x": speed, "spd_y": 0.0,
             "senderPseudo": pseudo} for i in range(n)]


class TestParsedTraceFilename:
    def test_parses_a_well_formed_name(self) -> None:
        parsed = pd_mod.ParsedTraceFilename.parse("traceJSON-21615-21613-A16-28800-8.json")
        assert parsed == pd_mod.ParsedTraceFilename(21615, 21613, 16, 28800, 8)

    def test_rejects_a_malformed_name(self) -> None:
        assert pd_mod.ParsedTraceFilename.parse("traceGroundTruthJSON-8.json") is None


class TestScenarioIndex:
    def test_enumerates_family_group_and_subfolder(self, tmp_path: pathlib.Path) -> None:
        _write_run(tmp_path, "GridSybil", "0709", "run1", {1: (16, _drive(25))})
        _write_run(tmp_path, "GridSybil", "1416", "run1", {2: (16, _drive(25))})
        runs = pd_mod.ScenarioIndex(tmp_path).runs()
        assert {(r.family, r.group, r.subfolder) for r in runs} == {
            ("GridSybil", "0709", "run1"), ("GridSybil", "1416", "run1")}

    def test_raises_when_the_base_dir_is_empty(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(FileNotFoundError):
            pd_mod.ScenarioIndex(tmp_path).runs()


class TestGridSybilSpliceFix:
    """Reproduces the exact defect: senderPseudo==1 shared by many physical vehicles."""

    def _build_collision_run(self, tmp_path: pathlib.Path, n_colliding_senders: int = 5) -> pd_mod.ScenarioRun:
        vehicles = {
            # Each "attacker" broadcasts its own distinct, physically continuous drive,
            # but all of them use the ground-truth sentinel senderPseudo == 1 -- the collision.
            1000 + s: (16, _drive(25, t0=s * 0.01, x0=s * 500.0, pseudo=1))
            for s in range(n_colliding_senders)
        }
        return _write_run(tmp_path, "GridSybil", "0709", "run1", vehicles)

    def test_old_notebook_grouping_would_splice_colliding_senders(self, tmp_path: pathlib.Path) -> None:
        """Demonstrates the bug exists in the data shape the notebook's key would group on."""
        run = self._build_collision_run(tmp_path)
        labels = pd_mod.VehicleLabelResolver().resolve([run])
        ground_truth = pd_mod.GroundTruthLoader(labels).load([run])
        old_keys = ["family", "group", "subfolder", "senderPseudo"]
        groups = ground_truth.groupby(old_keys)
        assert len(groups) == 1  # every colliding sender falls into ONE group under the old key
        merged = groups.get_group(("GridSybil", "0709", "run1", 1))
        assert merged["sender"].nunique() == 5  # confirmed: 5 physical vehicles spliced into "one" sequence

    def test_fixed_grouping_key_separates_every_sender(self, tmp_path: pathlib.Path) -> None:
        run = self._build_collision_run(tmp_path)
        labels = pd_mod.VehicleLabelResolver().resolve([run])
        ground_truth = pd_mod.GroundTruthLoader(labels).load([run])
        features = pd_mod.FeatureEngineer().add_deltas(ground_truth)
        window_set = pd_mod.WindowBuilder(pd_mod.PipelineConfig(min_seq_len=20)).build(features)
        # 5 senders x 25 steps, window=20 stride=10 -> 1 window each (starts: 0) since 25-20=5<10
        assert window_set.metadata["sender"].nunique() == 5
        assert len(window_set.metadata) == 5
        # every window's dpos/dt must be a real, small in-sequence displacement now, not a teleport
        assert window_set.max_implied_speed == pytest.approx(12.0, abs=1e-3)

    def test_window_builder_rejects_a_multi_sender_group_defensively(self, tmp_path: pathlib.Path) -> None:
        run = self._build_collision_run(tmp_path, n_colliding_senders=2)
        labels = pd_mod.VehicleLabelResolver().resolve([run])
        ground_truth = pd_mod.GroundTruthLoader(labels).load([run])
        features = pd_mod.FeatureEngineer().add_deltas(ground_truth)
        broken = pd_mod.WindowBuilder(pd_mod.PipelineConfig(), sequence_keys=["family", "group", "subfolder", "senderPseudo"])
        with pytest.raises(ValueError, match="mixes"):
            broken.build(features)


class TestVehicleLabelResolverAndGroundTruthLoader:
    def test_labels_attach_correctly_by_physical_sender(self, tmp_path: pathlib.Path) -> None:
        run = _write_run(tmp_path, "DataReplaySybil", "0709", "run1", {
            9: (0, _drive(20, pseudo=555)),
            4833: (17, _drive(20, pseudo=999)),
        })
        labels = pd_mod.VehicleLabelResolver().resolve([run])
        assert set(labels["attack_label"]) == {"Benign", "DataReplaySybil"}
        ground_truth = pd_mod.GroundTruthLoader(labels).load([run])
        assert set(ground_truth[ground_truth["sender"] == 9]["attack_label"]) == {"Benign"}
        assert set(ground_truth[ground_truth["sender"] == 4833]["attack_label"]) == {"DataReplaySybil"}


class TestFeatureEngineer:
    def test_deltas_are_zero_only_at_each_sequences_first_row(self, tmp_path: pathlib.Path) -> None:
        run = _write_run(tmp_path, "GridSybil", "0709", "run1", {1: (16, _drive(20, pseudo=1))})
        labels = pd_mod.VehicleLabelResolver().resolve([run])
        ground_truth = pd_mod.GroundTruthLoader(labels).load([run])
        features = pd_mod.FeatureEngineer().add_deltas(ground_truth)
        assert features["dt"].iloc[0] == 0.0
        assert (features["dt"].iloc[1:] > 0).all()


class TestWindowBuilder:
    def test_windows_tile_at_the_configured_stride(self, tmp_path: pathlib.Path) -> None:
        run = _write_run(tmp_path, "GridSybil", "0709", "run1", {1: (16, _drive(40, pseudo=1))})
        labels = pd_mod.VehicleLabelResolver().resolve([run])
        ground_truth = pd_mod.GroundTruthLoader(labels).load([run])
        features = pd_mod.FeatureEngineer().add_deltas(ground_truth)
        window_set = pd_mod.WindowBuilder(pd_mod.PipelineConfig(window_size=20, stride=10, min_seq_len=20)).build(features)
        assert window_set.features.shape == (3, 20, 13)  # starts 0, 10, 20
        assert list(window_set.metadata["window_start_idx"]) == [0, 10, 20]

    def test_sequences_shorter_than_min_seq_len_are_dropped(self, tmp_path: pathlib.Path) -> None:
        run = _write_run(tmp_path, "GridSybil", "0709", "run1", {1: (16, _drive(10, pseudo=1))})
        labels = pd_mod.VehicleLabelResolver().resolve([run])
        ground_truth = pd_mod.GroundTruthLoader(labels).load([run])
        features = pd_mod.FeatureEngineer().add_deltas(ground_truth)
        with pytest.raises(ValueError, match="no windows"):
            pd_mod.WindowBuilder(pd_mod.PipelineConfig(min_seq_len=20)).build(features)


class TestSplitBuilder:
    def _metadata(self, n_senders: int, windows_per_sender: int) -> pd.DataFrame:
        rows = []
        for s in range(n_senders):
            for w in range(windows_per_sender):
                rows.append({"family": "GridSybil", "group": "0709" if s % 2 == 0 else "1416",
                             "subfolder": "run1", "sender": s, "senderPseudo": s,
                             "attack_code": 0, "attack_label": "Benign", "is_attacker": False,
                             "window_start_idx": w * 10, "start_time": float(w), "end_time": float(w + 20)})
        return pd.DataFrame(rows)

    def test_splits_are_sender_disjoint_and_complete(self) -> None:
        metadata = self._metadata(n_senders=40, windows_per_sender=3)
        splits = pd_mod.SplitBuilder(pd_mod.PipelineConfig(seed=42)).build(metadata)
        all_idx = np.concatenate([splits.train_idx, splits.val_idx, splits.test_idx])
        assert sorted(all_idx) == list(range(len(metadata)))
        train_s = set(splits.sender_uid.iloc[splits.train_idx])
        test_s = set(splits.sender_uid.iloc[splits.test_idx])
        assert not (train_s & test_s)

    def test_scenario_holdout_partitions_by_group(self) -> None:
        metadata = self._metadata(n_senders=10, windows_per_sender=2)
        splits = pd_mod.SplitBuilder(pd_mod.PipelineConfig(seed=1)).build(metadata)
        assert len(splits.group_0709_idx) + len(splits.group_1416_idx) == len(metadata)
        assert set(splits.group_0709_idx) & set(splits.group_1416_idx) == set()


class TestNormalizer:
    def test_fits_only_on_the_train_split(self) -> None:
        windows = np.zeros((4, 5, 2), dtype=np.float32)
        windows[:2] = 1.0  # would-be train
        windows[2:] = 100.0  # val/test -- must not influence the stats
        mean, std = pd_mod.Normalizer.fit(windows, train_idx=np.array([0, 1]))
        assert np.allclose(mean, 1.0)
        assert np.allclose(std, 0.0) is False  # guarded away from 0
        assert (std >= 1.0 - 1e-8).all()

    def test_guards_against_zero_variance_features(self) -> None:
        windows = np.full((3, 4, 2), 7.0, dtype=np.float32)
        _, std = pd_mod.Normalizer.fit(windows, train_idx=np.array([0, 1, 2]))
        assert np.allclose(std, 1.0)


class TestDataPreparationPipelineEndToEnd:
    def test_runs_end_to_end_and_writes_a_consistent_manifest(self, tmp_path: pathlib.Path) -> None:
        base = tmp_path / "raw"
        out = tmp_path / "prepared"
        _write_run(base, "GridSybil", "0709", "run1", {
            **{1000 + s: (16, _drive(30, x0=s * 300.0, pseudo=1)) for s in range(4)},  # the collision, fixed
            9001: (0, _drive(30, pseudo=42)),
        })
        config = pd_mod.DataPreparationPipeline(
            pd_mod.PipelineConfig(base_dir=base, out_dir=out, min_seq_len=20, seed=7)).run()

        assert (out / "X_windows.npy").exists()
        windows = np.load(out / "X_windows.npy")
        assert windows.shape == (config["n_windows"], 20, 13)
        assert config["n_train"] + config["n_val"] + config["n_test"] == config["n_windows"]
        metadata = pd.read_parquet(out / "window_metadata.parquet")
        assert metadata["sender"].nunique() == 5  # 4 colliding GridSybil senders + 1 benign, never merged
        with open(out / "config.json", encoding="utf-8") as fh:
            written = json.load(fh)
        assert written["sequence_keys"] == pd_mod._SEQUENCE_KEYS
        assert written["max_implied_speed_mps"] < pd_mod._TELEPORT_SPEED_MPS  # no residual splice
