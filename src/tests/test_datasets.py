"""The two training sets: compact T24 shard reader, label firewall, `--dataset` defaults, run-folder collisions."""

from __future__ import annotations

import dataclasses
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.model.benign_gridsybil.timesnet.config import DATASETS, PretrainConfig
from src.model.benign_gridsybil.timesnet.data import (
    FEATURES,
    CompactWindowReader,
    FeatureShards,
    FeatureSpace,
    LabelFirewall,
    PaddedWindowReader,
    ShardReader,
)
from src.model.benign_gridsybil.timesnet.heads import PretrainingModel
from src.model.benign_gridsybil.timesnet.losses import Objective
from src.model.benign_gridsybil.timesnet.train import ensure_new_run_dir, parse_args, settings_from_args

from .conftest import synthetic_metadata, synthetic_windows


def compact_window(k: int, n: int) -> dict:
    """A compact window with n real rows whose values encode (window, row, feature)."""
    rows = [[k * 1000 + r * 13 + f for f in range(len(FEATURES))] for r in range(n)]
    return {"id": f"DoSRandomSybil_1416/run/traceJSON-1-2-A0-50400-1.json#{100 + k}-{k % 3}#0", "n": n, "x": rows}


def write_compact(path: Path, lengths: list[int], first: int = 0, seq_len: int = 24) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    windows = [compact_window(first + i, n) for i, n in enumerate(lengths)]
    with gzip.open(path, "wt") as f:
        json.dump({"seq_len": seq_len, "features": FEATURES, "windows": windows}, f)
    with gzip.open(path.with_name(path.name.replace(".json.gz", "_info.json.gz")), "wt") as f:
        json.dump({"labels": ["poison"] * len(windows)}, f)  # a label file the loader must never open


class TestCompactReader:
    """Compact gzipped shards are padded to seq_len with a prefix mask, in shard order."""

    def test_pads_and_masks(self, tmp_path: Path) -> None:
        write_compact(tmp_path / "train" / "part-00000.json.gz", [1, 24, 3])
        write_compact(tmp_path / "train" / "part-00001.json.gz", [5, 2], first=3)
        data = FeatureShards(tmp_path, "train", LabelFirewall(), None, workers=2, seq_len=24)
        assert data.x.shape == (5, 24, 13) and data.x.dtype == torch.float32
        assert data.mask.shape == (5, 24) and data.mask.dtype == torch.bool
        lengths = [1, 24, 3, 5, 2]
        assert data.mask.sum(1).tolist() == lengths
        for k, n in enumerate(lengths):
            assert bool(data.mask[k, :n].all()) and not bool(data.mask[k, n:].any())
            expected = torch.tensor([[k * 1000 + r * 13 + f for f in range(13)] for r in range(n)], dtype=torch.float32)
            assert torch.equal(data.x[k, :n], expected)
            assert bool((data.x[k, n:] == 0).all())
        assert data.ids == [compact_window(k, 1)["id"] for k in range(5)]
        assert "compact" in data.reader

    def test_seq_len_mismatch_is_refused(self, tmp_path: Path) -> None:
        write_compact(tmp_path / "train" / "part-00000.json.gz", [2], seq_len=24)
        with pytest.raises(ValueError, match="seq_len"):
            CompactWindowReader(64).read(str(tmp_path / "train" / "part-00000.json.gz"))

    def test_reader_is_picked_by_layout(self, tmp_path: Path) -> None:
        write_compact(tmp_path / "compact" / "train" / "part-00000.json.gz", [2])
        padded = tmp_path / "padded" / "train" / "b64" / "part-00000.json"
        padded.parent.mkdir(parents=True)
        padded.write_text(json.dumps({"windows": [{"id": "a", "x": [[0.0] * 13] * 64, "mask": [1] * 64}]}))
        assert isinstance(ShardReader.pick(tmp_path / "compact", "train", 24), CompactWindowReader)
        assert isinstance(ShardReader.pick(tmp_path / "padded", "train", 64), PaddedWindowReader)
        with pytest.raises(FileNotFoundError):
            ShardReader.pick(tmp_path / "missing", "train", 24)

    def test_in_place_hold_out_matches_the_copy(self, tmp_path: Path) -> None:
        lengths = [(i % 24) + 1 for i in range(90)]
        write_compact(tmp_path / "train" / "part-00000.json.gz", lengths)
        a = FeatureShards(tmp_path, "train", LabelFirewall(), None, workers=1, seq_len=24)
        b = FeatureShards(tmp_path, "train", LabelFirewall(), None, workers=1, seq_len=24)
        rest_copy, check_copy = a.hold_out(0.34, seed=0)
        rest, check = b.hold_out(0.34, seed=0, in_place=True)
        assert rest is b and rest.ids == rest_copy.ids and check.ids == check_copy.ids
        assert torch.equal(rest.x, rest_copy.x) and torch.equal(rest.mask, rest_copy.mask)
        assert len(rest) + len(check) == 90 and len(rest) == len(rest.ids)

    def test_ram_estimate_full_all_t24(self) -> None:
        assert FeatureShards.ram_estimate_gb(5_071_584, 24) == pytest.approx(7.21, abs=0.05)


class TestFirewall:
    """Both layouts pass; label files and test splits never do."""

    @pytest.mark.parametrize(
        "path",
        ["T24/train/part-00000.json.gz", "T64/train/b64/part-00012.json", "T24/pretrain_val/part-00001.json.gz"],
    )
    def test_feature_shards_pass(self, path: str) -> None:
        assert LabelFirewall().check(Path(path)) == Path(path)

    @pytest.mark.parametrize(
        "path",
        [
            "T24/train/part-00000_info.json.gz",
            "T64/train/b64/part-00000_info.json",
            "T24/test/part-00000.json.gz",
            "T64/test/b64/part-00000.json",
            "T24/val/part-00000.json.gz",
            "T24/train/metadata.json",
        ],
    )
    def test_labels_and_held_out_splits_are_refused(self, path: str) -> None:
        with pytest.raises(PermissionError):
            LabelFirewall().check(Path(path))

    def test_glob_never_finds_info_files(self, tmp_path: Path) -> None:
        write_compact(tmp_path / "train" / "part-00000.json.gz", [2])
        assert [p.name for p in CompactWindowReader(24).find(tmp_path, "train")] == ["part-00000.json.gz"]


class TestDatasetDefaults:
    """`--dataset` sets data dir, seq_len, runs dir and the fixed run name."""

    @pytest.mark.parametrize(
        "dataset, data_dir, seq_len, runs_dir",
        [
            ("grid", "src/data/encoder_input/benign_gridsybil/T64", 64, "src/runs/pretraining/benign_gridsybil/T64"),
            ("all", "src/data/encoder_input/all/T24", 24, "src/runs/pretraining/all/T24"),
        ],
    )
    def test_full_run(self, dataset: str, data_dir: str, seq_len: int, runs_dir: str) -> None:
        settings, run_dir, resume = settings_from_args(parse_args(["--dataset", dataset]))
        assert (settings.dataset, settings.data_dir, settings.seq_len, settings.runs_dir) == (
            dataset,
            data_dir,
            seq_len,
            runs_dir,
        )
        assert run_dir == Path(runs_dir) / f"model-{dataset}"
        assert not resume

    def test_default_is_grid(self) -> None:
        settings, run_dir, _ = settings_from_args(parse_args([]))
        assert settings.dataset == "grid" and run_dir.name == "model-grid"
        assert PretrainConfig().dataset == "grid" and PretrainConfig().seq_len == 64

    @pytest.mark.parametrize("flag", ["--check", "--smoke"])
    def test_test_runs_never_take_the_real_name(self, flag: str) -> None:
        _, run_dir, _ = settings_from_args(parse_args(["--dataset", "all", flag]))
        assert run_dir.name.startswith(f"model-all-{flag[2:]}-")
        assert run_dir.parent == Path(DATASETS["all"].runs_dir)

    def test_explicit_flags_win(self, tmp_path: Path) -> None:
        settings, run_dir, _ = settings_from_args(
            parse_args(["--dataset", "all", "--runs-dir", str(tmp_path), "--run-id", "model-all-2", "--data-dir", "d"])
        )
        assert run_dir == tmp_path / "model-all-2"
        assert settings.data_dir == "d" and settings.seq_len == 24

    def test_resume_keeps_the_saved_dataset(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "model-all"
        run_dir.mkdir()
        saved = PretrainConfig().use_dataset("all")
        (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(saved)))
        settings, got, resume = settings_from_args(parse_args(["--resume", str(run_dir)]))
        assert resume and got == run_dir
        assert settings.dataset == "all" and settings.seq_len == 24

    def test_old_config_without_dataset_is_grid(self) -> None:
        old = {k: v for k, v in PretrainConfig().__dict__.items() if k != "dataset"}
        assert PretrainConfig.from_dict(old).dataset == "grid"

    def test_unknown_dataset(self) -> None:
        with pytest.raises(ValueError):
            PretrainConfig().use_dataset("dos")


class TestRunFolderCollision:
    """A new run never reuses an existing folder."""

    def test_existing_folder_stops_with_resume_hint(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "model-all"
        run_dir.mkdir()
        with pytest.raises(SystemExit) as stop:
            ensure_new_run_dir(run_dir)
        message = str(stop.value)
        assert f"npm run train:resume -- {run_dir}" in message
        assert "--run-id model-all-2" in message

    def test_new_folder_passes(self, tmp_path: Path) -> None:
        ensure_new_run_dir(tmp_path / "model-grid")


def test_short_windows_through_the_objective(settings: PretrainConfig) -> None:
    """T = 24 with mostly 1-3-row windows: views stay on real rows, absent losses are skipped, nothing is NaN."""
    settings.use_dataset("all")
    lengths = [1, 1, 2, 3, 3, 4, 5, 8, 12, 24, 24, 17, 1, 2, 6, 24]
    x, mask = synthetic_windows(lengths, seq_len=24)
    space = FeatureSpace(synthetic_metadata())
    model, objective = PretrainingModel(settings), Objective(settings, space)
    losses, _ = objective(model, x, mask, torch.Generator().manual_seed(0), torch.arange(len(lengths)))
    assert set(losses) >= {"recon", "nce"} and all(torch.isfinite(v) for v in losses.values())
    short = [1, 1, 2, 3]
    xs, ms = synthetic_windows(short, seq_len=24)
    losses_short, _ = objective(model, xs, ms, torch.Generator().manual_seed(0), torch.arange(4))
    assert "nce" not in losses_short and not {"p1", "p2", "p3"} & set(losses_short)
    h, z = model.encoder(x, mask)
    assert h.shape == (16, 24, settings.d_model) and bool(torch.isfinite(z).all())
    garbage = x + torch.randn_like(x) * 100 * (~mask).unsqueeze(-1)
    assert float((model.encoder.eval()(garbage, mask)[1] - model.encoder(x, mask)[1]).detach().abs().max()) < 1e-4
    assert np.isfinite(float(sum(losses.values())))
