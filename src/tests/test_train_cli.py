"""Command-line settings of the pretraining run: `--runs-dir` for new runs and `--resume` from any folder."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from src.model.benign_gridsybil.timesnet.config import PretrainConfig
from src.model.benign_gridsybil.timesnet.train import parse_args, settings_from_args


class TestRunsDir:
    """Where run folders go and what config.json records."""

    def test_default_is_the_config_value(self) -> None:
        settings, run_dir, resume = settings_from_args(parse_args(["--run-id", "a"]))
        assert settings.runs_dir == PretrainConfig().runs_dir
        assert run_dir == Path(PretrainConfig().runs_dir) / "a"
        assert not resume

    def test_runs_dir_outside_the_repo(self, tmp_path: Path) -> None:
        target = tmp_path / "drive" / "runs"
        settings, run_dir, resume = settings_from_args(
            parse_args(["--runs-dir", str(target), "--run-id", "remote1", "--smoke"])
        )
        assert run_dir == target / "remote1"
        assert not resume
        assert dataclasses.asdict(settings)["runs_dir"] == str(target)  # what config.json stores

    def test_resume_from_a_folder_outside_the_repo(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "drive" / "runs" / "remote1"
        run_dir.mkdir(parents=True)
        saved = PretrainConfig(runs_dir=str(run_dir.parent), batch_size=128, max_hours=30.0)
        (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(saved)))
        settings, got_dir, resume = settings_from_args(
            parse_args(["--resume", str(run_dir), "--max-hours", "12", "--no-resource-guard"])
        )
        assert resume
        assert got_dir == run_dir
        assert settings.runs_dir == str(run_dir.parent)
        assert settings.batch_size == 128
        assert settings.max_hours == 12.0
        assert settings.resource_guard is False
