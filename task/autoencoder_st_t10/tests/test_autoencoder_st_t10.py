"""Tests for the supervised T = 10 autoencoder (synthetic shards, CPU only); the checks are the T = 20 package's."""

from __future__ import annotations

from pathlib import Path

from task.autoencoder_st_t10.config import N_DOWN, SEQ_LEN, TOTAL_PARAMS, SupervisedAEConfig
from task.autoencoder_st_t10.train import Experiment
from task.autoencoder_st_t10.trainer import Trainer
from task.autoencoder_st_t20.tests.test_autoencoder_st_t20 import (
    check_config,
    check_feature_log,
    check_model,
    check_resume,
    check_smoke,
)


def test_config_equals_t32_except_input_length_and_depth() -> None:
    check_config(SupervisedAEConfig, 10, 1, "autoencoder_st_t10")
    assert (SEQ_LEN, N_DOWN) == (10, 1)


def test_model_shapes_params_and_padding() -> None:
    check_model(SupervisedAEConfig, TOTAL_PARAMS)


def test_one_run_smoke_reads_only_train_and_val(tmp_path: Path) -> None:
    check_smoke(Experiment, SupervisedAEConfig, Trainer, tmp_path)


def test_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    check_resume(Experiment, SupervisedAEConfig, tmp_path)


def test_feature_error_log_is_opt_in(tmp_path: Path) -> None:
    check_feature_log(Experiment, SupervisedAEConfig, tmp_path)
