"""Tests for the one-class T = 5 autoencoder (synthetic shards, CPU only); the checks are the T = 20 package's."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from task.autoencoder_benign_t20.tests.test_autoencoder_benign_t20 import (
    check_config,
    check_feature_log,
    check_model,
    check_resume,
    check_smoke,
    make_data_dir,
)
from task.autoencoder_benign_t5.config import N_DOWN, SEQ_LEN, TOTAL_PARAMS, AutoencoderConfig
from task.autoencoder_benign_t5.data import WindowDataset
from task.autoencoder_benign_t5.train import Experiment
from task.autoencoder_benign_t5.trainer import Trainer


def test_config_equals_t32_except_input_length_and_depth() -> None:
    check_config(AutoencoderConfig, 5, 0, "autoencoder_benign_t5")
    assert (SEQ_LEN, N_DOWN) == (5, 0)


def test_model_shapes_params_and_padding() -> None:
    check_model(AutoencoderConfig, TOTAL_PARAMS)


def test_dataset_reads_b05_shards(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path, SEQ_LEN)
    assert (data_dir / "train" / "b05").is_dir()
    data = WindowDataset.load(data_dir, "train", workers=1)
    assert data.x.shape[1] == SEQ_LEN and np.all(data.x[~data.mask] == 0.0)


def test_experiment_smoke_writes_run_folder(tmp_path: Path) -> None:
    check_smoke(Experiment, AutoencoderConfig, Trainer, tmp_path)


def test_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    check_resume(Experiment, AutoencoderConfig, tmp_path)


def test_feature_error_log_is_opt_in(tmp_path: Path) -> None:
    check_feature_log(Experiment, AutoencoderConfig, tmp_path)
