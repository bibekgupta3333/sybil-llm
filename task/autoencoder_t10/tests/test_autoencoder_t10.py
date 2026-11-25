"""Tests for the unlabelled T = 10 autoencoder (synthetic shards, CPU only); the checks are the T = 20 package's."""

from __future__ import annotations

from pathlib import Path

from task.autoencoder_benign_t20.model import ConvAutoencoder, count_params
from task.autoencoder_t10.config import N_DOWN, SEQ_LEN, TOTAL_PARAMS, AutoencoderT10Config
from task.autoencoder_t10.evaluate import TestEvaluation
from task.autoencoder_t10.train import Experiment
from task.autoencoder_t10.trainer import Trainer
from task.autoencoder_t20.tests.test_autoencoder_t20 import (
    check_cli_guard,
    check_config,
    check_end_to_end,
    check_label_free_imports,
    check_resume,
    check_train_reads_no_label_and_no_test,
)


def test_config_equals_t32_except_input_length_and_depth() -> None:
    check_config(AutoencoderT10Config, 10, 1, "autoencoder_t10")
    assert (SEQ_LEN, N_DOWN) == (10, 1)


def test_model_parameter_count() -> None:
    assert count_params(ConvAutoencoder(AutoencoderT10Config())) == TOTAL_PARAMS


def test_label_free_modules_do_not_import_labels() -> None:
    check_label_free_imports("autoencoder_t10")


def test_train_reads_no_label_and_no_test(tmp_path: Path) -> None:
    check_train_reads_no_label_and_no_test(Experiment, AutoencoderT10Config, tmp_path)


def test_command_line_run_installs_the_guard(tmp_path: Path) -> None:
    check_cli_guard("task.autoencoder_t10.train", SEQ_LEN, tmp_path)


def test_resume_continues_the_same_run(tmp_path: Path) -> None:
    check_resume(Experiment, AutoencoderT10Config, tmp_path)


def test_end_to_end_evaluation_scores_before_labels(tmp_path: Path) -> None:
    check_end_to_end(Experiment, TestEvaluation, AutoencoderT10Config, Trainer, tmp_path)
