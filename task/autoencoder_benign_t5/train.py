"""Trains the one-class T = 5 autoencoder on the benign train windows of all 8 scenario folders.

Data: `src/data/encoder_input/all/T5_benign/` (the all/T32_benign split, 80 / 10 / 10 by sender vehicle; benign
train / val / test + `test_<attack>`). Only `train` and `val` are read. The experiment is
`task.autoencoder_benign_t20.train.Experiment` (= the `task/autoencoder_benign_t32` protocol) with this package's
config, trainer and parameter count; see that module for what is written.

Writes `task/autoencoder_benign_t5/runs/<run_id>/`; decoded shards are cached under `task/autoencoder_benign_t5/cache/`.

Usage (repo root):
    .venv-train/bin/python -m task.autoencoder_benign_t5.train --seed 0
    .venv-train/bin/python -m task.autoencoder_benign_t5.train --seed 0 --log-feature-error
    .venv-train/bin/python -m task.autoencoder_benign_t5.train --seed 0 --resume
    .venv-train/bin/python -m task.autoencoder_benign_t5.train --seed 1 --epochs 2 --max-shards 1 --device cpu
"""

from __future__ import annotations

from task.autoencoder_benign_t20 import train as t20
from task.autoencoder_benign_t5.config import N_DOWN, N_FEATURES, SEQ_LEN, TOTAL_PARAMS, AutoencoderConfig
from task.autoencoder_benign_t5.trainer import Trainer

TRAINING_SPLITS = t20.TRAINING_SPLITS


class Experiment(t20.Experiment):
    """One run of the T = 5 package (see `task.autoencoder_benign_t20.train.Experiment`)."""

    config_cls = AutoencoderConfig
    trainer_cls = Trainer
    default_shape = (SEQ_LEN, N_DOWN, N_FEATURES, 16)
    total_params = TOTAL_PARAMS


def main(argv: list[str] | None = None) -> None:
    t20.main(argv, Experiment, __doc__.splitlines()[0])


if __name__ == "__main__":
    main()
