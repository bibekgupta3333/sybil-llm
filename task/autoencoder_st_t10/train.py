"""Trains the supervised T = 10 autoencoder (classifier head on z) on the benign + GridSybil train windows.

Data: `src/data/encoder_input/all/T10_benign_grid/` (the all/T32_benign_grid split, 85 / 7.5 / 7.5 by vehicle key).
The experiment is `task.autoencoder_st_t20.train.Experiment` (= the `task/autoencoder_st_t32` protocol) with this
package's config, trainer and parameter count; see that module for what is written. The test split is never read here.

Usage (repo root):
    .venv-train/bin/python -m task.autoencoder_st_t10.train --seed 0
    .venv-train/bin/python -m task.autoencoder_st_t10.train --seed 0 --log-feature-error
    .venv-train/bin/python -m task.autoencoder_st_t10.train --seed 0 --resume
    .venv-train/bin/python -m task.autoencoder_st_t10.train --seed 1 --epochs 2 --max-shards 1 --device cpu
"""

from __future__ import annotations

from task.autoencoder_st_t20 import train as t20
from task.autoencoder_st_t10.config import N_DOWN, N_FEATURES, SEQ_LEN, TOTAL_PARAMS, SupervisedAEConfig
from task.autoencoder_st_t10.trainer import Trainer

TRAIN_SPLIT, VAL_SPLIT = t20.TRAIN_SPLIT, t20.VAL_SPLIT


class Experiment(t20.Experiment):
    """One run of the T = 10 package (see `task.autoencoder_st_t20.train.Experiment`)."""

    config_cls = SupervisedAEConfig
    trainer_cls = Trainer
    default_shape = (SEQ_LEN, N_DOWN, N_FEATURES, 16, 32)
    total_params = TOTAL_PARAMS


def main(argv: list[str] | None = None) -> None:
    t20.main(argv, Experiment, __doc__.splitlines()[0])


if __name__ == "__main__":
    main()
