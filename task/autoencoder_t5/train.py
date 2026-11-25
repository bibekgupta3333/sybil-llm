"""Trains the unlabelled T = 5 autoencoder on every train window of all/T5_benign_grid (benign + GridSybil pooled).

Data: `src/data/encoder_input/all/T5_benign_grid/` (the all/T32_benign_grid split, 85 / 7.5 / 7.5 by vehicle key).
No label is read (feature shards only; the command line installs `LabelFileGuard` in this process and in every
decoding worker). The experiment is `task.autoencoder_t20.train.Experiment` (= the `task/autoencoder_t32` protocol)
with this package's config, trainer and parameter count; see that module for what is written. The test evaluation is
`task.autoencoder_t5.evaluate`.

Usage (repo root):
    .venv-train/bin/python -m task.autoencoder_t5.train --seed 0
    .venv-train/bin/python -m task.autoencoder_t5.train --seed 0 --resume
    .venv-train/bin/python -m task.autoencoder_t5.train --seed 1 --epochs 2 --max-shards 1 --device cpu
"""

from __future__ import annotations

from task.autoencoder_t20 import train as t20
from task.autoencoder_t5.config import N_DOWN, N_FEATURES, SEQ_LEN, TOTAL_PARAMS, AutoencoderT5Config
from task.autoencoder_t5.trainer import Trainer

TRAINING_SPLITS = t20.TRAINING_SPLITS


class Experiment(t20.Experiment):
    """One run of the T = 5 package (see `task.autoencoder_t20.train.Experiment`)."""

    config_cls = AutoencoderT5Config
    trainer_cls = Trainer
    default_shape = (SEQ_LEN, N_DOWN, N_FEATURES, 16)
    total_params = TOTAL_PARAMS


def main(argv: list[str] | None = None) -> None:
    """Command line entry: installs the label-file guard for the whole process, then runs."""
    t20.main(argv, Experiment, __doc__.splitlines()[0])


if __name__ == "__main__":
    main()
