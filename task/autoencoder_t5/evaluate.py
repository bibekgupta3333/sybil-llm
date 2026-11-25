"""Final evaluation of a finished run on the test split of all/T5_benign_grid; writes `evaluation.json`.

`task.autoencoder_t20.evaluate` (= the `task/autoencoder_t32` procedure: score test first, write
`scores/test_*.npy`, only then open the labels) with this package's Trainer.

Usage (repo root):
    .venv-train/bin/python -m task.autoencoder_t5.evaluate --run task/autoencoder_t5/runs/seed0
"""

from __future__ import annotations

from task.autoencoder_t20 import evaluate as t20
from task.autoencoder_t5.trainer import Trainer


class TestEvaluation(t20.TestEvaluation):
    """Scores the test split of a finished T = 5 run, then reads the labels and builds `evaluation.json`."""

    __test__ = False  # not a pytest class
    trainer_cls = Trainer


def main(argv: list[str] | None = None) -> None:
    t20.main(argv, TestEvaluation, "task/autoencoder_t5/runs/seed0", __doc__.splitlines()[0])


if __name__ == "__main__":
    main()
