"""Settings of one run; written to `config.json` in the run folder.

Every setting is `task/autoencoder_benign_t32`'s (via `task/autoencoder_benign_t20`, subclassed, not copied); only the
input (all/T10_benign), the window length (10), the run / cache folders and `n_down` = 1 (stride-2 stages, so that
10 is divisible by 2 ** n_down) differ.
"""

from __future__ import annotations

import dataclasses

from task.autoencoder_benign_t20 import config as t20

DATA_DIR = "src/data/encoder_input/all/T10_benign"
RUNS_DIR = "task/autoencoder_benign_t10/runs"
CACHE_DIR = "task/autoencoder_benign_t10/cache"
SEQ_LEN = 10
N_DOWN = 1
N_FEATURES = t20.N_FEATURES
TOTAL_PARAMS = 52_861
EVAL_SPLITS = t20.EVAL_SPLITS


@dataclasses.dataclass(frozen=True)
class AutoencoderConfig(t20.AutoencoderConfig):
    """Every setting of a run (see `task.autoencoder_benign_t20.config.AutoencoderConfig`); T = 10, n_down = 1."""

    data_dir: str = DATA_DIR
    runs_dir: str = RUNS_DIR
    cache_dir: str = CACHE_DIR
    seq_len: int = SEQ_LEN
    n_down: int = N_DOWN
