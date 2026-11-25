"""Settings of one unlabelled T = 5 autoencoder run; written to `config.json` in the run folder.

Every setting is `task/autoencoder_t32`'s (via `task/autoencoder_t20`, subclassed, not copied); only the input
(all/T5_benign_grid), the window length (5), the run / cache folders and `n_down` = 0 (stride-2 stages, so that 5
is divisible by 2 ** n_down) differ.
"""

from __future__ import annotations

import dataclasses

from task.autoencoder_t20 import config as t20

DATA_DIR = "src/data/encoder_input/all/T5_benign_grid"
RUNS_DIR = "task/autoencoder_t5/runs"
CACHE_DIR = "task/autoencoder_t5/cache"
SEQ_LEN = 5
N_DOWN = 0
N_FEATURES = t20.N_FEATURES
TOTAL_PARAMS = 50_813


@dataclasses.dataclass(frozen=True)
class AutoencoderT5Config(t20.AutoencoderT20Config):
    """Every setting of a run (see `task.autoencoder_t20.config.AutoencoderT20Config`); T = 5, n_down = 0."""

    data_dir: str = DATA_DIR
    runs_dir: str = RUNS_DIR
    cache_dir: str = CACHE_DIR
    seq_len: int = SEQ_LEN
    n_down: int = N_DOWN
