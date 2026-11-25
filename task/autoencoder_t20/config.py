"""Settings of one unlabelled T = 20 autoencoder run; written to `config.json` in the run folder.

Every setting is `task/autoencoder_t32`'s (the dataclass is subclassed, not copied; model, optimiser, LR schedule and
early stopping equal `task/autoencoder_benign_t32`); only the input (all/T20_benign_grid), the window length (20), the
run / cache folders and `n_down` (stride-2 stages of the ConvAutoencoder; 2 = the T = 32 architecture) differ.
"""

from __future__ import annotations

import dataclasses

from task.autoencoder_t32 import config as t32

DATA_DIR = "src/data/encoder_input/all/T20_benign_grid"
RUNS_DIR = "task/autoencoder_t20/runs"
CACHE_DIR = "task/autoencoder_t20/cache"
SEQ_LEN = 20
N_DOWN = 2
N_FEATURES = t32.N_FEATURES
TOTAL_PARAMS = 56_957


@dataclasses.dataclass(frozen=True)
class AutoencoderT20Config(t32.AutoencoderT32Config):
    """Every setting of a run: the `task/autoencoder_t32` settings plus `n_down`.

    Attributes (beyond the T = 32 ones, see `task.autoencoder_t32.config.AutoencoderT32Config`):
        data_dir: all/T20_benign_grid encoder input (read only; train / val / test = 85 / 7.5 / 7.5 by vehicle key,
            fixed on disk, the all/T32_benign_grid split).
        seq_len: window length (20).
        n_down: stride-2 stages of the ConvAutoencoder (seq_len divisible by 2 ** n_down; 2 = the T = 32 model).
    """

    data_dir: str = DATA_DIR
    runs_dir: str = RUNS_DIR
    cache_dir: str = CACHE_DIR
    seq_len: int = SEQ_LEN
    n_down: int = N_DOWN
