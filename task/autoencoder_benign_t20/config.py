"""Settings of one run; written to `config.json` in the run folder.

Every training setting is `task/autoencoder_benign_t32`'s (the dataclass is subclassed, not copied); only the input
(all/T20_benign), the window length (20), the run / cache folders and the new `n_down` (number of stride-2 stages of
the ConvAutoencoder; 2 = the T = 32 architecture) differ.
"""

from __future__ import annotations

import dataclasses

from task.autoencoder_benign_t32 import config as t32

DATA_DIR = "src/data/encoder_input/all/T20_benign"
RUNS_DIR = "task/autoencoder_benign_t20/runs"
CACHE_DIR = "task/autoencoder_benign_t20/cache"
SEQ_LEN = 20
N_DOWN = 2
N_FEATURES = t32.N_FEATURES
TOTAL_PARAMS = 56_957
EVAL_SPLITS = t32.EVAL_SPLITS


@dataclasses.dataclass(frozen=True)
class AutoencoderConfig(t32.AutoencoderConfig):
    """Every setting of a run: the `task/autoencoder_benign_t32` settings plus `n_down`.

    Attributes (beyond the T = 32 ones, see `task.autoencoder_benign_t32.config.AutoencoderConfig`):
        data_dir: all/T20_benign encoder input (read only; train / val / test fixed on disk by sender vehicle).
        seq_len: window length (20).
        n_down: stride-2 stages in the encoder (and upsampling stages in the decoder); seq_len must be divisible by
            2 ** n_down. 2 = the T = 32 model (T -> T/2 -> T/4).
    """

    data_dir: str = DATA_DIR
    runs_dir: str = RUNS_DIR
    cache_dir: str = CACHE_DIR
    seq_len: int = SEQ_LEN
    n_down: int = N_DOWN
