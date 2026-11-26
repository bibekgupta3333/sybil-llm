"""Settings of one supervised T = 64 autoencoder run; written to `config.json` in the run folder.

The training settings are those of `task/autoencoder_st` (inherited from its `SupervisedAEConfig`); the data, the
model shape (the `task/encoder` ConvAutoencoder, input 14 x 64) and the in-code validation draw are the T = 64 ones
of `task/autoencoder_t64` / `task/encoder` supervised mode.
"""

from __future__ import annotations

import dataclasses

from task.autoencoder_st.config import SupervisedAEConfig
from task.encoder.config import DATA_DIR, EncoderTaskConfig

RUNS_DIR = "task/autoencoder_st_t64/runs"
ENCODER_CACHE_DIR = "task/encoder/runs/cache"
SEQ_LEN = 64
N_FEATURES = 13
AUTOENCODER_PARAMS = 80_189
HEAD_PARAMS = 577
TOTAL_PARAMS = AUTOENCODER_PARAMS + HEAD_PARAMS


@dataclasses.dataclass(frozen=True)
class SupervisedAET64Config(SupervisedAEConfig):
    """Every setting of a run (fields of `task.autoencoder_st.config.SupervisedAEConfig` plus the ones below).

    Attributes:
        cache_dir: decoded shards as .npy, shared with `task/encoder` (opened read-only when complete).
        val_frac, split_seed: share of train sender vehicles held out as validation and the seed of that draw
            (`task.encoder.data.VehicleSplit` on all train windows = the val vehicles of `task/encoder` supervised
            mode and of `task/autoencoder_t64`). The on-disk T64 input has train / test only.
        channels: conv widths of `task.encoder.model.ConvAutoencoder`.
    """

    data_dir: str = DATA_DIR
    runs_dir: str = RUNS_DIR
    seq_len: int = SEQ_LEN
    cache_dir: str = ENCODER_CACHE_DIR
    val_frac: float = 0.1
    split_seed: int = 0
    channels: tuple[int, int, int] = (32, 64, 64)

    def architecture(self) -> EncoderTaskConfig:
        """The `task/encoder` settings its ConvAutoencoder is built from (only the model fields matter)."""
        return EncoderTaskConfig(mode="ae", latent_dim=self.latent_dim, channels=tuple(self.channels))

    @classmethod
    def from_dict(cls, d: dict) -> "SupervisedAET64Config":
        """Inverse of `to_dict` (lists from JSON become tuples again)."""
        d = dict(d)
        d["channels"] = tuple(d["channels"])
        return cls(**d)
