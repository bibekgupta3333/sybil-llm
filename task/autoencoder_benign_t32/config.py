"""Settings of one run; written to `config.json` in the run folder."""

from __future__ import annotations

import dataclasses

DATA_DIR = "src/data/encoder_input/all/T32_benign"
RUNS_DIR = "task/autoencoder_benign_t32/runs"
CACHE_DIR = "task/autoencoder_benign_t32/cache"
SEQ_LEN = 32
N_FEATURES = 13
TOTAL_PARAMS = 63_293
EVAL_SPLITS = ("test", "test_gridsybil", "test_datareplay", "test_dosrandom", "test_dosdisruptive")


@dataclasses.dataclass(frozen=True)
class AutoencoderConfig:
    """Every setting of a run.

    Attributes:
        data_dir: all/T32_benign encoder input (read only; train / val / test fixed on disk by sender vehicle).
        runs_dir: parent of the run folders.
        cache_dir: decoded shards as .npy (gitignored); "" disables the cache.
        seq_len, n_features: window shape (32 x 13).
        latent_dim: size of z.
        lr, weight_decay: Adam.
        batch_size: training batch size.
        max_epochs: upper bound on epochs.
        patience: early stopping on validation loss (epochs without improvement).
        plateau_factor, plateau_patience: ReduceLROnPlateau on validation loss.
        min_delta: relative validation-loss decrease needed to count as an improvement (resets early stopping).
        grad_clip: max global gradient L2 norm (the logged norm is the pre-clip value).
        threshold_quantiles: quantiles of the validation scores used as alarm thresholds.
        mahalanobis_max_cond: condition number above which the z covariance is shrunk towards a scaled identity.
        mahalanobis_shrinkage: shrinkage weight applied in that case.
        seed: weights and batch order.
        device: "auto" (mps > cuda > cpu) or an explicit torch device.
        eval_batch: batch size for validation and scoring.
        workers: processes that decode the JSON shards.
    """

    data_dir: str = DATA_DIR
    runs_dir: str = RUNS_DIR
    cache_dir: str = CACHE_DIR
    seq_len: int = SEQ_LEN
    n_features: int = N_FEATURES
    latent_dim: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 300
    patience: int = 10
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    min_delta: float = 1e-4
    grad_clip: float = 1.0
    threshold_quantiles: tuple[float, ...] = (0.95, 0.99)
    mahalanobis_max_cond: float = 1e6
    mahalanobis_shrinkage: float = 0.01
    seed: int = 0
    device: str = "auto"
    eval_batch: int = 4096
    workers: int = 8

    @property
    def run_id(self) -> str:
        """Run folder name, e.g. `seed0`."""
        return f"seed{self.seed}"

    def to_dict(self) -> dict:
        """JSON-ready settings."""
        d = dataclasses.asdict(self)
        d["threshold_quantiles"] = list(self.threshold_quantiles)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AutoencoderConfig":
        """Inverse of `to_dict`."""
        d = dict(d)
        if "threshold_quantiles" in d:
            d["threshold_quantiles"] = tuple(d["threshold_quantiles"])
        return cls(**d)
