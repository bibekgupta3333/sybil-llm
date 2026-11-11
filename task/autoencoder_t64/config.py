"""Settings of one run of the unlabelled T = 64 autoencoder; written to `config.json` in the run folder."""

from __future__ import annotations

import dataclasses

from task.encoder.config import DATA_DIR, EncoderTaskConfig

RUNS_DIR = "task/autoencoder_t64/runs"
ENCODER_CACHE_DIR = "task/encoder/runs/cache"


@dataclasses.dataclass(frozen=True)
class AutoencoderT64Config:
    """Every setting of a run.

    Attributes:
        data_dir: T64 encoder input (read only; 90 / 10 train / test by sender vehicle on disk).
        runs_dir: parent of the run folders.
        cache_dir: decoded shards as .npy, shared with `task/encoder` (opened read-only when complete; built by
            `task.encoder.data.WindowStore` only if missing).
        val_frac, split_seed: share of train sender vehicles held out as validation and the seed of that draw; the
            rule is `task.encoder.data.VehicleSplit` on all train windows (= `task/encoder` supervised mode).
        latent_dim, channels: architecture of `task.encoder.model.ConvAutoencoder`.
        lr, weight_decay: Adam.
        batch_size: training batch size.
        max_epochs: upper bound on epochs.
        patience: early stopping on validation loss (epochs without improvement).
        plateau_factor, plateau_patience: ReduceLROnPlateau on validation loss.
        min_delta: relative validation-loss decrease that counts as an improvement (0 = any decrease).
        maha_ridge: ridge added to the z covariance diagonal before inversion (the `task/encoder` value).
        threshold_quantiles: quantiles of the unlabelled validation scores used as alarm thresholds.
        seed: weights and batch order.
        device: "auto" (mps > cuda > cpu) or an explicit torch device.
        eval_batch: batch size for validation and scoring.
    """

    data_dir: str = DATA_DIR
    runs_dir: str = RUNS_DIR
    cache_dir: str = ENCODER_CACHE_DIR
    val_frac: float = 0.1
    split_seed: int = 0
    latent_dim: int = 16
    channels: tuple[int, int, int] = (32, 64, 64)
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 300
    patience: int = 10
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    min_delta: float = 0.0
    maha_ridge: float = 1e-6
    threshold_quantiles: tuple[float, ...] = (0.95, 0.99)
    seed: int = 0
    device: str = "auto"
    eval_batch: int = 4096

    @property
    def run_id(self) -> str:
        """Run folder name, e.g. `seed0`."""
        return f"seed{self.seed}"

    def architecture(self) -> EncoderTaskConfig:
        """The `task/encoder` settings object its ConvAutoencoder is built from (only the model fields matter)."""
        return EncoderTaskConfig(mode="ae", latent_dim=self.latent_dim, channels=tuple(self.channels))

    def to_dict(self) -> dict:
        """JSON-ready settings."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "AutoencoderT64Config":
        """Inverse of `to_dict` (lists from JSON become tuples again)."""
        values = dict(values)
        for key in ("channels", "threshold_quantiles"):
            values[key] = tuple(values[key])
        return cls(**values)
