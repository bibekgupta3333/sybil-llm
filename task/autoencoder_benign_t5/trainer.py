"""Training loop and scoring: the `task/autoencoder_benign_t20` Trainer (= the `task/autoencoder_benign_t32` loop),
with this package's config class for checkpoints and its plot title."""

from __future__ import annotations

from task.autoencoder_benign_t20 import trainer as t20
from task.autoencoder_benign_t20.trainer import EarlyStopping, RunLog, pick_device, seed_everything
from task.autoencoder_benign_t5.config import AutoencoderConfig

__all__ = ["EarlyStopping", "RunLog", "Trainer", "pick_device", "seed_everything"]


class Trainer(t20.Trainer):
    """Fits the T = 5 ConvAutoencoder (n_down = 0) on (x, mask) arrays; scores windows by error and z."""

    config_cls = AutoencoderConfig
    title = "One-class autoencoder T = 5 (all/T5_benign): reconstruction loss"
