"""Training loop of the unlabelled autoencoder: the `task/autoencoder_t20` Trainer (= the `task/autoencoder_t32` loop)
with this package's config class for checkpoints and its plot title. Inputs are (x, mask) arrays only."""

from __future__ import annotations

from task.autoencoder_t20 import trainer as t20
from task.autoencoder_t20.trainer import EarlyStopping, RunLog, pick_device, seed_everything  # noqa: F401
from task.autoencoder_t5.config import AutoencoderT5Config


class Trainer(t20.Trainer):
    """Fits the T = 5 ConvAutoencoder (n_down = 0) on unlabelled (x, mask) arrays."""

    config_cls = AutoencoderT5Config
    title = "Unlabelled autoencoder T = 5 (all/T5_benign_grid): reconstruction loss"
