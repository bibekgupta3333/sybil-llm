"""Training loop of the unlabelled autoencoder: the `task/autoencoder_t20` Trainer (= the `task/autoencoder_t32` loop)
with this package's config class for checkpoints and its plot title. Inputs are (x, mask) arrays only."""

from __future__ import annotations

from task.autoencoder_t20 import trainer as t20
from task.autoencoder_t20.trainer import EarlyStopping, RunLog, pick_device, seed_everything  # noqa: F401
from task.autoencoder_t10.config import AutoencoderT10Config


class Trainer(t20.Trainer):
    """Fits the T = 10 ConvAutoencoder (n_down = 1) on unlabelled (x, mask) arrays."""

    config_cls = AutoencoderT10Config
    title = "Unlabelled autoencoder T = 10 (all/T10_benign_grid): reconstruction loss"
