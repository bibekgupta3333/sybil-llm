"""Training loop and scoring: the `task/autoencoder_st_t20` Trainer (= the `task/autoencoder_st_t32` loop) with this
package's config class for checkpoints and its plot title."""

from __future__ import annotations

from task.autoencoder_st_t20 import trainer as t20
from task.autoencoder_st_t20.trainer import EarlyStopping, RunLog, pick_device, seed_everything
from task.autoencoder_st_t5.config import SupervisedAEConfig

__all__ = ["EarlyStopping", "RunLog", "Trainer", "pick_device", "seed_everything"]


class Trainer(t20.Trainer):
    """Fits the supervised T = 5 autoencoder (n_down = 0) on (x, mask, y) arrays and scores windows."""

    config_cls = SupervisedAEConfig
    title = "Supervised autoencoder T = 5 (all/T5_benign_grid): BCE + recon_weight x masked MSE"
