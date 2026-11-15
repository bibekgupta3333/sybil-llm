"""Convolutional autoencoder (`convae`): the `task/autoencoder_benign_t32` model under the protocol.

The model class, the masked loss, the Mahalanobis scorer and every setting are imported from that package (63,293
parameters; settings = the defaults of its `AutoencoderConfig`, frozen as they were before this protocol). Scores:
`recon` = masked reconstruction error of the window, `maha` = Mahalanobis distance of z from the z of the regime's
training windows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch import nn

from baselinepaper.experiments import config as cfg
from baselinepaper.experiments.detectors.base import register
from baselinepaper.experiments.detectors.neural_common import (
    NeuralDetector,
    NeuralSettings,
    Objective,
    ReconstructionObjective,
)
from task.autoencoder_benign_t32.config import TOTAL_PARAMS, AutoencoderConfig
from task.autoencoder_benign_t32.model import ConvAutoencoder, count_params
from task.autoencoder_benign_t32.scoring import Mahalanobis

TASK_CONFIG = AutoencoderConfig()
RECON, MAHA = "recon", "maha"


@register
class ConvAutoencoderDetector(NeuralDetector):
    """1D-conv autoencoder trained on the regime's pool; reconstruction error and Mahalanobis distance on z.

    Attributes:
        mahalanobis: Gaussian model of the training z (None before `fit`).
    """

    name = "convae"
    regimes = cfg.UNLABELLED_REGIMES
    score_names = (RECON, MAHA)
    reused_modules = ("task.autoencoder_benign_t32",)
    settings = NeuralSettings.from_task_config(TASK_CONFIG)

    def __init__(self, regime: str, device: str = "cpu", statistics=None) -> None:
        super().__init__(regime=regime, device=device, statistics=statistics)
        self.mahalanobis: Mahalanobis | None = None

    def build_model(self) -> nn.Module:
        model = ConvAutoencoder(TASK_CONFIG)
        if count_params(model) != TOTAL_PARAMS:
            raise RuntimeError(f"ConvAutoencoder has {count_params(model):,} parameters, expected {TOTAL_PARAMS:,}")
        return model

    def objective(self, train) -> Objective:
        return ReconstructionObjective()

    def architecture(self) -> dict:
        return {
            "model": "task.autoencoder_benign_t32.model.ConvAutoencoder",
            "input": "14 x 32 (13 features with padding rows zeroed + mask channel)",
            "latent_dim": TASK_CONFIG.latent_dim,
            "parameters": TOTAL_PARAMS,
            "loss": "masked MSE over real rows",
            "model_selection": "lowest validation loss on the regime's validation pool",
            "mahalanobis": {
                "fitted_on": "z of every training window of the regime (best-epoch weights)",
                "max_cond": TASK_CONFIG.mahalanobis_max_cond,
                "shrinkage_if_above_max_cond": TASK_CONFIG.mahalanobis_shrinkage,
            },
        }

    def after_fit(self, x_train: np.ndarray, mask_train: np.ndarray) -> None:
        z = self.trainer.forward_batches(x_train, mask_train, lambda x, mask: x)["z"]
        self.mahalanobis = Mahalanobis(TASK_CONFIG.mahalanobis_max_cond, TASK_CONFIG.mahalanobis_shrinkage).fit(z)
        self._info["mahalanobis"] = {
            "fitted_windows": len(z),
            "cond": self.mahalanobis.cond,
            "shrinkage": self.mahalanobis.shrinkage,
        }

    def score(self, x: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        out = self.forward(x, mask)
        return {RECON: out["recon"], MAHA: self.mahalanobis.score(out["z"])}

    def save(self, run_dir: Path) -> None:
        super().save(run_dir)
        if self.mahalanobis is not None:
            np.savez(Path(run_dir) / "mahalanobis.npz", **self.mahalanobis.state())
