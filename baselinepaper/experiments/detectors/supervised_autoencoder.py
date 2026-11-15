"""Supervised autoencoder (`supae`): the `task/autoencoder_st_t32` autoencoder + classifier head under the protocol.

The model class and every setting are imported from that package (63,293 + 577 = 63,870 parameters; settings = the
defaults of its `SupervisedAEConfig`, frozen as they were before this protocol). Loss = BCEWithLogits (pos_weight =
benign / GridSybil count of the training labels) + recon_weight x masked MSE; model selection on the validation
AUROC (benign vs GridSybil validation windows). Scores: `prob` = P(GridSybil), `recon` = masked reconstruction
error. Labels are read in `fit` only; `score` gets x and mask.
"""

from __future__ import annotations

import numpy as np
from torch import nn

from baselinepaper.experiments import config as cfg
from baselinepaper.experiments.detectors.base import register
from baselinepaper.experiments.detectors.neural_common import (
    NeuralDetector,
    NeuralSettings,
    Objective,
    SupervisedObjective,
)
from task.autoencoder_st_t32.config import TOTAL_PARAMS, SupervisedAEConfig
from task.autoencoder_st_t32.model import SupervisedAutoencoder, count_params

TASK_CONFIG = SupervisedAEConfig()
PROB, RECON = "prob", "recon"


@register
class SupervisedAutoencoderDetector(NeuralDetector):
    """Conv autoencoder + classifier head on z, trained on labelled benign + GridSybil windows."""

    name = "supae"
    regimes = (cfg.SUPERVISED,)
    score_names = (PROB, RECON)
    reused_modules = ("task.autoencoder_st_t32", "task.autoencoder_benign_t32")
    settings = NeuralSettings.from_task_config(TASK_CONFIG)

    def build_model(self) -> nn.Module:
        model = SupervisedAutoencoder(TASK_CONFIG)
        if count_params(model) != TOTAL_PARAMS:
            raise RuntimeError(
                f"SupervisedAutoencoder has {count_params(model):,} parameters, expected {TOTAL_PARAMS:,}"
            )
        return model

    def objective(self, train) -> Objective:
        labels = np.asarray(train.labels)
        positives = int((labels == 1).sum())
        negatives = int((labels == 0).sum())
        if positives == 0 or negatives == 0:
            raise ValueError("the supervised regime needs both classes in the training labels")
        self._class_counts = {"benign": negatives, "gridsybil": positives}
        return SupervisedObjective(pos_weight=negatives / positives, recon_weight=TASK_CONFIG.recon_weight)

    def architecture(self) -> dict:
        return {
            "model": "task.autoencoder_st_t32.model.SupervisedAutoencoder",
            "input": "14 x 32 (13 features with padding rows zeroed + mask channel)",
            "latent_dim": TASK_CONFIG.latent_dim,
            "head": f"Linear 16 -> {TASK_CONFIG.head_hidden} -> ReLU -> Dropout {TASK_CONFIG.head_dropout} -> 1",
            "parameters": TOTAL_PARAMS,
            "loss": "BCEWithLogits(pos_weight = n_benign / n_gridsybil of the training labels) + recon_weight * MSE",
            "recon_weight": TASK_CONFIG.recon_weight,
            "model_selection": "highest validation AUROC (benign vs GridSybil validation windows)",
        }

    def after_fit(self, x_train: np.ndarray, mask_train: np.ndarray) -> None:
        self._info["train_class_counts"] = self._class_counts
        self._info["pos_weight"] = self.trainer.objective.pos_weight

    def score(self, x: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        out = self.forward(x, mask)
        return {PROB: 1.0 / (1.0 + np.exp(-out["logit"])), RECON: out["recon"]}
