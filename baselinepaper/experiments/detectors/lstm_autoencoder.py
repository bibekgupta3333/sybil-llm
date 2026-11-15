"""LSTM autoencoder (`lstmae`): a recurrent reference of matched size on the same input as `convae`.

Not a replication of a published model. Encoder: one LSTM layer over the rows (13 features with padding rows zeroed
+ the mask channel, as in the convolutional model); its hidden state at the last real row - what a packed sequence
would return - goes through a linear layer to z (16). Decoder: z repeated at every row -> one LSTM layer -> linear
-> 13 features. The loss is the masked MSE over real rows. Both LSTMs run forwards in time and the decoder never
sees x, so neither z nor the reconstruction of a real row depends on the padding rows.

Optimiser, batch size, LR schedule, gradient clipping and early-stopping rule are those of `convae`; the only
differences are the architecture and the epoch cap (`MAX_EPOCHS`, see `05-protocol/hyperparameters-neural.md`).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from baselinepaper.experiments import config as cfg
from baselinepaper.experiments.detectors.base import register
from baselinepaper.experiments.detectors.neural_common import (
    NeuralDetector,
    NeuralSettings,
    Objective,
    ReconstructionObjective,
    count_params,
)
from task.autoencoder_benign_t32.config import AutoencoderConfig

TASK_CONFIG = AutoencoderConfig()
HIDDEN = 80
LATENT = TASK_CONFIG.latent_dim
MAX_EPOCHS = 100  # wall-clock cap, 05-protocol/hyperparameters-neural.md section 5 (convae: 300)
TOTAL_PARAMS = 64_429
RECON = "recon"


class LSTMAutoencoder(nn.Module):
    """`forward(x (B, T, F), mask (B, T))` returns (reconstruction (B, T, F), z (B, latent))."""

    def __init__(self, n_features: int = cfg.N_FEATURES, hidden: int = HIDDEN, latent: int = LATENT) -> None:
        super().__init__()
        self.encoder = nn.LSTM(n_features + 1, hidden, batch_first=True)
        self.to_z = nn.Linear(hidden, latent)
        self.decoder = nn.LSTM(latent, hidden, batch_first=True)
        self.to_x = nn.Linear(hidden, n_features)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """z (B, latent) from the encoder state at the last real row of each window."""
        m = mask.unsqueeze(-1).to(x.dtype)
        states, _ = self.encoder(torch.cat([x * m, m], dim=-1))
        last = mask.sum(dim=1).clamp_min(1) - 1
        return self.to_z(states[torch.arange(len(x), device=x.device), last])

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Reconstruction (B, T, F)."""
        states, _ = self.decoder(z.unsqueeze(1).expand(-1, seq_len, -1).contiguous())
        return self.to_x(states)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, mask)
        return self.decode(z, x.shape[1]), z


@register
class LSTMAutoencoderDetector(NeuralDetector):
    """LSTM autoencoder trained on the regime's pool; score = masked reconstruction error."""

    name = "lstmae"
    regimes = cfg.UNLABELLED_REGIMES
    score_names = (RECON,)
    reused_modules = ("task.autoencoder_benign_t32",)
    settings = NeuralSettings.from_task_config(TASK_CONFIG, max_epochs=MAX_EPOCHS)

    def build_model(self) -> nn.Module:
        model = LSTMAutoencoder()
        if count_params(model) != TOTAL_PARAMS:
            raise RuntimeError(f"LSTMAutoencoder has {count_params(model):,} parameters, expected {TOTAL_PARAMS:,}")
        return model

    def objective(self, train) -> Objective:
        return ReconstructionObjective()

    def architecture(self) -> dict:
        return {
            "model": "baselinepaper.experiments.detectors.lstm_autoencoder.LSTMAutoencoder",
            "input": "32 x 14 (13 features with padding rows zeroed + mask channel)",
            "encoder": f"LSTM 14 -> {HIDDEN} (1 layer), state at the last real row -> Linear {HIDDEN} -> {LATENT}",
            "decoder": f"z repeated 32 times -> LSTM {LATENT} -> {HIDDEN} (1 layer) -> Linear {HIDDEN} -> 13",
            "latent_dim": LATENT,
            "parameters": TOTAL_PARAMS,
            "loss": "masked MSE over real rows",
            "model_selection": "lowest validation loss on the regime's validation pool",
        }

    def score(self, x: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        return {RECON: self.forward(x, mask)["recon"]}
