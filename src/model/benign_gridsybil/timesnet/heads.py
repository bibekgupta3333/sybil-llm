"""The three pretraining heads the plan asks for, and the model that bundles them with the encoder.

Reconstruction (MLP on H), physics heads P1-P3 (binary, on z, detect injected violations, D4) and the SimCLR
projection (on z). All hidden widths are d (no bottleneck, D10). `PretrainingModel` = TimesNet encoder + learned mask
token + the heads; only the encoder is kept after pretraining.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PretrainConfig
from .encoder import TimesNetEncoder


class ReconstructionHead(nn.Module):
    """2-layer MLP decoder from per-message vectors H back to the 13 input features."""

    def __init__(self, d_model: int, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_features))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(B, T, d) -> (B, T, n_features)."""
        return self.net(h)


class PhysicsHeads(nn.Module):
    """Three binary heads on z, one per injected-violation type (P1-P3, D4); hidden width d (no bottleneck, D10)."""

    NAMES = ("p1_speed_jump", "p2_speed_scaled", "p3_sharp_turn")

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {name: nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)) for name in self.NAMES}
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """(B, d) -> {name: (B,) logit}."""
        return {name: head(z).squeeze(-1) for name, head in self.heads.items()}


class ProjectionHead(nn.Module):
    """SimCLR projection d -> d -> proj_dim, L2-normalised (discarded after pretraining)."""

    def __init__(self, d_model: int, proj_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, proj_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, d) -> (B, proj_dim) with unit L2 norm per row."""
        return F.normalize(self.net(z), dim=-1)


class PretrainingModel(nn.Module):
    """TimesNet encoder + learned mask token + the three pretraining heads (heads are dropped after pretraining)."""

    def __init__(self, settings: PretrainConfig) -> None:
        super().__init__()
        self.encoder = TimesNetEncoder(settings)
        self.mask_token = nn.Parameter(torch.zeros(settings.n_features))  # replaces hidden rows (S1.3.A2)
        self.recon_head = ReconstructionHead(settings.d_model, settings.n_features)
        self.physics_heads = PhysicsHeads(settings.d_model)
        self.proj_head = ProjectionHead(settings.d_model, settings.proj_dim)

    def hide(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Replaces the hidden real rows with the learned mask token."""
        return torch.where(hidden.unsqueeze(-1), self.mask_token.to(x.dtype).expand_as(x), x)
