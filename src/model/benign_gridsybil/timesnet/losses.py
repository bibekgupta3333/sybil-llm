"""Pretraining losses.

`nt_xent` is the SimCLR contrastive loss, `LossScales` keeps each loss on a comparable scale (running mean during a
warm-up, then frozen, D7), and `Objective` builds the three views of a batch (masked, clean augmented, injected) and
returns every raw loss plus detached extras for the monitors.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .config import PretrainConfig
from .data import FeatureSpace
from .encoder import MaskedOps
from .heads import PhysicsHeads, PretrainingModel
from .views import Augmenter, ReconMasker, ViolationInjector


def nt_xent(
    p1: torch.Tensor, p2: torch.Tensor, temperature: float, groups: torch.Tensor | None = None
) -> tuple[torch.Tensor, int]:
    """SimCLR NT-Xent on L2-normalised projections (m, d) x2 with in-batch negatives.

    `groups` (m,): windows with the same number are copies / crops of the same broadcasts; they are removed from each
    other's negatives (false negatives), only the true positive pair is kept. Returns (loss, masked pairs).
    """
    m = p1.shape[0]
    reps = torch.cat([p1, p2], 0)
    sim = reps @ reps.T / temperature
    eye = torch.eye(2 * m, dtype=torch.bool, device=sim.device)
    target = torch.cat([torch.arange(m, 2 * m), torch.arange(0, m)]).to(sim.device)
    drop = eye
    masked = 0
    if groups is not None:
        g = torch.cat([groups, groups]).to(sim.device)
        positive = torch.zeros_like(eye)
        positive[torch.arange(2 * m, device=sim.device), target] = True
        same = (g.unsqueeze(0) == g.unsqueeze(1)) & ~eye & ~positive
        masked = int(same.sum()) // 2
        drop = eye | same
    sim = sim.masked_fill(drop, float("-inf"))
    return F.cross_entropy(sim, target), masked


class LossScales:
    """Running mean of each raw loss for the first `warmup` steps, then frozen (comparable scales, D7)."""

    NAMES = ("recon", "nce", "p1", "p2", "p3")

    def __init__(self, warmup: int) -> None:
        self.warmup = warmup
        self.sum = {k: 0.0 for k in self.NAMES}
        self.count = {k: 0 for k in self.NAMES}
        self.frozen = False

    def update(self, raw: dict[str, float], step: int) -> None:
        """Adds the losses present this step; each loss's scale is the mean of its first `warmup` values."""
        for k, v in raw.items():
            if math.isfinite(v) and self.count[k] < self.warmup:
                self.sum[k] += v
                self.count[k] += 1
        seen = [k for k in self.NAMES if self.count[k] > 0]
        self.frozen = bool(seen) and all(self.count[k] >= self.warmup for k in seen)

    def scale(self, name: str) -> float:
        """Divisor for loss `name` (1 before any update)."""
        c = self.count[name]
        return max(self.sum[name] / c, 1e-8) if c else 1.0

    def state(self) -> dict[str, Any]:
        """Checkpointable state."""
        return {"sum": self.sum, "count": self.count, "frozen": self.frozen, "warmup": self.warmup}

    def load(self, state: dict[str, Any]) -> None:
        """Restores a state from `state()`."""
        self.sum, self.count, self.frozen = state["sum"], state["count"], state["frozen"]


class Objective:
    """Builds the three views of a batch and computes every pretraining loss."""

    def __init__(self, settings: PretrainConfig, space: FeatureSpace) -> None:
        self.s = settings
        self.masker = ReconMasker(settings)
        self.augment = Augmenter(settings, space)
        self.inject = ViolationInjector(settings, space)

    def weights(self) -> dict[str, float]:
        """lambda per loss name."""
        s = self.s
        return {"recon": s.lambda_recon, "nce": s.lambda_nce, "p1": s.lambda_p1, "p2": s.lambda_p2, "p3": s.lambda_p3}

    def __call__(
        self,
        model: PretrainingModel,
        x: torch.Tensor,
        mask: torch.Tensor,
        gen: torch.Generator,
        groups: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Returns raw losses (tensors) and extras for monitoring (detached).

        A loss is only returned when the batch has windows it applies to: with length-bucketed batches, a batch of
        very short windows can have no hidden rows (recon), fewer than two long-enough windows (NT-Xent) or no
        injectable window (P1-P3). Absent losses are skipped, never counted as 0.
        """
        s = self.s
        losses: dict[str, torch.Tensor] = {}
        extra: dict[str, Any] = {}
        v1, m1 = self.augment(x, mask, gen)
        v2, m2 = self.augment(x, mask, gen)
        hidden = self.masker(m1, gen)
        if s.use_recon or s.use_nce:
            h1, z1 = model.encoder(model.hide(v1, hidden), m1)
            if s.use_recon and bool(hidden.any()):  # a batch of 1-row windows has nothing to hide
                pred = model.recon_head(h1)
                losses["recon"] = MaskedOps.masked_mse(pred, v1, hidden)
                rows = hidden.unsqueeze(-1).to(pred.dtype)
                extra["recon_per_feature"] = (((pred - v1) ** 2) * rows).sum((0, 1)).detach() / rows.sum().clamp(min=1)
                extra["pred"], extra["target"], extra["hidden"], extra["mask1"] = pred.detach(), v1.detach(), hidden, m1
            extra["z1"] = z1.detach()
        if s.use_nce:
            _, z2 = model.encoder(v2, m2)
            keep = (m1.sum(1) >= s.min_rows_aux) & (m2.sum(1) >= s.min_rows_aux)
            p1, p2 = model.proj_head(z1[keep]), model.proj_head(z2[keep])
            extra["proj"] = (p1.detach(), p2.detach())
            g = groups[keep.cpu()] if (groups is not None and s.mask_same_broadcast) else None
            if keep.sum() >= 2:
                losses["nce"], extra["nce_masked_pairs"] = nt_xent(p1, p2, s.temperature, g)
            else:  # too few windows with >= min_rows_aux real rows: no contrastive loss this batch
                extra["nce_masked_pairs"] = 0
        if s.use_physics:
            xi, targets, valid = self.inject(x, mask, gen)
            _, zi = model.encoder(xi, mask)
            logits = model.physics_heads(zi)
            if bool(valid.any()):  # windows with < min_rows_aux real rows are never injected or scored
                for j, name in enumerate(("p1", "p2", "p3")):
                    key = PhysicsHeads.NAMES[j]
                    losses[name] = F.binary_cross_entropy_with_logits(logits[key][valid], targets[valid, j])
            extra["physics"] = ({k: v.detach() for k, v in logits.items()}, targets, valid)
        return losses, extra

    def joint(self, losses: dict[str, torch.Tensor], scales: LossScales) -> torch.Tensor:
        """L = sum_i lambda_i * L_i / scale_i over the objectives that are on."""
        weight = self.weights()
        return sum(weight[k] * v / scales.scale(k) for k, v in losses.items())
