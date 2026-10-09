"""The views of a batch that the objectives train on.

`ReconMasker` picks the rows to hide for reconstruction, `Augmenter` builds physically consistent views for NT-Xent
(S1.3.C1), `ViolationInjector` injects P1-P3 violations for the physics heads (D4). All of them touch real rows only
and draw their randomness from a seeded CPU generator.
"""

from __future__ import annotations

import math

import torch

from .config import PretrainConfig
from .data import FeatureSpace


def uniform(gen: torch.Generator, shape: tuple[int, ...], low: float, high: float) -> torch.Tensor:
    """Uniform samples on CPU from a seeded generator."""
    return torch.rand(shape, generator=gen) * (high - low) + low


class ReconMasker:
    """Hides real rows: per window either exactly `recon_ratio` of them at random or one block of 20-30%."""

    def __init__(self, settings: PretrainConfig) -> None:
        self.ratio, self.block = settings.recon_ratio, settings.block_ratio

    def __call__(self, mask: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """(B, T) bool real rows -> (B, T) bool hidden rows (subset of the real rows; none if < 2 real rows)."""
        device = mask.device
        m = mask.cpu()
        b, t = m.shape
        n = m.sum(1)
        pos = torch.arange(t).expand(b, t)
        k = torch.clamp(torch.round(self.ratio * n).long(), min=1)
        scores = torch.rand(b, t, generator=gen).masked_fill(~m, 2.0)
        rank = scores.argsort(1).argsort(1)
        random_rows = rank < k.unsqueeze(1)
        length = torch.clamp(torch.round(uniform(gen, (b,), *self.block) * n).long(), min=1)
        start = (torch.rand(b, generator=gen) * (n - length + 1).clamp(min=1)).long()
        block_rows = (pos >= start.unsqueeze(1)) & (pos < (start + length).unsqueeze(1))
        use_block = torch.rand(b, generator=gen) < 0.5
        hidden = torch.where(use_block.unsqueeze(1), block_rows, random_rows) & m & (n >= 2).unsqueeze(1)
        return hidden.to(device)


class Augmenter:
    """Physically consistent views (S1.3.C1): common-mode shift, time stretch, contiguous crop."""

    def __init__(self, settings: PretrainConfig, space: FeatureSpace) -> None:
        self.s, self.space = settings, space

    def __call__(self, x: torch.Tensor, mask: torch.Tensor, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, 13) normalised + (B, T) mask -> an augmented view and its mask."""
        s, sp, device = self.s, self.space, x.device
        b, t, _ = x.shape
        raw = sp.raw(x)
        # common-mode shift of sender and receiver positions: range and bearing stay the same
        angle = uniform(gen, (b,), 0.0, 2 * math.pi)
        dist = uniform(gen, (b,), *s.shift_m)
        shift = torch.stack([dist * torch.cos(angle), dist * torch.sin(angle)], 1).to(device).unsqueeze(1)
        raw[..., 0:2] += shift
        raw[..., 2:4] += shift
        # time stretch: v * s, a * s^2, dtau / s (positions unchanged, so dpos = v * dtau still holds)
        k = uniform(gen, (b,), *s.stretch).to(device).view(b, 1, 1)
        raw[..., 4:8] *= k
        raw[..., 8:10] *= k**2
        raw[..., 12:13] -= torch.log(k)
        x_aug = sp.norm(raw, mask)
        # contiguous crop of 80-100% of the real rows (keeps the true time gaps)
        n = mask.sum(1).cpu()
        keep = torch.clamp(torch.ceil(uniform(gen, (b,), *s.crop) * n).long(), min=1)
        keep = torch.minimum(keep, n.clamp(min=1))
        start = (torch.rand(b, generator=gen) * (n - keep + 1).clamp(min=1)).long()
        index = (start.unsqueeze(1) + torch.arange(t)).clamp(max=t - 1).to(device)
        x_aug = x_aug.gather(1, index.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        new_mask = (torch.arange(t).expand(b, t) < keep.unsqueeze(1)).to(device) & (n > 0).to(device).unsqueeze(1)
        return x_aug * new_mask.unsqueeze(-1), new_mask


class ViolationInjector:
    """Injects one physics violation into half of the eligible windows (D4), in raw units, real rows only.

    P1 position jump (claimed position teleports 50-150 m and stays shifted; range and bearing recomputed),
    P2 speed scaled without matching positions (claimed velocity x2-3 or x0.2-0.5),
    P3 impossible sharp turn (claimed velocity and acceleration rotated by 90-180 degrees).
    Windows with fewer than `min_rows_aux` real rows are not eligible and are excluded from the loss.
    """

    def __init__(self, settings: PretrainConfig, space: FeatureSpace) -> None:
        self.s, self.space = settings, space

    def __call__(
        self, x: torch.Tensor, mask: torch.Tensor, gen: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (injected x, targets (B, 3) float, valid (B,) bool)."""
        s, sp, device = self.s, self.space, x.device
        b, t, _ = x.shape
        raw = sp.raw(x)
        n = mask.sum(1).cpu()
        valid = n >= s.min_rows_aux
        speed = (torch.linalg.vector_norm(raw[..., 4:6], dim=-1) * mask).sum(1).cpu() / n.clamp(min=1)
        inject = (torch.rand(b, generator=gen) < s.inject_prob) & valid
        kind = torch.randint(0, 3, (b,), generator=gen)
        kind = torch.where((kind > 0) & (speed < s.min_speed_mps), torch.zeros_like(kind), kind)
        first = 1 + (torch.rand(b, generator=gen) * (n - 2).clamp(min=1)).long()  # rows first..n-1 are changed
        seg = ((torch.arange(t).expand(b, t) >= first.unsqueeze(1)) & mask.cpu()).to(device)
        p1 = (inject & (kind == 0)).to(device).view(b, 1)
        p2 = (inject & (kind == 1)).to(device).view(b, 1)
        p3 = (inject & (kind == 2)).to(device).view(b, 1)
        # P1: jump
        angle = uniform(gen, (b,), 0.0, 2 * math.pi)
        dist = uniform(gen, (b,), *s.jump_m)
        jump = torch.stack([dist * torch.cos(angle), dist * torch.sin(angle)], 1).to(device).unsqueeze(1)
        on1 = (seg & p1).unsqueeze(-1).to(raw.dtype)
        raw[..., 0:2] = raw[..., 0:2] + jump * on1
        rng, bearing = FeatureSpace.range_bearing(raw)
        raw[..., 10] = torch.where(seg & p1, rng, raw[..., 10])
        raw[..., 11] = torch.where(seg & p1, bearing, raw[..., 11])
        # P2: speed scaled
        up = torch.rand(b, generator=gen) < 0.5
        factor = torch.where(up, uniform(gen, (b,), *s.speed_up), uniform(gen, (b,), *s.speed_down))
        factor = factor.to(device).view(b, 1, 1)
        on2 = (seg & p2).unsqueeze(-1)
        raw[..., 4:6] = torch.where(on2, raw[..., 4:6] * factor, raw[..., 4:6])
        # P3: sharp turn
        theta = uniform(gen, (b,), *s.turn_rad) * torch.where(torch.rand(b, generator=gen) < 0.5, -1.0, 1.0)
        c, si = torch.cos(theta).to(device).view(b, 1), torch.sin(theta).to(device).view(b, 1)
        on3 = seg & p3
        for lo in (4, 8):  # claimed velocity and claimed acceleration
            vx, vy = raw[..., lo].clone(), raw[..., lo + 1].clone()
            raw[..., lo] = torch.where(on3, c * vx - si * vy, vx)
            raw[..., lo + 1] = torch.where(on3, si * vx + c * vy, vy)
        targets = torch.stack([inject & (kind == k) for k in range(3)], 1).float().to(device)
        return sp.norm(raw, mask), targets, valid.to(device)
