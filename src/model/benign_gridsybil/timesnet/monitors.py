"""Label-free monitors computed on the pretrain check set.

Small numpy metrics (linear-interpolation baseline for reconstruction, effective rank, ridge R^2, AUROC) and the
`Evaluator`, which runs the objective on a fixed subset with fixed views and reports losses, collapse monitors,
length-shortcut monitors and P-head AUROC. No attack label is ever read.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .config import PretrainConfig
from .data import FeatureShards
from .heads import PhysicsHeads, PretrainingModel
from .losses import LossScales, Objective


def interpolation_baseline(target: np.ndarray, hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-feature MSE of filling each hidden row by linear interpolation between the nearest visible real rows."""
    total, count = np.zeros(target.shape[-1]), 0
    for x, hid, real in zip(target, hidden, mask):
        rows = np.flatnonzero(real)
        if rows.size == 0 or not hid.any():
            continue
        visible = rows[~hid[rows]]
        for r in np.flatnonzero(hid):
            if visible.size == 0:
                guess = np.zeros(x.shape[-1])
            else:
                before, after = visible[visible < r], visible[visible > r]
                if before.size and after.size:
                    a, c = before[-1], after[0]
                    w = (r - a) / (c - a)
                    guess = (1 - w) * x[a] + w * x[c]
                else:
                    guess = x[before[-1]] if before.size else x[after[0]]
            total += (x[r] - guess) ** 2
            count += 1
    return total / max(count, 1)


def effective_rank(z: np.ndarray) -> float:
    """exp(entropy) of the normalised singular values of centred z (1 = collapsed, d = full)."""
    sv = np.linalg.svd(z - z.mean(0), compute_uv=False)
    p = sv / max(sv.sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def ridge_r2(z: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> float:
    """Held-out R^2 of a ridge regression y ~ z (first half fit, second half test)."""
    half = len(y) // 2
    a, b = z[:half], z[half:]
    mu, sd = a.mean(0), a.std(0) + 1e-6
    a, b = (a - mu) / sd, (b - mu) / sd
    a1 = np.hstack([a, np.ones((len(a), 1))])
    w = np.linalg.solve(a1.T @ a1 + alpha * np.eye(a1.shape[1]), a1.T @ y[:half])
    pred = np.hstack([b, np.ones((len(b), 1))]) @ w
    resid = ((y[half:] - pred) ** 2).sum()
    return float(1 - resid / max(((y[half:] - y[half:].mean()) ** 2).sum(), 1e-12))


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based AUROC (NaN if one class is missing)."""
    pos, neg = label == 1, label == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = np.empty(len(score))
    ranks[np.argsort(score, kind="stable")] = np.arange(1, len(score) + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


class Evaluator:
    """Fixed pretrain check subset with fixed views; every number is label-free."""

    def __init__(self, settings: PretrainConfig, data: FeatureShards, objective: Objective, device: torch.device):
        self.s, self.objective, self.device = settings, objective, device
        gen = torch.Generator().manual_seed(settings.seed + 7)
        idx = torch.randperm(len(data), generator=gen)[: settings.eval_windows]
        self.x, self.mask = data.x[idx], data.mask[idx]
        self.groups = data.broadcast_groups()[idx]
        self.baseline: np.ndarray | None = None

    @torch.no_grad()
    def __call__(self, model: PretrainingModel, scales: LossScales) -> dict[str, Any]:
        """Runs the model over the check subset and returns every monitor (val_* losses, collapse, length, AUROC)."""
        s = self.s
        model.eval()
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}  # batches in which each loss was present
        batches = 0
        feat_sum = torch.zeros(s.n_features)
        z_clean, n_real, proj1, proj2, ph_logits, ph_targets, ph_valid = [], [], [], [], [], [], []
        targets_np, hidden_np, mask_np = [], [], []
        for start in range(0, len(self.x), s.eval_batch_size):
            x = self.x[start : start + s.eval_batch_size].to(self.device)
            mask = self.mask[start : start + s.eval_batch_size].to(self.device)
            gen = torch.Generator().manual_seed(s.seed * 1_000_003 + 999_983 + start)  # same views every eval
            groups = self.groups[start : start + s.eval_batch_size]
            losses, extra = self.objective(model, x, mask, gen, groups)
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1
            batches += 1
            if "recon_per_feature" in extra:
                feat_sum += extra["recon_per_feature"].cpu()
                if self.baseline is None:
                    targets_np.append(extra["target"].cpu().numpy())
                    hidden_np.append(extra["hidden"].cpu().numpy())
                    mask_np.append(extra["mask1"].cpu().numpy())
            _, z = model.encoder(x, mask)
            z_clean.append(z.cpu().numpy())
            n_real.append(mask.sum(1).cpu().numpy())
            if "proj" in extra:
                proj1.append(extra["proj"][0].cpu().numpy())
                proj2.append(extra["proj"][1].cpu().numpy())
            if "physics" in extra:
                logits, targets, valid = extra["physics"]
                ph_logits.append(np.stack([logits[k].cpu().numpy() for k in PhysicsHeads.NAMES], 1))
                ph_targets.append(targets.cpu().numpy())
                ph_valid.append(valid.cpu().numpy())
        out: dict[str, Any] = {f"val_{k}": v / counts[k] for k, v in sums.items()}
        weight = self.objective.weights()
        out["val_joint"] = float(sum(weight[k] * (v / counts[k]) / scales.scale(k) for k, v in sums.items()))
        out["scales_frozen"] = scales.frozen
        if s.use_recon:
            out["val_recon_per_feature"] = (feat_sum / max(counts.get("recon", 0), 1)).tolist()
            if self.baseline is None and targets_np:
                self.baseline = interpolation_baseline(
                    np.concatenate(targets_np), np.concatenate(hidden_np), np.concatenate(mask_np)
                )
            if self.baseline is not None:
                out["val_recon_interp_baseline_per_feature"] = self.baseline.tolist()
                out["val_recon_interp_baseline"] = float(self.baseline.mean())
        z = np.concatenate(z_clean)
        n = np.concatenate(n_real).astype(float)
        out["z_std_mean"], out["z_std_min"] = float(z.std(0).mean()), float(z.std(0).min())
        out["z_effective_rank"] = effective_rank(z)
        out["length_probe_r2"] = ridge_r2(z, n)
        out["corr_znorm_length"] = float(np.corrcoef(np.linalg.norm(z, axis=1), n)[0, 1])
        if proj1:
            a, b = np.concatenate(proj1), np.concatenate(proj2)
            out["align"] = float(((a - b) ** 2).sum(1).mean())
            m = min(len(a), 2048)
            sa, sb = a[:m], b[:m]
            d2 = np.clip(2.0 - 2.0 * (sa @ sa.T), 0.0, None)  # unit vectors: |u - v|^2 = 2 - 2 u.v
            off = ~np.eye(m, dtype=bool)
            out["uniformity"] = float(np.log(np.exp(-2.0 * d2[off]).mean()))
            out["cos_pos"] = float((a * b).sum(1).mean())
            out["cos_neg"] = float((sa @ sb.T)[off].mean())
        if ph_logits:
            lg, tg, vd = np.concatenate(ph_logits), np.concatenate(ph_targets), np.concatenate(ph_valid)
            for j, name in enumerate(("p1", "p2", "p3")):
                out[f"auroc_{name}"] = auroc(lg[vd, j], tg[vd, j])
        model.train()
        return out
