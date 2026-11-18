"""Label-free window scores of a trained masked autoencoder.

Three scores per window, all over real cells only (padding never scored):
  * masked_recon: masked-reconstruction error. The T x 13 cells are partitioned once into `n_eval_draws` equal groups by
    a fixed RNG (`EvalMasks`, seed `EVAL_MASK_SEED`, the same for every window, run and seed); in draw k the cells of
    group k are hidden (zeroed in the input) and their reconstruction error is kept. Every real cell is hidden in exactly
    one draw, so the score is the mean over the window's real cells of the squared error of a cell reconstructed while
    hidden. With p = 0.2 there are 5 draws, each hiding 20% of the cells, as in training.
  * full_recon: reconstruction error with nothing hidden (widedeep's eval mode, mask of ones), mean over real cells.
  * mahalanobis: distance of z (clean input) to a Gaussian fitted on benign train z.

Mahalanobis, per-track aggregation and quantile thresholds are imported from `task.autoencoder_benign_t32.scoring`.
"""

from __future__ import annotations

import numpy as np
import torch

from task.autoencoder_benign_t32.scoring import Mahalanobis, TrackScores, quantile_thresholds
from task.masked_autoencoder.model import MaskedTabularAutoencoder, masked_cell_mse

__all__ = ["SCORE_NAMES", "EvalMasks", "Mahalanobis", "TrackScores", "WindowScorer", "quantile_thresholds"]

SCORE_NAMES = ("masked_recon", "full_recon", "mahalanobis")


class EvalMasks:
    """Fixed partition of the T x F cells into `n_draws` equal groups (deterministic scoring masks).

    Attributes:
        groups: (T, F) int, group index of every cell.
    """

    def __init__(self, seq_len: int, n_features: int, n_draws: int, seed: int) -> None:
        n = seq_len * n_features
        order = np.random.default_rng(seed).permutation(n)
        groups = np.empty(n, dtype=np.int64)
        groups[order] = np.arange(n) % n_draws
        self.groups = groups.reshape(seq_len, n_features)
        self.n_draws = n_draws

    def draw(self, k: int) -> np.ndarray:
        """(T, F) bool: cells hidden in draw k."""
        return self.groups == k


class WindowScorer:
    """Scores windows with a trained model in eval mode (dropout off, BatchNorm running statistics)."""

    def __init__(self, model: MaskedTabularAutoencoder, masks: EvalMasks, device: torch.device, batch: int) -> None:
        self.model, self.masks, self.device, self.batch = model, masks, device, batch
        self._draws = [torch.from_numpy(masks.draw(k)).to(device) for k in range(masks.n_draws)]

    @torch.no_grad()
    def batch_scores(self, xb: torch.Tensor, mb: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-window masked-recon SSE / count, full-recon SSE / count and z of one batch."""
        real = mb.unsqueeze(-1).expand_as(xb)
        sse_m = torch.zeros(len(xb), device=self.device)
        cnt_m = torch.zeros(len(xb), device=self.device)
        for draw in self._draws:
            hidden = draw.unsqueeze(0) & real
            recon, _ = self.model(xb, mb, hidden)
            sse, cnt = masked_cell_mse(xb, recon, hidden)
            sse_m += sse
            cnt_m += cnt
        recon, z = self.model(xb, mb, None)
        sse_f, cnt_f = masked_cell_mse(xb, recon, real)
        return {"sse_masked": sse_m, "cnt_masked": cnt_m, "sse_full": sse_f, "cnt_full": cnt_f, "z": z}

    @torch.no_grad()
    def __call__(self, x: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        """{"masked_recon": (N,), "full_recon": (N,), "z": (N, d)} (float64 errors)."""
        self.model.eval()
        out = {"masked_recon": [], "full_recon": [], "z": []}
        for i in range(0, len(x), self.batch):
            xb = torch.from_numpy(np.ascontiguousarray(x[i : i + self.batch])).to(self.device)
            mb = torch.from_numpy(np.ascontiguousarray(mask[i : i + self.batch])).to(self.device)
            s = self.batch_scores(xb, mb)
            if not torch.equal(s["cnt_masked"], s["cnt_full"]):
                raise AssertionError("scoring masks did not hide every real cell exactly once")
            out["masked_recon"].append((s["sse_masked"] / s["cnt_masked"].clamp_min(1.0)).cpu().double().numpy())
            out["full_recon"].append((s["sse_full"] / s["cnt_full"].clamp_min(1.0)).cpu().double().numpy())
            out["z"].append(s["z"].float().cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}

    @torch.no_grad()
    def pooled(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
        """Masked / full reconstruction MSE pooled over every real cell of a set (validation loss)."""
        self.model.eval()
        tot = {"sse_masked": 0.0, "cnt_masked": 0.0, "sse_full": 0.0, "cnt_full": 0.0}
        for i in range(0, len(x), self.batch):
            s = self.batch_scores(x[i : i + self.batch], mask[i : i + self.batch])
            for k in tot:
                tot[k] += float(s[k].sum())
        return {
            "masked_mse": tot["sse_masked"] / max(tot["cnt_masked"], 1.0),
            "full_mse": tot["sse_full"] / max(tot["cnt_full"], 1.0),
        }
