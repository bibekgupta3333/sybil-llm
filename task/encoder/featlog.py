"""Per-feature masked reconstruction error per epoch, train and val (opt-in: `--log-feature-error`).

Train values are accumulated from the training batches during the epoch, with the model in train mode (BatchNorm uses
batch statistics, dropout is on, and the weights change from batch to batch). Val values come from one extra pass over
the full val split in eval mode after the epoch. Both use the definition of `window_errors`: per window, the squared
error of each feature averaged over the window's real rows. `*_window_mean` averages that over windows;
`*_row_pooled` pools the squared error over every real row of the split (its mean over the features equals the
pooled masked MSE of the epoch). One JSON line per epoch goes to `feature_history.jsonl` in the run folder.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

FEATURES = (
    "claimed_pos_x",
    "claimed_pos_y",
    "rx_pos_x",
    "rx_pos_y",
    "claimed_vel_x",
    "claimed_vel_y",
    "rx_vel_x",
    "rx_vel_y",
    "claimed_acl_x",
    "claimed_acl_y",
    "range",
    "bearing",
    "log_dtau",
)
FILE_NAME = "feature_history.jsonl"
TRAIN_MODE_NOTE = "train mode, accumulated from the epoch's training batches while the weights change"
VAL_MODE_NOTE = "eval mode, full val split after the epoch"


class FeatureErrorSum:
    """Running sums of the per-feature squared error over real rows (stays on the device until `result`)."""

    def __init__(self, n_features: int, device: torch.device) -> None:
        self.n_features = n_features
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.window_sum = torch.zeros(self.n_features, device=self.device)
        self.square_sum = torch.zeros(self.n_features, device=self.device)
        self.windows = 0
        self.rows = torch.zeros((), device=self.device)

    @torch.no_grad()
    def add(self, recon: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> None:
        """Adds one batch: recon, x (B, T, F), mask (B, T)."""
        m = mask.unsqueeze(-1).to(recon.dtype)
        square = (recon.detach() - x) ** 2 * m
        rows = m.sum(dim=1)
        self.window_sum += (square.sum(dim=1) / rows.clamp_min(1.0)).sum(dim=0)
        self.square_sum += square.sum(dim=(0, 1))
        self.windows += x.shape[0]
        self.rows += rows.sum()

    def result(self) -> dict[str, list[float]]:
        """{"window_mean": F values, "row_pooled": F values}."""
        return {
            "window_mean": (self.window_sum / max(self.windows, 1)).tolist(),
            "row_pooled": (self.square_sum / self.rows.clamp_min(1.0)).tolist(),
        }


class FeatureErrorLog:
    """Collects the train sums during an epoch, runs the val pass after it and appends one row per epoch.

    Attributes:
        path: `feature_history.jsonl` of the run.
        train: running train sums of the current epoch.
    """

    def __init__(self, run_dir: Path, n_features: int, device: torch.device, resume_epoch: int = 0) -> None:
        """Starts a new file, or keeps the rows up to `resume_epoch` when a run resumes."""
        self.path = Path(run_dir) / FILE_NAME
        self.n_features = n_features
        self.device = device
        self.train = FeatureErrorSum(n_features, device)
        rows = []
        if resume_epoch and self.path.exists():
            rows = [r for r in self.path.read_text().splitlines() if r and json.loads(r)["epoch"] <= resume_epoch]
        self.path.write_text("".join(r + "\n" for r in rows))

    def train_batch(self, recon: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> None:
        """Adds one training batch (call right after the forward pass)."""
        self.train.add(recon, x, mask)

    @torch.no_grad()
    def end_epoch(
        self, epoch: int, model: torch.nn.Module, x_val: torch.Tensor, m_val: torch.Tensor, batch: int
    ) -> dict:
        """Val pass in eval mode, writes the row, clears the train sums. The model is left in eval mode."""
        model.eval()
        val = FeatureErrorSum(self.n_features, self.device)
        for i in range(0, len(x_val), batch):
            xb, mb = x_val[i : i + batch], m_val[i : i + batch]
            val.add(model(xb, mb)[0], xb, mb)
        tr, va = self.train.result(), val.result()
        row = {
            "epoch": epoch,
            "features": list(FEATURES[: self.n_features]),
            "train_window_mean": tr["window_mean"],
            "train_row_pooled": tr["row_pooled"],
            "val_window_mean": va["window_mean"],
            "val_row_pooled": va["row_pooled"],
            "train_windows": self.train.windows,
            "val_windows": val.windows,
            "train_mode": TRAIN_MODE_NOTE,
            "val_mode": VAL_MODE_NOTE,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.train.reset()
        return row
