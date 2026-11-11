"""Training loop and scoring: the `task/autoencoder_benign_t32` Trainer (masked MSE, gradient clipping, LR schedule,
early stopping with relative min_delta, best / last checkpoints, resume, logs) around the T = 48 model.

Only the checkpoint loader (this package's config + the T = 48 ConvAutoencoder) and the plot title differ, so the
optimisation is the sibling's code line for line.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from task.autoencoder_benign_t32 import trainer as benign_t32  # noqa: E402
from task.autoencoder_benign_t32.trainer import EarlyStopping, RunLog, pick_device, seed_everything  # noqa: E402
from task.autoencoder_benign_t48.config import AutoencoderConfig  # noqa: E402
from task.autoencoder_benign_t48.model import ConvAutoencoder  # noqa: E402

__all__ = ["EarlyStopping", "RunLog", "Trainer", "pick_device", "seed_everything"]


class Trainer(benign_t32.Trainer):
    """Fits a T = 48 ConvAutoencoder on (x, mask) arrays and scores windows by reconstruction error and z."""

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> "Trainer":
        """Trainer around the model stored in `best.pt` (or `last.pt`)."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        settings = AutoencoderConfig.from_dict(state["config"])
        model = ConvAutoencoder(settings)
        model.load_state_dict(state["model"])
        return cls(model, settings, pick_device(device))

    @staticmethod
    def plot(history: list[dict], best_epoch: int, lr_changes: list[dict], path: Path) -> None:
        """Top: train and validation loss (log scale). Bottom: gradient norm (mean / max) and LR. Best epoch and LR
        drops marked on both."""
        epochs = [h["epoch"] for h in history]
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, height_ratios=(3, 2))
        ax.plot(epochs, [h["train_loss"] for h in history], label="train loss (benign)")
        ax.plot(epochs, [h["val_loss"] for h in history], label="validation loss (benign)")
        ax.set_yscale("log")
        ax.set_ylabel("masked MSE (real rows)")
        ax.set_title("One-class autoencoder T = 48 (benign_gridsybil/T48_benign): reconstruction loss")
        ax2.plot(epochs, [h["grad_norm_mean"] for h in history], color="tab:green", label="grad norm mean (pre-clip)")
        ax2.plot(epochs, [h["grad_norm_max"] for h in history], color="tab:olive", ls=":", label="grad norm max")
        ax2.set_ylabel("gradient L2 norm")
        ax2.set_xlabel("epoch")
        lr_ax = ax2.twinx()
        lr_ax.plot(epochs, [h["lr"] for h in history], color="tab:purple", label="learning rate")
        lr_ax.set_yscale("log")
        lr_ax.set_ylabel("learning rate")
        for a in (ax, ax2):
            for k, c in enumerate(lr_changes):
                a.axvline(c["after_epoch"] + 0.5, color="grey", ls=":", lw=1, label="LR halved" if k == 0 else None)
            if best_epoch:
                a.axvline(best_epoch, color="red", ls="--", lw=1, label=f"best epoch {best_epoch}")
        ax.legend()
        lines, lr_lines = ax2.get_legend_handles_labels(), lr_ax.get_legend_handles_labels()
        ax2.legend(lines[0] + lr_lines[0], lines[1] + lr_lines[1], fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
