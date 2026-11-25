"""Training loop and scoring: the `task/autoencoder_benign_t32` Trainer (masked MSE, gradient clipping, LR schedule,
early stopping with relative min_delta, best / last checkpoints, resume, per-feature log, curves) around the
depth-configurable ConvAutoencoder.

Only the checkpoint loader (this package's config class + `model.ConvAutoencoder`) and the plot title differ, so the
optimisation is the sibling's code line for line. The T = 10 / 5 packages subclass this Trainer and set `config_cls`
and `title`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from task.autoencoder_benign_t20.config import AutoencoderConfig  # noqa: E402
from task.autoencoder_benign_t20.model import ConvAutoencoder  # noqa: E402
from task.autoencoder_benign_t32 import trainer as benign_t32  # noqa: E402
from task.autoencoder_benign_t32.trainer import EarlyStopping, RunLog, pick_device, seed_everything  # noqa: E402

__all__ = ["EarlyStopping", "RunLog", "Trainer", "pick_device", "seed_everything"]


class Trainer(benign_t32.Trainer):
    """Fits a ConvAutoencoder (`n_down` stride-2 stages) on (x, mask) arrays; scores windows by error and z.

    Class attributes:
        config_cls: settings class used to rebuild a run from its checkpoint.
        title: title of curves.png.
    """

    config_cls = AutoencoderConfig
    title = "One-class autoencoder T = 20 (all/T20_benign): reconstruction loss"

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> "Trainer":
        """Trainer around the model stored in `best.pt` (or `last.pt`)."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        settings = cls.config_cls.from_dict(state["config"])
        model = ConvAutoencoder(settings)
        model.load_state_dict(state["model"])
        return cls(model, settings, pick_device(device))

    @classmethod
    def plot(cls, history: list[dict], best_epoch: int, lr_changes: list[dict], path: Path) -> None:
        """Top: train and validation loss (log scale). Bottom: gradient norm (mean / max) and LR. Best epoch and LR
        drops marked on both."""
        epochs = [h["epoch"] for h in history]
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, height_ratios=(3, 2))
        ax.plot(epochs, [h["train_loss"] for h in history], label="train loss (benign)")
        ax.plot(epochs, [h["val_loss"] for h in history], label="validation loss (benign)")
        ax.set_yscale("log")
        ax.set_ylabel("masked MSE (real rows)")
        ax.set_title(cls.title)
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
