"""Training loop and scoring: the `task/autoencoder_st_t32` Trainer (BCE with pos_weight + recon_weight x masked MSE,
gradient clipping, ReduceLROnPlateau and early stopping on val AUROC, best / last checkpoints, resume, per-feature log,
curves, `predict`) around the supervised autoencoder with the depth-configurable ConvAutoencoder.

Only the checkpoint loader (this package's config class + model) and the plot title differ, so the optimisation is
the sibling's code line for line. The T = 10 / 5 packages subclass this Trainer and set `config_cls` and `title`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from task.autoencoder_st_t20.config import SupervisedAEConfig  # noqa: E402
from task.autoencoder_st_t20.model import SupervisedAutoencoder  # noqa: E402
from task.autoencoder_st_t32 import trainer as st_t32  # noqa: E402
from task.autoencoder_st_t32.trainer import EarlyStopping, RunLog, pick_device, seed_everything  # noqa: E402

__all__ = ["EarlyStopping", "RunLog", "Trainer", "pick_device", "seed_everything"]


class Trainer(st_t32.Trainer):
    """Fits a SupervisedAutoencoder (`n_down` stride-2 stages) on (x, mask, y) arrays and scores windows.

    Class attributes:
        config_cls: settings class used to rebuild a run from its checkpoint.
        title: title of curves.png.
    """

    config_cls = SupervisedAEConfig
    title = "Supervised autoencoder T = 20 (all/T20_benign_grid): BCE + recon_weight x masked MSE"

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> "Trainer":
        """Trainer around the model stored in `best.pt` (or `last.pt`)."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        settings = cls.config_cls.from_dict(state["config"])
        model = SupervisedAutoencoder(settings)
        model.load_state_dict(state["model"])
        return cls(model, settings, pick_device(device), state["pos_weight"])

    @classmethod
    def plot(cls, history: list[dict], best_epoch: int, lr_changes: list[dict], path: Path) -> None:
        """Losses (train vs val), val AUROC / PR-AUC / F1, and LR + gradient norm; best epoch marked."""
        epochs = [h["epoch"] for h in history]
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        for key, color in (("loss", "tab:blue"), ("bce", "tab:orange"), ("recon", "tab:green")):
            ax1.plot(epochs, [h[f"train_{key}"] for h in history], color=color, ls="--", label=f"train {key}")
            ax1.plot(epochs, [h[f"val_{key}"] for h in history], color=color, label=f"val {key}")
        ax1.set_yscale("log")
        ax1.set_ylabel("loss")
        ax1.set_title(cls.title)
        ax1.legend(fontsize=8, ncol=3)
        for key, label in (("auroc", "val AUROC"), ("pr_auc", "val PR-AUC"), ("f1", "val F1 @ 0.5")):
            ax2.plot(epochs, [h[f"val_{key}"] for h in history], label=label)
        ax2.plot(epochs, [h["train_auroc"] for h in history], color="grey", ls="--", label="train AUROC (running)")
        ax2.set_ylabel("metric")
        ax2.legend(fontsize=8)
        ax3.plot(epochs, [h["grad_norm_mean"] for h in history], color="tab:green", label="grad norm mean (pre-clip)")
        ax3.plot(epochs, [h["grad_norm_max"] for h in history], color="tab:olive", ls=":", label="grad norm max")
        ax3.set_ylabel("gradient L2 norm")
        ax3.set_xlabel("epoch")
        lr_ax = ax3.twinx()
        lr_ax.plot(epochs, [h["lr"] for h in history], color="tab:purple", label="learning rate")
        lr_ax.set_yscale("log")
        lr_ax.set_ylabel("learning rate")
        for a in (ax1, ax2, ax3):
            for k, c in enumerate(lr_changes):
                a.axvline(c["after_epoch"] + 0.5, color="grey", ls=":", lw=1, label="LR halved" if k == 0 else None)
            if best_epoch:
                a.axvline(best_epoch, color="red", ls="--", lw=1, label=f"best epoch {best_epoch}")
        lines, lr_lines = ax3.get_legend_handles_labels(), lr_ax.get_legend_handles_labels()
        ax3.legend(lines[0] + lr_lines[0], lines[1] + lr_lines[1], fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
