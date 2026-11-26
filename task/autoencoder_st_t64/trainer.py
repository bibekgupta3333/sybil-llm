"""The `task/autoencoder_st` training loop (class-weighted BCE + masked MSE, gradient clipping, LR schedule and early
stopping on val AUROC, best / last checkpoints, resume, logs, curves) around the T = 64 model.

Only two things differ from `task.autoencoder_st.trainer.Trainer`: `from_checkpoint` rebuilds the T = 64 model and
config, and the curve title names T = 64.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from task.autoencoder_st import trainer as st_trainer
from task.autoencoder_st.trainer import EarlyStopping, RunLog, pick_device, seed_everything
from task.autoencoder_st_t64.config import SupervisedAET64Config
from task.autoencoder_st_t64.model import SupervisedAutoencoder

__all__ = ["EarlyStopping", "RunLog", "Trainer", "pick_device", "seed_everything"]

TITLE = "Supervised autoencoder T = 64 (benign_gridsybil/T64): BCE + recon_weight x masked MSE"


class Trainer(st_trainer.Trainer):
    """`task.autoencoder_st.trainer.Trainer` for the T = 64 model."""

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> "Trainer":
        """Trainer around the model stored in `best.pt` (or `last.pt`)."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        settings = SupervisedAET64Config.from_dict(state["config"])
        model = SupervisedAutoencoder(settings)
        model.load_state_dict(state["model"])
        return cls(model, settings, pick_device(device), state["pos_weight"])

    @staticmethod
    def plot(history: list[dict], best_epoch: int, lr_changes: list[dict], path: Path) -> None:
        """The parent's curves with the T = 64 title (the parent saves and closes the figure)."""
        original = plt.Figure.savefig

        def savefig(fig, *args, **kwargs):
            fig.axes[0].set_title(TITLE)
            return original(fig, *args, **kwargs)

        plt.Figure.savefig = savefig
        try:
            st_trainer.Trainer.plot(history, best_epoch, lr_changes, path)
        finally:
            plt.Figure.savefig = original
