"""Training loop of the unlabelled autoencoder: the `task/autoencoder_benign_t32` loop (masked MSE, gradient clipping,
ReduceLROnPlateau, early stopping with relative min_delta, best / last checkpoints, resume, logs, curves) on pooled,
unlabelled windows. Inputs are (x, mask) arrays only."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from task.autoencoder_benign_t32 import trainer as base  # noqa: E402
from task.autoencoder_benign_t32.model import ConvAutoencoder, masked_mse  # noqa: E402
from task.autoencoder_benign_t32.trainer import (  # noqa: E402, F401
    EarlyStopping,
    RunLog,
    pick_device,
    rng_state,
    seed_everything,
    set_rng_state,
)
from task.autoencoder_t32.config import AutoencoderT32Config  # noqa: E402


class Trainer(base.Trainer):
    """Fits a ConvAutoencoder on unlabelled (x, mask) arrays; `score` (inherited) gives error, per-feature error, z.

    Attributes:
        model: the autoencoder.
        settings: run settings.
        device: torch device.
    """

    def __init__(self, model: ConvAutoencoder, settings: AutoencoderT32Config, device: torch.device) -> None:
        super().__init__(model, settings, device)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> "Trainer":
        """Trainer around the model stored in `best.pt` (or `last.pt`)."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        settings = AutoencoderT32Config.from_dict(state["config"])
        model = ConvAutoencoder(settings)
        model.load_state_dict(state["model"])
        return cls(model, settings, pick_device(device))

    def fit(
        self, train: tuple[np.ndarray, np.ndarray], val: tuple[np.ndarray, np.ndarray], run_dir: Path, resume=False
    ) -> dict:
        """Trains until early stopping or `max_epochs`; writes history.jsonl, train.log, curves.png, best.pt, last.pt.

        Args:
            train, val: (x (N, T, F) float32, mask (N, T) bool); unlabelled windows, classes mixed.
            run_dir: run folder.
            resume: continue from `run_dir/last.pt` (model, optimizer, scheduler, early stopping, history, RNG).

        Returns:
            summary fields: best_epoch, best_val_loss, epochs_run, stop_reason, lr_changes, train_seconds.
        """
        s = self.settings
        log = RunLog(run_dir / "train.log")
        history_path = run_dir / "history.jsonl"
        xt, mt = self._tensors(*train)
        xv, mv = self._tensors(*val)
        params = list(self.model.parameters())
        opt = torch.optim.Adam(params, lr=s.lr, weight_decay=s.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=s.plateau_factor, patience=s.plateau_patience
        )
        gen = torch.Generator().manual_seed(s.seed)
        stopper = EarlyStopping(s.patience, s.min_delta)
        history, lr_changes, first_epoch, elapsed = [], [], 1, 0.0
        if resume:
            state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            self.model.load_state_dict(state["model"])
            opt.load_state_dict(state["optimizer"])
            sched.load_state_dict(state["scheduler"])
            stopper.load_state_dict(state["early_stopping"])
            history, lr_changes = state["history"], state["lr_changes"]
            first_epoch, elapsed = state["epoch"] + 1, state["train_seconds"]
            set_rng_state(state["rng"], self.device, gen)
            history_path.write_text("".join(json.dumps(r) + "\n" for r in history))
            log(f"resumed from last.pt after epoch {state['epoch']} (best epoch {stopper.best_epoch})")
        stop_reason = "early_stopping" if stopper.stop else "max_epochs"
        start = time.time() - elapsed
        log(
            f"fit: {len(xt):,} unlabelled train windows, val: {len(xv):,} unlabelled windows, batch {s.batch_size}, "
            f"device {self.device}, grad_clip {s.grad_clip:g}, early stopping on val loss "
            f"(min_delta {s.min_delta:g} relative, patience {s.patience})"
        )
        for epoch in range(first_epoch, s.max_epochs + 1):
            if stopper.stop:
                break
            t0 = time.time()
            self.model.train()
            order = torch.randperm(len(xt), generator=gen).to(self.device)
            total = torch.zeros((), device=self.device)
            norm_sum = torch.zeros((), device=self.device)
            norm_max = torch.zeros((), device=self.device)
            n = 0
            for i in range(0, len(order), s.batch_size):
                idx = order[i : i + s.batch_size]
                xb, mb = xt[idx], mt[idx]
                recon, _ = self.model(xb, mb)
                loss = masked_mse(recon, xb, mb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(params, s.grad_clip)  # pre-clip global norm
                opt.step()
                total += loss.detach()
                norm_sum += norm
                norm_max = torch.maximum(norm_max, norm)
                n += 1
            train_loss = float(total) / n
            val_loss = self._val_loss(xv, mv)
            lr = opt.param_groups[0]["lr"]
            sched.step(val_loss)
            new_lr = opt.param_groups[0]["lr"]
            improved = stopper.update(epoch, val_loss)
            if improved:
                torch.save(
                    {
                        "model": copy.deepcopy(self.model.state_dict()),
                        "config": s.to_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                    },
                    run_dir / "best.pt",
                )
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "grad_norm_mean": float(norm_sum) / n,
                "grad_norm_max": float(norm_max),
                "seconds": round(time.time() - t0, 2),
            }
            history.append(row)
            with history_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            mark = " *best" if improved else f" (no improvement {stopper.waited}/{s.patience})"
            log(
                f"epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}  lr {lr:.2e}  "
                f"grad_norm {row['grad_norm_mean']:.4f}/{row['grad_norm_max']:.4f}  {row['seconds']:.1f}s{mark}"
            )
            if new_lr < lr:
                lr_changes.append({"after_epoch": epoch, "from": lr, "to": new_lr})
                log(f"lr reduced {lr:.2e} -> {new_lr:.2e} (validation loss flat for {s.plateau_patience + 1} epochs)")
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "early_stopping": stopper.state_dict(),
                    "history": history,
                    "lr_changes": lr_changes,
                    "epoch": epoch,
                    "train_seconds": time.time() - start,
                    "rng": rng_state(self.device, gen),
                    "config": s.to_dict(),
                },
                run_dir / "last.pt",
            )
            self.plot(history, stopper.best_epoch, lr_changes, run_dir / "curves.png")
            if stopper.stop:
                stop_reason = "early_stopping"
                log(
                    f"early stopping: no improvement for {s.patience} epochs; "
                    f"best epoch {stopper.best_epoch} val {stopper.best:.5f}"
                )
        best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        self.model.load_state_dict(best["model"])
        return {
            "best_epoch": stopper.best_epoch,
            "best_val_loss": stopper.best,
            "epochs_run": len(history),
            "stop_reason": stop_reason,
            "lr_changes": lr_changes,
            "train_seconds": round(time.time() - start, 1),
        }

    @staticmethod
    def plot(history: list[dict], best_epoch: int, lr_changes: list[dict], path: Path) -> None:
        """Top: train and validation loss (log scale). Bottom: gradient norm (mean / max) and LR. Best epoch and LR
        drops marked on both."""
        epochs = [h["epoch"] for h in history]
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, height_ratios=(3, 2))
        ax.plot(epochs, [h["train_loss"] for h in history], label="train loss (unlabelled mix)")
        ax.plot(epochs, [h["val_loss"] for h in history], label="validation loss (unlabelled mix)")
        ax.set_yscale("log")
        ax.set_ylabel("masked MSE (real rows)")
        ax.set_title("Unlabelled autoencoder T = 32 (all/T32_benign_grid): reconstruction loss")
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
