"""Training loop (early stopping with min_delta, LR schedule, gradient norm, best checkpoint, logs) and scoring."""

from __future__ import annotations

import copy
import datetime as dt
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from task.autoencoder_overlap.config import AutoencoderConfig  # noqa: E402
from task.autoencoder_overlap.model import ConvAutoencoder, masked_mse, window_errors  # noqa: E402


def pick_device(name: str) -> torch.device:
    """`auto` -> mps, then cuda, then cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def grad_norm(module: torch.nn.Module) -> torch.Tensor:
    """L2 norm of all parameter gradients together (stays on the device)."""
    norms = [p.grad.detach().norm(2) for p in module.parameters() if p.grad is not None]
    return torch.linalg.vector_norm(torch.stack(norms), 2)


class RunLog:
    """Timestamped lines to stdout and to `train.log` (written as the run goes)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        with self.path.open("a") as f:
            f.write(line + "\n")


class EarlyStopping:
    """Counts epochs without a relative improvement of at least `min_delta` on the validation loss.

    Attributes:
        best: best validation loss so far.
        best_epoch: epoch of `best` (0 before the first update).
        waited: epochs since the last improvement.
    """

    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_epoch = 0
        self.waited = 0

    def update(self, epoch: int, loss: float) -> bool:
        """Records one epoch; True if it counts as an improvement (loss < best * (1 - min_delta))."""
        if loss < self.best * (1.0 - self.min_delta):
            self.best, self.best_epoch, self.waited = loss, epoch, 0
            return True
        self.waited += 1
        return False

    @property
    def stop(self) -> bool:
        return self.waited >= self.patience


class Trainer:
    """Fits a ConvAutoencoder on (x, mask) arrays and scores windows by reconstruction error and z.

    Attributes:
        model: the autoencoder.
        settings: run settings.
        device: torch device.
    """

    def __init__(self, model: ConvAutoencoder, settings: AutoencoderConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.settings = settings
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> "Trainer":
        """Trainer around the model stored in `best.pt`."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        settings = AutoencoderConfig.from_dict(state["config"])
        model = ConvAutoencoder(settings)
        model.load_state_dict(state["model"])
        return cls(model, settings, pick_device(device))

    def _tensors(self, x: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(np.ascontiguousarray(x)).to(self.device), torch.from_numpy(mask).to(self.device)

    @torch.no_grad()
    def _val_loss(self, x: torch.Tensor, m: torch.Tensor) -> float:
        """Masked MSE pooled over every real row of the validation set (eval mode)."""
        self.model.eval()
        sq, rows = 0.0, 0.0
        for i in range(0, len(x), self.settings.eval_batch):
            xb, mb = x[i : i + self.settings.eval_batch], m[i : i + self.settings.eval_batch]
            recon, _ = self.model(xb, mb)
            w = mb.unsqueeze(-1).float()
            sq += float(((recon - xb) ** 2 * w).sum())
            rows += float(w.sum()) * xb.shape[-1]
        return sq / max(rows, 1.0)

    def fit(self, train: tuple[np.ndarray, np.ndarray], val: tuple[np.ndarray, np.ndarray], run_dir: Path) -> dict:
        """Trains until early stopping or `max_epochs`; writes history.jsonl, train.log, loss_curve.png, best.pt.

        Returns:
            summary fields: best_epoch, best_val_loss, epochs_run, stop_reason, lr_changes, train_seconds.
        """
        s = self.settings
        log = RunLog(run_dir / "train.log")
        history_path = run_dir / "history.jsonl"
        xt, mt = self._tensors(*train)
        xv, mv = self._tensors(*val)
        opt = torch.optim.Adam(self.model.parameters(), lr=s.lr, weight_decay=s.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=s.plateau_factor, patience=s.plateau_patience
        )
        gen = torch.Generator().manual_seed(s.seed)
        stopper = EarlyStopping(s.patience, s.min_delta)
        best_state = None
        history, lr_changes, stop_reason = [], [], "max_epochs"
        start = time.time()
        log(
            f"fit: {len(xt):,} windows, val: {len(xv):,} windows, batch {s.batch_size}, device {self.device}, "
            f"min_delta {s.min_delta:g} (relative), patience {s.patience}"
        )
        for epoch in range(1, s.max_epochs + 1):
            t0 = time.time()
            self.model.train()
            order = torch.randperm(len(xt), generator=gen).to(self.device)
            total = torch.zeros((), device=self.device)
            total_norm = torch.zeros((), device=self.device)
            n = 0
            for i in range(0, len(order), s.batch_size):
                idx = order[i : i + s.batch_size]
                xb, mb = xt[idx], mt[idx]
                recon, _ = self.model(xb, mb)
                loss = masked_mse(recon, xb, mb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                total_norm += grad_norm(self.model)
                opt.step()
                total += loss.detach()
                n += 1
            train_loss = float(total) / n
            mean_grad_norm = float(total_norm) / n
            val_loss = self._val_loss(xv, mv)
            lr = opt.param_groups[0]["lr"]
            sched.step(val_loss)
            new_lr = opt.param_groups[0]["lr"]
            improved = stopper.update(epoch, val_loss)
            if improved:
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save(
                    {"model": best_state, "config": s.to_dict(), "epoch": epoch, "val_loss": val_loss},
                    run_dir / "best.pt",
                )
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "grad_norm": mean_grad_norm,
                "seconds": round(time.time() - t0, 2),
            }
            history.append(row)
            with history_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            mark = " *best" if improved else f" (no improvement {stopper.waited}/{s.patience})"
            log(
                f"epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}  lr {lr:.2e}  "
                f"grad_norm {mean_grad_norm:.4f}  {row['seconds']:.1f}s{mark}"
            )
            if new_lr < lr:
                lr_changes.append({"after_epoch": epoch, "from": lr, "to": new_lr})
                log(f"lr reduced {lr:.2e} -> {new_lr:.2e} (validation loss flat for {s.plateau_patience + 1} epochs)")
            self.plot(history, stopper.best_epoch, lr_changes, run_dir / "loss_curve.png")
            if stopper.stop:
                stop_reason = "early_stopping"
                log(
                    f"early stopping: no improvement for {s.patience} epochs; "
                    f"best epoch {stopper.best_epoch} val {stopper.best:.5f}"
                )
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
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
        """Top: train and validation loss (log scale), best epoch and LR drops. Bottom: gradient norm and LR."""
        epochs = [h["epoch"] for h in history]
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, height_ratios=(3, 2))
        ax.plot(epochs, [h["train_loss"] for h in history], label="train loss")
        ax.plot(epochs, [h["val_loss"] for h in history], label="validation loss")
        for a in (ax, ax2):
            for k, c in enumerate(lr_changes):
                a.axvline(c["after_epoch"] + 0.5, color="grey", ls=":", lw=1, label="LR halved" if k == 0 else None)
            if best_epoch:
                a.axvline(best_epoch, color="red", ls="--", lw=1, label=f"best epoch {best_epoch}")
        ax.set_yscale("log")
        ax.set_ylabel("masked MSE (real rows)")
        ax.set_title("Autoencoder T = 48, stride-24 overlap: reconstruction loss")
        ax.legend()
        ax2.plot(epochs, [h["grad_norm"] for h in history], color="tab:green", label="mean gradient L2 norm")
        ax2.set_ylabel("gradient L2 norm")
        ax2.set_xlabel("epoch")
        lr_ax = ax2.twinx()
        lr_ax.plot(epochs, [h["lr"] for h in history], color="tab:purple", label="learning rate")
        lr_ax.set_yscale("log")
        lr_ax.set_ylabel("learning rate")
        lines = ax2.get_legend_handles_labels()
        lr_lines = lr_ax.get_legend_handles_labels()
        ax2.legend(lines[0] + lr_lines[0], lines[1] + lr_lines[1], fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    @torch.no_grad()
    def score(self, x: np.ndarray, mask: np.ndarray, keep_recon: bool = False) -> dict[str, np.ndarray]:
        """Per-window reconstruction error (N,), per-feature error (N, F), latent z (N, latent) [, recon (N, T, F)]."""
        self.model.eval()
        err, feat, zs, recons = [], [], [], []
        for i in range(0, len(x), self.settings.eval_batch):
            xb, mb = self._tensors(x[i : i + self.settings.eval_batch], mask[i : i + self.settings.eval_batch])
            recon, z = self.model(xb, mb)
            e, f = window_errors(recon, xb, mb)
            err.append(e.cpu().numpy())
            feat.append(f.cpu().numpy())
            zs.append(z.cpu().numpy())
            if keep_recon:
                recons.append(recon.cpu().numpy())
        out = {"error": np.concatenate(err), "error_feature": np.concatenate(feat), "z": np.concatenate(zs)}
        if keep_recon:
            out["recon"] = np.concatenate(recons)
        return out
