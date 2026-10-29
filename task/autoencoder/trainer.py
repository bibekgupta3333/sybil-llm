"""Training loop (early stopping, LR schedule, best checkpoint, per-epoch logs) and reconstruction scoring."""

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

from task.autoencoder.config import AutoencoderConfig  # noqa: E402
from task.autoencoder.model import ConvAutoencoder, masked_mse, window_errors  # noqa: E402


def pick_device(name: str) -> torch.device:
    """`auto` -> mps, then cuda, then cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


class Trainer:
    """Fits a ConvAutoencoder on (x, mask) arrays and scores windows by reconstruction error.

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
        settings = AutoencoderConfig(**state["config"])
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
        best, best_epoch, best_state, waited = float("inf"), 0, None, 0
        history, lr_changes, stop_reason = [], [], "max_epochs"
        start = time.time()
        log(f"fit: {len(xt):,} windows, val: {len(xv):,} windows, batch {s.batch_size}, device {self.device}")
        for epoch in range(1, s.max_epochs + 1):
            t0 = time.time()
            self.model.train()
            order = torch.randperm(len(xt), generator=gen).to(self.device)
            total, n = torch.zeros((), device=self.device), 0
            for i in range(0, len(order), s.batch_size):
                idx = order[i : i + s.batch_size]
                xb, mb = xt[idx], mt[idx]
                recon, _ = self.model(xb, mb)
                loss = masked_mse(recon, xb, mb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.detach()
                n += 1
            train_loss = float(total) / n
            val_loss = self._val_loss(xv, mv)
            lr = opt.param_groups[0]["lr"]
            sched.step(val_loss)
            new_lr = opt.param_groups[0]["lr"]
            improved = val_loss < best * (1.0 - s.min_delta)
            if improved:
                best, best_epoch, waited = val_loss, epoch, 0
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save(
                    {"model": best_state, "config": s.to_dict(), "epoch": epoch, "val_loss": val_loss},
                    run_dir / "best.pt",
                )
            else:
                waited += 1
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "seconds": round(time.time() - t0, 2),
            }
            history.append(row)
            with history_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            mark = " *best" if improved else f" (no improvement {waited}/{s.patience})"
            log(
                f"epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}  lr {lr:.2e}  "
                f"{row['seconds']:.1f}s{mark}"
            )
            if new_lr < lr:
                lr_changes.append({"after_epoch": epoch, "from": lr, "to": new_lr})
                log(f"lr reduced {lr:.2e} -> {new_lr:.2e} (validation loss flat for {s.plateau_patience + 1} epochs)")
            self.plot(history, best_epoch, lr_changes, run_dir / "loss_curve.png")
            if waited >= s.patience:
                stop_reason = "early_stopping"
                log(f"early stopping: no improvement for {s.patience} epochs; best epoch {best_epoch} val {best:.5f}")
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {
            "best_epoch": best_epoch,
            "best_val_loss": best,
            "epochs_run": len(history),
            "stop_reason": stop_reason,
            "lr_changes": lr_changes,
            "train_seconds": round(time.time() - start, 1),
        }

    @staticmethod
    def plot(history: list[dict], best_epoch: int, lr_changes: list[dict], path: Path) -> None:
        """Train and validation loss per epoch (log scale), best epoch and LR drops marked."""
        epochs = [h["epoch"] for h in history]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(epochs, [h["train_loss"] for h in history], label="train loss")
        ax.plot(epochs, [h["val_loss"] for h in history], label="validation loss")
        for k, c in enumerate(lr_changes):
            ax.axvline(c["after_epoch"] + 0.5, color="grey", ls=":", lw=1, label="LR halved" if k == 0 else None)
        if best_epoch:
            ax.axvline(best_epoch, color="red", ls="--", lw=1, label=f"best epoch {best_epoch}")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("masked MSE (real rows)")
        ax.set_title("Autoencoder T = 48: reconstruction loss")
        ax.legend()
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
