"""Training loop (class-weighted BCE + masked MSE, gradient clipping, LR schedule and early stopping on val AUROC,
best / last checkpoints, resume, logs, curves) and scoring."""

from __future__ import annotations

import copy
import datetime as dt
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from task.autoencoder_st.config import SupervisedAEConfig  # noqa: E402
from task.autoencoder_st.metrics import epoch_metrics  # noqa: E402
from task.autoencoder_st.model import SupervisedAutoencoder, masked_mse, window_errors  # noqa: E402

METRICS = ("auroc", "pr_auc", "f1", "balanced_accuracy")


def pick_device(name: str) -> torch.device:
    """`auto` -> mps, then cuda, then cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Python, numpy and torch (all devices) random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    """Counts epochs without an absolute increase of at least `min_delta` in validation AUROC.

    Attributes:
        best: best validation AUROC so far.
        best_epoch: epoch of `best` (0 before the first update).
        waited: epochs since the last improvement.
    """

    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("-inf")
        self.best_epoch = 0
        self.waited = 0

    def update(self, epoch: int, score: float) -> bool:
        """Records one epoch; True if it counts as an improvement (score > best + min_delta)."""
        if score > self.best + self.min_delta:
            self.best, self.best_epoch, self.waited = score, epoch, 0
            return True
        self.waited += 1
        return False

    @property
    def stop(self) -> bool:
        return self.waited >= self.patience

    def state_dict(self) -> dict:
        return {"best": self.best, "best_epoch": self.best_epoch, "waited": self.waited}

    def load_state_dict(self, state: dict) -> None:
        self.best, self.best_epoch, self.waited = state["best"], state["best_epoch"], state["waited"]


def rng_state(device: torch.device, gen: torch.Generator) -> dict:
    """Every random generator the loop uses, for an exact resume."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "batch_order": gen.get_state(),
    }
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def set_rng_state(state: dict, device: torch.device, gen: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    gen.set_state(state["batch_order"])
    if device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])
    elif device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


class Trainer:
    """Fits a SupervisedAutoencoder on (x, mask, y) arrays and scores windows.

    Attributes:
        model: the supervised autoencoder.
        settings: run settings.
        device: torch device.
        pos_weight: BCE weight of the GridSybil class (n_benign / n_gridsybil of the train split).
    """

    def __init__(
        self, model: SupervisedAutoencoder, settings: SupervisedAEConfig, device: torch.device, pos_weight: float
    ) -> None:
        self.model = model.to(device)
        self.settings = settings
        self.device = device
        self.pos_weight = float(pos_weight)
        self._pw = torch.tensor(self.pos_weight, device=device)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> "Trainer":
        """Trainer around the model stored in `best.pt` (or `last.pt`)."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        settings = SupervisedAEConfig.from_dict(state["config"])
        model = SupervisedAutoencoder(settings)
        model.load_state_dict(state["model"])
        return cls(model, settings, pick_device(device), state["pos_weight"])

    def _tensors(self, *arrays: np.ndarray) -> list[torch.Tensor]:
        return [torch.from_numpy(np.ascontiguousarray(a)).to(self.device) for a in arrays]

    def losses(
        self, recon: torch.Tensor, logit: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(total, BCE, masked MSE) of one batch; total = BCE + recon_weight * MSE."""
        bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=self._pw)
        mse = masked_mse(recon, x, mask)
        return bce + self.settings.recon_weight * mse, bce, mse

    @torch.no_grad()
    def _evaluate(self, x: torch.Tensor, m: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """Validation losses (BCE mean over windows, MSE pooled over real rows) and metrics, eval mode."""
        self.model.eval()
        bce_sum, sq, rows, logits = 0.0, 0.0, 0.0, []
        for i in range(0, len(x), self.settings.eval_batch):
            xb, mb, yb = (t[i : i + self.settings.eval_batch] for t in (x, m, y))
            recon, _, logit = self.model(xb, mb)
            bce_sum += float(F.binary_cross_entropy_with_logits(logit, yb, pos_weight=self._pw, reduction="sum"))
            w = mb.unsqueeze(-1).float()
            sq += float(((recon - xb) ** 2 * w).sum())
            rows += float(w.sum()) * xb.shape[-1]
            logits.append(logit)
        bce, mse = bce_sum / len(x), sq / max(rows, 1.0)
        prob = torch.sigmoid(torch.cat(logits)).cpu().numpy()
        return {
            "loss": bce + self.settings.recon_weight * mse,
            "bce": bce,
            "recon": mse,
            **epoch_metrics(y.cpu().numpy(), prob),
        }

    def fit(
        self,
        train: tuple[np.ndarray, np.ndarray, np.ndarray],
        val: tuple[np.ndarray, np.ndarray, np.ndarray],
        run_dir: Path,
        resume: bool = False,
    ) -> dict:
        """Trains until early stopping or `max_epochs`; writes history.jsonl, train.log, curves.png, best.pt, last.pt.

        Args:
            train, val: (x (N, T, F) float32, mask (N, T) bool, y (N,) float32).
            run_dir: run folder.
            resume: continue from `run_dir/last.pt` (model, optimizer, scheduler, early stopping, history, RNG).

        Returns:
            summary fields: best_epoch, best_val_auroc, epochs_run, stop_reason, lr_changes, train_seconds.
        """
        s = self.settings
        log = RunLog(run_dir / "train.log")
        history_path = run_dir / "history.jsonl"
        xt, mt, yt = self._tensors(*train)
        xv, mv, yv = self._tensors(*val)
        params = list(self.model.parameters())
        opt = torch.optim.Adam(params, lr=s.lr, weight_decay=s.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=s.plateau_factor, patience=s.plateau_patience
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
            f"fit: {len(xt):,} windows, val: {len(xv):,} windows, batch {s.batch_size}, device {self.device}, "
            f"pos_weight {self.pos_weight:.4f}, recon_weight {s.recon_weight:g}, grad_clip {s.grad_clip:g}, "
            f"early stopping on val AUROC (min_delta {s.min_delta:g} absolute, patience {s.patience})"
        )
        for epoch in range(first_epoch, s.max_epochs + 1):
            if stopper.stop:
                break
            t0 = time.time()
            self.model.train()
            order = torch.randperm(len(xt), generator=gen).to(self.device)
            sums = torch.zeros(3, device=self.device)
            norm_sum = torch.zeros((), device=self.device)
            norm_max = torch.zeros((), device=self.device)
            logits, n_batches = [], 0
            for i in range(0, len(order), s.batch_size):
                idx = order[i : i + s.batch_size]
                xb, mb, yb = xt[idx], mt[idx], yt[idx]
                recon, _, logit = self.model(xb, mb)
                total, bce, mse = self.losses(recon, logit, xb, mb, yb)
                opt.zero_grad(set_to_none=True)
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(params, s.grad_clip)  # pre-clip global norm
                opt.step()
                sums += torch.stack([total.detach(), bce.detach(), mse.detach()]) * len(idx)
                norm_sum += norm
                norm_max = torch.maximum(norm_max, norm)
                logits.append(logit.detach())
                n_batches += 1
            # Train metrics come from the running epoch outputs (model in train mode: dropout on, batch statistics,
            # weights changing during the epoch), so they are not directly comparable with the val metrics.
            prob = torch.sigmoid(torch.cat(logits)).cpu().numpy()
            loss, bce, mse = (sums / len(order)).tolist()
            tr = {"loss": loss, "bce": bce, "recon": mse, **epoch_metrics(yt[order].cpu().numpy(), prob)}
            va = self._evaluate(xv, mv, yv)
            lr = opt.param_groups[0]["lr"]
            sched.step(va["auroc"])
            new_lr = opt.param_groups[0]["lr"]
            improved = stopper.update(epoch, va["auroc"])
            if improved:
                self.save(run_dir / "best.pt", epoch, va)
            row = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in tr.items()},
                **{f"val_{k}": v for k, v in va.items()},
                "lr": lr,
                "grad_norm_mean": float(norm_sum) / n_batches,
                "grad_norm_max": float(norm_max),
                "seconds": round(time.time() - t0, 2),
            }
            history.append(row)
            with history_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            mark = " *best" if improved else f" (no improvement {stopper.waited}/{s.patience})"
            log(
                f"epoch {epoch:3d}  loss {tr['loss']:.4f}/{va['loss']:.4f}  bce {tr['bce']:.4f}/{va['bce']:.4f}  "
                f"recon {tr['recon']:.4f}/{va['recon']:.4f}  auroc {tr['auroc']:.4f}/{va['auroc']:.4f}  "
                f"pr_auc {va['pr_auc']:.4f}  f1 {va['f1']:.4f}  bal_acc {va['balanced_accuracy']:.4f}  "
                f"lr {lr:.2e}  grad_norm {row['grad_norm_mean']:.3f}/{row['grad_norm_max']:.3f}  "
                f"{row['seconds']:.1f}s{mark}"
            )
            if new_lr < lr:
                lr_changes.append({"after_epoch": epoch, "from": lr, "to": new_lr})
                log(f"lr reduced {lr:.2e} -> {new_lr:.2e} (val AUROC flat for {s.plateau_patience + 1} epochs)")
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
                    "pos_weight": self.pos_weight,
                },
                run_dir / "last.pt",
            )
            self.plot(history, stopper.best_epoch, lr_changes, run_dir / "curves.png")
            if stopper.stop:
                stop_reason = "early_stopping"
                log(
                    f"early stopping: no improvement for {s.patience} epochs; "
                    f"best epoch {stopper.best_epoch} val AUROC {stopper.best:.5f}"
                )
        best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        self.model.load_state_dict(best["model"])
        return {
            "best_epoch": stopper.best_epoch,
            "best_val_auroc": stopper.best,
            "best_val": best["val"],
            "epochs_run": len(history),
            "stop_reason": stop_reason,
            "lr_changes": lr_changes,
            "train_seconds": round(time.time() - start, 1),
        }

    def save(self, path: Path, epoch: int, val: dict) -> None:
        """Model weights with the settings, pos_weight and the val metrics of that epoch."""
        torch.save(
            {
                "model": copy.deepcopy(self.model.state_dict()),
                "config": self.settings.to_dict(),
                "pos_weight": self.pos_weight,
                "epoch": epoch,
                "val": val,
            },
            path,
        )

    @staticmethod
    def plot(history: list[dict], best_epoch: int, lr_changes: list[dict], path: Path) -> None:
        """Losses (train vs val), val AUROC / PR-AUC / F1, and LR + gradient norm; best epoch marked."""
        epochs = [h["epoch"] for h in history]
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        for key, color in (("loss", "tab:blue"), ("bce", "tab:orange"), ("recon", "tab:green")):
            ax1.plot(epochs, [h[f"train_{key}"] for h in history], color=color, ls="--", label=f"train {key}")
            ax1.plot(epochs, [h[f"val_{key}"] for h in history], color=color, label=f"val {key}")
        ax1.set_yscale("log")
        ax1.set_ylabel("loss")
        ax1.set_title("Supervised autoencoder T = 48 (T48_tvt): BCE + recon_weight x masked MSE")
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

    @torch.no_grad()
    def predict(self, x: np.ndarray, mask: np.ndarray, keep_recon: bool = False) -> dict[str, np.ndarray]:
        """Per window: GridSybil probability and logit (N,), reconstruction error (N,), per-feature error (N, F),
        z (N, latent) [, recon (N, T, F)]. Eval mode; labels not needed."""
        self.model.eval()
        out: dict[str, list] = {"prob": [], "logit": [], "error": [], "error_feature": [], "z": [], "recon": []}
        for i in range(0, len(x), self.settings.eval_batch):
            xb, mb = self._tensors(x[i : i + self.settings.eval_batch], mask[i : i + self.settings.eval_batch])
            recon, z, logit = self.model(xb, mb)
            e, f = window_errors(recon, xb, mb)
            for key, value in (("prob", torch.sigmoid(logit)), ("logit", logit), ("error", e)):
                out[key].append(value.cpu().numpy())
            out["error_feature"].append(f.cpu().numpy())
            out["z"].append(z.cpu().numpy())
            if keep_recon:
                out["recon"].append(recon.cpu().numpy())
        if not keep_recon:
            del out["recon"]
        return {k: np.concatenate(v) for k, v in out.items()}
