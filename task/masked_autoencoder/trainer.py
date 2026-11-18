"""Training loop of the masked autoencoder (widedeep EncoderDecoderTrainer semantics, sibling ConvAE protocol).

Per step (widedeep `EncoderDecoderTrainer._train_step`, self_supervised_training/encoder_decoder_trainer.py L217-229):
draw the obfuscation mask on real cells, reconstruct, `EncoderDecoderLoss` on the hidden cells, backward, optimizer step.
Optimizer AdamW (widedeep's default when none is passed, _base_encoder_decoder_trainer.py L59-63). After every epoch the
benign val masked-reconstruction MSE (deterministic `EvalMasks`, pooled over every real cell) drives ReduceLROnPlateau
and early stopping, with the sibling ConvAE's patience / min_delta. `EarlyStopping`, `RunLog`, `pick_device` and
`seed_everything` are imported from `task.autoencoder_benign_t32.trainer`.

The trainer sees only (x, mask) arrays; it has no access to labels.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from task.autoencoder_benign_t32.trainer import EarlyStopping, RunLog, pick_device, seed_everything
from task.masked_autoencoder.config import MAEConfig
from task.masked_autoencoder.model import EncoderDecoderLoss, MaskedTabularAutoencoder
from task.masked_autoencoder.scoring import EvalMasks, WindowScorer

__all__ = ["MaskedTrainer", "RunLog", "pick_device", "seed_everything"]


class MaskedTrainer:
    """Fits a `MaskedTabularAutoencoder` on benign (x, mask) arrays.

    Attributes:
        model: the autoencoder.
        settings: run settings.
        device: torch device.
        scorer: deterministic-mask scorer (validation loss and window scores).
    """

    def __init__(self, model: MaskedTabularAutoencoder, settings: MAEConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.settings = settings
        self.device = device
        masks = EvalMasks(settings.seq_len, settings.n_features, settings.n_eval_draws, settings.eval_mask_seed)
        self.scorer = WindowScorer(self.model, masks, device, settings.eval_batch)
        self.loss_fn = EncoderDecoderLoss()

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> "MaskedTrainer":
        state = torch.load(path, map_location="cpu", weights_only=False)
        settings = MAEConfig.from_dict(state["config"])
        model = MaskedTabularAutoencoder(settings)
        model.load_state_dict(state["model"])
        return cls(model, settings, pick_device(device))

    def _tensors(self, x: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(np.ascontiguousarray(x)).to(self.device),
            torch.from_numpy(np.ascontiguousarray(mask)).to(self.device),
        )

    def train_step(
        self, xb: torch.Tensor, mb: torch.Tensor, opt: torch.optim.Optimizer
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One widedeep training step; returns (loss, pre-clip gradient norm) as detached device scalars."""
        self.model.train()
        hidden = self.model.masker(xb, mb)
        recon, _ = self.model(xb, mb, hidden)
        loss = self.loss_fn(xb, recon, hidden, mb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        params = [p for p in self.model.parameters() if p.grad is not None]
        max_norm = self.settings.grad_clip if self.settings.grad_clip > 0 else float("inf")
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
        opt.step()
        return loss.detach(), norm.detach()

    def fit(self, train: tuple[np.ndarray, np.ndarray], val: tuple[np.ndarray, np.ndarray], run_dir: Path) -> dict:
        """Trains until early stopping or `max_epochs`; writes history.jsonl, train.log, best.pt; loads best weights.

        Args:
            train, val: (x (N, T, F) float32, mask (N, T) bool); benign windows only.
            run_dir: run folder.

        Returns:
            best_epoch, best_val_loss, epochs_run, stop_reason, lr_changes, train_seconds.
        """
        s = self.settings
        log = RunLog(run_dir / "train.log")
        history_path = run_dir / "history.jsonl"
        xt, mt = self._tensors(*train)
        xv, mv = self._tensors(*val)
        opt = torch.optim.AdamW(self.model.parameters(), lr=s.lr, weight_decay=s.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=s.plateau_factor, patience=s.plateau_patience
        )
        gen = torch.Generator().manual_seed(s.seed)
        stopper = EarlyStopping(s.patience, s.min_delta)
        lr_changes, start, epochs_run = [], time.time(), 0
        stop_reason = "max_epochs"
        log(
            f"fit: {len(xt):,} benign windows, val: {len(xv):,} benign windows, batch {s.batch_size}, device "
            f"{self.device}, masked_prob {s.masked_prob}, AdamW lr {s.lr:g} wd {s.weight_decay:g}, early stopping on "
            f"val masked MSE (min_delta {s.min_delta:g} relative, patience {s.patience})"
        )
        for epoch in range(1, s.max_epochs + 1):
            t0 = time.time()
            order = torch.randperm(len(xt), generator=gen).to(self.device)
            losses, norms = [], []
            for i in range(0, len(order), s.batch_size):
                idx = order[i : i + s.batch_size]
                if len(idx) < 2:  # BatchNorm (resnet) needs > 1 sample; the same rule for both encoders
                    continue
                loss, norm = self.train_step(xt[idx], mt[idx], opt)
                losses.append(loss)
                norms.append(norm)
            losses_t, norms_t = torch.stack(losses), torch.stack(norms)
            if not bool(torch.isfinite(losses_t).all()):
                raise FloatingPointError(f"epoch {epoch}: non-finite training loss")
            val = self.scorer.pooled(xv, mv)
            val_loss = val["masked_mse"]
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
                "train_loss": float(losses_t.mean()),
                "val_masked_mse": val_loss,
                "val_full_mse": val["full_mse"],
                "lr": lr,
                "grad_norm_mean": float(norms_t.mean()),
                "grad_norm_max": float(norms_t.max()),
                "seconds": round(time.time() - t0, 2),
            }
            epochs_run = epoch
            with history_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            mark = " *best" if improved else f" (no improvement {stopper.waited}/{s.patience})"
            log(
                f"epoch {epoch:3d}  train {row['train_loss']:.5f}  val masked {val_loss:.5f}  val full "
                f"{val['full_mse']:.5f}  lr {lr:.2e}  grad_norm {row['grad_norm_mean']:.3f}/{row['grad_norm_max']:.3f}"
                f"  {row['seconds']:.1f}s{mark}"
            )
            if new_lr < lr:
                lr_changes.append({"after_epoch": epoch, "from": lr, "to": new_lr})
                log(f"lr reduced {lr:.2e} -> {new_lr:.2e}")
            if stopper.stop:
                stop_reason = "early_stopping"
                log(f"early stopping: best epoch {stopper.best_epoch} val masked MSE {stopper.best:.5f}")
                break
        best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        self.model.load_state_dict(best["model"])
        return {
            "best_epoch": stopper.best_epoch,
            "best_val_loss": stopper.best,
            "epochs_run": epochs_run,
            "stop_reason": stop_reason,
            "lr_changes": lr_changes,
            "train_seconds": round(time.time() - start, 1),
        }
