"""Shared training code of the neural detectors (`convae`, `lstmae`, `supae`).

One loop for all three: Adam, gradient clipping, ReduceLROnPlateau and early stopping on the regime's validation
pool, best weights restored at the end. It follows `task/autoencoder_benign_t32/trainer.py` and
`task/autoencoder_st_t32/trainer.py` operation for operation (seeding, batch order, optimiser, scheduler,
early-stopping rule - the two `EarlyStopping` classes are imported from those packages), without their run-folder
side effects (checkpoints, plots, logs), which this package's runner replaces. The tests check that the loop gives
the same validation history as the two task trainers on CPU.

What a model is trained for is an `Objective`: reconstruction (masked MSE, selection on validation loss) or
supervised (class-weighted BCE + masked MSE, selection on validation AUROC).
"""

from __future__ import annotations

import abc
import copy
import dataclasses
import time
from pathlib import Path
from typing import Callable, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from baselinepaper.experiments.detectors.base import Detector
from task.autoencoder_benign_t32.model import masked_mse, window_errors
from task.autoencoder_benign_t32.trainer import EarlyStopping as RelativeDecreaseStopping
from task.autoencoder_benign_t32.trainer import seed_everything
from task.autoencoder_st_t32.metrics import ranking_metrics
from task.autoencoder_st_t32.trainer import EarlyStopping as AbsoluteIncreaseStopping

__all__ = [
    "NeuralSettings",
    "Objective",
    "ReconstructionObjective",
    "SupervisedObjective",
    "FitResult",
    "NeuralTrainer",
    "NeuralDetector",
    "count_params",
    "to_float32",
]

# Batches between two `torch.mps.synchronize()` calls. Without them the Metal command queue grows over an epoch and
# the LSTM step slows from ~4 ms to ~60 ms (measured); the call only waits for pending work, results are unchanged.
MPS_SYNC_EVERY = 50


def count_params(module: nn.Module) -> int:
    """Trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def to_float32(x: np.ndarray) -> np.ndarray:
    """A writable, contiguous float32 array (copies a read-only memory map)."""
    x = np.asarray(x)
    if x.dtype != np.float32 or not x.flags.writeable or not x.flags.c_contiguous:
        return np.array(x, dtype=np.float32, order="C")
    return x


@dataclasses.dataclass(frozen=True)
class NeuralSettings:
    """Optimisation settings of one neural detector (frozen in `05-protocol/hyperparameters-neural.md`).

    Attributes:
        lr, weight_decay: Adam.
        batch_size: training batch size.
        max_epochs: upper bound on epochs.
        patience: early stopping (epochs without improvement of the monitored validation value).
        min_delta: improvement needed to reset early stopping (relative decrease of the validation loss for a
            reconstruction objective, absolute increase of the validation AUROC for the supervised one).
        plateau_factor, plateau_patience: ReduceLROnPlateau on the monitored validation value.
        grad_clip: max global gradient L2 norm.
        eval_batch: batch size for validation and scoring.
    """

    lr: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    min_delta: float
    plateau_factor: float
    plateau_patience: int
    grad_clip: float
    eval_batch: int

    @classmethod
    def from_task_config(cls, task_config, **overrides) -> "NeuralSettings":
        """Settings read from a `task/` package config (`AutoencoderConfig`, `SupervisedAEConfig`)."""
        values = {f.name: getattr(task_config, f.name) for f in dataclasses.fields(cls)}
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class Objective(abc.ABC):
    """What the loop minimises and what it selects the model on.

    Attributes:
        mode: "min" or "max" - direction of the monitored validation value (scheduler and early stopping).
        monitor: key of the monitored value in the dict `validate` returns.
    """

    mode: str
    monitor: str

    @abc.abstractmethod
    def stopper(self, settings: NeuralSettings):
        """A fresh early-stopping counter (`update(epoch, value) -> improved`, `stop`, `best`, `best_epoch`)."""

    @abc.abstractmethod
    def loss(self, model: nn.Module, x: torch.Tensor, mask: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        """Training loss of one batch (model in train mode)."""

    @abc.abstractmethod
    def validate(
        self, model: nn.Module, x: torch.Tensor, mask: torch.Tensor, y: torch.Tensor | None, eval_batch: int
    ) -> dict[str, float]:
        """Validation values of the whole validation pool (model in eval mode); must contain `monitor`."""


class ReconstructionObjective(Objective):
    """Masked MSE over real rows; selection on the validation loss (pooled over every real row).

    The model's `forward(x, mask)` returns a tuple whose first element is the reconstruction (B, T, F).
    """

    mode = "min"
    monitor = "val_loss"

    def stopper(self, settings: NeuralSettings) -> RelativeDecreaseStopping:
        return RelativeDecreaseStopping(settings.patience, settings.min_delta)

    def loss(self, model, x, mask, y=None) -> torch.Tensor:
        return masked_mse(model(x, mask)[0], x, mask)

    @torch.no_grad()
    def validate(self, model, x, mask, y, eval_batch) -> dict[str, float]:
        sq, rows = 0.0, 0.0
        for i in range(0, len(x), eval_batch):
            xb, mb = x[i : i + eval_batch], mask[i : i + eval_batch]
            w = mb.unsqueeze(-1).float()
            sq += float(((model(xb, mb)[0] - xb) ** 2 * w).sum())
            rows += float(w.sum()) * xb.shape[-1]
        return {"val_loss": sq / max(rows, 1.0)}


class SupervisedObjective(Objective):
    """BCEWithLogits (pos_weight) + recon_weight * masked MSE; selection on the validation AUROC.

    The model's `forward(x, mask)` returns (reconstruction, z, logit).

    Attributes:
        pos_weight: BCE weight of the positive class (n_benign / n_gridsybil of the training labels).
        recon_weight: weight of the reconstruction term.
    """

    mode = "max"
    monitor = "val_auroc"

    def __init__(self, pos_weight: float, recon_weight: float) -> None:
        self.pos_weight = float(pos_weight)
        self.recon_weight = float(recon_weight)
        self._pw: torch.Tensor | None = None

    def _pos_weight(self, device: torch.device) -> torch.Tensor:
        if self._pw is None or self._pw.device != device:
            self._pw = torch.tensor(self.pos_weight, device=device)
        return self._pw

    def stopper(self, settings: NeuralSettings) -> AbsoluteIncreaseStopping:
        return AbsoluteIncreaseStopping(settings.patience, settings.min_delta)

    def loss(self, model, x, mask, y) -> torch.Tensor:
        recon, _, logit = model(x, mask)
        pw = self._pos_weight(x.device)
        return F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw) + self.recon_weight * masked_mse(
            recon, x, mask
        )

    @torch.no_grad()
    def validate(self, model, x, mask, y, eval_batch) -> dict[str, float]:
        pw = self._pos_weight(x.device)
        bce_sum, sq, rows, logits = 0.0, 0.0, 0.0, []
        for i in range(0, len(x), eval_batch):
            xb, mb, yb = x[i : i + eval_batch], mask[i : i + eval_batch], y[i : i + eval_batch]
            recon, _, logit = model(xb, mb)
            bce_sum += float(F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pw, reduction="sum"))
            w = mb.unsqueeze(-1).float()
            sq += float(((recon - xb) ** 2 * w).sum())
            rows += float(w.sum()) * xb.shape[-1]
            logits.append(logit)
        bce, mse = bce_sum / len(x), sq / max(rows, 1.0)
        ranking = ranking_metrics(y.cpu().numpy(), torch.sigmoid(torch.cat(logits)).cpu().numpy())
        return {
            "val_loss": bce + self.recon_weight * mse,
            "val_bce": bce,
            "val_recon": mse,
            "val_auroc": ranking["auroc"],
            "val_pr_auc": ranking["pr_auc"],
        }


@dataclasses.dataclass
class FitResult:
    """Outcome of one training run.

    Attributes:
        history: one dict per epoch (epoch, train_loss, validation values, lr, grad norms, seconds).
        best_epoch: epoch whose weights the model now holds.
        best_value: monitored validation value at `best_epoch`.
        epochs_run: epochs trained.
        stop_reason: "early_stopping" or "max_epochs".
        lr_changes: learning-rate reductions ({after_epoch, from, to}).
        train_seconds: wall-clock of the loop.
    """

    history: list[dict]
    best_epoch: int
    best_value: float
    epochs_run: int
    stop_reason: str
    lr_changes: list[dict]
    train_seconds: float

    def info(self, monitor: str) -> dict:
        """JSON-ready summary (without the per-epoch history)."""
        return {
            "best_epoch": self.best_epoch,
            f"best_{monitor}": self.best_value,
            "epochs_run": self.epochs_run,
            "stop_reason": self.stop_reason,
            "lr_changes": self.lr_changes,
            "train_seconds": self.train_seconds,
        }


class NeuralTrainer:
    """Builds a model from a seed, trains it on arrays held on the device, and runs it in batches.

    Attributes:
        settings: optimisation settings.
        objective: loss and model-selection rule.
        device: torch device.
        model: the trained model (None before `fit`); holds the best-epoch weights after `fit`.
    """

    def __init__(self, settings: NeuralSettings, objective: Objective, device: str) -> None:
        self.settings = settings
        self.objective = objective
        self.device = torch.device(device)
        self.model: nn.Module | None = None

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).to(self.device)

    def fit(
        self,
        build_model: Callable[[], nn.Module],
        train: tuple[np.ndarray, ...],
        val: tuple[np.ndarray, ...],
        seed: int,
        log: Callable[[str], None] | None = None,
    ) -> FitResult:
        """Trains until early stopping or `max_epochs` and restores the best-epoch weights.

        Args:
            build_model: creates the model; called right after seeding, so the seed fixes the initial weights.
            train, val: (x (N, T, F) float32, mask (N, T) bool[, y (N,) float32]) - already standardised.
            seed: seeds python / numpy / torch (initial weights, dropout) and the batch-order generator.
            log: optional sink of one line per epoch.

        Returns:
            the `FitResult`.
        """
        s, objective = self.settings, self.objective
        seed_everything(seed)
        self.model = build_model().to(self.device)
        model = self.model
        xt, mt = self._tensor(train[0]), self._tensor(train[1])
        xv, mv = self._tensor(val[0]), self._tensor(val[1])
        yt = self._tensor(train[2]) if len(train) > 2 else None
        yv = self._tensor(val[2]) if len(val) > 2 else None
        params = list(model.parameters())
        opt = torch.optim.Adam(params, lr=s.lr, weight_decay=s.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=objective.mode, factor=s.plateau_factor, patience=s.plateau_patience
        )
        gen = torch.Generator().manual_seed(seed)
        stopper = objective.stopper(s)
        history: list[dict] = []
        lr_changes: list[dict] = []
        best_state = None
        stop_reason = "max_epochs"
        start = time.time()
        for epoch in range(1, s.max_epochs + 1):
            t0 = time.time()
            model.train()
            order = torch.randperm(len(xt), generator=gen).to(self.device)
            total = torch.zeros((), device=self.device)
            norm_sum = torch.zeros((), device=self.device)
            norm_max = torch.zeros((), device=self.device)
            batches = 0
            for i in range(0, len(order), s.batch_size):
                idx = order[i : i + s.batch_size]
                loss = objective.loss(model, xt[idx], mt[idx], None if yt is None else yt[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(params, s.grad_clip)  # pre-clip global norm
                opt.step()
                total += loss.detach()
                norm_sum += norm
                norm_max = torch.maximum(norm_max, norm)
                batches += 1
                if self.device.type == "mps" and batches % MPS_SYNC_EVERY == 0:
                    torch.mps.synchronize()
            model.eval()
            values = objective.validate(model, xv, mv, yv, s.eval_batch)
            monitored = values[objective.monitor]
            lr = opt.param_groups[0]["lr"]
            sched.step(monitored)
            new_lr = opt.param_groups[0]["lr"]
            improved = stopper.update(epoch, monitored)
            if improved:
                best_state = copy.deepcopy(model.state_dict())
            row = {
                "epoch": epoch,
                "train_loss": float(total) / batches,
                **values,
                "lr": lr,
                "grad_norm_mean": float(norm_sum) / batches,
                "grad_norm_max": float(norm_max),
                "improved": improved,
                "seconds": round(time.time() - t0, 2),
            }
            history.append(row)
            if new_lr < lr:
                lr_changes.append({"after_epoch": epoch, "from": lr, "to": new_lr})
            if log is not None:
                log(
                    f"epoch {epoch:3d}  train {row['train_loss']:.5f}  {objective.monitor} {monitored:.5f}  "
                    f"lr {lr:.2e}  {row['seconds']:.1f}s" + (" *best" if improved else "")
                )
            if stopper.stop:
                stop_reason = "early_stopping"
                break
        if best_state is None:
            raise RuntimeError("no epoch improved the monitored validation value (non-finite loss?)")
        model.load_state_dict(best_state)
        model.eval()
        del xt, mt, xv, mv, yt, yv
        return FitResult(
            history=history,
            best_epoch=stopper.best_epoch,
            best_value=float(stopper.best),
            epochs_run=len(history),
            stop_reason=stop_reason,
            lr_changes=lr_changes,
            train_seconds=round(time.time() - start, 1),
        )

    @torch.no_grad()
    def forward_batches(
        self, x: np.ndarray, mask: np.ndarray, standardise: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Runs the trained model over windows in batches of `eval_batch` (eval mode).

        Args:
            x: (N, T, F) in the cached normalisation (may be a read-only memory map).
            mask: (N, T) bool.
            standardise: the detector's `standardise`, applied to every batch.

        Returns:
            {"recon": per-window masked MSE (N,), "z": (N, latent)[, "logit": (N,)]} as float64 / float32 arrays.
        """
        if self.model is None:
            raise RuntimeError("fit() first")
        self.model.eval()
        out: dict[str, list[np.ndarray]] = {"recon": [], "z": [], "logit": []}
        for i in range(0, len(x), self.settings.eval_batch):
            mb_np = np.ascontiguousarray(mask[i : i + self.settings.eval_batch], dtype=bool)
            xb_np = to_float32(standardise(x[i : i + self.settings.eval_batch], mb_np))
            xb, mb = self._tensor(xb_np), self._tensor(mb_np)
            outputs = self.model(xb, mb)
            out["recon"].append(window_errors(outputs[0], xb, mb)[0].cpu().numpy().astype(np.float64))
            out["z"].append(outputs[1].cpu().numpy())
            if len(outputs) > 2:
                out["logit"].append(outputs[2].cpu().numpy().astype(np.float64))
        return {k: np.concatenate(v) for k, v in out.items() if v}


class NeuralDetector(Detector):
    """Base of the neural detectors: standardise, train with `NeuralTrainer`, score in batches.

    A subclass sets `settings` and implements `build_model`, `objective`, `architecture` and `score`; it may
    extend `after_fit` (e.g. to fit a Mahalanobis model on the training z).

    Class attributes:
        settings: frozen optimisation settings (tests replace them on the instance with smaller ones).

    Instance attributes:
        trainer: the `NeuralTrainer` (None before `fit`).
        result: the `FitResult` (None before `fit`).
    """

    is_stochastic = True
    settings: ClassVar[NeuralSettings]

    def __init__(self, regime: str, device: str = "cpu", statistics=None) -> None:
        super().__init__(regime=regime, device=device, statistics=statistics)
        self.trainer: NeuralTrainer | None = None
        self.result: FitResult | None = None
        self._info: dict = {}
        self.log: Callable[[str], None] | None = lambda line: print(line, flush=True)

    @abc.abstractmethod
    def build_model(self) -> nn.Module:
        """A freshly initialised model (called after seeding)."""

    @abc.abstractmethod
    def objective(self, train) -> Objective:
        """The training objective; `train` is the regime's training set (labels only if supervised)."""

    @abc.abstractmethod
    def architecture(self) -> dict:
        """Architecture description and parameter count for config.json."""

    def hyperparameters(self) -> dict:
        return {
            "architecture": self.architecture(),
            "optimiser": "Adam",
            **self.settings.to_dict(),
            "lr_schedule": "ReduceLROnPlateau on the monitored validation value",
            "seed_controls": "initial weights, dropout, batch order (python / numpy / torch seeded with the seed)",
            "frozen_in": "baselinepaper/05-protocol/hyperparameters-neural.md",
        }

    def fit(self, train, val, seed: int) -> None:
        objective = self.objective(train)
        self.trainer = NeuralTrainer(self.settings, objective, self.device)
        x_train = to_float32(self.standardise(train.x, train.mask))
        x_val = to_float32(self.standardise(val.x, val.mask))
        train_arrays = [x_train, np.ascontiguousarray(train.mask, dtype=bool)]
        val_arrays = [x_val, np.ascontiguousarray(val.mask, dtype=bool)]
        if hasattr(train, "labels"):
            train_arrays.append(np.asarray(train.labels, dtype=np.float32))
            val_arrays.append(np.asarray(val.labels, dtype=np.float32))
        self.result = self.trainer.fit(self.build_model, tuple(train_arrays), tuple(val_arrays), seed, self.log)
        self._info = {
            "parameters": count_params(self.trainer.model),
            "device": str(self.trainer.device),
            "seed": seed,
            "train_windows": len(x_train),
            "val_windows": len(x_val),
            "monitor": objective.monitor,
            **self.result.info(objective.monitor),
        }
        self.after_fit(x_train, train_arrays[1])

    def after_fit(self, x_train: np.ndarray, mask_train: np.ndarray) -> None:
        """Hook run after training with the standardised training windows (no labels)."""

    def forward(self, x: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        """Model outputs for windows in the cached normalisation (standardised here, batch by batch)."""
        if self.trainer is None:
            raise RuntimeError("fit() first")
        return self.trainer.forward_batches(x, mask, self.standardise)

    def history(self) -> list[dict]:
        return [] if self.result is None else self.result.history

    def fit_info(self) -> dict:
        return dict(self._info)

    def save(self, run_dir: Path) -> None:
        """Best-epoch weights (CPU tensors) with the hyperparameters."""
        if self.trainer is None or self.trainer.model is None:
            return
        state = {k: v.detach().cpu() for k, v in self.trainer.model.state_dict().items()}
        torch.save(
            {"model": state, "hyperparameters": self.hyperparameters(), "fit_info": self.fit_info()},
            Path(run_dir) / "best.pt",
        )
