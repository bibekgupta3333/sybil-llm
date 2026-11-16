"""Per-epoch and per-step training logs of the T32_benign TimesNet run, for plots of every loss over time.

Mirrors the conventions of `task/autoencoder_benign_t32` (`history.jsonl`, `feature_history.jsonl`, `summary.json`)
and adds a light per-update file. Every file is JSON lines (one object per line) in the run folder:

- `history.jsonl`: one line per epoch plus an `epoch: 0` line for the random-init evaluation (train fields null).
  Train fields are means over the epoch's optimiser updates; a loss's mean uses only the updates in which that loss
  was present (length-bucketed batches of short windows can lack NT-Xent or the physics losses). Validation fields
  are the scalars of the existing `Evaluator` with their names unchanged (`val_joint`, `auroc_p1`, ...).
- `feature_history.jsonl`: one line per history line. `train_recon_row_pooled` pools the squared reconstruction
  error over every hidden real row of the epoch's training batches (train mode, taken from the prediction and target
  the training step already computed: no extra forward pass), so its mean over the 13 features is the row-weighted
  masked MSE of the epoch, not the batch mean `train_recon`. `val_recon` and `val_interp_baseline` are the Evaluator's
  per-feature lists; the Evaluator averages per-batch per-feature MSEs over batches, so the mean of `val_recon` over
  the 13 features equals the history's `val_recon` (float32 rounding aside).
- `steps.jsonl`: one line per real optimiser update (skipped batches have no line), floats to 6 significant digits.
- `summary.json`: written when a run ends (epochs done, early stopping, or a step / time / memory budget).

A run stopped by a budget inside an epoch writes that epoch's lines with `partial: true`; a resume continues from
`last.pt` and first truncates every file after the checkpoint's epoch / step, so no epoch or step appears twice.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from src.model.all_t32_benign.timesnet.base_data import FEATURES
from src.model.all_t32_benign.timesnet.losses import LossScales

LOSS_NAMES = LossScales.NAMES  # ("recon", "nce", "p1", "p2", "p3")
HISTORY_FILE = "history.jsonl"
FEATURE_FILE = "feature_history.jsonl"
STEPS_FILE = "steps.jsonl"
SUMMARY_FILE = "summary.json"
EVAL_KEYS = (
    "val_joint",
    "val_recon",
    "val_recon_interp_baseline",
    "val_nce",
    "val_p1",
    "val_p2",
    "val_p3",
    "auroc_p1",
    "auroc_p2",
    "auroc_p3",
    "z_std_mean",
    "z_std_min",
    "z_effective_rank",
    "length_probe_r2",
    "corr_znorm_length",
    "align",
    "uniformity",
    "cos_pos",
    "cos_neg",
)
TRAIN_KEYS = tuple(f"train_{k}" for k in ("joint", *LOSS_NAMES))
SUMMARY_KEYS = (
    "run_id",
    "seed",
    "params_encoder",
    "params_total",
    "best_epoch",
    "best_val_joint",
    "epochs_run",
    "steps",
    "stop_reason",
    "wall_seconds",
    "device",
    "train_windows",
    "val_windows",
    "min_train_rows",
    "short",
)
_EVAL_SKIP = frozenset({"kind", "tag", "step", "epoch"})  # set by the history itself


def clean(value: Any) -> Any:
    """JSON-safe scalar: NaN / inf become None, numpy and torch scalars become Python numbers."""
    if isinstance(value, torch.Tensor):
        value = value.item()
    if hasattr(value, "item") and not isinstance(value, (list, tuple, dict, str)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def round_sig(value: float | None, digits: int = 6) -> float | None:
    """`value` rounded to `digits` significant digits (None and non-finite values give None)."""
    value = clean(value)
    if value is None:
        return None
    return float(f"{float(value):.{digits}g}")


class JsonLines:
    """An append-only JSON-lines file that can be cut back for a resume.

    Args:
        path: the file (created on the first append).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, record: Mapping[str, Any]) -> None:
        """Writes one line (NaN / inf written as null)."""
        with self.path.open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")

    def read(self) -> list[dict[str, Any]]:
        """Every record of the file ([] if it does not exist)."""
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def keep(self, predicate: Callable[[dict[str, Any]], bool]) -> int:
        """Rewrites the file with the records for which `predicate` holds; returns how many were dropped."""
        if not self.path.is_file():
            return 0
        lines = [line for line in self.path.read_text().splitlines() if line.strip()]
        kept = [line for line in lines if predicate(json.loads(line))]
        self.path.write_text("".join(line + "\n" for line in kept))
        return len(lines) - len(kept)


class EpochStats:
    """Running sums of one epoch's training updates (checkpointed, so a resumed epoch keeps its earlier updates).

    Per-loss sums count only the updates where the loss was present. The per-feature reconstruction sums stay on the
    training device until `result` (no extra host synchronisation per step).

    Args:
        n_features: features per row (13).
        device: device of the training tensors.
    """

    def __init__(self, n_features: int, device: torch.device) -> None:
        self.n_features, self.device = n_features, device
        self.dtype = torch.float32 if device.type == "mps" else torch.float64  # MPS has no float64
        self.reset()

    def reset(self) -> None:
        """Clears every sum (called after each epoch's line is written)."""
        self.loss_sum = {k: 0.0 for k in ("joint", *LOSS_NAMES)}
        self.loss_count = {k: 0 for k in ("joint", *LOSS_NAMES)}
        self.grad_sum, self.grad_max = 0.0, None
        self.steps, self.skipped, self.seconds = 0, 0, 0.0
        self.square_sum = torch.zeros(self.n_features, dtype=self.dtype, device=self.device)
        self.hidden_rows = torch.zeros((), dtype=self.dtype, device=self.device)

    def add_update(self, joint: float, raw: Mapping[str, float], grad_norm: float) -> None:
        """Adds one optimiser update: the joint loss, the raw losses present and the pre-clip gradient norm."""
        self.steps += 1
        for name, value in (("joint", joint), *raw.items()):
            if name in self.loss_sum and math.isfinite(value):
                self.loss_sum[name] += value
                self.loss_count[name] += 1
        if math.isfinite(grad_norm):
            self.grad_sum += grad_norm
            self.grad_max = grad_norm if self.grad_max is None else max(self.grad_max, grad_norm)

    @torch.no_grad()
    def add_reconstruction(self, pred: torch.Tensor, target: torch.Tensor, hidden: torch.Tensor) -> None:
        """Adds the squared error of the hidden real rows: pred, target (B, T, F), hidden (B, T) bool."""
        rows = hidden.unsqueeze(-1).to(pred.dtype)
        self.square_sum += (((pred - target) ** 2) * rows).sum((0, 1)).to(self.dtype)
        self.hidden_rows += hidden.sum().to(self.dtype)

    def add_skip(self) -> None:
        """Counts a planned batch that produced no loss (no update)."""
        self.skipped += 1

    def add_seconds(self, seconds: float) -> None:
        """Adds training wall time of this epoch (excludes evaluation)."""
        self.seconds += seconds

    def means(self) -> dict[str, float | None]:
        """{"train_joint", "train_recon", ..., "train_p3"}: per-loss means (None if the loss never appeared)."""
        return {
            f"train_{k}": (self.loss_sum[k] / self.loss_count[k] if self.loss_count[k] else None)
            for k in ("joint", *LOSS_NAMES)
        }

    def row_pooled(self) -> list[float] | None:
        """Per-feature squared error pooled over every hidden real row of the epoch (None if no row was hidden)."""
        rows = float(self.hidden_rows)
        if rows == 0:
            return None
        return [float(v) for v in (self.square_sum / rows).tolist()]

    def state(self) -> dict[str, Any]:
        """Checkpointable state (CPU values)."""
        return {
            "loss_sum": dict(self.loss_sum),
            "loss_count": dict(self.loss_count),
            "grad_sum": self.grad_sum,
            "grad_max": self.grad_max,
            "steps": self.steps,
            "skipped": self.skipped,
            "seconds": self.seconds,
            "square_sum": self.square_sum.cpu().tolist(),
            "hidden_rows": float(self.hidden_rows),
        }

    def load(self, state: Mapping[str, Any]) -> None:
        """Restores `state()` (missing in checkpoints written at an epoch end: then nothing to restore)."""
        self.loss_sum, self.loss_count = dict(state["loss_sum"]), dict(state["loss_count"])
        self.grad_sum, self.grad_max = state["grad_sum"], state["grad_max"]
        self.steps, self.skipped, self.seconds = state["steps"], state["skipped"], state["seconds"]
        self.square_sum = torch.tensor(state["square_sum"], dtype=self.dtype, device=self.device)
        self.hidden_rows = torch.tensor(state["hidden_rows"], dtype=self.dtype, device=self.device)


class TrainingHistory:
    """Writes history.jsonl, feature_history.jsonl, steps.jsonl and summary.json of one run folder.

    Args:
        run_dir: the run folder.
        features: feature names in input order (13).
    """

    def __init__(self, run_dir: Path, features: Sequence[str] = tuple(FEATURES)) -> None:
        self.run_dir = Path(run_dir)
        self.features = list(features)
        self.history = JsonLines(self.run_dir / HISTORY_FILE)
        self.feature_history = JsonLines(self.run_dir / FEATURE_FILE)
        self.steps = JsonLines(self.run_dir / STEPS_FILE)

    def touch(self) -> None:
        """Creates the three JSON-lines files if missing (a reader finds them even before the first line)."""
        for log in (self.history, self.feature_history, self.steps):
            log.path.touch()

    def resume_from(self, epoch: int, step: int) -> dict[str, int]:
        """Drops every line written after the checkpoint (epoch > `epoch`, step > `step`); returns drop counts."""
        return {
            HISTORY_FILE: self.history.keep(lambda r: r["epoch"] <= epoch),
            FEATURE_FILE: self.feature_history.keep(lambda r: r["epoch"] <= epoch),
            STEPS_FILE: self.steps.keep(lambda r: r["step"] <= step),
        }

    def step(
        self,
        step: int,
        epoch: int,
        lr: float,
        grad_norm: float,
        joint: float,
        raw: Mapping[str, float],
        step_seconds: float,
    ) -> None:
        """One steps.jsonl line for one optimiser update (absent losses written as null)."""
        record: dict[str, Any] = {"step": step, "epoch": epoch, "lr": round_sig(lr), "grad_norm": round_sig(grad_norm)}
        record["joint"] = round_sig(joint)
        record.update({k: round_sig(raw.get(k)) for k in LOSS_NAMES})
        record["step_seconds"] = round_sig(step_seconds, 4)
        self.steps.append(record)

    def epoch_end(
        self,
        epoch: int,
        step: int,
        lr: float,
        evaluation: Mapping[str, Any],
        stats: EpochStats | None,
        scales: LossScales,
        is_best: bool,
        bad_epochs: int,
        partial: bool = False,
    ) -> dict[str, Any]:
        """Appends the history and feature-history lines of one epoch; `stats` None = the init line (epoch 0).

        Returns:
            the history.jsonl record.
        """
        record: dict[str, Any] = {"epoch": epoch, "step": step, "lr": clean(lr)}
        if stats is None:
            record.update({k: None for k in TRAIN_KEYS})
            record.update({"train_steps": 0, "train_skipped": 0, "grad_norm_mean": None, "grad_norm_max": None})
            record["epoch_seconds"] = None
        else:
            record.update(stats.means())
            record["train_steps"], record["train_skipped"] = stats.steps, stats.skipped
            record["grad_norm_mean"] = stats.grad_sum / stats.steps if stats.steps else None
            record["grad_norm_max"] = stats.grad_max
            record["epoch_seconds"] = round(stats.seconds, 2)
        record.update({k: clean(evaluation.get(k)) for k in EVAL_KEYS})
        for key, value in evaluation.items():  # every other scalar the evaluation reports (memory, eval_seconds, ...)
            if key not in record and key not in _EVAL_SKIP and not isinstance(value, (list, tuple, dict)):
                record[key] = clean(value)
        record["is_best"], record["bad_epochs"], record["partial"] = bool(is_best), int(bad_epochs), bool(partial)
        record.update({f"scale_{k}": (scales.scale(k) if scales.count[k] else None) for k in LOSS_NAMES})
        record["scales_frozen"] = bool(scales.frozen)
        self.history.append(record)
        self.feature_history.append(
            {
                "epoch": epoch,
                "features": self.features,
                "train_recon_row_pooled": stats.row_pooled() if stats is not None else None,
                "val_recon": [clean(v) for v in evaluation.get("val_recon_per_feature") or []] or None,
                "val_interp_baseline": [clean(v) for v in evaluation.get("val_recon_interp_baseline_per_feature") or []]
                or None,
                "partial": bool(partial),
            }
        )
        return record

    def write_summary(self, summary: Mapping[str, Any]) -> Path:
        """Writes summary.json (every key of SUMMARY_KEYS required, extra keys kept); returns its path.

        Raises:
            KeyError: a required key is missing.
        """
        missing = [k for k in SUMMARY_KEYS if k not in summary]
        if missing:
            raise KeyError(f"summary.json misses {missing}")
        path = self.run_dir / SUMMARY_FILE
        path.write_text(json.dumps({k: clean(v) for k, v in summary.items()}, indent=2, allow_nan=False))
        return path
