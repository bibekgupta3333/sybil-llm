"""Loss-curve image of a masked-autoencoder run, drawn from its `history.jsonl` and `summary.json`.

One panel per run: the training loss (`EncoderDecoderLoss` on the hidden cells, divided by the batch variance of each
feature, dropout on) and the benign val masked / full reconstruction MSE (deterministic scoring masks, pooled over every
real cell, dropout off) per epoch, the best epoch, and the LR (secondary axis, log scale) with every ReduceLROnPlateau
halving marked. The training loss and the val MSEs are different quantities (variance-normalised, per-window averaged
vs pooled raw MSE), so their gap is not a train / val generalisation gap.

Uses `matplotlib.figure.Figure` directly (no pyplot), so it works under any backend, also inside a notebook.

Usage (repo root):
    .venv-train/bin/python -m task.masked_autoencoder.plots --all
    .venv-train/bin/python -m task.masked_autoencoder.plots --run-dir task/masked_autoencoder/runs/T32-mlp-seed0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from task.masked_autoencoder.config import RUNS_DIR

PNG_NAME = "loss_curve.png"


class RunHistory:
    """Per-epoch history and summary of one finished run.

    Attributes:
        run_dir: the run folder.
        rows: one dict per epoch (history.jsonl).
        summary: summary.json.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.rows = [json.loads(line) for line in (self.run_dir / "history.jsonl").read_text().splitlines() if line]
        if not self.rows:
            raise ValueError(f"{self.run_dir}: history.jsonl is empty")
        self.summary = json.loads((self.run_dir / "summary.json").read_text())

    def column(self, key: str) -> list[float]:
        return [r[key] for r in self.rows]

    @property
    def epochs(self) -> list[int]:
        return self.column("epoch")

    @property
    def best_epoch(self) -> int:
        return int(self.summary["best_epoch"])

    @property
    def lr_drops(self) -> list[int]:
        """Epochs after which the LR was halved."""
        return [int(c["after_epoch"]) for c in self.summary.get("lr_changes", [])]


class LossCurvePlot:
    """Draws the loss curve of one run onto an axes, or into a stand-alone PNG."""

    def __init__(self, run_dir: str | Path) -> None:
        self.history = RunHistory(run_dir)

    def draw(self, ax: Axes, legend: bool = True, title: bool = True) -> Axes:
        """Draws the curves on `ax` and the LR on a twin axis; returns the twin axis."""
        h, s = self.history, self.history.summary
        ep = h.epochs
        ax.plot(ep, h.column("train_loss"), color="tab:blue", label="train loss (hidden cells, /var, dropout on)")
        ax.plot(ep, h.column("val_masked_mse"), color="tab:orange", label="val masked recon MSE (early stopping)")
        ax.plot(ep, h.column("val_full_mse"), color="tab:green", ls="--", label="val full recon MSE")
        best_val = h.rows[h.best_epoch - 1]["val_masked_mse"]
        ax.axvline(h.best_epoch, color="red", ls="--", lw=1, label=f"best epoch {h.best_epoch}")
        ax.plot([h.best_epoch], [best_val], "o", color="red", ms=4)
        for k, e in enumerate(h.lr_drops):
            ax.axvline(e + 0.5, color="grey", ls=":", lw=1, label="LR halved" if k == 0 else None)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss / MSE")
        lr_ax = ax.twinx()
        lr_ax.step(ep, h.column("lr"), where="post", color="tab:purple", lw=1, alpha=0.6, label="LR (right axis)")
        lr_ax.set_yscale("log")
        lr_ax.set_ylabel("learning rate", color="tab:purple", fontsize=8)
        lr_ax.tick_params(axis="y", labelcolor="tab:purple", labelsize=7)
        if title:
            ax.set_title(
                f"{s['run_id']}: best {h.best_epoch} / {s['epochs_run']} epochs ({s['stop_reason']}), "
                f"val masked {s['best_val_loss']:.4f}, {s['train_seconds'] / 60:.1f} min",
                fontsize=9,
            )
        if legend:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = lr_ax.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
        return lr_ax

    def figure(self) -> Figure:
        fig = Figure(figsize=(9, 4.8), layout="tight")
        self.draw(fig.add_subplot(1, 1, 1))
        return fig

    def save(self, path: str | Path | None = None) -> Path:
        """Writes the PNG (default `<run>/loss_curve.png`) and returns its path."""
        path = Path(path) if path else self.history.run_dir / PNG_NAME
        self.figure().savefig(path, dpi=120)
        return path


def finished_runs(runs_dir: str | Path = RUNS_DIR) -> list[Path]:
    """Run folders holding both history.jsonl and summary.json, sorted by name."""
    return sorted(p.parent for p in Path(runs_dir).glob("*/summary.json") if (p.parent / "history.jsonl").is_file())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", help="one run folder")
    group.add_argument("--all", action="store_true", help="every finished run under --runs-dir")
    p.add_argument("--runs-dir", default=RUNS_DIR)
    a = p.parse_args(argv)
    runs = finished_runs(a.runs_dir) if a.all else [Path(a.run_dir)]
    if not runs:
        raise SystemExit(f"no finished run under {a.runs_dir}")
    for run in runs:
        print(LossCurvePlot(run).save())


if __name__ == "__main__":
    main()
