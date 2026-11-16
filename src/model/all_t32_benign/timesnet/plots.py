"""Read-only loaders and plots for the analysis notebook of a TimesNet run on all/T32_benign.

Used by `src/model/all_t32_benign/timesnet_T32_benign.ipynb`. Nothing here trains, writes into a run folder or opens a
label file: `RunFolder` reads what `train.py` wrote, `EvalFolder` reads what `evaluate.py` wrote (score files were
saved before labels were read; the attack / benign identity of a window is its split name, checked by evaluate's
`LabelReader`), and `BaselineContext` reads the stored outputs of the one-class ConvAE notebook.

Every plotting method returns a matplotlib Figure, or None after printing a note when the file or key it needs is
missing (a smoke run with one epoch, a run without an evaluation yet, an older run without steps.jsonl).
"""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter

RUNS_DIR = Path("src/runs/pretraining/all/T32_benign")
EVAL_ROOT = Path("src/runs/evaluation/all/T32_benign")
BASELINE_NOTEBOOK = Path("task/autoencoder_benign_t32/autoencoder_benign_T32.ipynb")
BASELINE_RESULTS_JSON = Path("baselinepaper/01-our-studies/results.json")
BASELINE_KEY = "task/autoencoder_benign_t32"

ATTACKS = {
    "test_gridsybil": "GridSybil",
    "test_datareplay": "DataReplay",
    "test_dosrandom": "DoSRandom",
    "test_dosdisruptive": "DoSDisruptive",
}
SUBSETS = ("all", "n_ge_2", "n_le_3", "n_ge_4")
SUBSET_LABELS = {"all": "all", "n_ge_2": "n ≥ 2", "n_le_3": "n ≤ 3", "n_ge_4": "n ≥ 4"}
LOSSES = ("joint", "recon", "nce", "p1", "p2", "p3")
LOSS_TITLES = {
    "joint": "joint (weighted, scaled)",
    "recon": "reconstruction (masked MSE)",
    "nce": "contrastive (NT-Xent)",
    "p1": "P1 speed jump (BCE)",
    "p2": "P2 speed scaled (BCE)",
    "p3": "P3 sharp turn (BCE)",
}


def note(message: str) -> None:
    """Prints a skip note (the notebook shows it in place of the missing plot)."""
    print(f"[skipped] {message}")


def read_json(path: Path) -> dict[str, Any] | None:
    """Parsed JSON file, or None if it does not exist."""
    return json.loads(path.read_text()) if path.is_file() else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Records of a JSON-lines file ([] if missing); a half-written last line is skipped."""
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"skipped an unreadable line in {path.name}: {line[:60]}")
    return out


def column(df: pd.DataFrame | None, key: str) -> pd.Series | None:
    """Numeric column `key` of `df`, or None if absent or all null."""
    if df is None or df.empty or key not in df.columns:
        return None
    s = pd.to_numeric(df[key], errors="coerce")
    return None if s.isna().all() else s


class RunFolder:
    """One pretraining run folder, read-only.

    Attributes:
        path: the folder.
        config, env, summary: config.json, env.json, summary.json ({} if missing).
        history: one row per epoch (history.jsonl; epoch 0 = the random-init evaluation, train fields null).
        feature_history: per-epoch per-feature records (feature_history.jsonl).
        steps: one row per optimiser update (steps.jsonl).
        metrics: the raw metrics.jsonl records.
        log: train.log lines.
    """

    REQUIRED = "history.jsonl"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.config = read_json(self.path / "config.json") or {}
        self.env = read_json(self.path / "env.json") or {}
        self.summary = read_json(self.path / "summary.json") or {}
        self.history = pd.DataFrame(read_jsonl(self.path / "history.jsonl"))
        if not self.history.empty and "epoch" in self.history:
            self.history = self.history.sort_values("epoch").reset_index(drop=True)
        self.feature_history = read_jsonl(self.path / "feature_history.jsonl")
        self.steps = pd.DataFrame(read_jsonl(self.path / "steps.jsonl"))
        self.metrics = read_jsonl(self.path / "metrics.jsonl")
        log = self.path / "train.log"
        self.log = log.read_text().splitlines() if log.is_file() else []

    @classmethod
    def newest(cls, runs_dir: Path = RUNS_DIR) -> "RunFolder":
        """The newest run folder under `runs_dir` that has a history.jsonl (by that file's mtime).

        Raises:
            FileNotFoundError: no such run folder.
        """
        found = sorted(Path(runs_dir).glob(f"*/{cls.REQUIRED}"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise FileNotFoundError(f"no run folder with {cls.REQUIRED} under {runs_dir}")
        return cls(found[-1].parent)

    @classmethod
    def pick(cls, name: str | Path | None, runs_dir: Path = RUNS_DIR) -> "RunFolder":
        """`name` (a folder name under runs_dir or a path), or the newest run when None."""
        if name is None:
            return cls.newest(runs_dir)
        path = Path(name)
        path = path if path.is_dir() else Path(runs_dir) / str(name)
        if not path.is_dir():
            raise FileNotFoundError(f"run folder {path} not found")
        return cls(path)

    @property
    def run_id(self) -> str:
        """The folder name (= run id)."""
        return self.path.name

    @property
    def epochs(self) -> pd.DataFrame:
        """History rows of trained epochs (epoch ≥ 1)."""
        if self.history.empty or "epoch" not in self.history:
            return self.history
        return self.history[self.history.epoch >= 1]

    @property
    def best_epoch(self) -> int | None:
        """summary.json best_epoch, else the last history row flagged is_best, else argmin val_joint (epoch ≥ 1)."""
        if self.summary.get("best_epoch") is not None:
            return int(self.summary["best_epoch"])
        ep = self.epochs
        if "is_best" in ep and ep.is_best.fillna(False).astype(bool).any():
            return int(ep[ep.is_best.fillna(False).astype(bool)].epoch.iloc[-1])
        val = column(ep, "val_joint")
        if val is not None:
            return int(ep.epoch.loc[val.idxmin()])
        return None

    def eval_records(self) -> pd.DataFrame:
        """metrics.jsonl evaluation records (scalar fields only)."""
        rows = [{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in self.metrics]
        df = pd.DataFrame([r for r in rows if r.get("kind") == "eval"])
        return df

    def is_short(self) -> bool:
        """True for a smoke / short / check run (numbers are plumbing checks, not results)."""
        name = self.run_id
        flagged = self.summary.get("short") or self.config.get("short")
        return bool(flagged) or any(t in name for t in ("smoke", "short", "check"))


class EvalFolder:
    """One evaluation folder written by evaluate.py, read-only.

    Attributes:
        path: the folder.
        summary: summary.json.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.summary = read_json(self.path / "summary.json") or {}
        self._scores: dict[str, dict[str, np.ndarray]] = {}

    @staticmethod
    def candidates(run_id: str, eval_root: Path = EVAL_ROOT, random_init: bool = False) -> list[Path]:
        """Evaluation folders of `run_id` with a summary.json: full ones first (newest first), then smoke ones."""
        base = f"{run_id}-random-init" if random_init else run_id
        full, smoke = [], []
        for summary in Path(eval_root).glob("*/summary.json"):
            name = summary.parent.name
            if name == base:
                full.append(summary.parent)
            elif name.startswith(base + "-smoke-"):
                smoke.append(summary.parent)
        newest = lambda paths: sorted(paths, key=lambda p: (p / "summary.json").stat().st_mtime, reverse=True)
        return newest(full) + newest(smoke)

    @classmethod
    def find(
        cls, run_id: str, name: str | Path | None = None, eval_root: Path = EVAL_ROOT, random_init: bool = False
    ) -> "EvalFolder | None":
        """`name` if given, else the best match for `run_id` (a full evaluation before a smoke one); None if none."""
        if name is not None:
            path = Path(name)
            path = path if path.is_dir() else Path(eval_root) / str(name)
            return cls(path) if (path / "summary.json").is_file() else None
        found = cls.candidates(run_id, eval_root, random_init)
        return cls(found[0]) if found else None

    @property
    def smoke(self) -> bool:
        """True for a smoke evaluation (a few windows per split: plumbing only)."""
        return bool(self.summary.get("smoke"))

    def scores(self, split: str) -> dict[str, np.ndarray] | None:
        """Arrays of scores/<split>.npz (ids, n_real_rows, recon, recon_raw, recon_defined, mahalanobis, knn_k*)."""
        if split not in self._scores:
            path = self.path / "scores" / f"{split}.npz"
            if not path.is_file():
                return None
            with np.load(path) as f:
                self._scores[split] = {k: f[k] for k in f.files}
        return self._scores[split]

    def score_names(self) -> list[str]:
        """Score names in the summary metrics (recon, mahalanobis, knn_k*, length_only), in summary order."""
        for m in self.summary.get("metrics", {}).values():
            return list(m.get("window", {}))
        return []

    def threshold(self, score: str, q: str = "p95", level: str = "window") -> float | None:
        """Benign val threshold of a score (summary thresholds)."""
        return self.summary.get("thresholds", {}).get(level, {}).get(score, {}).get(q)

    def label_checks(self) -> dict[str, Any]:
        """The evidence that labels were read after scores and aligned (from summary.json)."""
        events = self.summary.get("events", {})
        align = self.summary.get("alignment", {})
        meta = self.summary.get("metadata_check", {})
        return {
            "scores_written_utc": events.get("scores_written_utc"),
            "labels_first_read_utc": events.get("labels_first_read_utc"),
            "labels_read_after_scores": events.get("labels_read_after_scores"),
            "alignment checks": align.get("checks"),
            "info shards opened": align.get("info_shards_opened"),
            "metadata problems": meta.get("problems"),
            "smoke (plumbing only)": self.smoke,
            "weights": self.summary.get("checkpoint", {}).get("weights"),
        }


class AurocTable:
    """AUROC tables from an evaluation summary (no recomputation, no labels)."""

    @staticmethod
    def long(summary: dict[str, Any]) -> pd.DataFrame:
        """One row per (attack, score, subset): AUROC, PR-AUC, recall / benign FPR at val p95, counts."""
        rows = []
        for split, m in summary.get("metrics", {}).items():
            for score, subsets in m.get("window", {}).items():
                for subset, v in subsets.items():
                    p95 = (v.get("at_threshold") or {}).get("p95", {})
                    rows.append(
                        {
                            "attack": ATTACKS.get(split, split),
                            "score": score,
                            "subset": subset,
                            "AUROC": v.get("auroc"),
                            "PR-AUC": v.get("pr_auc"),
                            "chance": v.get("chance"),
                            "recall@p95": p95.get("recall"),
                            "benign FPR@p95": p95.get("benign_fpr"),
                            "n benign": v.get("n_benign"),
                            "n attack": v.get("n_attack"),
                        }
                    )
        return pd.DataFrame(rows)

    @classmethod
    def wide(cls, summary: dict[str, Any]) -> pd.DataFrame:
        """Rows (attack, score); per subset the score's AUROC and the length-only AUROC on the same windows."""
        df = cls.long(summary)
        if df.empty:
            return df
        out = {}
        for (attack, score), g in df[df.score != "length_only"].groupby(["attack", "score"], sort=False):
            row = {}
            length = df[(df.attack == attack) & (df.score == "length_only")].set_index("subset")["AUROC"]
            for sub in SUBSETS:
                hit = g[g.subset == sub]
                row[(SUBSET_LABELS[sub], "AUROC")] = hit.AUROC.iloc[0] if len(hit) else np.nan
                row[(SUBSET_LABELS[sub], "length only")] = length.get(sub, np.nan)
            out[(attack, score)] = row
        wide = pd.DataFrame(out).T
        wide.columns = pd.MultiIndex.from_tuples(wide.columns)
        wide.index.names = ["attack", "score"]
        fill = cls.fill_shares(summary)
        if not fill.empty:
            wide[("recon fill share", "benign")] = [fill.loc[a, "benign test"] for a, _ in wide.index]
            wide[("recon fill share", "attack")] = [fill.loc[a, "attack"] for a, _ in wide.index]
        return wide.astype(float)

    @staticmethod
    def fill_shares(summary: dict[str, Any]) -> pd.DataFrame:
        """Share of windows whose reconstruction score is filled (< 2 real rows), benign test vs each attack."""
        rows = {}
        for split, m in summary.get("metrics", {}).items():
            share = m.get("counts", {}).get("recon_filled_share", {})
            rows[ATTACKS.get(split, split)] = {"benign test": share.get("benign"), "attack": share.get("attack")}
        return pd.DataFrame(rows).T

    @staticmethod
    def tracks(summary: dict[str, Any]) -> pd.DataFrame:
        """Per-track max / mean AUROC per attack and score."""
        rows = {}
        for split, m in summary.get("metrics", {}).items():
            for score, levels in m.get("track", {}).items():
                rows[(ATTACKS.get(split, split), score)] = {lv: v.get("auroc") for lv, v in levels.items()}
        df = pd.DataFrame(rows).T
        if not df.empty:
            df.index.names = ["attack", "score"]
        return df


class BaselineContext:
    """AUROCs of the one-class ConvAE `task/autoencoder_benign_t32` (same split and test windows), read from disk.

    Source order: the stored output of the notebook's results cell (the numbers as computed there), else the sourced
    inventory `baselinepaper/01-our-studies/results.json`. Nothing is hard-coded.
    """

    RESULTS_CELL_MARKER = 'results = {"run": RUN.name'

    def __init__(self, metrics: dict[str, Any], source: str) -> None:
        self.metrics, self.source = metrics, source

    @classmethod
    def load(
        cls, notebook: Path = BASELINE_NOTEBOOK, results_json: Path = BASELINE_RESULTS_JSON
    ) -> "BaselineContext | None":
        """The baseline metrics, or None if neither source is present."""
        if notebook.is_file():
            nb = json.loads(notebook.read_text())
            for cell in nb.get("cells", []):
                if cell.get("cell_type") != "code" or cls.RESULTS_CELL_MARKER not in "".join(cell.get("source", [])):
                    continue
                for out in cell.get("outputs", []):
                    text = "".join(out.get("data", {}).get("text/plain", []))
                    if text.startswith("{"):
                        try:
                            return cls(ast.literal_eval(text), f"{notebook} (stored output of the results cell)")
                        except (ValueError, SyntaxError):
                            break
        data = read_json(results_json)
        if data and BASELINE_KEY in data:
            test = data[BASELINE_KEY].get("test", {})
            if "metrics" in test:
                return cls(test["metrics"], f"{results_json} ({test.get('source', '')})")
        return None

    def table(self) -> pd.DataFrame:
        """Rows (attack, score): baseline window AUROC on all windows and on windows with ≥ 4 real rows."""
        names = {"reconstruction": "recon", "mahalanobis": "mahalanobis", "length only (reference)": "length_only"}
        rows = {}
        for key, v in self.metrics.items():
            parts = key.split("/")
            if len(parts) != 3 or parts[2] != "window" or not isinstance(v, dict):
                continue
            rows[(parts[0], names.get(parts[1], parts[1]))] = {
                "ConvAE all": v.get("AUROC"),
                "ConvAE n ≥ 4": v.get("AUROC (>= 4 rows)"),
            }
        df = pd.DataFrame(rows).T
        if not df.empty:
            df.index.names = ["attack", "score"]
        return df

    def compare(self, summary: dict[str, Any]) -> pd.DataFrame:
        """Baseline AUROCs next to TimesNet's (all / n ≥ 4) for the scores both have."""
        base = self.table()
        tn = AurocTable.long(summary)
        if base.empty or tn.empty:
            return pd.DataFrame()
        rows = {}
        for (attack, score), b in base.iterrows():
            hit = tn[(tn.attack == attack) & (tn.score == score)].set_index("subset")["AUROC"]
            rows[(attack, score)] = {
                "ConvAE all": b["ConvAE all"],
                "TimesNet all": hit.get("all", np.nan),
                "ConvAE n ≥ 4": b["ConvAE n ≥ 4"],
                "TimesNet n ≥ 4": hit.get("n_ge_4", np.nan),
            }
        df = pd.DataFrame(rows).T.astype(float)
        df.index.names = ["attack", "score"]
        return df


def _mark_best(ax: plt.Axes, best: int | None) -> None:
    if best is not None:
        ax.axvline(best, color="red", ls="--", lw=1, label=f"best epoch {best}")


def _empty(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=9, color="grey")
    ax.set_xticks([])
    ax.set_yticks([])


class TrainingPlots:
    """Plots of the training record (history.jsonl, steps.jsonl, metrics.jsonl)."""

    def __init__(self, run: RunFolder) -> None:
        self.run = run

    def losses(self) -> Figure | None:
        """Each loss, train vs val per epoch (val includes the epoch-0 random-init point), best epoch marked."""
        h = self.run.history
        if h.empty or column(h, "val_joint") is None:
            note("history.jsonl missing or without val_joint")
            return None
        best = self.run.best_epoch
        fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), squeeze=False)
        for ax, key in zip(axes.ravel(), LOSSES):
            tr, va = column(h, f"train_{key}"), column(h, f"val_{key}")
            if tr is None and va is None:
                _empty(ax, f"{key}: not logged")
                ax.set_title(LOSS_TITLES[key], fontsize=10)
                continue
            if tr is not None:
                ax.plot(h.epoch, tr, marker=".", label="train (epoch mean)")
            if va is not None:
                ax.plot(h.epoch, va, marker=".", label="val (benign)")
            if key in ("p1", "p2", "p3"):
                ax.axhline(math.log(2), color="grey", ls=":", lw=1, label="chance (ln 2)")
            _mark_best(ax, best)
            ax.set_title(LOSS_TITLES[key], fontsize=10)
            ax.set_xlabel("epoch (0 = random init, val only)")
            ax.set_ylabel("loss")
            ax.legend(fontsize=7)
        fig.suptitle(f"{self.run.run_id}: losses per epoch, train vs val")
        fig.tight_layout()
        return fig

    def recon_vs_interpolation(self) -> Figure | None:
        """Val reconstruction per epoch against the linear-interpolation baseline on the same hidden rows."""
        h = self.run.history
        val, base = column(h, "val_recon"), column(h, "val_recon_interp_baseline")
        if val is None or base is None:
            note("history.jsonl has no val_recon / val_recon_interp_baseline")
            return None
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(h.epoch, val, marker=".", label="model (val masked MSE)")
        ax.plot(h.epoch, base, ls="--", color="black", label="linear interpolation of hidden rows (val)")
        tr = column(h, "train_recon")
        if tr is not None:
            ax.plot(h.epoch, tr, marker=".", alpha=0.6, label="model (train, epoch mean)")
        _mark_best(ax, self.run.best_epoch)
        ax.set_yscale("log")
        ax.set_xlabel("epoch (0 = random init)")
        ax.set_ylabel("masked MSE on hidden real rows (log)")
        ax.set_title("Reconstruction vs the interpolation baseline: the model must go below the dashed line")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def steps(self, smooth: int | None = None) -> Figure | None:
        """Per-update losses (raw + rolling mean), LR overlay on the joint panel, gradient norm."""
        s = self.run.steps
        if s.empty or "step" not in s:
            note("steps.jsonl missing or empty")
            return None
        smooth = smooth or max(1, len(s) // 50)
        fig, axes = plt.subplots(2, 4, figsize=(18, 7.5), squeeze=False)
        panels = [*LOSSES, "grad_norm"]
        for ax, key in zip(axes.ravel(), panels):
            y = column(s, key)
            title = LOSS_TITLES.get(key, "gradient L2 norm (pre-clip)")
            ax.set_title(title, fontsize=10)
            if y is None:
                _empty(ax, f"{key}: not logged")
                continue
            ax.plot(s.step, y, lw=0.5, alpha=0.35, label="per update")
            ax.plot(s.step, y.rolling(smooth, min_periods=1).mean(), lw=1.4, label=f"rolling mean ({smooth})")
            ax.set_xlabel("optimiser update")
            ax.set_ylabel("loss" if key != "grad_norm" else "L2 norm")
            if key == "grad_norm" and self.run.config.get("grad_clip"):
                ax.axhline(self.run.config["grad_clip"], color="black", ls="--", lw=1, label="clip")
            if key == "joint" and column(s, "lr") is not None:
                lr_ax = ax.twinx()
                lr_ax.plot(s.step, column(s, "lr"), color="tab:purple", lw=1, label="LR")
                lr_ax.set_ylabel("learning rate", color="tab:purple")
            ax.legend(fontsize=7, loc="upper right")
        ax = axes.ravel()[-1]
        lr = column(s, "lr")
        if lr is not None:
            ax.plot(s.step, lr, color="tab:purple")
            ax.set_title("learning rate schedule", fontsize=10)
            ax.set_xlabel("optimiser update")
            ax.set_ylabel("learning rate")
        else:
            _empty(ax, "lr: not logged")
        fig.suptitle(f"{self.run.run_id}: per-update curves ({len(s)} updates)")
        fig.tight_layout()
        return fig

    def physics_auroc(self) -> Figure | None:
        """P1-P3 AUROC on val per epoch (detection of violations injected into benign windows; no attack labels)."""
        h = self.run.history
        keys = [k for k in ("auroc_p1", "auroc_p2", "auroc_p3") if column(h, k) is not None]
        if not keys:
            note("history.jsonl has no auroc_p1..p3")
            return None
        fig, ax = plt.subplots(figsize=(8, 4))
        for k in keys:
            ax.plot(h.epoch, column(h, k), marker=".", label=k.replace("auroc_", "").upper())
        ax.axhline(0.5, color="black", ls="--", lw=1, label="chance")
        _mark_best(ax, self.run.best_epoch)
        ax.set_ylim(0.3, 1.02)
        ax.set_xlabel("epoch (0 = random init)")
        ax.set_ylabel("AUROC (injected vs clean)")
        ax.set_title("Physics heads on val: injected-violation detection (not attack detection)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def representation(self) -> Figure | None:
        """Collapse and length-shortcut monitors per epoch."""
        h = self.run.history
        panels = [
            ("effective rank of z", ["z_effective_rank"], "rank (max d)"),
            ("z std per dimension", ["z_std_mean", "z_std_min"], "std"),
            ("length shortcut: ridge R² of n rows from z", ["length_probe_r2"], "R²"),
            ("corr(|z|, n rows)", ["corr_znorm_length"], "Pearson r"),
            ("alignment / uniformity", ["align", "uniformity"], "value (lower = better)"),
            ("cosine: two views vs other windows", ["cos_pos", "cos_neg"], "mean cosine"),
        ]
        if not any(column(h, k) is not None for _, keys, _ in panels for k in keys):
            note("history.jsonl has no representation monitors")
            return None
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), squeeze=False)
        for ax, (title, keys, ylab) in zip(axes.ravel(), panels):
            ax.set_title(title, fontsize=10)
            present = [k for k in keys if column(h, k) is not None]
            if not present:
                _empty(ax, "not logged")
                continue
            for k in present:
                ax.plot(h.epoch, column(h, k), marker=".", label=k)
            if keys == ["z_effective_rank"] and self.run.config.get("d_model"):
                ax.axhline(self.run.config["d_model"], color="grey", ls=":", lw=1, label="d")
            if keys == ["corr_znorm_length"]:
                ax.axhline(0, color="grey", ls=":", lw=1)
            _mark_best(ax, self.run.best_epoch)
            ax.set_xlabel("epoch (0 = random init)")
            ax.set_ylabel(ylab)
            ax.legend(fontsize=7)
        fig.suptitle(f"{self.run.run_id}: representation health on val")
        fig.tight_layout()
        return fig

    def time_memory(self) -> Figure | None:
        """Training and evaluation seconds per epoch, and memory per evaluation (GB; history, else metrics.jsonl)."""
        h = self.run.history
        secs, eval_secs = column(h, "epoch_seconds"), column(h, "eval_seconds")
        mem_src = h
        mem_keys = [c for c in h.columns if c.endswith("_gb") and column(h, c) is not None] if not h.empty else []
        if not mem_keys:
            mem_src = self.run.eval_records()
            mem_keys = [c for c in mem_src.columns if c.endswith("_gb") and column(mem_src, c) is not None]
        if secs is None and eval_secs is None and not mem_keys:
            note("no epoch_seconds / eval_seconds / memory fields in history.jsonl or metrics.jsonl")
            return None
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 3.8))
        if secs is not None or eval_secs is not None:
            train_min = (secs.fillna(0) if secs is not None else pd.Series(0.0, index=h.index)) / 60.0
            a1.bar(h.epoch, train_min, color="tab:blue", label="training updates")
            if eval_secs is not None:
                a1.bar(
                    h.epoch, eval_secs.fillna(0) / 60.0, bottom=train_min, color="tab:orange", label="val evaluation"
                )
            a1.set_xlabel("epoch (0 = random-init evaluation)")
            a1.set_ylabel("minutes")
            a1.set_title("wall-clock time per epoch")
            a1.legend(fontsize=7)
        else:
            _empty(a1, "epoch_seconds: not logged")
        if mem_keys:
            x = mem_src["epoch"] if "epoch" in mem_src else np.arange(len(mem_src))
            for k in mem_keys:
                a2.plot(x, column(mem_src, k), marker="o", label=k)
            a2.set_xlabel("epoch of the evaluation")
            a2.set_ylabel("GB")
            a2.set_title("memory at each evaluation (CUDA: peak since the previous one)")
            a2.legend(fontsize=7)
        else:
            _empty(a2, "memory: not logged")
        fig.tight_layout()
        return fig

    def loss_scales(self) -> Figure | None:
        """Running loss scales (normalisation of each objective, D7) per epoch, if logged."""
        h = self.run.history
        keys = [c for c in h.columns if c.startswith("scale_") and column(h, c) is not None] if not h.empty else []
        if not keys:
            note("history.jsonl has no scale_* fields")
            return None
        fig, ax = plt.subplots(figsize=(8, 3.8))
        for k in keys:
            ax.plot(h.epoch, column(h, k), marker=".", label=k)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("scale (log)")
        ax.set_title("Loss normalisation scales (each raw loss is divided by its scale)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig


class FeaturePlots:
    """Per-feature reconstruction error over epochs (feature_history.jsonl)."""

    def __init__(self, run: RunFolder, features: Sequence[str]) -> None:
        self.run, self.features = run, list(features)

    def frame(self, key: str) -> pd.DataFrame | None:
        """(epochs × 13) of one per-feature field; None if never logged."""
        rows = {r["epoch"]: r[key] for r in self.run.feature_history if r.get(key) is not None and "epoch" in r}
        if not rows:
            return None
        names = self.run.feature_history[0].get("features") or self.features
        return pd.DataFrame(rows, index=list(names)).T.sort_index()

    def curves(self) -> Figure | None:
        """13 small multiples: train (row-pooled), val and val interpolation baseline per epoch, best epoch marked."""
        val = self.frame("val_recon")
        if val is None:
            note("feature_history.jsonl missing or without val_recon")
            return None
        train, base = self.frame("train_recon_row_pooled"), self.frame("val_interp_baseline")
        ncols = 5
        nrows = math.ceil(len(val.columns) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.1 * nrows), squeeze=False)
        best = self.run.best_epoch
        for ax, f in zip(axes.ravel(), val.columns):
            if train is not None and f in train:
                ax.plot(train.index, train[f], marker=".", ms=3, label="train (row-pooled)")
            ax.plot(val.index, val[f], marker=".", ms=3, label="val")
            if base is not None and f in base:
                ax.plot(base.index, base[f], ls="--", color="black", lw=1, label="val interpolation")
            _mark_best(ax, best)
            ax.set_yscale("log")
            ax.set_title(f, fontsize=9)
            ax.set_xlabel("epoch")
            ax.set_ylabel("MSE (log)")
        for ax in axes.ravel()[len(val.columns) :]:
            ax.axis("off")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("Per-feature reconstruction error on hidden rows, per epoch")
        fig.tight_layout()
        return fig

    def best_bar(self, epoch: int | None = None) -> tuple[Figure | None, pd.DataFrame | None]:
        """Per-feature val error at `epoch` (default the best epoch) next to the interpolation baseline."""
        val, base = self.frame("val_recon"), self.frame("val_interp_baseline")
        if val is None:
            note("feature_history.jsonl missing or without val_recon")
            return None, None
        epoch = epoch if epoch is not None else self.run.best_epoch
        if epoch not in val.index:
            epoch = int(val.index[-1])
        table = pd.DataFrame({"model (val)": val.loc[epoch]})
        if base is not None and epoch in base.index:
            table["interpolation (val)"] = base.loc[epoch]
            table["model / interpolation"] = table["model (val)"] / table["interpolation (val)"]
        fig, ax = plt.subplots(figsize=(11, 4))
        pos = np.arange(len(table))
        width = 0.4 if "interpolation (val)" in table else 0.8
        ax.bar(pos - (width / 2 if width < 0.8 else 0), table["model (val)"], width, label="model")
        if "interpolation (val)" in table:
            ax.bar(pos + width / 2, table["interpolation (val)"], width, label="linear interpolation", color="grey")
        ax.set_xticks(pos, table.index, rotation=45, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("val MSE on hidden rows (log)")
        ax.set_title(f"Per-feature val reconstruction error at epoch {epoch}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig, table


class ReconExamples:
    """Masked reconstruction of a few val windows with the run's checkpoint, on the CPU (seeded masks).

    Uses the package pieces evaluate.py uses: `PretrainingModel`, `ReconMasker`, the run's scaling (`restore_space`)
    and `Rescaler` (stored train z-scores -> the trained scaling). Values are shown in the trained scaling.
    """

    def __init__(self, checkpoint: Path, metadata: dict[str, Any]) -> None:
        import torch

        from src.model.all_t32_benign.timesnet.data import restore_space
        from src.model.all_t32_benign.timesnet.evaluate import Rescaler
        from src.model.all_t32_benign.timesnet.base_config import PretrainConfig
        from src.model.all_t32_benign.timesnet.base_data import FeatureSpace
        from src.model.all_t32_benign.timesnet.heads import PretrainingModel
        from src.model.all_t32_benign.timesnet.views import ReconMasker

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.settings = PretrainConfig.from_dict(ckpt["settings"])
        self.model = PretrainingModel(self.settings)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.masker = ReconMasker(self.settings)
        self.rescaler = Rescaler(FeatureSpace(metadata), restore_space(metadata, ckpt["space"]))
        self.epoch, self.step = ckpt.get("epoch"), ckpt.get("step")

    def run(self, x: Any, mask: Any, seed: int = 0) -> dict[str, np.ndarray]:
        """x (B, T, 13) stored scaling + mask -> {"x", "pred", "hidden", "mask"} in the trained scaling."""
        import torch

        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            mask = torch.as_tensor(mask).bool()
            xt = self.rescaler(x, mask)
            hidden = self.masker(mask, torch.Generator().manual_seed(seed))
            h, _ = self.model.encoder(self.model.hide(xt, hidden), mask)
            pred = self.model.recon_head(h)
        return {"x": xt.numpy(), "pred": pred.numpy(), "hidden": hidden.numpy(), "mask": mask.numpy()}

    @staticmethod
    def interpolate(x: np.ndarray, hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Linear interpolation of the hidden rows from the visible real rows (the baseline of the training monitor)."""
        out = x.copy()
        n = int(mask.sum())
        vis = np.flatnonzero(mask[:n] & ~hidden[:n])
        hid = np.flatnonzero(hidden[:n])
        if len(vis) and len(hid):
            for j in range(x.shape[1]):
                out[hid, j] = np.interp(hid, vis, x[vis, j])
        return out

    @classmethod
    def plot(
        cls, out: dict[str, np.ndarray], features: Sequence[str], show: Sequence[str], ids: Sequence[str]
    ) -> Figure:
        """Rows = windows, columns = features: input, model output, interpolation; hidden rows shaded."""
        b = len(out["x"])
        fig, axes = plt.subplots(b, len(show), figsize=(4.4 * len(show), 2.6 * b), squeeze=False)
        for r in range(b):
            n = int(out["mask"][r].sum())
            interp = cls.interpolate(out["x"][r], out["hidden"][r], out["mask"][r])
            hid = np.flatnonzero(out["hidden"][r][:n])
            for c, f in enumerate(show):
                j = list(features).index(f)
                ax = axes[r, c]
                for t in hid:
                    ax.axvspan(t - 0.5, t + 0.5, color="grey", alpha=0.2, lw=0)
                ax.plot(np.arange(n), out["x"][r, :n, j], marker=".", label="input")
                ax.plot(np.arange(n), out["pred"][r, :n, j], ls="--", label="model output")
                ax.plot(
                    hid, interp[hid, j], "x", color="black", label="linear interpolation (hidden rows; edge value held)"
                )
                ax.set_title(f"{f} ({n} rows, {len(hid)} hidden)", fontsize=9)
                ax.set_xlabel("message (1 Hz)")
                ax.set_ylabel("trained scaling")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("Val windows: hidden rows shaded; the model sees the mask token there, not the input")
        fig.tight_layout()
        return fig


class TestPlots:
    """Plots of saved score files (benign test vs each attack split; the split name is the class)."""

    def __init__(self, ev: EvalFolder) -> None:
        self.ev = ev

    def histograms(self, scores: Sequence[str] = ("recon", "mahalanobis", "knn_k10")) -> Figure | None:
        """Score histograms per window, benign test vs each attack, with the benign val p95 threshold."""
        benign = self.ev.scores("test")
        if benign is None:
            note("no scores/test.npz in the evaluation folder")
            return None
        scores = [s for s in scores if s in benign]
        attacks = [a for a in ATTACKS if self.ev.scores(a) is not None]
        if not scores or not attacks:
            note("no attack score files or none of the requested scores")
            return None
        fig, axes = plt.subplots(
            len(scores), len(attacks), figsize=(4.2 * len(attacks), 3.2 * len(scores)), squeeze=False
        )
        for r, score in enumerate(scores):
            vals = np.concatenate([benign[score]] + [self.ev.scores(a)[score] for a in attacks])
            vals = vals[np.isfinite(vals)]
            positive = vals[vals > 0]
            log = score != "length_only" and len(positive) == len(vals) and len(vals) > 0
            if log:
                bins = np.logspace(np.log10(positive.min()), np.log10(max(positive.max(), positive.min() * 1.01)), 60)
            else:
                bins = np.linspace(vals.min(), max(vals.max(), vals.min() + 1e-9), 60)
            th = self.ev.threshold(score)
            for c, split in enumerate(attacks):
                ax = axes[r, c]
                ax.hist(benign[score], bins=bins, density=True, alpha=0.55, label="benign test")
                ax.hist(
                    self.ev.scores(split)[score],
                    bins=bins,
                    density=True,
                    alpha=0.55,
                    color="tab:red",
                    label=ATTACKS[split],
                )
                if th is not None:
                    ax.axvline(th, color="black", ls="--", lw=1, label="val p95")
                if log:
                    ax.set_xscale("log")
                    ax.xaxis.set_minor_formatter(NullFormatter())
                auroc = (
                    self.ev.summary.get("metrics", {})
                    .get(split, {})
                    .get("window", {})
                    .get(score, {})
                    .get("all", {})
                    .get("auroc")
                )
                title = f"{score}: {ATTACKS[split]}" + (f", AUROC {auroc:.3f}" if auroc is not None else "")
                ax.set_title(title, fontsize=9)
                ax.set_xlabel(f"{score} score" + (" (log)" if log else ""))
                ax.set_ylabel("density")
                ax.legend(fontsize=7)
        fig.suptitle("Window scores, benign test vs attack (recon of < 2-row windows = filled val median)")
        fig.tight_layout()
        return fig

    def recon_by_rows(self) -> Figure | None:
        """Measured reconstruction error (recon_raw, ≥ 2 real rows) vs real rows: median and IQR per length."""
        splits = [("test", "benign test")] + [(a, n) for a, n in ATTACKS.items()]
        data = [(s, lab, self.ev.scores(s)) for s, lab in splits if self.ev.scores(s) is not None]
        if not data or "recon_raw" not in data[0][2]:
            note("no score files with recon_raw")
            return None
        fig, ax = plt.subplots(figsize=(10, 4.2))
        for _, lab, sc in data:
            ok = np.isfinite(sc["recon_raw"])
            if not ok.any():
                continue
            df = pd.DataFrame({"n": sc["n_real_rows"][ok], "e": sc["recon_raw"][ok]})
            g = df.groupby("n").e
            med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
            line = ax.plot(med.index, med.values, marker=".", label=f"{lab} ({int(ok.sum()):,} windows)")[0]
            ax.fill_between(med.index, lo.values, hi.values, color=line.get_color(), alpha=0.12)
        th = self.ev.threshold("recon")
        if th is not None:
            ax.axhline(th, color="black", ls="--", lw=1, label="val p95 (recon)")
        ax.set_yscale("log")
        ax.set_xlabel("real messages per window")
        ax.set_ylabel("masked MSE on hidden rows (log)")
        ax.set_title("Measured reconstruction error by window length (median, IQR band; < 2-row windows excluded)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        return fig

    def length_mix(self) -> pd.DataFrame:
        """Windows per split and their share with ≤ 3 / ≥ 4 / < 2 real rows (from the score files)."""
        rows = {}
        for split, lab in [("val", "benign val"), ("test", "benign test"), *ATTACKS.items()]:
            sc = self.ev.scores(split)
            if sc is None:
                continue
            n = sc["n_real_rows"]
            rows[lab] = {
                "windows": len(n),
                "share n < 2 (recon filled)": float((n < 2).mean()) if len(n) else np.nan,
                "share n ≤ 3": float((n <= 3).mean()) if len(n) else np.nan,
                "share n ≥ 4": float((n >= 4).mean()) if len(n) else np.nan,
            }
        return pd.DataFrame(rows).T
