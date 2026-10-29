"""Evaluation of the benign vs GridSybil baseline encoders on the test split.

Reads the run folders written by ``task/encoder/train.py`` (``scores_test.npz``, ``scores_val.npz``,
``config.json``) and the shortcut-baseline folder written by ``task/encoder/baselines.py``, and writes
``results.json`` + PNG figures to ``task/encoder/runs/evaluation/``.

Metrics (GridSybil = positive): AUROC, PR-AUC, precision / recall / F1 / FPR + confusion matrix at the 95th and
99th percentile of validation-benign scores (one-class scores) or at 0.5 (supervised probability), mean +- std over
seeds, a 95% bootstrap CI of AUROC over sender vehicles, AUROC per length stratum next to the length-only AUROC in
the same stratum, and per-feature reconstruction error benign vs GridSybil. Thresholds never see attack data.

Run from the repo root::

    .venv-train/bin/python -m task.encoder.evaluate --runs task/encoder/runs/<run_id> ...
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINES = REPO_ROOT / "task" / "encoder" / "runs" / "baselines"
DEFAULT_OUT = REPO_ROOT / "task" / "encoder" / "runs" / "evaluation"
FEATURES = (
    "claimed_pos_x",
    "claimed_pos_y",
    "rx_pos_x",
    "rx_pos_y",
    "claimed_vel_x",
    "claimed_vel_y",
    "rx_vel_x",
    "rx_vel_y",
    "claimed_acl_x",
    "claimed_acl_y",
    "range",
    "bearing",
    "log_dtau",
)
STRATA = (("n<=3", 1, 3), ("4-16", 4, 16), ("17-48", 17, 48), ("49-64", 49, 64))
SCORE_KEYS = ("recon_error", "mahalanobis", "supervised_prob")
DISPLAY = {
    "ae:recon_error": "AE reconstruction error",
    "ae:mahalanobis": "AE latent Mahalanobis",
    "supervised:supervised_prob": "Supervised encoder + head",
    "supervised:recon_error": "Supervised run: reconstruction error",
    "supervised:mahalanobis": "Supervised encoder: latent Mahalanobis",
    "baseline:length": "Length only (n_messages)",
    "baseline:stats_lr": "Stats + logistic regression",
}


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    """Settings of the evaluation.

    Attributes:
        runs: Run folders of the trained models.
        baselines_dir: Folder written by ``baselines.py`` (supplies ``sender`` for the bootstrap + the baselines).
        out_dir: Output folder for ``results.json`` and figures.
        n_boot: Bootstrap repetitions over sender vehicles.
        seed: Seed of the bootstrap and the plot subsample.
        percentiles: Validation-benign percentiles used as one-class thresholds.
        pca_sample: Windows per class in the z PCA plot.
    """

    runs: tuple[str, ...] = ()
    baselines_dir: str = str(DEFAULT_BASELINES)
    out_dir: str = str(DEFAULT_OUT)
    n_boot: int = 1000
    seed: int = 0
    percentiles: tuple[float, ...] = (95.0, 99.0)
    pca_sample: int = 2500


@dataclasses.dataclass
class ScoreSet:
    """One score of one run on the test split, plus its thresholds."""

    method: str
    run_id: str
    seed: int | None
    scores: np.ndarray
    thresholds: dict[str, float]


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Precision / recall / F1 / FPR + confusion matrix for ``score >= threshold`` = GridSybil."""
    pred = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def safe_auroc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None) -> float | None:
    """AUROC, or ``None`` when only one class is present."""
    if len(np.unique(labels[weights > 0] if weights is not None else labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores, sample_weight=weights))


class VehicleBootstrap:
    """Cluster bootstrap over sender vehicles: resample vehicles with replacement, weight windows by draw count."""

    def __init__(self, senders: np.ndarray, n_boot: int, seed: int) -> None:
        self._codes = np.unique(senders, return_inverse=True)[1]
        n_vehicles = int(self._codes.max()) + 1
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, n_vehicles, size=(n_boot, n_vehicles))
        self._counts = np.stack([np.bincount(d, minlength=n_vehicles) for d in draws])
        self.n_vehicles = n_vehicles

    def auroc_ci(self, labels: np.ndarray, scores: np.ndarray) -> list[float]:
        """2.5 / 97.5 percentiles of the bootstrap AUROC."""
        values = []
        for counts in self._counts:
            weights = counts[self._codes].astype(np.float64)
            auc = safe_auroc(labels, scores, weights)
            if auc is not None:
                values.append(auc)
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


class RunLoader:
    """Loads run folders + the baseline folder and aligns them on window id."""

    def __init__(self, config: EvalConfig) -> None:
        self._config = config
        base = Path(config.baselines_dir)
        b = np.load(base / "scores_test.npz", allow_pickle=False)
        self.ids = b["id"].astype(str)
        self.labels = b["label"].astype(np.int64)
        self.n_messages = b["n_messages"].astype(np.int64)
        self.senders = b["sender"].astype(np.int64)
        self.baselines = {"baseline:length": b["length_score"], "baseline:stats_lr": b["stats_prob"]}
        self.baseline_summary = (
            json.loads((base / "summary.json").read_text()) if (base / "summary.json").exists() else {}
        )
        self._index = {i: k for k, i in enumerate(self.ids)}
        self.runs: list[dict] = []

    def _align(self, run_ids: np.ndarray, run_dir: Path) -> np.ndarray:
        """Permutation that puts a run's windows into the baseline order."""
        if len(run_ids) != len(self.ids):
            raise ValueError(f"{run_dir}: {len(run_ids)} test windows, baselines have {len(self.ids)}")
        try:
            pos = np.fromiter((self._index[i] for i in run_ids), dtype=np.int64, count=len(run_ids))
        except KeyError as err:
            raise ValueError(f"{run_dir}: test id {err} not in the baseline file") from err
        order = np.empty_like(pos)
        order[pos] = np.arange(len(pos))
        return order

    def load(self, run_dir: Path) -> dict:
        """One run: mode, seed, aligned test arrays and validation scores."""
        config = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
        test = np.load(run_dir / "scores_test.npz", allow_pickle=False)
        order = self._align(test["id"].astype(str), run_dir)
        arrays = {k: test[k][order] for k in test.files if k != "id"}
        if not np.array_equal(arrays["label"].astype(np.int64), self.labels):
            raise ValueError(f"{run_dir}: labels differ from the baseline file")
        if not np.array_equal(arrays["n_messages"].astype(np.int64), self.n_messages):
            raise ValueError(f"{run_dir}: n_messages differ from the baseline file")
        val_path = run_dir / "scores_val.npz"
        val = dict(np.load(val_path, allow_pickle=False)) if val_path.exists() else {}
        sup = arrays.get("supervised_prob")
        mode = config.get("mode") or ("supervised" if sup is not None and np.isfinite(sup).any() else "ae")
        run = {
            "dir": run_dir,
            "run_id": config.get("run_id", run_dir.name),
            "mode": mode,
            "seed": config.get("seed"),
            "test": arrays,
            "val": val,
        }
        self.runs.append(run)
        return run

    def score_sets(self, percentiles: tuple[float, ...]) -> list[ScoreSet]:
        """Every finite score of every loaded run, plus the baselines."""
        sets = []
        for run in self.runs:
            for key in SCORE_KEYS:
                values = run["test"].get(key)
                if values is None or not np.isfinite(values).all():
                    continue
                if key == "supervised_prob":
                    thresholds = {"p0.5": 0.5}
                elif key in run["val"] and np.isfinite(run["val"][key]).all():
                    thresholds = {f"val_p{p:g}": float(np.percentile(run["val"][key], p)) for p in percentiles}
                else:
                    thresholds = {}
                sets.append(ScoreSet(f"{run['mode']}:{key}", run["run_id"], run["seed"], values, thresholds))
        sets.append(ScoreSet("baseline:length", "baselines", None, self.baselines["baseline:length"], {}))
        sets.append(
            ScoreSet("baseline:stats_lr", "baselines", None, self.baselines["baseline:stats_lr"], {"p0.5": 0.5})
        )
        return sets


def mean_std(values: list[float]) -> dict:
    """Mean and sample std (0 for one value)."""
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, "n": int(len(arr))}


class Evaluator:
    """Computes all metrics and figures."""

    def __init__(self, config: EvalConfig) -> None:
        self._config = config
        self._loader = RunLoader(config)
        for run_dir in config.runs:
            self._loader.load(Path(run_dir))
        self._boot = VehicleBootstrap(self._loader.senders, config.n_boot, config.seed)
        self._out = Path(config.out_dir)

    def _strata_masks(self) -> dict[str, np.ndarray]:
        n = self._loader.n_messages
        return {name: (n >= lo) & (n <= hi) for name, lo, hi in STRATA}

    def per_score(self, s: ScoreSet) -> dict:
        """Metrics of one score set."""
        y = self._loader.labels
        entry = {
            "run_id": s.run_id,
            "seed": s.seed,
            "auroc": float(roc_auc_score(y, s.scores)),
            "pr_auc": float(average_precision_score(y, s.scores)),
            "auroc_ci95_vehicle_bootstrap": self._boot.auroc_ci(y, s.scores),
            "thresholds": {name: binary_metrics(y, s.scores, thr) for name, thr in s.thresholds.items()},
            "strata_auroc": {},
        }
        for name, m in self._strata_masks().items():
            entry["strata_auroc"][name] = safe_auroc(y[m], s.scores[m])
        return entry

    def summarise(self, entries: list[dict]) -> dict:
        """Mean +- std over seeds of every scalar metric."""
        summary = {"auroc": mean_std([e["auroc"] for e in entries]), "pr_auc": mean_std([e["pr_auc"] for e in entries])}
        summary["thresholds"] = {}
        for name in entries[0]["thresholds"]:
            summary["thresholds"][name] = {
                k: mean_std([e["thresholds"][name][k] for e in entries]) for k in ("precision", "recall", "f1", "fpr")
            }
        summary["strata_auroc"] = {}
        for name in entries[0]["strata_auroc"]:
            vals = [e["strata_auroc"][name] for e in entries if e["strata_auroc"][name] is not None]
            summary["strata_auroc"][name] = mean_std(vals) if vals else None
        return summary

    def strata_table(self) -> dict:
        """Window counts and length-only AUROC (both directions) per stratum."""
        y, n = self._loader.labels, self._loader.n_messages.astype(np.float64)
        table = {}
        for name, m in self._strata_masks().items():
            auc = safe_auroc(y[m], n[m])
            table[name] = {
                "n_windows": int(m.sum()),
                "n_benign": int((m & (y == 0)).sum()),
                "n_gridsybil": int((m & (y == 1)).sum()),
                "gridsybil_share": float(y[m].mean()) if m.any() else None,
                "length_auroc_plus_n": auc,
                "length_auroc_best_direction": None if auc is None else max(auc, 1 - auc),
            }
        return table

    def per_feature(self) -> dict:
        """Mean per-feature reconstruction error benign vs GridSybil (mean over AE seeds) + per-feature AUROC."""
        runs = [r for r in self._loader.runs if r["mode"] == "ae" and "recon_error_feat" in r["test"]]
        runs = [r for r in runs if np.isfinite(r["test"]["recon_error_feat"]).all()]
        if not runs:
            return {}
        y = self._loader.labels
        out = {}
        for j, feat in enumerate(FEATURES):
            ben = [float(r["test"]["recon_error_feat"][y == 0, j].mean()) for r in runs]
            grid = [float(r["test"]["recon_error_feat"][y == 1, j].mean()) for r in runs]
            aucs = [float(roc_auc_score(y, r["test"]["recon_error_feat"][:, j])) for r in runs]
            out[feat] = {"benign_mean": mean_std(ben), "gridsybil_mean": mean_std(grid), "auroc": mean_std(aucs)}
        return out

    def run(self) -> dict:
        """Computes everything, writes ``results.json`` and the figures, returns the results."""
        self._out.mkdir(parents=True, exist_ok=True)
        sets = self._loader.score_sets(self._config.percentiles)
        methods: dict[str, dict] = {}
        for s in sets:
            methods.setdefault(s.method, {"name": DISPLAY.get(s.method, s.method), "per_seed": []})
            methods[s.method]["per_seed"].append(self.per_score(s))
        for m in methods.values():
            m["summary"] = self.summarise(m["per_seed"])
        y = self._loader.labels
        results = {
            "config": dataclasses.asdict(self._config),
            "n_test": int(len(y)),
            "test_counts": {"benign": int((y == 0).sum()), "gridsybil": int((y == 1).sum())},
            "n_sender_vehicles": self._boot.n_vehicles,
            "runs": [
                {"run_id": r["run_id"], "mode": r["mode"], "seed": r["seed"], "dir": str(r["dir"])}
                for r in self._loader.runs
            ],
            "baseline_summary": self._loader.baseline_summary,
            "methods": methods,
            "strata": self.strata_table(),
            "per_feature_recon_error": self.per_feature(),
        }
        (self._out / "results.json").write_text(json.dumps(results, indent=2))
        FigureWriter(self._out, self._loader, sets, results, self._config).write_all()
        return results


class FigureWriter:
    """The PNG figures of the evaluation."""

    def __init__(self, out: Path, loader: RunLoader, sets: list[ScoreSet], results: dict, config: EvalConfig) -> None:
        self._out = out
        self._loader = loader
        self._sets = sets
        self._results = results
        self._config = config

    def _first(self, method: str) -> ScoreSet | None:
        cands = [s for s in self._sets if s.method == method]
        return sorted(cands, key=lambda s: (s.seed is None, s.seed or 0))[0] if cands else None

    def write_all(self) -> list[Path]:
        """Writes every figure that has data."""
        paths = [self.roc_curves(), self.strata_bars()]
        paths += [p for p in (self.error_histograms(), self.per_feature_bars(), self.z_pca()) if p]
        return paths

    def error_histograms(self) -> Path | None:
        """log10 score histograms benign vs GridSybil with the validation thresholds."""
        panels = [s for s in (self._first("ae:recon_error"), self._first("ae:mahalanobis")) if s is not None]
        if not panels:
            return None
        y = self._loader.labels
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4), squeeze=False)
        for ax, s in zip(axes[0], panels):
            v = np.log10(np.maximum(s.scores, 1e-12))
            bins = np.linspace(np.percentile(v, 0.1), np.percentile(v, 99.9), 80)
            ax.hist(v[y == 0], bins=bins, density=True, alpha=0.55, label="Benign", color="#3b75af")
            ax.hist(v[y == 1], bins=bins, density=True, alpha=0.55, label="GridSybil", color="#d1495b")
            for (name, thr), ls in zip(s.thresholds.items(), ("--", ":")):
                ax.axvline(np.log10(max(thr, 1e-12)), color="black", ls=ls, label=f"{name} = {thr:.3g}")
            ax.set_xlabel(f"log10 {DISPLAY.get(s.method, s.method)}")
            ax.set_ylabel("density")
            ax.set_title(f"{s.run_id} (seed {s.seed})", fontsize=9)
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = self._out / "error_histograms.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path

    def roc_curves(self) -> Path:
        """ROC curve of every method (first seed)."""
        y = self._loader.labels
        fig, ax = plt.subplots(figsize=(6, 5.5))
        for method in dict.fromkeys(s.method for s in self._sets):
            s = self._first(method)
            fpr, tpr, _ = roc_curve(y, s.scores)
            auc = roc_auc_score(y, s.scores)
            ax.plot(fpr, tpr, label=f"{DISPLAY.get(method, method)} ({auc:.3f})", lw=1.5)
        ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
        ax.set_xlabel("false positive rate (benign flagged)")
        ax.set_ylabel("true positive rate (GridSybil flagged)")
        ax.set_title("ROC, test split (first seed of each model)", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        path = self._out / "roc_curves.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path

    def strata_bars(self) -> Path:
        """AUROC per length stratum (mean over seeds), with the length-only AUROC in the same stratum."""
        methods = self._results["methods"]
        names = [n for n, _, _ in STRATA]
        keys = [k for k in methods if k != "baseline:length"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        width = 0.8 / (len(keys) + 1)
        x = np.arange(len(names))
        for i, key in enumerate(keys):
            vals = [(methods[key]["summary"]["strata_auroc"].get(n) or {}).get("mean", np.nan) for n in names]
            ax.bar(x + i * width, vals, width, label=DISPLAY.get(key, key))
        length = [self._results["strata"][n]["length_auroc_best_direction"] or np.nan for n in names]
        ax.bar(x + len(keys) * width, length, width, label="Length only (best direction in stratum)", color="grey")
        ax.axhline(0.5, color="black", lw=0.8, ls=":")
        counts = [self._results["strata"][n]["n_windows"] for n in names]
        ax.set_xticks(
            x + 0.4 - width / 2,
            [f"{n if n.startswith('n') else 'n ' + n}\n({c:,} windows)" for n, c in zip(names, counts)],
            fontsize=8,
        )
        ax.set_ylabel("AUROC (test)")
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = self._out / "strata_auroc.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path

    def per_feature_bars(self) -> Path | None:
        """Mean per-feature reconstruction error benign vs GridSybil."""
        pf = self._results["per_feature_recon_error"]
        if not pf:
            return None
        x = np.arange(len(FEATURES))
        ben = [pf[f]["benign_mean"]["mean"] for f in FEATURES]
        grid = [pf[f]["gridsybil_mean"]["mean"] for f in FEATURES]
        ben_e = [pf[f]["benign_mean"]["std"] for f in FEATURES]
        grid_e = [pf[f]["gridsybil_mean"]["std"] for f in FEATURES]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(x - 0.2, ben, 0.4, yerr=ben_e, label="Benign", color="#3b75af")
        ax.bar(x + 0.2, grid, 0.4, yerr=grid_e, label="GridSybil", color="#d1495b")
        ax.set_yscale("log")
        ax.set_xticks(x, FEATURES, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("mean masked MSE (normalised units)")
        ax.set_title("Per-feature reconstruction error, test split (mean over AE seeds, bars = std)", fontsize=10)
        ax.legend()
        fig.tight_layout()
        path = self._out / "per_feature_error.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path

    def z_pca(self) -> Path | None:
        """2-D PCA of z on a class-balanced test sample, one panel per model (first seed)."""
        y = self._loader.labels
        panels = []
        for mode in ("ae", "supervised"):
            runs = sorted((r for r in self._loader.runs if r["mode"] == mode), key=lambda r: r["seed"] or 0)
            runs = [r for r in runs if "z" in r["test"] and np.isfinite(r["test"]["z"]).all()]
            if runs:
                panels.append(runs[0])
        if not panels:
            return None
        rng = np.random.default_rng(self._config.seed)
        idx = np.concatenate(
            [
                rng.choice(np.flatnonzero(y == c), min(self._config.pca_sample, int((y == c).sum())), replace=False)
                for c in (0, 1)
            ]
        )
        fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5), squeeze=False)
        for ax, run in zip(axes[0], panels):
            z = run["test"]["z"][idx]
            pca = PCA(n_components=2, random_state=self._config.seed).fit(z)
            p = pca.transform(z)
            for c, name, color in ((0, "Benign", "#3b75af"), (1, "GridSybil", "#d1495b")):
                sel = y[idx] == c
                ax.scatter(p[sel, 0], p[sel, 1], s=3, alpha=0.4, label=name, color=color)
            ev = pca.explained_variance_ratio_
            ax.set_xlabel(f"PC1 ({ev[0]:.0%})")
            ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
            ax.set_title(f"z of {run['mode']} {run['run_id']} (seed {run['seed']})", fontsize=9)
            ax.legend(markerscale=4, fontsize=8)
        fig.tight_layout()
        path = self._out / "z_pca.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path


def discover_runs(runs_root: Path) -> list[Path]:
    """Run folders under ``runs_root`` with a ``scores_test.npz``; quick-check runs (``*-check``) are skipped."""
    found = [p.parent for p in sorted(runs_root.glob("*/scores_test.npz"))]
    skip = ("baselines", "evaluation")
    return [p for p in found if p.name not in skip and not p.name.endswith("-check") and not p.name.startswith("_")]


def parse_args(argv: list[str] | None = None) -> EvalConfig:
    """CLI -> config (no ``--runs``: every finished run under ``task/encoder/runs/``)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="*", default=[], help="run folders with scores_test.npz")
    parser.add_argument("--baselines", default=str(DEFAULT_BASELINES))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    runs = args.runs or [str(p) for p in discover_runs(DEFAULT_OUT.parent)]
    return EvalConfig(
        runs=tuple(runs), baselines_dir=args.baselines, out_dir=args.out, n_boot=args.n_boot, seed=args.seed
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point: prints the headline table."""
    results = Evaluator(parse_args(argv)).run()
    for key, m in results["methods"].items():
        s = m["summary"]
        print(
            f"{m['name']:<42} AUROC {s['auroc']['mean']:.3f} +- {s['auroc']['std']:.3f}  "
            f"PR-AUC {s['pr_auc']['mean']:.3f}  (n={s['auroc']['n']})"
        )


if __name__ == "__main__":
    main()
