"""Final evaluation of a finished run on the test split of all/T32_benign_grid; writes `evaluation.json`.

Order of work: (1) the best checkpoint, the Mahalanobis model and the val thresholds are loaded from the run folder,
the test windows are read without labels and scored, and `scores/test_*.npy` is written; (2) only then are the
`_info.json` files opened and the scores compared with the labels (GridSybil = positive). Nothing is fitted or chosen
on test: thresholds are the val p95 / p99 of `summary.json`, the length rule is fixed (score = -n_messages).

Reports, for all test windows and again for the test windows of the two GridSybil folders only (benign traffic
repeats in every family, F1): window AUROC / PR-AUC, recall and benign FPR at val p95 / p99, per-track max / mean,
1-3-row and >= 4-row windows, per group 0709 / 1416, benign FPR per source folder, length alone; plus the class
counts of train / val / test and the sibling studies' numbers for context.

Usage (repo root):
    .venv-train/bin/python -m task.autoencoder_t32.evaluate --run task/autoencoder_t32/runs/seed0
"""

from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np

from task.autoencoder_t32.data import UnlabelledWindows
from task.autoencoder_t32.labels import WindowLabels
from task.autoencoder_t32.scoring import SCORES, Evaluation, Mahalanobis, ScoreFiles
from task.autoencoder_t32.trainer import Trainer

GRIDSYBIL_FAMILY = "GridSybil"
ST_T32_NOTEBOOK = "task/autoencoder_st_t32/autoencoder_st_T32.ipynb"
BENIGN_T32_NOTEBOOK = "task/autoencoder_benign_t32/autoencoder_benign_T32.ipynb"


class SiblingContext:
    """Existing numbers of the two T = 32 sibling studies, read from the result cell of their executed notebooks
    (their `runs/seed0/summary.json` hold training / val fields only). Context, not a ranking."""

    @staticmethod
    def results_cell(notebook: str) -> dict | None:
        """The dict displayed by the notebook's last `results` cell, or None if it cannot be read."""
        path = Path(notebook)
        if not path.is_file():
            return None
        for cell in reversed(json.loads(path.read_text())["cells"]):
            if cell["cell_type"] != "code" or "results" not in "".join(cell["source"]):
                continue
            for out in cell.get("outputs", []):
                text = "".join(out.get("data", {}).get("text/plain", ""))
                try:
                    value = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(value, dict):
                    return value
        return None

    @classmethod
    def collect(cls) -> dict:
        st, oc = cls.results_cell(ST_T32_NOTEBOOK), cls.results_cell(BENIGN_T32_NOTEBOOK)
        out = {"note": "one seed each; context only. The one-class study uses a different split and test set."}
        if st:
            out["supervised_st_t32"] = {
                "source": ST_T32_NOTEBOOK,
                "data": "all/T32_benign_grid, same 85 / 7.5 / 7.5 split and test windows as this study",
                "training_signal": "GridSybil labels (classifier head on z)",
                "window_auroc": st["window/max_f1"]["auroc"],
                "window_pr_auc": st["window/max_f1"]["pr_auc"],
                "reconstruction_error_auroc": st.get("reconstruction_error_auroc"),
                "length_only_auroc_best_direction": st.get("length_only_auroc"),
            }
        if oc:
            row = oc["GridSybil/reconstruction/window"]
            out["one_class_benign_t32"] = {
                "source": BENIGN_T32_NOTEBOOK,
                "data": "all/T32_benign: a different 80 / 10 / 10 split; benign test vs test_gridsybil windows",
                "training_signal": "benign windows only",
                "window_auroc_reconstruction": row["AUROC"],
                "window_auroc_reconstruction_ge_4_rows": row.get("AUROC (>= 4 rows)"),
                "window_auroc_mahalanobis": oc["GridSybil/mahalanobis/window"]["AUROC"],
                "length_only_auroc": oc["GridSybil/length only (reference)/window"]["AUROC"],
            }
        return out


class TestEvaluation:
    """Scores the test split of a finished run, then reads the labels and builds `evaluation.json`.

    Attributes:
        run_dir: the run folder (config.json, best.pt, mahalanobis.npz, summary.json).
        device: torch device name for scoring.
    """

    __test__ = False  # not a pytest class

    def __init__(self, run_dir: str | Path, device: str = "auto") -> None:
        self.run_dir = Path(run_dir)
        self.device = device
        self.summary = json.loads((self.run_dir / "summary.json").read_text())
        self.trainer = Trainer.from_checkpoint(self.run_dir / "best.pt", device)
        self.maha = Mahalanobis.from_state(np.load(self.run_dir / "mahalanobis.npz"))
        self.files = ScoreFiles(self.run_dir)

    def score_test(self) -> UnlabelledWindows:
        """Reads the test windows (features only), scores them, writes `scores/test_*.npy`."""
        s = self.trainer.settings
        test = UnlabelledWindows.load(s.data_dir, "test", s.workers, cache_dir=s.cache_dir)
        out = self.trainer.score(test.x, test.mask)
        self.files.save("test", test.ids, out["error"], self.maha.score(out["z"]), test.n_messages, out["z"])
        return test

    def evaluate(self, test: UnlabelledWindows) -> dict:
        """Labels are opened here, after `score_test` has written the scores it reads back."""
        s = self.trainer.settings
        scores = self.files.load("test")
        labels = WindowLabels.load(s.data_dir, "test", s.workers)
        labels.check(scores["ids"], scores["n_messages"])
        evaluation = Evaluation(
            {name: scores[name] for name in SCORES},
            labels.y,
            scores["n_messages"],
            test.tracks(),
            labels.group,
            labels.scenario,
            self.summary["thresholds"],
        )
        counts = {"test": labels.class_counts()}
        for split in ("train", "val"):
            counts[split] = WindowLabels.load(s.data_dir, split, s.workers).class_counts()
        for c in counts.values():
            c["gridsybil_share"] = c["gridsybil"] / (c["benign"] + c["gridsybil"])
        return {
            "run_id": self.summary["run_id"],
            "note": "one seed, side experiment, not a thesis result; labels read after scoring; thresholds from val",
            "positive_class": "GridSybil (is_attack = 1)",
            "alarm_rule": "score > val threshold",
            "threshold_source": self.summary["threshold_source"],
            "thresholds": self.summary["thresholds"],
            "class_counts": counts,
            "test_all": evaluation.report(),
            "test_gridsybil_folders": evaluation.report(labels.family == GRIDSYBIL_FAMILY),
            "context": SiblingContext.collect(),
        }

    def run(self) -> dict:
        t0 = time.time()
        result = self.evaluate(self.score_test())
        result["evaluation_seconds"] = round(time.time() - t0, 1)
        (self.run_dir / "evaluation.json").write_text(json.dumps(result, indent=2))
        return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", default="task/autoencoder_t32/runs/seed0", help="run folder")
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    result = TestEvaluation(a.run, a.device).run()
    for part in ("test_all", "test_gridsybil_folders"):
        r = result[part]
        line = ", ".join(
            f"{n} AUROC {r['window'][n]['auroc']:.4f} PR-AUC {r['window'][n]['pr_auc']:.4f}" for n in SCORES
        )
        print(
            f"{part}: {r['windows']:,} windows (GridSybil share {r['gridsybil_share']:.3f}); {line}; "
            f"length alone {r['length_only']['auroc']:.4f}"
        )
    print(f"wrote {Path(a.run) / 'evaluation.json'}")


if __name__ == "__main__":
    main()
