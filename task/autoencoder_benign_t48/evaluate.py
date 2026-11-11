"""Evaluates a finished run on the test vehicles: benign `test` vs `test_gridsybil`. Scores first, labels after.

Every window of val, test and test_gridsybil is scored with `best.pt` (reconstruction error) and the run's
`mahalanobis.npz` (distance of z, fitted on benign train z), and the scores are written to `<run>/scores/` before any
label is read:

    <split>_ids.npy             (N,) window ids
    <split>_n_messages.npy      (N,) real rows per window
    <split>_reconstruction.npy  (N,) masked MSE per window
    <split>_mahalanobis.npy     (N,) Mahalanobis distance of z

The four files of a split are aligned row by row. Then the labels in the info files are checked against the split
names (test = benign only, test_gridsybil = code 16 only; test vehicles in neither train nor val) and the metrics are
computed from the score files and the val thresholds stored in `summary.json` alone, so `test_metrics.json` can be
rebuilt without the model or the notebook (`--from-scores`). Positive = GridSybil. No threshold, direction or setting
is chosen here: thresholds are the benign val p95 / p99 of the run, and the length-only reference uses the fixed rule
score = -n_messages (shorter = more anomalous).

Usage (repo root):
    .venv-train/bin/python -m task.autoencoder_benign_t48.evaluate --seed 0
    .venv-train/bin/python -m task.autoencoder_benign_t48.evaluate --seed 0 --from-scores
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from task.autoencoder_benign_t48.config import ATTACK_TEST_SPLIT, BENIGN_TEST_SPLIT, EVAL_SPLITS, AutoencoderConfig
from task.autoencoder_benign_t48.data import WindowDataset, track_id
from task.autoencoder_benign_t48.scoring import Mahalanobis, TrackScores
from task.autoencoder_benign_t48.trainer import Trainer

SCORE_NAMES = ("reconstruction", "mahalanobis")
SCORED_SPLITS = ("val", *EVAL_SPLITS)
ATTACK_CODE = 16
SHORT_MAX_ROWS = 3


@dataclasses.dataclass
class SplitScores:
    """Label-free scores of one split, aligned row by row.

    Attributes:
        split: split name.
        ids: (N,) window ids.
        n_messages: (N,) real rows per window.
        scores: {"reconstruction": (N,), "mahalanobis": (N,)}.
    """

    split: str
    ids: np.ndarray
    n_messages: np.ndarray
    scores: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def tracks(self) -> np.ndarray:
        """(N,) track id (link) of each window."""
        return np.asarray([track_id(i) for i in self.ids])

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / f"{self.split}_ids.npy", self.ids)
        np.save(folder / f"{self.split}_n_messages.npy", self.n_messages)
        for name, values in self.scores.items():
            np.save(folder / f"{self.split}_{name}.npy", values)

    @classmethod
    def load(cls, folder: Path, split: str) -> "SplitScores":
        out = cls(
            split,
            np.load(folder / f"{split}_ids.npy"),
            np.load(folder / f"{split}_n_messages.npy"),
            {name: np.load(folder / f"{split}_{name}.npy") for name in SCORE_NAMES},
        )
        if any(len(v) != len(out) for v in (out.n_messages, *out.scores.values())):
            raise ValueError(f"{split}: score files are not aligned with the window ids")
        return out


class DetectionMetrics:
    """Benign test (0) vs attack test (1) metrics from two `SplitScores` and the run's val thresholds."""

    def __init__(self, benign: SplitScores, attack: SplitScores, thresholds: dict) -> None:
        self.benign, self.attack, self.thresholds = benign, attack, thresholds
        self.y = np.r_[np.zeros(len(benign), int), np.ones(len(attack), int)]
        self.n_messages = np.r_[benign.n_messages, attack.n_messages]
        self.benign_tracks, self.attack_tracks = TrackScores(benign.tracks), TrackScores(attack.tracks)
        self.shared_tracks = len(set(self.benign_tracks.tracks) & set(self.attack_tracks.tracks))

    @staticmethod
    def ranking(y: np.ndarray, s: np.ndarray) -> dict:
        """AUROC, PR-AUC and the class counts; AUROC / PR-AUC are None when one class is missing."""
        n_attack = int(y.sum())
        out = {"n_benign": int(len(y) - n_attack), "n_attack": n_attack, "chance": float(y.mean()) if len(y) else None}
        both = 0 < n_attack < len(y)
        out["auroc"] = float(roc_auc_score(y, s)) if both else None
        out["pr_auc"] = float(average_precision_score(y, s)) if both else None
        return out

    @staticmethod
    def at_thresholds(y: np.ndarray, s: np.ndarray, thresholds: dict[str, float]) -> dict:
        """Attack recall and benign FPR of the alarm `score > threshold`, per val quantile."""
        out = {}
        for q, th in thresholds.items():
            out[f"recall_{q}"] = float((s[y == 1] > th).mean())
            out[f"benign_fpr_{q}"] = float((s[y == 0] > th).mean())
            out[f"threshold_{q}"] = float(th)
        return out

    def window(self, name: str) -> dict:
        """Window-level metrics of one score, plus AUROC on the 1-3-row and the >= 4-row windows (both classes cut)."""
        s = np.r_[self.benign.scores[name], self.attack.scores[name]]
        out = {**self.ranking(self.y, s), **self.at_thresholds(self.y, s, self.thresholds["window"][name])}
        short = self.n_messages <= SHORT_MAX_ROWS
        out["rows_1_3"] = self.ranking(self.y[short], s[short])
        out["rows_ge_4"] = self.ranking(self.y[~short], s[~short])
        return out

    def track(self, name: str, how: str) -> dict:
        """Track-level metrics of one score aggregated by `how` ("max" or "mean") over each link's windows."""
        b = self.benign_tracks.aggregate(self.benign.scores[name])[how]
        a = self.attack_tracks.aggregate(self.attack.scores[name])[how]
        y, s = np.r_[np.zeros(len(b), int), np.ones(len(a), int)], np.r_[b, a]
        return {**self.ranking(y, s), **self.at_thresholds(y, s, self.thresholds["track"][f"{name}_{how}"])}

    def length_only(self) -> dict:
        """Reference: score = -n_messages (shorter window = more anomalous), a rule fixed before any test label."""
        out = self.ranking(self.y, -self.n_messages.astype(np.float64))
        out["rule"] = "score = -n_messages (shorter = more anomalous); fixed in advance, not chosen on test labels"
        out["sibling_convention"] = "task/autoencoder_benign_t32 uses the same score = -n_messages; no difference"
        return out

    def all(self) -> dict:
        out = {
            "positive": "GridSybil (test_gridsybil)",
            "negative": "benign (test)",
            "threshold_source": "benign val p95 / p99 from summary.json",
            "tracks_in_both_splits": self.shared_tracks,
            "window": {name: self.window(name) for name in SCORE_NAMES},
            "track": {f"{name}_{how}": self.track(name, how) for name in SCORE_NAMES for how in ("max", "mean")},
            "length_only": self.length_only(),
        }
        short_b = float((self.benign.n_messages <= SHORT_MAX_ROWS).mean())
        short_a = float((self.attack.n_messages <= SHORT_MAX_ROWS).mean())
        out["share_rows_1_3"] = {"benign": short_b, "attack": short_a}
        return out


class Evaluation:
    """Scores val and the two test splits of a finished run, checks the labels, writes `test_metrics.json`."""

    def __init__(self, run_dir: str | Path, device: str = "auto") -> None:
        self.run_dir = Path(run_dir)
        self.scores_dir = self.run_dir / "scores"
        self.device = device
        if not (self.run_dir / "summary.json").exists():
            raise FileNotFoundError(f"{self.run_dir} holds no finished run (summary.json missing)")
        self.summary = json.loads((self.run_dir / "summary.json").read_text())
        self.settings = AutoencoderConfig.from_dict(json.loads((self.run_dir / "config.json").read_text()))

    def load(self, split: str) -> WindowDataset:
        s = self.settings
        return WindowDataset.load(s.data_dir, split, s.workers, cache_dir=s.cache_dir)

    def score(self) -> dict[str, WindowDataset]:
        """Scores val, test and test_gridsybil and writes the score files; no label is read."""
        trainer = Trainer.from_checkpoint(self.run_dir / "best.pt", self.device)
        maha = Mahalanobis.from_state(np.load(self.run_dir / "mahalanobis.npz"))
        data = {}
        for split in SCORED_SPLITS:
            d = self.load(split)
            out = trainer.score(d.x, d.mask)
            scores = {"reconstruction": out["error"].astype(np.float64), "mahalanobis": maha.score(out["z"])}
            SplitScores(split, d.ids, d.n_messages, scores).save(self.scores_dir)
            data[split] = d
            print(f"scored {split}: {len(d):,} windows")
        return data

    def check_labels(self, data: dict[str, WindowDataset]) -> dict:
        """Labels vs split names, and test vehicles vs train / val vehicles (called after the scores are on disk)."""
        for split, code in (("val", 0), (BENIGN_TEST_SPLIT, 0), (ATTACK_TEST_SPLIT, ATTACK_CODE)):
            if not np.all(data[split].info["label"].astype(int) == code):
                raise ValueError(f"{split}: labels other than {code}")
        fit_vehicles = set(self.load("train").vehicles) | set(data["val"].vehicles)
        test_vehicles = set(data[BENIGN_TEST_SPLIT].vehicles) | set(data[ATTACK_TEST_SPLIT].vehicles)
        if fit_vehicles & test_vehicles:
            raise ValueError(f"{len(fit_vehicles & test_vehicles)} test sender vehicles are in train or val")
        return {
            "benign_test_vehicles": len(set(data[BENIGN_TEST_SPLIT].vehicles)),
            "attack_test_vehicles": len(set(data[ATTACK_TEST_SPLIT].vehicles)),
            "test_vehicles_in_train_or_val": 0,
        }

    def metrics(self) -> dict:
        """Metrics from the score files and the val thresholds of `summary.json` (no model, no data folder)."""
        val = SplitScores.load(self.scores_dir, "val")
        benign = SplitScores.load(self.scores_dir, BENIGN_TEST_SPLIT)
        attack = SplitScores.load(self.scores_dir, ATTACK_TEST_SPLIT)
        thresholds = self.summary["thresholds"]
        val_fpr = {
            name: {q: float((val.scores[name] > th).mean()) for q, th in thresholds["window"][name].items()}
            for name in SCORE_NAMES
        }
        return {
            "run_id": self.summary["run_id"],
            "windows": {"val": len(val), BENIGN_TEST_SPLIT: len(benign), ATTACK_TEST_SPLIT: len(attack)},
            "val_alarm_rate": val_fpr,
            **DetectionMetrics(benign, attack, thresholds).all(),
        }

    def run(self, from_scores: bool = False) -> dict:
        checks = None if from_scores else self.check_labels(self.score())
        result = self.metrics()
        path = self.run_dir / "test_metrics.json"
        if checks is None and path.exists():
            checks = json.loads(path.read_text()).get("label_checks")
        result["label_checks"] = checks
        path.write_text(json.dumps(result, indent=2))
        print(f"wrote {path}")
        return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-dir", default=None, help="run folder (default: <runs_dir>/seed<seed>)")
    p.add_argument("--device", default="auto")
    p.add_argument("--from-scores", action="store_true", help="rebuild test_metrics.json from the score files only")
    a = p.parse_args(argv)
    run_dir = a.run_dir or Path(AutoencoderConfig().runs_dir) / f"seed{a.seed}"
    result = Evaluation(run_dir, a.device).run(a.from_scores)
    brief = {k: {m: v[m] for m in ("auroc", "pr_auc")} for k, v in result["window"].items()}
    print(json.dumps({"window": brief, "length_only_auroc": result["length_only"]["auroc"]}, indent=2))


if __name__ == "__main__":
    main()
