"""Evaluates a finished masked-autoencoder run on the test vehicles. Scores first, labels after.

Every window of val and of the test splits is scored with `best.pt` (masked_recon, full_recon) and the run's
`mahalanobis.npz`, and written to `<run>/scores/<split>.npz` (compressed; ids as UTF-8 bytes, n_messages int16,
masked_recon / full_recon / mahalanobis float32; rows aligned) before any label is read. Then the labels are read (logged), checked (val and benign test = code 0, each
attack split = its code; no test vehicle in train or val), and every attack is compared with the benign test windows:
AUROC, PR-AUC, recall and benign FPR at the benign val p95 / p99 thresholds of `summary.json`, AUROC on 1-3-row and
>= 4-row windows, per-track max / mean, and the length-only reference score = -n_messages (fixed rule, not chosen on
test labels). Metrics come from `task.autoencoder_benign_t48.evaluate.DetectionMetrics` (imported).

T = 64: the on-disk `test` split holds benign and GridSybil windows together; they are separated by label after scoring.

Usage (repo root):
    .venv-train/bin/python -m task.masked_autoencoder.evaluate --run-dir task/masked_autoencoder/runs/T32-mlp-seed0
    .venv-train/bin/python -m task.masked_autoencoder.evaluate --run-dir <run> --from-scores
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from task.autoencoder_benign_t48.evaluate import DetectionMetrics, SplitScores
from task.masked_autoencoder.config import MAEConfig
from task.masked_autoencoder.data import OneClassSource, make_source
from task.masked_autoencoder.scoring import SCORE_NAMES, Mahalanobis
from task.masked_autoencoder.trainer import MaskedTrainer

ATTACK_CODES = {"test_gridsybil": 16, "test_datareplay": 17, "test_dosrandom": 18, "test_dosdisruptive": 19}
ATTACK_NAMES = {16: "GridSybil", 17: "DataReplay", 18: "DoSRandom", 19: "DoSDisruptive"}


class MultiScoreMetrics(DetectionMetrics):
    """`DetectionMetrics` over this package's three scores."""

    def __init__(self, benign: SplitScores, attack: SplitScores, thresholds: dict, attack_name: str) -> None:
        super().__init__(benign, attack, thresholds)
        self.attack_name = attack_name

    def length_only(self) -> dict:
        out = self.ranking(self.y, -self.n_messages.astype(np.float64))
        out["rule"] = "score = -n_messages (shorter = more anomalous); fixed in advance, not chosen on test labels"
        return out

    def all(self) -> dict:
        return {
            "positive": self.attack_name,
            "negative": "benign test",
            "threshold_source": "benign val p95 / p99 from summary.json",
            "tracks_in_both_splits": self.shared_tracks,
            "window": {name: self.window(name) for name in SCORE_NAMES},
            "track": {f"{name}_{how}": self.track(name, how) for name in SCORE_NAMES for how in ("max", "mean")},
            "length_only": self.length_only(),
            "share_rows_1_3": {
                "benign": float((self.benign.n_messages <= 3).mean()),
                "attack": float((self.attack.n_messages <= 3).mean()),
            },
        }


def save_scores(folder: Path, split: str, ids, n_messages, scores: dict[str, np.ndarray]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    scores = {k: np.asarray(v, dtype=np.float32) for k, v in scores.items()}
    np.savez_compressed(
        folder / f"{split}.npz",
        ids=np.char.encode(np.asarray(ids).astype(str), "utf-8"),
        n_messages=np.asarray(n_messages, dtype=np.int16),
        **scores,
    )


def load_scores(folder: Path, split: str) -> SplitScores:
    f = np.load(folder / f"{split}.npz")
    ids = np.char.decode(f["ids"], "utf-8")
    scores = {name: f[name].astype(np.float64) for name in SCORE_NAMES}
    out = SplitScores(split, ids, f["n_messages"].astype(np.int32), scores)
    if any(len(v) != len(out) for v in (out.n_messages, *out.scores.values())):
        raise ValueError(f"{split}: score arrays are not aligned with the window ids")
    return out


def subset(s: SplitScores, rows: np.ndarray, name: str) -> SplitScores:
    return SplitScores(name, s.ids[rows], s.n_messages[rows], {k: v[rows] for k, v in s.scores.items()})


class Evaluation:
    """Scores val + test splits of a finished run, then reads labels and writes `test_metrics.json`."""

    def __init__(self, run_dir: str | Path, device: str = "auto", source: OneClassSource | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.scores_dir = self.run_dir / "scores"
        if not (self.run_dir / "summary.json").exists():
            raise FileNotFoundError(f"{self.run_dir} holds no finished run (summary.json missing)")
        self.summary = json.loads((self.run_dir / "summary.json").read_text())
        self.settings = MAEConfig.from_dict(json.loads((self.run_dir / "config.json").read_text()))
        self.device = device
        self._source = source

    @property
    def source(self) -> OneClassSource:
        if self._source is None:
            self._source = make_source(self.settings)
        return self._source

    @property
    def splits(self) -> tuple[str, ...]:
        return ("val", *self.source.eval_splits())

    @property
    def attack_splits(self) -> tuple[str, ...]:
        """On-disk attack test splits (none at T = 64, where `test` is mixed)."""
        return tuple(s for s in self.source.eval_splits() if s != "test")

    def score(self) -> None:
        """Writes `scores/<split>.npz` for val and every test split; reads no label."""
        trainer = MaskedTrainer.from_checkpoint(self.run_dir / "best.pt", self.device)
        maha = Mahalanobis.from_state(np.load(self.run_dir / "mahalanobis.npz"))
        n_before = len(self.source.label_reads)
        for split in self.splits:
            w = self.source.load(split)
            out = trainer.scorer(w.x, w.mask)
            scores = {"masked_recon": out["masked_recon"], "full_recon": out["full_recon"]}
            scores["mahalanobis"] = maha.score(out["z"])
            save_scores(self.scores_dir, split, w.ids, w.n_messages, scores)
            print(f"scored {split}: {len(w):,} windows")
        if self.settings.window != 64 and len(self.source.label_reads) != n_before:
            raise AssertionError("labels were read while scoring")

    def labels(self) -> dict[str, np.ndarray]:
        """Attack codes per scored split (read after every score file exists) + consistency checks."""
        codes = {}
        for split in self.splits:
            codes[split] = self.source.labels(split, "evaluation, after every score file is on disk")
            n = len(load_scores(self.scores_dir, split))
            if len(codes[split]) != n:
                raise ValueError(f"{split}: {len(codes[split])} labels for {n} scored windows")
        if np.any(codes["val"] != 0):
            raise ValueError("val holds attack windows")
        if self.settings.window == 64:
            if not set(np.unique(codes["test"])) <= {0, 16}:
                raise ValueError(f"T64 test: unexpected codes {np.unique(codes['test'])}")
        else:
            if np.any(codes["test"] != 0):
                raise ValueError("benign test holds attack windows")
            for split in self.attack_splits:
                if np.any(codes[split] != ATTACK_CODES[split]):
                    raise ValueError(f"{split}: codes other than {ATTACK_CODES[split]}")
        return codes

    def vehicle_check(self) -> dict:
        train, val = self.source.fit_val()
        fit_vehicles = set(train.vehicles) | set(val.vehicles)
        test_vehicles = set()
        for split in self.source.eval_splits():
            test_vehicles |= set(self.source.load(split).vehicles)
        shared = fit_vehicles & test_vehicles
        if shared:
            raise ValueError(f"{len(shared)} test sender vehicles are in train or val")
        return {"test_vehicles": len(test_vehicles), "test_vehicles_in_train_or_val": 0}

    def groups(self, codes: dict[str, np.ndarray]) -> tuple[SplitScores, dict[str, SplitScores]]:
        """Benign test scores and one score set per attack."""
        if self.settings.window == 64:
            test = load_scores(self.scores_dir, "test")
            c = codes["test"]
            return subset(test, np.flatnonzero(c == 0), "test_benign"), {
                "GridSybil": subset(test, np.flatnonzero(c == 16), "test_gridsybil")
            }
        benign = load_scores(self.scores_dir, "test")
        attacks = {
            ATTACK_NAMES[ATTACK_CODES[split]]: load_scores(self.scores_dir, split) for split in self.attack_splits
        }
        return benign, attacks

    def metrics(self, codes: dict[str, np.ndarray]) -> dict:
        thresholds = self.summary["thresholds"]
        val = load_scores(self.scores_dir, "val")
        benign, attacks = self.groups(codes)
        val_alarm = {
            name: {q: float((val.scores[name] > th).mean()) for q, th in thresholds["window"][name].items()}
            for name in SCORE_NAMES
        }
        return {
            "run_id": self.summary["run_id"],
            "windows": {"val": len(val), "benign_test": len(benign), **{k: len(v) for k, v in attacks.items()}},
            "val_alarm_rate": val_alarm,
            "attacks": {
                name: MultiScoreMetrics(benign, attack, thresholds, name).all() for name, attack in attacks.items()
            },
        }

    def run(self, from_scores: bool = False) -> dict:
        if not from_scores:
            self.score()
        codes = self.labels()
        checks = self.vehicle_check()
        result = self.metrics(codes)
        result["label_checks"] = checks
        result["label_reads"] = list(self.source.label_reads)
        (self.run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2))
        brief = {
            name: {s: round(m["window"][s]["auroc"], 4) for s in SCORE_NAMES}
            | {"length": round(m["length_only"]["auroc"], 4)}
            for name, m in result["attacks"].items()
        }
        print(json.dumps({"run": result["run_id"], "window_auroc": brief}, indent=2))
        return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--from-scores", action="store_true", help="rebuild test_metrics.json from the score files")
    a = p.parse_args(argv)
    Evaluation(a.run_dir, a.device).run(a.from_scores)


if __name__ == "__main__":
    main()
