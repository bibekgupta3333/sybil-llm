"""Test evaluation of one finished run: scores every on-disk test window with `best.pt` (x and mask only), then reads
the labels and writes `test_results.json` and `test_scores.npz` into the run folder.

Thresholds come from the run's `summary.json` (picked on val by `train.py`). Reported per window and per track
(max / mean of the window probabilities): AUROC, PR-AUC, and precision / recall / F1 / balanced accuracy / benign FPR
at the val max-F1 threshold, the val benign-FPR-5% threshold and a fixed 0.5. Length-only references: the AUROC of
the real-row count in the best direction (picked on test labels, as in the other `autoencoder_st*` notebooks) and with
the fixed rule score = -n_messages (shorter = more suspicious; not picked on test).

Usage (repo root):
    .venv-train/bin/python -m task.autoencoder_st_t64.evaluate --run task/autoencoder_st_t64/runs/seed0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from task.autoencoder_st.metrics import TrackScores, ranking_metrics, threshold_metrics
from task.autoencoder_st_t64.data import Split, load_test
from task.autoencoder_st_t64.trainer import Trainer

LENGTH_BINS = ((1, 3), (4, 16), (17, 63), (64, 64), (4, 64))


def length_auroc(y: np.ndarray, n: np.ndarray) -> float:
    """AUROC of the real-row count, best direction."""
    a = roc_auc_score(y, n)
    return float(max(a, 1 - a))


def level_rows(y: np.ndarray, scores: np.ndarray, thresholds: dict) -> dict:
    """{threshold name: metrics} at the val thresholds and 0.5, each with the threshold-free AUROC / PR-AUC."""
    base = ranking_metrics(y, scores)
    picks = {**{k: v["threshold"] for k, v in thresholds.items()}, "fixed 0.5": 0.5}
    out = {}
    for name, t in picks.items():
        m = threshold_metrics(y, scores, t)
        out[name] = {**base, "θ": t, **{k: m[k] for k in ("precision", "recall", "f1", "balanced_accuracy", "benign_fpr")},
                     "confusion": m["confusion"]}
    return out


def evaluate(run_dir: str | Path, test: Split | None = None, write: bool = True) -> tuple[dict, dict]:
    """(results, raw outputs). `test` can be passed in to avoid a second load."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines() if line]
    trainer = Trainer.from_checkpoint(run_dir / "best.pt")
    test = test if test is not None else load_test(trainer.settings)
    # scoring first: the model sees x and mask only
    out = trainer.predict(test.x, test.mask)
    tracks = TrackScores(test.tracks)
    track_scores = tracks.aggregate(out["prob"])
    # labels after scoring
    y = test.y.astype(int)
    y_trk, mixed = tracks.track_labels(y)
    thr = summary["thresholds"]
    results: dict = {
        "run": run_dir.name,
        "params": summary["params"],
        "best_epoch": summary["best_epoch"],
        "best_val_auroc": round(summary["best_val_auroc"], 4),
        "epoch1_val_auroc": round(float(history[0]["val_auroc"]), 4),
        "epochs_run": summary["epochs_run"],
        "stop_reason": summary["stop_reason"],
        "lr_changes": len(summary["lr_changes"]),
        "train_minutes": round(summary["train_seconds"] / 60, 1),
        "pos_weight": round(summary["pos_weight"], 4),
        "mixed_label_tracks": mixed,
        "test_windows": len(test),
        "test_tracks": len(tracks),
        "test_gridsybil_share": round(float(y.mean()), 4),
        "length_only_auroc": round(length_auroc(y, test.n_messages), 4),
        "length_only_auroc_fixed_neg_n": round(float(roc_auc_score(y, -test.n_messages)), 4),
        "reconstruction_error_auroc": round(float(roc_auc_score(y, out["error"])), 4),
    }
    levels = {
        "window": (y, out["prob"], thr["window"]),
        "track max": (y_trk, track_scores["max"], thr["track"]["max"]),
        "track mean": (y_trk, track_scores["mean"], thr["track"]["mean"]),
    }
    for level, (yy, ss, tt) in levels.items():
        for name, m in level_rows(yy, ss, tt).items():
            results[f"{level}/{name}"] = {k: (v if k == "confusion" else round(float(v), 4)) for k, v in m.items()}
    by_length = {}
    for lo, hi in LENGTH_BINS:
        s = (test.n_messages >= lo) & (test.n_messages <= hi)
        yy = y[s]
        row = {"windows": int(s.sum()), "gridsybil_share": round(float(yy.mean()), 4)}
        if 0 < yy.mean() < 1:
            row["auroc"] = round(float(roc_auc_score(yy, out["prob"][s])), 4)
            if lo != hi:
                row["length_only_auroc"] = round(length_auroc(yy, test.n_messages[s]), 4)
        by_length[f"{lo}-{hi}"] = row
    results["by_length"] = by_length
    by_period = {}
    for p in sorted(set(test.period.tolist())):
        s = test.period == p
        by_period[p] = {"windows": int(s.sum()), "gridsybil_share": round(float(y[s].mean()), 4),
                        **{k: round(v, 4) for k, v in ranking_metrics(y[s], out["prob"][s]).items()}}
    results["by_period"] = by_period
    results["threshold_source"] = summary.get("threshold_source", "val")
    if write:
        (run_dir / "test_results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
        np.savez_compressed(
            run_dir / "test_scores.npz",
            ids=test.ids.astype(str),
            n_messages=test.n_messages,
            prob=out["prob"],
            recon_error=out["error"],
            label=y.astype(np.int8),
        )
    return results, {"out": out, "track_scores": track_scores, "y": y, "y_trk": y_trk, "test": test, "tracks": tracks}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True)
    a = p.parse_args(argv)
    results, _ = evaluate(a.run)
    print(json.dumps({k: v for k, v in results.items() if not isinstance(v, dict)}, indent=1))
    print(json.dumps({k: results[k] for k in ("window/max_f1", "window/benign_fpr_5pct")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
