"""Unit tests for the pretraining summariser (tiny synthetic run folder; no src/data or real run access)."""

from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import summarize_pretraining as sp  # noqa: E402

_STEPS_PER_EPOCH = 100


def _eval(tag: str, epoch: int, joint: float, recon: float, auroc: float) -> dict:
    return {
        "kind": "eval",
        "tag": tag,
        "epoch": epoch,
        "step": epoch * _STEPS_PER_EPOCH,
        "val_joint": joint,
        "val_recon": recon,
        "val_recon_interp_baseline": 0.4,
        "val_recon_per_feature": [recon, recon * 2],
        "val_recon_interp_baseline_per_feature": [0.4, 0.3],
        "val_nce": 0.1,
        "val_p1": 0.2,
        "val_p2": 0.2,
        "val_p3": 0.2,
        "auroc_p1": auroc,
        "auroc_p2": auroc,
        "auroc_p3": auroc,
        "z_effective_rank": 40.0 + epoch,
        "z_std_mean": 0.5,
        "z_std_min": 0.3,
        "length_probe_r2": 0.9,
        "corr_znorm_length": -0.5,
        "cos_pos": 0.9,
        "cos_neg": 0.1,
        "uniformity": -3.0,
        "eval_seconds": 1.0,
    }


def _make_run(root: pathlib.Path) -> pathlib.Path:
    run = root / "runs" / "pretraining" / "all" / "T4" / "model-x"
    run.mkdir(parents=True)
    data = root / "enc"
    (data / "train").mkdir(parents=True)
    windows = [{"id": i, "n": n, "x": [[0.0, 0.0]] * n} for i, n in enumerate([1, 1, 1, 2, 3, 4, 4, 9])]
    with gzip.open(data / "train" / "part-00000.json.gz", "wt") as f:
        json.dump({"windows": windows}, f)
    with gzip.open(data / "train" / "part-00000_info.json.gz", "wt") as f:
        f.write("not json: the scan must never open label files")
    config = {
        "dataset": "all",
        "data_dir": str(data),
        "seq_len": 4,
        "d_model": 8,
        "batch_size": 4,
        "lr": 1e-3,
        "warmup_steps": 10,
        "max_epochs": 2,
        "max_steps": None,
        "min_rows_aux": 4,
        "seed": 0,
    }
    (run / "config.json").write_text(json.dumps(config))
    (run / "env.json").write_text(json.dumps({"device": "cpu", "rows": {"train": 400}, "scaling": {"a": {}, "b": {}}}))
    records = [_eval("init", 0, 2.0, 0.9, 0.5)]
    skipped = 0
    for step in range(1, 2 * _STEPS_PER_EPOCH + 1):
        if step % 3 == 0:  # every third batch skipped, never logged
            skipped += 1
            continue
        if step % 10 == 0:
            full = step % 20 == 0
            rec = {"kind": "train", "step": step, "epoch": (step - 1) // _STEPS_PER_EPOCH, "joint": 1.0, "recon": 0.5}
            rec.update({"nce": 0.3, "p1": 0.2, "p2": 0.2, "p3": 0.2} if full else {})
            real = step - skipped
            factor = real / 10 if real < 10 else 1.0  # LR counted on real updates (as train.py does)
            rec.update(
                {
                    "lr": 1e-3 * factor,
                    "grad_norm": 0.5,
                    "mem_gb": 1.0,
                    "step_seconds": 0.1,
                    "skipped_steps": skipped,
                    "losses_present": ["nce", "p1", "p2", "p3", "recon"] if full else ["recon"],
                }
            )
            records.append(rec)
        if step % _STEPS_PER_EPOCH == 0:
            e = step // _STEPS_PER_EPOCH
            records.append(_eval(f"epoch{e}", e, 1.0 / e, 0.2, 0.95))
    records.append(
        {"kind": "summary", "stop_reason": "max_epochs", "step": 200, "epoch": 2, "best_val_joint": 0.5, "hours": 0.1}
    )
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (run / "train.log").write_text("> train:all\n> python -m train --dataset all\n\nrun folder\n")
    sums = [f"{hashlib.sha256((run / n).read_bytes()).hexdigest()}  {n}" for n in ("config.json", "metrics.jsonl")]
    (run / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return run


def test_summary_schema_skips_and_verdicts(tmp_path: pathlib.Path) -> None:
    run = _make_run(tmp_path)
    out = tmp_path / "runs" / "evaluation" / "all" / "T4" / "model-x" / "training_summary.json"
    assert sp.default_out(run) == out
    summary = sp.SummaryWriter(run, out, None, scan=True, max_points=5).write()
    on_disk = json.loads(out.read_text())
    assert on_disk["kind"] == "training_summary"
    for key in ("checksums", "settings", "steps", "batch_kinds", "lr_schedule", "eval_per_epoch", "curves", "verdicts"):
        assert key in on_disk
    assert summary["checksums"]["all_ok"] is True
    assert set(summary["checksums"]["uncovered"]) == {"env.json", "train.log"}
    assert summary["command"] == "python -m train --dataset all"
    # true skipped = 66 of 200; the logged-counter bracket must contain it
    lo, hi = summary["steps"]["skipped_bracket"]
    assert lo <= 66 <= hi
    assert [e["epoch"] for e in summary["steps"]["per_epoch"]] == [1, 2]
    # lengths: 3 of 8 windows have n = 1, 2 have 2-3, 3 have >= 4 (label file never parsed)
    scan = summary["data"]["length_scan"]
    assert scan["windows"] == 8 and scan["share_n1_no_loss"] == pytest.approx(3 / 8, abs=1e-4)
    assert scan["share_n_ge_4_all_losses"] == pytest.approx(3 / 8, abs=1e-4)
    curve = summary["curves"]["train"]["all"]["recon"]
    assert len(curve["step"]) <= 5 and sum(curve["n"]) == len([r for r in sp.RunFolder(run).train])
    checks = {v["check"]: v for v in summary["verdicts"]}
    assert checks["reconstruction beats interpolation?"]["status"] == "ok"
    assert checks["converged?"]["status"] == "look"  # val_joint halved in the last epoch
    assert checks["LR schedule as configured?"]["status"] == "look"  # LR never decays in the fake log
    assert checks["training values finite?"]["status"] == "ok"
    assert summary["batch_kinds"]["counts"]["recon_only"] > 0


def test_refuses_to_write_inside_run_folder(tmp_path: pathlib.Path) -> None:
    run = _make_run(tmp_path)
    with pytest.raises(ValueError):
        sp.SummaryWriter(run, run / "training_summary.json", None, scan=False, max_points=10)


def test_curve_sampler_keeps_short_series_and_bins_long_ones() -> None:
    sampler = sp.CurveSampler(4)
    short = sampler([1, 2], [0.5, float("nan")])
    assert short == {"step": [1], "value": [0.5], "n": [1]}
    long = sampler(list(range(100)), [float(i) for i in range(100)])
    assert len(long["step"]) == 4 and sum(long["n"]) == 100
