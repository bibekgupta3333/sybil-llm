"""Unit tests for the model-all report generator (tiny synthetic inputs; no real run, data or results read)."""

from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import build_model_report as bmr  # noqa: E402

_SECRET_PROBE_AUROC = 0.4242424242


def _eval_row(epoch: int) -> dict:
    return {
        "tag": "init" if epoch == 0 else f"epoch{epoch}",
        "epoch": epoch,
        "step": epoch * 10,
        "val_joint": 2.0 - 0.1 * epoch,
        "val_recon": 0.8 - 0.05 * epoch,
        "val_recon_interp_baseline": 0.4,
        "auroc_p1": 0.5 + 0.04 * epoch,
        "auroc_p2": 0.5 + 0.04 * epoch,
        "auroc_p3": 0.5 + 0.04 * epoch,
        "z_effective_rank": 40.0 + epoch,
        "length_probe_r2": 0.9,
    }


def _curve(n: int = 5) -> dict:
    return {"step": list(range(0, n * 10, 10)), "value": [1.0 / (i + 1) for i in range(n)], "n": [1] * n}


def _summary() -> dict:
    curves = {k: _curve() for k in ("joint", "recon", "nce", "p1", "p2", "p3")}
    return {
        "created_utc": "2026-01-01T00:00:00+00:00",
        "run_dir": "src/runs/pretraining/all/T24/model-x",
        "scope_note": "synthetic",
        "command": "python -m train --dataset all",
        "checksums": {"files": {"best.pt": {"sha256": "ab" * 32, "bytes": 10, "ok": True}}},
        "hardware": {"device": "cpu", "torch": "x", "git_commit": "deadbeef", "created_utc": "2026-01-01"},
        "settings": {"dataset": "all", "seq_len": 24, "d_model": 128, "n_blocks": 4, "seed": 0, "min_rows_aux": 4},
        "data": {
            "rows": {"train": 90, "pretrain_check": 10},
            "label_files_opened": 0,
            "length_scan": {"windows": 100, "shards": 1, "hist_n": {"1": 70, "2": 10, "24": 20}},
        },
        "params": {"params_encoder": 1, "params_total": 2},
        "run_summary": {"epoch": 2, "step": 40, "hours": 0.1, "best_val_joint": 1.8},
        "steps": {"total_batches": 40, "skipped_bracket": [20, 22], "real_updates_bracket": [18, 20]},
        "batch_kinds": {"share_of_all_batches_est": {"skipped_no_loss": 0.5, "recon_only": 0.2, "all_losses": 0.3}},
        "timing": {"wall_hours": 0.1},
        "lr_schedule": {"peak_lr": 0.001, "warmup_steps": 5, "planned_total_steps": 40, "final_lr_over_peak": 0.8},
        "eval_per_epoch": [_eval_row(e) for e in range(3)],
        "curves": {
            "method": "bins",
            "train": {"all_losses_batches": curves, "recon_only_batches": curves, "lr": _curve()},
        },
        "verdicts": [{"check": "converged?", "verdict": "No", "status": "look"}],
    }


def _manifest() -> dict:
    full = {c: {"n": 5, "auroc_vs_benign": 0.6, "length_baseline_auroc": 0.5, "flag_rate": 0.1} for c in bmr.CLASSES}
    full["Benign"]["auroc_vs_benign"] = None
    return {
        "created": "2026-01-01",
        "model": {"sha256": "cd" * 32, "checkpoint": "best.pt", "epoch": 2},
        "method": {"name": "kNN", "k": 10, "bank_size": 100, "theta": 0.2},
        "metrics": {"full_test_1416": full, "benign_fpr_at_theta": 0.1},
        "caveats": ["synthetic"],
    }


def _findings() -> dict:
    return {
        "title": "Synthetic findings",
        "date": "2026-01-01",
        "author": "test",
        "summary": [{"q": "Did it work?", "a": "Partly.", "status": "partly"}],
        "issues": [{"id": "I1", "title": "LR bug", "severity": "high", "evidence": ["e1"], "impact": "big"}],
        "fixes": [
            {
                "id": "X1",
                "title": "Fix LR",
                "addresses": ["I1"],
                "what": "w",
                "expected_effect": "e",
                "cost": "c",
                "priority": 1,
                "plan_task": "S1.3",
                "needs_user_ok": True,
            }
        ],
        "direction": ["Fix the LR.", "Rerun."],
        "limitations": ["one seed"],
        "pending": ["eval:all"],
    }


def _cell(v: float) -> dict:
    return {"mean": v, "std": 0.01, "ci95": {"lo": v - 0.02, "hi": v + 0.02, "n": 10}}


def _report_data() -> dict:
    main = []
    for rep in bmr.REPRESENTATIONS:
        row = {"representation": rep}
        for key in ("lr", "knn"):
            for m in ("binary_auroc", "binary_f1", "macro_f1"):
                row[f"{key}_{m}"] = _cell(_SECRET_PROBE_AUROC)
        row["anomaly_binary_auroc"] = _cell(_SECRET_PROBE_AUROC)
        main.append(row)
    fam = []
    for rep in bmr.REPRESENTATIONS:
        for probe in ("lr", "knn", "anomaly"):
            for f in bmr.FAMILIES:
                strata = {
                    s: {"n": 3, "auroc_mean": _SECRET_PROBE_AUROC, "auroc_std": 0.0, "auroc_ci95": None}
                    for s in ("all", "n<=3", "n>=4")
                }
                fam.append({"representation": rep, "probe": probe, "family": f, **strata})
    eye = [[1.0 if i == j else 0.0 for j in range(5)] for i in range(5)]
    return {
        "created": "2026-01-02",
        "scope": {"note": "synthetic"},
        "classes": list(bmr.CLASSES),
        "seeds": [0, 1, 2],
        "main_table": main,
        "per_family": fam,
        "confusion_row_normalised_mean": {r: {"lr": eye, "knn": eye} for r in bmr.REPRESENTATIONS},
        "benign_flagged_rate": {
            r: {p: {"all": 0.1, "n<=3": 0.1, "n>=4": 0.1} for p in ("lr", "knn", "anomaly")}
            for r in bmr.REPRESENTATIONS
        },
        "notes": ["synthetic"],
    }


def _write(path: pathlib.Path, obj: dict) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _build(tmp_path: pathlib.Path, eval_config: dict | None = None, findings: bool = True) -> str:
    summary = _write(tmp_path / "summary.json", _summary())
    manifest = _write(tmp_path / "manifest.json", _manifest())
    fpath = tmp_path / "findings.json"
    if findings:
        _write(fpath, _findings())
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    if eval_config is not None:
        _write(eval_dir / "report_data.json", _report_data())
        _write(eval_dir / "config.json", {"evaluation": eval_config})
    out = tmp_path / "out" / "report.html"
    argv = [
        "--summary",
        str(summary),
        "--manifest",
        str(manifest),
        "--findings",
        str(fpath),
        "--eval-dir",
        str(eval_dir),
        "--metadata",
        str(tmp_path / "missing-metadata.json"),
        "--out",
        str(out),
    ]
    assert bmr.main(argv) == 0
    files = list((tmp_path / "out").iterdir())
    assert files == [out]
    return out.read_text(encoding="utf-8")


def _assert_self_contained(page: str) -> None:
    assert page.startswith("<!doctype html>")
    assert "<script" not in page
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", page)
    assert "@import" not in page and "url(http" not in page
    for sid in bmr.SECTION_IDS:
        assert f'id="{sid}"' in page, sid


FULL = {"seeds": [0, 1, 2], "per_class": 20_000, "bootstrap": 1_000}


def test_renders_pending_without_results(tmp_path: pathlib.Path) -> None:
    page = _build(tmp_path)
    _assert_self_contained(page)
    assert "Pending." in page and "npm run eval:all" in page
    assert "Synthetic findings" in page and "LR bug" in page and "Fix LR" in page
    assert str(_SECRET_PROBE_AUROC) not in page


def test_renders_full_results(tmp_path: pathlib.Path) -> None:
    page = _build(tmp_path, FULL)
    _assert_self_contained(page)
    assert "Pending." not in page
    assert "Main table" in page and "5-class confusion matrices" in page
    assert str(_SECRET_PROBE_AUROC) in page  # exact value in a tooltip


@pytest.mark.parametrize(
    "config",
    [
        {"seeds": [0], "per_class": 20_000, "bootstrap": 1_000},
        {"seeds": [0, 1, 2], "per_class": 50, "bootstrap": 1_000},
        {"seeds": [0, 1, 2], "per_class": 20_000, "bootstrap": 10},
        {},
    ],
)
def test_reduced_run_numbers_never_shown(tmp_path: pathlib.Path, config: dict) -> None:
    page = _build(tmp_path, config)
    _assert_self_contained(page)
    assert "Pending." in page
    assert str(_SECRET_PROBE_AUROC) not in page
    assert "0.424" not in page


def test_missing_findings_is_visible(tmp_path: pathlib.Path) -> None:
    page = _build(tmp_path, findings=False)
    _assert_self_contained(page)
    assert page.count("Findings missing.") == 4


def test_missing_summary_fails(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        bmr.main(["--summary", str(tmp_path / "nope.json"), "--out", str(tmp_path / "x.html")])


def test_number_tooltip_keeps_exact_value() -> None:
    out = bmr.Num.f(0.123456789, 3)
    assert 'title="0.123456789"' in out and ">0.123<" in out
    assert "—" in bmr.Num.f(None)
    assert bmr.tick_label(0.00025) == "0.00025"
