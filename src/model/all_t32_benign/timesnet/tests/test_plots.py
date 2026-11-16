"""Loaders and plots of the analysis notebook, on a synthetic run folder and evaluation folder (Agg backend, CPU)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.model.all_t32_benign.timesnet import plots, scoring  # noqa: E402
from src.model.all_t32_benign.timesnet.data import space_state  # noqa: E402
from src.model.all_t32_benign.timesnet.base_config import PretrainConfig  # noqa: E402
from src.model.all_t32_benign.timesnet.base_data import FEATURES, FeatureSpace  # noqa: E402
from src.model.all_t32_benign.timesnet.heads import PretrainingModel  # noqa: E402

from .conftest import SYNTHETIC_MEAN, SYNTHETIC_STD  # noqa: E402

EPOCHS = 4
BEST = 3


def _jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _history(epochs: int = EPOCHS) -> list[dict]:
    rows = []
    for e in range(epochs + 1):
        train = None if e == 0 else 1.0 / (e + 1)
        row = {"epoch": e, "step": 10 * e, "lr": 1e-3 * (1 - e / (epochs + 1))}
        for k in ("joint", "recon", "nce", "p1", "p2", "p3"):
            row[f"train_{k}"] = train
            row[f"val_{k}"] = 1.2 / (e + 1)
        row.update(
            {
                "train_steps": 0 if e == 0 else 10,
                "train_skipped": 0,
                "grad_norm_mean": None if e == 0 else 0.5,
                "grad_norm_max": None if e == 0 else 0.9,
                "epoch_seconds": None if e == 0 else 30.0,
                "eval_seconds": 5.0,
                "val_recon_interp_baseline": 0.3,
                "auroc_p1": 0.5 + 0.1 * e,
                "auroc_p2": 0.5 + 0.08 * e,
                "auroc_p3": 0.5 + 0.05 * e,
                "z_std_mean": 0.7,
                "z_std_min": 0.4,
                "z_effective_rank": 50 + e,
                "length_probe_r2": 0.9,
                "corr_znorm_length": -0.2,
                "align": 0.05,
                "uniformity": -1.5,
                "cos_pos": 0.97,
                "cos_neg": 0.5,
                "is_best": e == BEST,
                "bad_epochs": max(0, e - BEST),
                "rss_gb": 6.0 + 0.1 * e,
                "scale_recon": None if e == 0 else 0.8,
                "scale_nce": None if e == 0 else 3.0,
            }
        )
        rows.append(row)
    return rows


def _features(epochs: int = EPOCHS) -> list[dict]:
    return [
        {
            "epoch": e,
            "features": list(FEATURES),
            "train_recon_row_pooled": None if e == 0 else [0.5 / (e + 1)] * 13,
            "val_recon": [0.6 / (e + 1) + 0.01 * j for j in range(13)],
            "val_interp_baseline": [0.2 + 0.01 * j for j in range(13)],
        }
        for e in range(epochs + 1)
    ]


def _steps(n: int = 40) -> list[dict]:
    rows = []
    for s in range(1, n + 1):
        row = {"step": s, "epoch": (s - 1) // 10, "lr": 1e-3 * s / n, "grad_norm": 0.5, "step_seconds": 0.2}
        row.update({k: 1.0 / s for k in ("joint", "recon", "p1", "p2", "p3")})
        row["nce"] = None if s % 7 == 0 else 2.0 / s  # a loss can be absent on some updates
        rows.append(row)
    return rows


def _metadata() -> dict:
    return {"features": list(FEATURES), "normalisation": {"mean": SYNTHETIC_MEAN, "std": SYNTHETIC_STD}}


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A synthetic finished run folder with every file of the trainer's contract and a tiny real checkpoint."""
    run = tmp_path / "runs" / "model-t32b-seed0"
    run.mkdir(parents=True)
    settings = PretrainConfig(d_model=16, d_ff=8, n_blocks=1, proj_dim=8, seq_len=32)
    (run / "config.json").write_text(json.dumps({"d_model": 16, "grad_clip": 1.0, "lambda_nce": 0.3}))
    (run / "env.json").write_text(json.dumps({"preflight": {"params_encoder": 1, "params_total": 2}}))
    (run / "summary.json").write_text(json.dumps({"run_id": run.name, "best_epoch": BEST, "short": False}))
    _jsonl(run / "history.jsonl", _history())
    _jsonl(run / "feature_history.jsonl", _features())
    _jsonl(run / "steps.jsonl", _steps())
    _jsonl(run / "metrics.jsonl", [{"kind": "eval", "epoch": 0, "rss_gb": 6.0}, {"kind": "train", "step": 1}])
    (run / "train.log").write_text("line 1\nline 2\n")
    torch.manual_seed(0)
    model = PretrainingModel(settings)
    space = FeatureSpace(_metadata())
    import dataclasses

    torch.save(
        {"model": model.state_dict(), "settings": dataclasses.asdict(settings), "space": space_state(space)},
        run / "best.pt",
    )
    return run


@pytest.fixture
def eval_dir(tmp_path: Path, run_dir: Path) -> Path:
    """A synthetic evaluation folder: score files + a summary.json built with the package's DetectionReport."""
    rng = np.random.default_rng(0)
    out = tmp_path / "evaluation" / f"{run_dir.name}-smoke-20261115-000000"
    (out / "scores").mkdir(parents=True)
    splits = {}
    for k, split in enumerate(("val", "test", *plots.ATTACKS)):
        n = 60
        rows = rng.integers(1, 33, size=n)
        recon_raw = np.where(rows >= 2, rng.gamma(2.0, 0.1 + 0.05 * (k > 1), size=n), np.nan)
        arrays = {
            "ids": np.array([f"f{split}#{1000 + i}-{i}#{i % 2}" for i in range(n)]),
            "n_real_rows": rows,
            "recon_raw": recon_raw,
            "recon_defined": np.isfinite(recon_raw),
            "recon": np.where(np.isfinite(recon_raw), recon_raw, 0.2),
            "mahalanobis": rng.gamma(3.0, 1.0 + 0.3 * (k > 1), size=n),
            **{f"knn_k{kk}": rng.uniform(0, 0.5 + 0.1 * (k > 1), size=n) for kk in (5, 10, 20)},
        }
        np.savez(out / "scores" / f"{split}.npz", **arrays)
        names = [a for a in arrays if a in ("recon", "mahalanobis") or a.startswith("knn_k")]
        splits[split] = scoring.ScoredSplit(arrays["ids"], rows, {a: arrays[a].astype(float) for a in names})
    report = scoring.DetectionReport(splits["val"], (0.95, 0.99))
    metrics = {
        a: report.compare(splits["test"], splits[a], np.zeros(60, int), np.full(60, code))
        for a, code in zip(plots.ATTACKS, (16, 17, 18, 19))
    }
    summary = {
        "smoke": True,
        "random_init": False,
        "metrics": metrics,
        "thresholds": report.thresholds,
        "events": {"scores_written_utc": "a", "labels_first_read_utc": "b", "labels_read_after_scores": True},
        "alignment": {"checks": "all passed", "info_shards_opened": 6},
        "metadata_check": {"problems": []},
        "checkpoint": {"weights": "checkpoint"},
    }
    (out / "summary.json").write_text(json.dumps(summary))
    return out


def _is_fig(fig: object) -> bool:
    ok = isinstance(fig, Figure)
    plt.close("all")
    return ok


def test_run_folder_loads_contract_files(run_dir):
    run = plots.RunFolder.newest(run_dir.parent)
    assert run.path == run_dir
    assert len(run.history) == EPOCHS + 1 and list(run.epochs.epoch) == list(range(1, EPOCHS + 1))
    assert run.best_epoch == BEST
    assert len(run.feature_history) == EPOCHS + 1 and len(run.steps) == 40
    assert not run.is_short()
    assert len(run.eval_records()) == 1


def test_best_epoch_fallbacks(run_dir):
    (run_dir / "summary.json").unlink()
    assert plots.RunFolder(run_dir).best_epoch == BEST  # from is_best
    _jsonl(run_dir / "history.jsonl", [{k: v for k, v in r.items() if k != "is_best"} for r in _history()])
    assert plots.RunFolder(run_dir).best_epoch == EPOCHS  # argmin val_joint (decreasing curve)


def test_newest_without_history_raises(tmp_path):
    (tmp_path / "r").mkdir()
    with pytest.raises(FileNotFoundError):
        plots.RunFolder.newest(tmp_path)


def test_training_plots_return_figures(run_dir):
    tp = plots.TrainingPlots(plots.RunFolder(run_dir))
    for fig in (
        tp.losses(),
        tp.recon_vs_interpolation(),
        tp.steps(),
        tp.physics_auroc(),
        tp.representation(),
        tp.time_memory(),
        tp.loss_scales(),
    ):
        assert _is_fig(fig)


def test_plots_skip_gracefully_on_an_empty_run(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    run = plots.RunFolder(empty)
    tp = plots.TrainingPlots(run)
    assert tp.losses() is None and tp.steps() is None and tp.recon_vs_interpolation() is None
    assert tp.physics_auroc() is None and tp.representation() is None and tp.time_memory() is None
    assert tp.loss_scales() is None
    fp = plots.FeaturePlots(run, FEATURES)
    assert fp.curves() is None and fp.best_bar() == (None, None)
    assert run.best_epoch is None
    assert "[skipped]" in capsys.readouterr().out


def test_one_epoch_smoke_history_still_plots(tmp_path):
    run_dir = tmp_path / "smoke"
    run_dir.mkdir()
    _jsonl(run_dir / "history.jsonl", _history(1))
    _jsonl(run_dir / "feature_history.jsonl", _features(1))
    run = plots.RunFolder(run_dir)
    assert run.is_short() and run.best_epoch == 1
    assert _is_fig(plots.TrainingPlots(run).losses())
    assert _is_fig(plots.FeaturePlots(run, FEATURES).curves())


def test_feature_plots(run_dir):
    fp = plots.FeaturePlots(plots.RunFolder(run_dir), FEATURES)
    assert fp.frame("val_recon").shape == (EPOCHS + 1, 13)
    assert fp.frame("train_recon_row_pooled").shape == (EPOCHS, 13)  # epoch 0 has no train rows
    assert _is_fig(fp.curves())
    fig, table = fp.best_bar()
    assert _is_fig(fig)
    assert list(table.index) == list(FEATURES)
    assert math.isclose(table.loc[FEATURES[0], "model (val)"], 0.6 / (BEST + 1))


def test_recon_examples_cpu(run_dir):
    ex = plots.ReconExamples(run_dir / "best.pt", _metadata())
    x = torch.randn(2, 32, 13)
    mask = torch.zeros(2, 32, dtype=torch.bool)
    mask[0, :32], mask[1, :12] = True, True
    out = ex.run(x, mask, seed=0)
    assert out["pred"].shape == (2, 32, 13)
    assert out["hidden"].sum() > 0 and not (out["hidden"] & ~out["mask"]).any()
    again = ex.run(x, mask, seed=0)
    assert np.array_equal(out["hidden"], again["hidden"]) and np.allclose(out["pred"], again["pred"])
    assert _is_fig(plots.ReconExamples.plot(out, FEATURES, ["claimed_pos_x", "range", "log_dtau"], ["a", "b"]))


def test_interpolation_fills_hidden_rows_only():
    x = np.arange(10, dtype=float)[:, None].repeat(2, 1)
    mask = np.array([True] * 8 + [False] * 2)
    hidden = np.zeros(10, bool)
    hidden[[3, 4]] = True
    xi = x.copy()
    xi[[3, 4]] = 0.0
    out = plots.ReconExamples.interpolate(xi, hidden, mask)
    assert np.allclose(out[[3, 4], 0], [3.0, 4.0]) and np.array_equal(out[~hidden], xi[~hidden])


def test_eval_folder_and_tables(run_dir, eval_dir):
    ev = plots.EvalFolder.find(run_dir.name, eval_root=eval_dir.parent)
    assert ev is not None and ev.path == eval_dir and ev.smoke
    assert plots.EvalFolder.find(run_dir.name, eval_root=eval_dir.parent, random_init=True) is None
    assert ev.scores("test")["recon"].shape == (60,)
    assert ev.scores("nope") is None
    assert "length_only" in ev.score_names() and ev.threshold("recon") is not None
    assert ev.label_checks()["labels_read_after_scores"] is True
    wide = plots.AurocTable.wide(ev.summary)
    assert ("GridSybil", "recon") in wide.index and ("GridSybil", "length_only") not in wide.index
    assert ("n ≥ 4", "length only") in wide.columns and ("recon fill share", "attack") in wide.columns
    long = plots.AurocTable.long(ev.summary)
    expected = ev.summary["metrics"]["test_gridsybil"]["window"]["mahalanobis"]["n_ge_4"]["auroc"]
    got = long[(long.attack == "GridSybil") & (long.score == "mahalanobis") & (long.subset == "n_ge_4")].AUROC
    assert math.isclose(got.iloc[0], expected)
    assert not plots.AurocTable.tracks(ev.summary).empty


def test_full_evaluation_is_preferred_over_smoke(run_dir, eval_dir):
    full = eval_dir.parent / run_dir.name
    full.mkdir()
    (full / "summary.json").write_text(json.dumps({"smoke": False}))
    assert plots.EvalFolder.find(run_dir.name, eval_root=eval_dir.parent).path == full


def test_test_plots(eval_dir):
    tp = plots.TestPlots(plots.EvalFolder(eval_dir))
    assert _is_fig(tp.histograms())
    assert _is_fig(tp.recon_by_rows())
    mix = tp.length_mix()
    assert mix.loc["benign test", "windows"] == 60


def test_test_plots_skip_without_scores(tmp_path):
    (tmp_path / "e").mkdir()
    tp = plots.TestPlots(plots.EvalFolder(tmp_path / "e"))
    assert tp.histograms() is None and tp.recon_by_rows() is None and tp.length_mix().empty


def test_baseline_context_from_notebook_output(tmp_path, eval_dir):
    results = {
        "run": "seed0",
        "GridSybil/reconstruction/window": {"AUROC": 0.7, "AUROC (>= 4 rows)": 0.75},
        "GridSybil/mahalanobis/window": {"AUROC": 0.5, "AUROC (>= 4 rows)": 0.55},
        "GridSybil/reconstruction/track max": {"AUROC": 0.6},
    }
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": ['results = {"run": RUN.name, "params": 1}\n'],
                "outputs": [{"output_type": "execute_result", "data": {"text/plain": [repr(results)]}}],
            }
        ]
    }
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(nb))
    ctx = plots.BaselineContext.load(path, tmp_path / "missing.json")
    assert ctx is not None and "stored output" in ctx.source
    table = ctx.table()
    assert table.loc[("GridSybil", "recon"), "ConvAE n ≥ 4"] == 0.75
    cmp = ctx.compare(plots.EvalFolder(eval_dir).summary)
    assert {"ConvAE all", "TimesNet all", "ConvAE n ≥ 4", "TimesNet n ≥ 4"} == set(cmp.columns)
    assert np.isfinite(cmp.loc[("GridSybil", "recon"), "TimesNet n ≥ 4"])


def test_baseline_context_fallback_and_missing(tmp_path):
    assert plots.BaselineContext.load(tmp_path / "no.ipynb", tmp_path / "no.json") is None
    data = {
        plots.BASELINE_KEY: {"test": {"source": "x", "metrics": {"GridSybil/reconstruction/window": {"AUROC": 0.7}}}}
    }
    (tmp_path / "r.json").write_text(json.dumps(data))
    ctx = plots.BaselineContext.load(tmp_path / "no.ipynb", tmp_path / "r.json")
    assert ctx is not None and ctx.table().loc[("GridSybil", "recon"), "ConvAE all"] == 0.7
