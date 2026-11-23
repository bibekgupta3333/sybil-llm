"""Tests for the one-class masked autoencoder on synthetic windows (CPU only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from task.masked_autoencoder.config import MAEConfig, is_finished, logs_path, run_path
from task.masked_autoencoder.data import OneClassSource, Windows
from task.masked_autoencoder.evaluate import Evaluation
from task.masked_autoencoder.model import EncoderDecoderLoss, MaskedTabularAutoencoder, count_params
from task.masked_autoencoder.plots import LossCurvePlot, finished_runs
from task.masked_autoencoder.run_all import GridRunner
from task.masked_autoencoder.scoring import EvalMasks, WindowScorer
from task.masked_autoencoder.train import Experiment

F = 13


def synthetic(split: str, n: int, t: int, seed: int, sender_base: int = 0, junk: float = 0.0) -> Windows:
    """Prefix-masked windows with `junk` in the padding rows (the real loaders write zeros there)."""
    rng = np.random.default_rng(seed)
    lengths = rng.choice([1, 2, 5, t // 2, t], size=n)
    mask = np.arange(t)[None, :] < lengths[:, None]
    x = rng.normal(size=(n, t, F)).astype(np.float32)
    x[~mask] = junk
    senders = sender_base + np.arange(n) // 3
    ids = np.asarray([f"S_0709/run/trace-{i}.json#{100 + s}-{s}#{i % 3}" for i, s in enumerate(senders)])
    vehicles = np.asarray([f"0709:{s}" for s in senders])
    return Windows(split, x, mask, ids, lengths.astype(np.int32), vehicles)


class FakeSource(OneClassSource):
    """Synthetic T = 32 source; `labels` raises unless `allow_labels` is set."""

    def __init__(self, settings: MAEConfig) -> None:
        super().__init__(settings)
        t = settings.seq_len
        self.sets = {
            "train": synthetic("train", 300, t, 0, 0),
            "val": synthetic("val", 90, t, 1, 1000),
            "test": synthetic("test", 90, t, 2, 2000),
        }
        attack = synthetic("test_gridsybil", 60, t, 3, 3000)
        attack.x[attack.mask] *= 3.0
        self.sets["test_gridsybil"] = attack
        self.allow_labels = False

    def fit_val(self):
        return self.sets["train"], self.sets["val"]

    def eval_splits(self):
        return ("test", "test_gridsybil")

    def load(self, split):
        return self.sets[split]

    def _labels(self, split):
        if not self.allow_labels:
            raise AssertionError(f"label read during training: {split}")
        return np.full(len(self.sets[split]), 16 if split == "test_gridsybil" else 0)


@pytest.fixture
def settings(tmp_path):
    return MAEConfig.for_window(
        32, runs_dir=str(tmp_path), device="cpu", max_epochs=2, eval_batch=64, threshold_quantiles=(0.95, 0.99)
    )


@pytest.mark.parametrize("window", [64, 48, 32])
@pytest.mark.parametrize("encoder", ["mlp", "resnet"])
def test_shapes(window, encoder):
    s = MAEConfig.for_window(window, encoder=encoder)
    model = MaskedTabularAutoencoder(s).eval()
    w = synthetic("x", 8, window, 0)
    recon, z = model(torch.from_numpy(w.x), torch.from_numpy(w.mask))
    assert recon.shape == (8, window, F)
    assert z.shape == (8, 16)
    assert count_params(model) > 0


def test_mlp_param_counts():
    counts = {w: count_params(MaskedTabularAutoencoder(MAEConfig.for_window(w))) for w in (64, 48, 32)}
    assert counts == {64: 113_616, 48: 85_760, 32: 57_904}


def test_resnet_param_counts():
    counts = {
        w: count_params(MaskedTabularAutoencoder(MAEConfig.for_window(w, encoder="resnet"))) for w in (64, 48, 32)
    }
    assert counts == {64: 128_224, 48: 100_576, 32: 72_928}


def test_run_layout(tmp_path):
    assert run_path("mlp", 32, 0) == Path("task/masked_autoencoder/runs/mlp/T32-seed0")
    assert run_path("resnet", 64, 3, tmp_path) == tmp_path / "resnet" / "T64-seed3"
    assert logs_path("resnet", tmp_path) == tmp_path / "resnet" / "logs"
    s = MAEConfig.for_window(48, encoder="resnet", seed=2, runs_dir=str(tmp_path))
    assert s.run_dir == tmp_path / "resnet" / "T48-seed2"
    assert s.run_id == "T48-resnet-seed2"
    assert MAEConfig.for_window(48, runs_dir=str(tmp_path), max_windows=10).run_dir.name == "T48-seed0-check"
    with pytest.raises(ValueError):
        run_path("transformer", 32, 0)


def test_grid_runner_skips_finished_runs(tmp_path):
    done, trained = run_path("resnet", 32, 0, tmp_path), run_path("resnet", 32, 1, tmp_path)
    for d in (done, trained):
        d.mkdir(parents=True)
        (d / "summary.json").write_text("{}")
    (done / "test_metrics.json").write_text("{}")
    assert is_finished(done) and not is_finished(trained)
    runner = GridRunner("resnet", windows=(32,), seeds=(0, 1, 2), runs_dir=str(tmp_path))
    assert runner.plan() == [(32, 0, "skip"), (32, 1, "evaluate"), (32, 2, "train")]
    assert "task.masked_autoencoder.evaluate" in runner.command(32, 1, "evaluate")
    assert runner.command(32, 2, "train")[-1] == "--evaluate"
    assert runner.run(dry_run=True) == 0
    assert not runner.log_path.exists()  # a dry run writes no log


def test_obfuscator_never_hides_padding():
    s = MAEConfig.for_window(32)
    model = MaskedTabularAutoencoder(s)
    w = synthetic("x", 64, 32, 0)
    torch.manual_seed(0)
    hidden = model.masker(torch.from_numpy(w.x), torch.from_numpy(w.mask))
    assert not hidden[~torch.from_numpy(w.mask)].any()
    share = hidden[torch.from_numpy(w.mask)].float().mean().item()
    assert 0.15 < share < 0.25


def test_padding_never_affects_scores_or_loss():
    s = MAEConfig.for_window(32)
    model = MaskedTabularAutoencoder(s).eval()
    masks = EvalMasks(32, F, 5, s.eval_mask_seed)
    scorer = WindowScorer(model, masks, torch.device("cpu"), 16)
    clean, junk = synthetic("x", 40, 32, 0, junk=0.0), synthetic("x", 40, 32, 0, junk=99.0)
    a, b = scorer(clean.x, clean.mask), scorer(junk.x, junk.mask)
    for k in ("masked_recon", "full_recon", "z"):
        np.testing.assert_array_equal(a[k], b[k])
    hidden = torch.from_numpy(clean.mask).unsqueeze(-1).expand(-1, -1, F).clone()
    loss = EncoderDecoderLoss()
    xt, mt = torch.from_numpy(clean.x), torch.from_numpy(clean.mask)
    pred = torch.zeros_like(xt)
    pred_junk = pred.clone()
    pred_junk[~mt] = 1e6
    assert torch.equal(loss(xt, pred, hidden, mt), loss(xt, pred_junk, hidden, mt))


def test_eval_masks_deterministic_partition():
    a, b = EvalMasks(48, F, 5, 123), EvalMasks(48, F, 5, 123)
    np.testing.assert_array_equal(a.groups, b.groups)
    counts = np.bincount(a.groups.ravel(), minlength=5)
    assert counts.max() - counts.min() <= 1
    covered = sum(a.draw(k).astype(int) for k in range(5))
    assert np.all(covered == 1)
    assert not np.array_equal(a.groups, EvalMasks(48, F, 5, 124).groups)


def test_scores_repeatable():
    s = MAEConfig.for_window(32, encoder="resnet")
    torch.manual_seed(0)
    model = MaskedTabularAutoencoder(s).eval()
    scorer = WindowScorer(model, EvalMasks(32, F, 5, s.eval_mask_seed), torch.device("cpu"), 16)
    w = synthetic("x", 30, 32, 0)
    a, b = scorer(w.x, w.mask), scorer(w.x, w.mask)
    np.testing.assert_array_equal(a["masked_recon"], b["masked_recon"])


def test_loss_only_on_hidden_cells():
    torch.manual_seed(0)
    x = torch.randn(6, 4, F)
    mask = torch.ones(6, 4, dtype=torch.bool)
    hidden = torch.rand(6, 4, F) < 0.3
    pred = torch.randn(6, 4, F, requires_grad=True)
    EncoderDecoderLoss()(x, pred, hidden, mask).backward()
    assert torch.all(pred.grad[~hidden] == 0)
    assert torch.any(pred.grad[hidden] != 0)


def test_loss_matches_widedeep_formula_without_padding():
    """T = 1: the per-feature variance equals widedeep's per-column variance (losses.py L954-970)."""
    torch.manual_seed(1)
    x, pred = torch.randn(32, 1, F), torch.randn(32, 1, F)
    hidden = torch.rand(32, 1, F) < 0.2
    ours = EncoderDecoderLoss()(x, pred, hidden, torch.ones(32, 1, dtype=torch.bool))
    xt, xp, m = x[:, 0], pred[:, 0], hidden[:, 0].float()
    err = torch.mul(xp - xt, m) ** 2
    var = torch.std(xt, dim=0) ** 2
    ref = torch.mean(torch.matmul(err, 1 / var) / (m.sum(dim=1) + 1e-9))
    assert torch.allclose(ours, ref, atol=1e-6)


def test_feed_masked_input_flag():
    w = synthetic("x", 4, 32, 0)
    xt, mt = torch.from_numpy(w.x), torch.from_numpy(w.mask)
    hidden = mt.unsqueeze(-1).expand(-1, -1, F).clone()
    on = MaskedTabularAutoencoder(MAEConfig.for_window(32)).inputs(xt, mt, hidden)
    off = MaskedTabularAutoencoder(MAEConfig.for_window(32, feed_masked_input=False)).inputs(xt, mt, hidden)
    assert torch.all(on[:, : 32 * F] == 0)
    assert torch.equal(off[:, : 32 * F], (xt * mt.unsqueeze(-1)).flatten(1))
    assert torch.equal(on[:, 32 * F :], mt.float())


def test_training_reads_no_label_and_evaluation_scores_first(settings):
    source = FakeSource(settings)
    exp = Experiment(settings, source)
    summary = exp.run()
    assert summary["label_reads"] == []
    for name in (
        "config.json",
        "env.json",
        "history.jsonl",
        "train.log",
        "best.pt",
        "mahalanobis.npz",
        "loss_curve.png",
    ):
        assert (exp.run_dir / name).exists()
    ev = Evaluation(exp.run_dir, "cpu", source=source)
    ev.score()  # raises inside FakeSource if a label were read
    assert sorted(p.name for p in ev.scores_dir.iterdir()) == ["test.npz", "test_gridsybil.npz", "val.npz"]
    source.allow_labels = True
    result = ev.run(from_scores=True)
    m = result["attacks"]["GridSybil"]["window"]
    assert set(m) == {"masked_recon", "full_recon", "mahalanobis"}
    assert 0.0 <= m["masked_recon"]["auroc"] <= 1.0
    assert json.loads((exp.run_dir / "test_metrics.json").read_text())["run_id"] == settings.run_id
    assert exp.run_dir == run_path("mlp", 32, 0, settings.runs_dir)


def test_loss_curve_from_history(tmp_path):
    run = run_path("mlp", 32, 0, tmp_path)
    run.mkdir(parents=True)
    run_path("resnet", 32, 0, tmp_path).mkdir(parents=True)  # unfinished: not listed
    rows = [
        {
            "epoch": e,
            "train_loss": 1.0 / e,
            "val_masked_mse": 0.5 / e,
            "val_full_mse": 0.4 / e,
            "lr": 1e-3 / (1 + (e > 3)),
        }
        for e in range(1, 7)
    ]
    (run / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {
        "run_id": run.name,
        "best_epoch": 6,
        "epochs_run": 6,
        "stop_reason": "max_epochs",
        "best_val_loss": 0.5 / 6,
        "train_seconds": 12.0,
        "lr_changes": [{"after_epoch": 3, "from": 1e-3, "to": 5e-4}],
    }
    (run / "summary.json").write_text(json.dumps(summary))
    assert finished_runs(tmp_path) == [run]
    assert finished_runs(tmp_path, ("resnet",)) == []
    path = LossCurvePlot(run).save()
    assert path == run / "loss_curve.png" and path.stat().st_size > 1000
