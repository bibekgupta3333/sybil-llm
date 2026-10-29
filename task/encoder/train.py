"""Trains one baseline run and writes its folder `task/encoder/runs/<run_id>/`.

Modes:
    ae          one-class ConvAutoencoder on benign train windows only (labels used only to select them);
                validation = 10% of benign train sender vehicles (split seed 0), early stopping on its masked MSE.
    supervised  the same encoder + SupervisedHead, BCE with class weight on all train windows; validation =
                10% of train sender vehicles (both classes).

Outputs (the contract with `evaluate.py`): scores_test.npz (id, label, n_messages, recon_error, recon_error_feat,
mahalanobis, supervised_prob, z), scores_val.npz (validation benign recon_error, mahalanobis), config.json,
env.json, history.jsonl; plus summary.json, train.log and best.pt. Test labels are read only after every test
score exists.

Usage (repo root):
    .venv-train/bin/python -m task.encoder.train --mode ae --seed 0
    .venv-train/bin/python -m task.encoder.train --mode supervised --seed 1 --epochs 50 --device mps
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as dt
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from task.encoder.config import EncoderTaskConfig
from task.encoder.data import VehicleSplit, WindowSet, WindowStore
from task.encoder.model import ConvAutoencoder, SupervisedHead, count_params, masked_mse, window_errors


class RunLog:
    """Timestamped lines to the run's train.log and to a shared progress log (survives an interrupted session)."""

    def __init__(self, *paths: Path) -> None:
        self.paths = paths
        for p in paths:
            p.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        for p in self.paths:
            with p.open("a") as f:
                f.write(line + "\n")


def pick_device(name: str) -> torch.device:
    """`auto` -> mps, then cuda, then cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def environment(device: torch.device) -> dict:
    """Versions, host and git state for env.json (RULE 4)."""

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = ""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn_version,
        "device": str(device),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "argv": sys.argv,
        "started": dt.datetime.now().isoformat(timespec="seconds"),
    }


class Mahalanobis:
    """Distance of z to a Gaussian fitted on benign train z (mean + ridge-regularised covariance)."""

    def __init__(self, ridge: float) -> None:
        self.ridge = ridge
        self.mean: np.ndarray | None = None
        self.precision: np.ndarray | None = None

    def fit(self, z: np.ndarray) -> "Mahalanobis":
        z = z.astype(np.float64)
        self.mean = z.mean(axis=0)
        cov = np.cov(z, rowvar=False) + self.ridge * np.eye(z.shape[1])
        self.precision = np.linalg.pinv(cov)
        return self

    def score(self, z: np.ndarray) -> np.ndarray:
        d = z.astype(np.float64) - self.mean
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.precision, d), 0.0))


@dataclasses.dataclass
class Scores:
    """Model outputs for a window set (NaN where the mode has no such output)."""

    recon_error: np.ndarray
    recon_error_feat: np.ndarray
    z: np.ndarray
    supervised_prob: np.ndarray


class Scorer:
    """Runs a trained model over a window set in batches (works on memory-mapped arrays)."""

    def __init__(self, model: nn.Module, mode: str, device: torch.device, batch: int) -> None:
        self.model, self.mode, self.device, self.batch = model, mode, device, batch

    @torch.no_grad()
    def __call__(self, data: WindowSet) -> Scores:
        self.model.eval()
        errs, feats, zs, probs = [], [], [], []
        for start in range(0, len(data), self.batch):
            x = torch.from_numpy(np.array(data.x[start : start + self.batch])).to(self.device)
            m = torch.from_numpy(np.array(data.mask[start : start + self.batch])).to(self.device)
            out, z = self.model(x, m)
            zs.append(z.float().cpu().numpy())
            if self.mode == "ae":
                err, feat = window_errors(out, x, m)
                errs.append(err.cpu().numpy())
                feats.append(feat.cpu().numpy())
            else:
                probs.append(torch.sigmoid(out).cpu().numpy())
        n = len(data)
        nan = np.full(n, np.nan, dtype=np.float32)
        return Scores(
            recon_error=np.concatenate(errs).astype(np.float32) if errs else nan,
            recon_error_feat=(
                np.concatenate(feats).astype(np.float32) if feats else np.full((n, 13), np.nan, dtype=np.float32)
            ),
            z=np.concatenate(zs).astype(np.float32),
            supervised_prob=np.concatenate(probs).astype(np.float32) if probs else nan,
        )


class Trainer:
    """Adam + ReduceLROnPlateau + early stopping on the validation loss; keeps the best weights."""

    def __init__(self, model: nn.Module, settings: EncoderTaskConfig, device: torch.device, log: RunLog) -> None:
        self.model, self.settings, self.device, self.log = model, settings, device, log
        self.pos_weight: torch.Tensor | None = None

    def loss(self, x: torch.Tensor, m: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        out, _ = self.model(x, m)
        if self.settings.mode == "ae":
            return masked_mse(out, x, m)
        return nn.functional.binary_cross_entropy_with_logits(out, y, pos_weight=self.pos_weight)

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor, m: torch.Tensor, y: torch.Tensor | None) -> float:
        """Validation loss, weighted the same way as training (sum over batches / weights)."""
        self.model.eval()
        total, weight = 0.0, 0.0
        bs = self.settings.eval_batch
        for start in range(0, len(x), bs):
            xb, mb = x[start : start + bs], m[start : start + bs]
            yb = None if y is None else y[start : start + bs]
            w = float(mb.sum()) if self.settings.mode == "ae" else float(len(xb))
            total += float(self.loss(xb, mb, yb)) * w
            weight += w
        return total / max(weight, 1.0)

    def fit(self, train: tuple, val: tuple, history_path: Path) -> dict:
        """train / val = (x, mask, y or None) tensors already on the device. Returns a summary."""
        s = self.settings
        opt = torch.optim.Adam(self.model.parameters(), lr=s.lr, weight_decay=s.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=s.plateau_factor, patience=s.plateau_patience)
        gen = torch.Generator().manual_seed(s.seed)
        x, m, y = train
        best, best_epoch, best_state, wait = float("inf"), 0, None, 0
        stop_reason = "max_epochs"
        epoch = 0
        for epoch in range(1, s.max_epochs + 1):
            t0 = time.time()
            self.model.train()
            perm = torch.randperm(len(x), generator=gen).to(self.device)
            running, steps = 0.0, 0
            for start in range(0, len(x), s.batch_size):
                idx = perm[start : start + s.batch_size]
                if len(idx) < 2:  # BatchNorm needs > 1 sample
                    continue
                loss = self.loss(x[idx], m[idx], None if y is None else y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                running += float(loss.detach())
                steps += 1
            train_loss = running / max(steps, 1)
            val_loss = self.evaluate(*val)
            sched.step(val_loss)
            improved = val_loss < best
            if improved:
                best, best_epoch, wait = val_loss, epoch, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                wait += 1
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": opt.param_groups[0]["lr"],
                "best_val_loss": best,
                "seconds": round(time.time() - t0, 2),
            }
            with history_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            self.log(
                f"epoch {epoch:2d} train {train_loss:.5f} val {val_loss:.5f} lr {row['lr']:.2e} "
                f"{'*' if improved else ' '} {row['seconds']:.1f}s"
            )
            if wait >= s.patience:
                stop_reason = f"early_stopping (patience {s.patience})"
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"epochs_run": epoch, "best_epoch": best_epoch, "best_val_loss": best, "stop_reason": stop_reason}


class Experiment:
    """One run: data, split, training, scoring, files."""

    def __init__(self, settings: EncoderTaskConfig) -> None:
        self.s = settings
        self.device = pick_device(settings.device)
        self.run_dir = Path(settings.runs_dir) / settings.run_id
        self.store = WindowStore(settings.data_dir, settings.cache_dir)

    @staticmethod
    def tensors(data: WindowSet, y: np.ndarray | None, device: torch.device) -> tuple:
        xt = torch.from_numpy(np.ascontiguousarray(data.x)).to(device)
        mt = torch.from_numpy(np.ascontiguousarray(data.mask)).to(device)
        yt = None if y is None else torch.from_numpy(y.astype(np.float32)).to(device)
        return xt, mt, yt

    def run(self) -> dict:
        s = self.s
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("history.jsonl", "train.log"):
            (self.run_dir / name).unlink(missing_ok=True)
        log = RunLog(self.run_dir / "train.log", Path(s.runs_dir) / "progress.log")
        t_start = time.time()
        torch.manual_seed(s.seed)
        np.random.seed(s.seed)
        (self.run_dir / "config.json").write_text(json.dumps(s.to_dict(), indent=2))
        (self.run_dir / "env.json").write_text(json.dumps(environment(self.device), indent=2))
        log(f"[{s.run_id}] start: mode {s.mode}, seed {s.seed}, device {self.device}, run dir {self.run_dir}")

        train = self.store.open("train", s.max_windows)
        train_labels = train.labels(
            "train: select benign windows (one-class)" if s.mode == "ae" else "train: supervised targets"
        )
        if s.mode == "ae":
            pool = train.take(np.flatnonzero(train_labels == 0))
            pool_y = np.zeros(len(pool), dtype=np.int8)
        else:
            pool = train.take(np.arange(len(train)))
            pool_y = np.asarray(train_labels, dtype=np.int8)
        fit_rows, val_rows = VehicleSplit(s.val_frac, s.split_seed).split(pool.vehicles)
        fit, val = pool.take(fit_rows), pool.take(val_rows)
        fit_y, val_y = pool_y[fit_rows], pool_y[val_rows]
        assert not set(fit.vehicles) & set(val.vehicles), "fit / val share a vehicle"
        log(
            f"[{s.run_id}] fit {len(fit)} windows ({len(set(fit.vehicles))} vehicles, {int(fit_y.sum())} GridSybil), "
            f"val {len(val)} windows ({len(set(val.vehicles))} vehicles, {int(val_y.sum())} GridSybil)"
        )

        model = (ConvAutoencoder(s) if s.mode == "ae" else SupervisedHead(s)).to(self.device)
        n_params = count_params(model)
        log(f"[{s.run_id}] model {type(model).__name__}: {n_params} trainable parameters")
        trainer = Trainer(model, s, self.device, log)
        if s.mode == "supervised":
            n_pos = max(int(fit_y.sum()), 1)
            trainer.pos_weight = torch.tensor((len(fit_y) - n_pos) / n_pos, device=self.device)
            log(f"[{s.run_id}] BCE pos_weight (benign / GridSybil in fit) = {float(trainer.pos_weight):.4f}")
        use_y = s.mode == "supervised"
        summary = trainer.fit(
            self.tensors(fit, fit_y if use_y else None, self.device),
            self.tensors(val, val_y if use_y else None, self.device),
            self.run_dir / "history.jsonl",
        )
        torch.save(model.state_dict(), self.run_dir / "best.pt")

        scorer = Scorer(model, s.mode, self.device, s.eval_batch)
        fit_benign = fit.take(np.flatnonzero(fit_y == 0))
        maha = Mahalanobis(s.maha_ridge).fit(scorer(fit_benign).z)
        val_benign = val.take(np.flatnonzero(val_y == 0))
        val_scores = scorer(val_benign)
        np.savez(
            self.run_dir / "scores_val.npz",
            recon_error=val_scores.recon_error,
            mahalanobis=maha.score(val_scores.z).astype(np.float32),
        )

        test = self.store.open("test", s.max_windows)
        test_scores = scorer(test)
        test_maha = maha.score(test_scores.z).astype(np.float32)
        test_labels = test.labels("test: evaluation labels, read after every test score exists")
        np.savez(
            self.run_dir / "scores_test.npz",
            id=np.asarray(test.ids).astype(str),
            label=np.asarray(test_labels, dtype=np.int8),
            n_messages=np.asarray(test.n_messages, dtype=np.int32),
            recon_error=test_scores.recon_error,
            recon_error_feat=test_scores.recon_error_feat,
            mahalanobis=test_maha,
            supervised_prob=test_scores.supervised_prob,
            z=test_scores.z,
        )
        wall = time.time() - t_start
        summary.update(
            {
                "run_id": s.run_id,
                "mode": s.mode,
                "seed": s.seed,
                "model": type(model).__name__,
                "n_params": n_params,
                "n_fit": len(fit),
                "n_val": len(val),
                "n_val_benign": len(val_benign),
                "n_fit_benign_for_mahalanobis": len(fit_benign),
                "n_test": len(test),
                "fit_vehicles": len(set(fit.vehicles)),
                "val_vehicles": len(set(val.vehicles)),
                "label_reads": train.label_reads + test.label_reads,
                "wall_seconds": round(wall, 1),
                "device": str(self.device),
            }
        )
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        log(
            f"[{s.run_id}] done: epochs {summary['epochs_run']} (best {summary['best_epoch']}), best val loss "
            f"{summary['best_val_loss']:.5f}, {summary['stop_reason']}, wall {wall / 60:.1f} min"
        )
        return summary


def parse_args(argv: list[str] | None = None) -> EncoderTaskConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("ae", "supervised"), default="ae")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=EncoderTaskConfig.max_epochs, help="maximum epochs")
    p.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--max-windows", type=int, default=0, help="quick check: evenly spaced windows per split")
    p.add_argument("--data-dir", default=EncoderTaskConfig.data_dir)
    p.add_argument("--runs-dir", default=EncoderTaskConfig.runs_dir)
    p.add_argument("--cache-dir", default=EncoderTaskConfig.cache_dir)
    a = p.parse_args(argv)
    return EncoderTaskConfig(
        mode=a.mode,
        seed=a.seed,
        max_epochs=a.epochs,
        device=a.device,
        max_windows=a.max_windows,
        data_dir=a.data_dir,
        runs_dir=a.runs_dir,
        cache_dir=a.cache_dir,
    )


def main(argv: list[str] | None = None) -> None:
    Experiment(parse_args(argv)).run()


if __name__ == "__main__":
    main()
