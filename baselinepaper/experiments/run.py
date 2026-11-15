"""Runs one detector in one regime with one seed and writes its scores.

    .venv-train/bin/python -m baselinepaper.experiments.run --detector length_only --regime none --seed 0

Run folder `runs/<regime>/<detector>/seed<k>/`: config.json, env.json, history.json,
`scores_<score name>_<split>.npy` for the two validation splits and the five test splits (rows in the cache order
of that split), summary.json (written last; its presence marks the run as finished).

The runner never reads a label: the detector gets the regime's training / validation set, then x and mask of each
scored split, one split at a time. Metrics are computed later by `evaluation` from the score files.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from baselinepaper.experiments import config as cfg
from baselinepaper.experiments.data import WindowCache
from baselinepaper.experiments.detectors import Detector, available, get_detector

PACKAGES = ("numpy", "scipy", "scikit-learn", "torch")
SUMMARY_QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def resolve_device(device: str) -> str:
    """ "auto" -> mps > cuda > cpu (cpu if torch is missing); anything else is returned as given."""
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def score_file(run_dir: Path, score_name: str, split: str) -> Path:
    """`<run_dir>/scores_<score name>_<split>.npy`."""
    return Path(run_dir) / f"scores_{score_name}_{split}.npy"


class Environment:
    """What a run was produced with: packages, device, git state and the sha256 of every source file."""

    def __init__(self, detector_cls: type[Detector], device: str, cache: WindowCache) -> None:
        self.detector_cls = detector_cls
        self.device = device
        self.cache = cache

    @staticmethod
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    @staticmethod
    def source_hashes(module_name: str) -> dict[str, str]:
        """{path: sha256} of every .py file of a package (or of the single module file)."""
        module = importlib.import_module(module_name)
        origin = Path(module.__file__)
        files = sorted(origin.parent.rglob("*.py")) if origin.name == "__init__.py" else [origin]
        out = {}
        for f in files:
            try:
                key = str(f.resolve().relative_to(Path.cwd().resolve()))
            except ValueError:
                key = str(f)
            out[key] = hashlib.sha256(f.read_bytes()).hexdigest()
        return out

    def record(self) -> dict:
        versions = {}
        for package in PACKAGES:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        status = self._git("status", "--porcelain")
        sources = {
            "baselinepaper/__init__.py": hashlib.sha256(
                Path(importlib.import_module("baselinepaper").__file__).read_bytes()
            ).hexdigest()
        }
        sources.update(self.source_hashes("baselinepaper.experiments"))
        reused = {m: self.source_hashes(m) for m in self.detector_cls.reused_modules}
        return {
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": versions,
            "device": self.device,
            "git_commit": self._git("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status),
            "cache_meta_sha256": self.cache.fingerprint(),
            "cache_built_utc": self.cache.meta["built_utc"],
            "source_sha256": sources,
            "reused_module_sha256": reused,
            "command": " ".join(sys.argv),
        }


class Runner:
    """Fits one detector and writes one run folder.

    Attributes:
        config: protocol settings.
        cache: the window cache.
        detector_cls, regime, seed, device: what is run.
        run_dir: `runs/<regime>/<detector>/seed<k>`.
    """

    def __init__(
        self, detector: str, regime: str, seed: int, device: str = "auto", config: cfg.ProtocolConfig | None = None
    ) -> None:
        self.config = config or cfg.ProtocolConfig()
        self.cache = WindowCache(self.config)
        self.detector_cls = get_detector(detector)
        if regime not in self.detector_cls.regimes:
            raise ValueError(f"detector {detector!r} supports regimes {self.detector_cls.regimes}, not {regime!r}")
        if seed not in self.config.seeds:
            raise ValueError(f"seed {seed} is not one of the protocol's seeds {self.config.seeds}")
        if not self.detector_cls.is_stochastic and seed != self.config.seeds[0]:
            raise ValueError(f"{detector!r} is deterministic: it is run once, with seed {self.config.seeds[0]}")
        self.regime, self.seed = regime, seed
        self.device = resolve_device(device)
        self.run_dir = self.config.run_dir(regime, detector, seed)

    def finished(self) -> bool:
        """True if summary.json and every score file exist."""
        return (self.run_dir / "summary.json").exists() and all(
            score_file(self.run_dir, s, split).exists()
            for s in self.detector_cls.score_names
            for split in cfg.SCORED_SPLITS
        )

    def run(self, force: bool = False, log=print) -> dict:
        """Fits, scores, writes the run folder; returns summary.json's content."""
        if self.finished() and not force:
            log(f"finished run found, skipped: {self.run_dir} (use --force to redo)")
            return json.loads((self.run_dir / "summary.json").read_text())
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for stale in [self.run_dir / "summary.json", *self.run_dir.glob("scores_*.npy")]:
            stale.unlink(missing_ok=True)

        restandardise = self.regime in cfg.RESTANDARDISED_REGIMES
        statistics = self.cache.regime_statistics(self.regime) if restandardise else None
        detector = self.detector_cls(regime=self.regime, device=self.device, statistics=statistics)
        run_config = {
            "detector": detector.name,
            "regime": self.regime,
            "seed": self.seed,
            "is_stochastic": detector.is_stochastic,
            "score_names": list(detector.score_names),
            "device_requested": self.device,
            "hyperparameters": detector.hyperparameters(),
            "restandardised_with_regime_statistics": statistics.to_dict() if statistics is not None else None,
            "training_pool": self.cache.regime_composition(self.regime) if self.regime in cfg.REGIMES else None,
            "validation_pool": (
                self.cache.regime_composition(self.regime, "val") if self.regime in cfg.REGIMES else None
            ),
            "protocol": self.config.to_dict(),
        }
        (self.run_dir / "config.json").write_text(json.dumps(run_config, indent=1))
        (self.run_dir / "env.json").write_text(
            json.dumps(Environment(self.detector_cls, self.device, self.cache).record(), indent=1)
        )

        train = val = None
        if self.regime in cfg.REGIMES:
            train, val = self.cache.training_set(self.regime), self.cache.validation_set(self.regime)
        started = time.time()
        detector.fit(train, val, self.seed)
        fit_seconds = time.time() - started
        del train, val
        (self.run_dir / "history.json").write_text(json.dumps(detector.history(), indent=1))
        detector.save(self.run_dir)

        started = time.time()
        score_stats: dict[str, dict] = {s: {} for s in detector.score_names}
        for split in cfg.SCORED_SPLITS:
            windows = self.cache.features(split)
            scores = detector.score(windows.x, windows.mask)
            if set(scores) != set(detector.score_names):
                raise RuntimeError(f"{detector.name}: scores {sorted(scores)}, declared {detector.score_names}")
            for name, values in scores.items():
                values = np.asarray(values, dtype=np.float64)
                if values.shape != (len(windows),) or not np.isfinite(values).all():
                    raise RuntimeError(f"{detector.name}/{name}/{split}: shape {values.shape} or non-finite scores")
                np.save(score_file(self.run_dir, name, split), values)
                quantiles = np.quantile(values, SUMMARY_QUANTILES)
                score_stats[name][split] = {
                    "windows": len(values),
                    "mean": float(values.mean()),
                    "quantiles": {f"q{q:g}": float(v) for q, v in zip(SUMMARY_QUANTILES, quantiles)},
                }
            log(f"scored {split}: {len(windows):,} windows")
        summary = {
            "status": "complete",
            "detector": detector.name,
            "regime": self.regime,
            "seed": self.seed,
            "device": self.device,
            "fit_seconds": round(fit_seconds, 2),
            "score_seconds": round(time.time() - started, 2),
            "fit_info": detector.fit_info(),
            "scored_splits": list(cfg.SCORED_SPLITS),
            "scores": score_stats,
            "note": "no metric here: labels are read only by baselinepaper.experiments.evaluation",
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        log(f"run written: {self.run_dir} (fit {fit_seconds:.1f} s)")
        return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--detector", required=True, help=f"one of {sorted(available())}")
    parser.add_argument("--regime", required=True, choices=(*cfg.REGIMES, cfg.CONTROL_REGIME))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto", help="auto (mps > cuda > cpu), cpu, mps or cuda")
    parser.add_argument("--force", action="store_true", help="redo a finished run")
    args = parser.parse_args(argv)
    Runner(args.detector, args.regime, args.seed, args.device).run(force=args.force)


if __name__ == "__main__":
    main()
