"""Runs every neural detector x regime x seed, one after the other, through the runner.

    .venv-train/bin/python -m baselinepaper.experiments.sweep_neural                # all 45 runs
    .venv-train/bin/python -m baselinepaper.experiments.sweep_neural --dry-run      # list them
    .venv-train/bin/python -m baselinepaper.experiments.sweep_neural --detectors lstmae --seeds 0 1

Each run is its own process (`python -m baselinepaper.experiments.run ...`), so memory is released between runs and
a crash stops one run only. Finished runs (summary.json + every score file) are skipped; a failed run is logged and
the sweep goes on; failures are listed at the end (exit code 1 if there are any). Everything is appended to
`runs/sweep_neural.log`. The sweep computes no metric and reads no label.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import subprocess
import sys
import time
from pathlib import Path

from baselinepaper.experiments import config as cfg
from baselinepaper.experiments.detectors import get_detector
from baselinepaper.experiments.run import Runner

NEURAL_DETECTORS = ("supae", "convae", "lstmae")
LOG_NAME = "sweep_neural.log"


@dataclasses.dataclass(frozen=True)
class Job:
    """One run of the sweep."""

    detector: str
    regime: str
    seed: int

    def __str__(self) -> str:
        return f"{self.detector} / {self.regime} / seed {self.seed}"


class NeuralSweep:
    """Runs the jobs sequentially and keeps a log.

    Attributes:
        config: protocol settings (seeds, run folders).
        jobs: the runs, ordered seed by seed so that every detector has complete seeds early.
        device: passed to the runner ("auto" = mps > cuda > cpu).
        log_path: `runs/sweep_neural.log`.
    """

    def __init__(
        self,
        detectors: tuple[str, ...] = NEURAL_DETECTORS,
        regimes: tuple[str, ...] | None = None,
        seeds: tuple[int, ...] | None = None,
        device: str = "auto",
        config: cfg.ProtocolConfig | None = None,
    ) -> None:
        self.config = config or cfg.ProtocolConfig()
        self.device = device
        self.log_path = Path(self.config.runs_dir) / LOG_NAME
        self.jobs = [
            Job(detector, regime, seed)
            for seed in (seeds if seeds is not None else self.config.seeds)
            for detector in detectors
            for regime in get_detector(detector).regimes
            if regimes is None or regime in regimes
        ]

    def log(self, message: str) -> None:
        line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(line + "\n")

    def finished(self, job: Job) -> bool:
        """True if the run folder holds summary.json and every score file."""
        return Runner(job.detector, job.regime, job.seed, self.device, self.config).finished()

    def command(self, job: Job) -> list[str]:
        return [
            sys.executable,
            "-m",
            "baselinepaper.experiments.run",
            "--detector",
            job.detector,
            "--regime",
            job.regime,
            "--seed",
            str(job.seed),
            "--device",
            self.device,
        ]

    def run_job(self, job: Job) -> int:
        """Runs one job as a child process; its output goes to the log. Returns the exit code."""
        with self.log_path.open("a") as f:
            f.flush()
            return subprocess.run(self.command(job), stdout=f, stderr=subprocess.STDOUT, check=False).returncode

    def run(self, dry_run: bool = False) -> list[Job]:
        """Runs every unfinished job; returns the failed ones."""
        pending = [job for job in self.jobs if not self.finished(job)]
        self.log(
            f"sweep: {len(self.jobs)} runs, {len(self.jobs) - len(pending)} already finished, device {self.device}"
        )
        if dry_run:
            for job in self.jobs:
                print(("pending   " if job in pending else "finished  ") + str(job))
            return []
        failed: list[Job] = []
        started = time.time()
        for position, job in enumerate(pending, start=1):
            self.log(f"[{position}/{len(pending)}] start {job}")
            t0 = time.time()
            try:
                code = self.run_job(job)
            except KeyboardInterrupt:
                self.log(f"interrupted during {job}; finished runs are kept, restart to continue")
                raise
            minutes = (time.time() - t0) / 60
            if code == 0 and self.finished(job):
                self.log(f"[{position}/{len(pending)}] done {job} in {minutes:.1f} min")
            else:
                failed.append(job)
                self.log(f"[{position}/{len(pending)}] FAILED {job} (exit code {code}) after {minutes:.1f} min")
        self.log(
            f"sweep finished in {(time.time() - started) / 3600:.2f} h: "
            f"{len(pending) - len(failed)} run, {len(failed)} failed"
        )
        for job in failed:
            self.log(f"failed: {job}  ->  {' '.join(self.command(job))}")
        return failed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--detectors", nargs="+", default=list(NEURAL_DETECTORS), choices=NEURAL_DETECTORS)
    parser.add_argument("--regimes", nargs="+", default=None, choices=cfg.REGIMES)
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="default: the protocol's seeds 0-4")
    parser.add_argument("--device", default="auto", help="auto (mps > cuda > cpu), cpu, mps or cuda")
    parser.add_argument("--dry-run", action="store_true", help="list the runs and whether each is finished")
    args = parser.parse_args(argv)
    sweep = NeuralSweep(
        tuple(args.detectors),
        None if args.regimes is None else tuple(args.regimes),
        None if args.seeds is None else tuple(args.seeds),
        args.device,
    )
    sys.exit(1 if sweep.run(dry_run=args.dry_run) else 0)


if __name__ == "__main__":
    main()
