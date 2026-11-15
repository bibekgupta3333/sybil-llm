"""Runs every classical detector in each of its regimes with each of its seeds, through `run.py`, on the CPU.

    .venv-train/bin/python -m baselinepaper.experiments.sweep_classical [--workers 4] [--detectors pca knn] [--dry-run]

One subprocess per run (`python -m baselinepaper.experiments.run ... --device cpu`), each limited to one numeric
thread, at most `--workers` at a time (default 4). Finished runs are skipped. Progress, wall-clock per run and per
detector, and the failed runs are appended to `runs/sweep_classical.log`. No metric is computed: labels are read
only by `evaluation`.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from baselinepaper.experiments import config as cfg
from baselinepaper.experiments.detectors import get_detector
from baselinepaper.experiments.run import Runner

CLASSICAL_DETECTORS = ("pca", "statslr", "iforest", "knn", "ocsvm")
LOG_NAME = "sweep_classical.log"
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
ERROR_TAIL_LINES = 15


@dataclasses.dataclass
class Job:
    """One run of the sweep.

    Attributes:
        detector, regime, seed: what is run.
        status: "pending", "skipped" (finished before the sweep), "done" or "failed".
        seconds: wall-clock of the subprocess (0 if skipped).
        error: the last lines of the subprocess output of a failed run.
    """

    detector: str
    regime: str
    seed: int
    status: str = "pending"
    seconds: float = 0.0
    error: str = ""

    @property
    def label(self) -> str:
        return f"{self.detector} / {self.regime} / seed {self.seed}"


class ClassicalSweep:
    """Runs the classical detectors of the protocol and logs what happened.

    Attributes:
        config: protocol settings (run folder, seeds).
        detectors: detector names, in run order.
        workers: subprocesses running at the same time.
        threads_per_worker: numeric threads each subprocess may use.
        log_path: `<runs_dir>/sweep_classical.log` (appended).
    """

    def __init__(
        self,
        detectors: tuple[str, ...] = CLASSICAL_DETECTORS,
        workers: int = 4,
        threads_per_worker: int = 1,
        config: cfg.ProtocolConfig | None = None,
    ) -> None:
        self.config = config or cfg.ProtocolConfig()
        self.detectors = tuple(detectors)
        self.workers = workers
        self.threads_per_worker = threads_per_worker
        self.log_path = Path(self.config.runs_dir) / LOG_NAME
        self._lock = threading.Lock()

    def jobs(self) -> list[Job]:
        """Every (detector, regime, seed) of the sweep: all seeds if stochastic, the first seed otherwise."""
        out = []
        for name in self.detectors:
            cls = get_detector(name)
            seeds = self.config.seeds if cls.is_stochastic else self.config.seeds[:1]
            out.extend(Job(name, regime, seed) for regime in cls.regimes for seed in seeds)
        return out

    def log(self, message: str) -> None:
        line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {message}"
        with self._lock:
            print(line, flush=True)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(line + "\n")

    def _environment(self) -> dict[str, str]:
        env = dict(os.environ)
        for variable in THREAD_VARIABLES:
            env[variable] = str(self.threads_per_worker)
        return env

    def _run_job(self, job: Job) -> Job:
        command = [
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
            "cpu",
        ]
        self.log(f"start   {job.label}")
        started = time.time()
        result = subprocess.run(command, capture_output=True, text=True, env=self._environment())
        job.seconds = time.time() - started
        if result.returncode == 0:
            job.status = "done"
            self.log(f"done    {job.label} ({job.seconds:.1f} s)")
        else:
            job.status = "failed"
            job.error = "\n".join((result.stderr or result.stdout).strip().splitlines()[-ERROR_TAIL_LINES:])
            self.log(f"FAILED  {job.label} ({job.seconds:.1f} s, exit {result.returncode})\n{job.error}")
        return job

    def run(self, dry_run: bool = False) -> list[Job]:
        """Runs every unfinished job; returns all jobs with their status and wall-clock."""
        jobs = self.jobs()
        for job in jobs:
            if Runner(job.detector, job.regime, job.seed, device="cpu", config=self.config).finished():
                job.status = "skipped"
        todo = [j for j in jobs if j.status == "pending"]
        self.log(
            f"sweep: {len(jobs)} runs of {list(self.detectors)}, {len(jobs) - len(todo)} finished before, "
            f"{len(todo)} to run, {self.workers} workers x {self.threads_per_worker} thread(s)"
        )
        if dry_run:
            for job in todo:
                self.log(f"would run {job.label}")
            return jobs
        started = time.time()
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(self._run_job, todo))
        self.report(jobs, time.time() - started)
        return jobs

    def report(self, jobs: list[Job], elapsed: float) -> None:
        """Logs the wall-clock per detector and the failed runs."""
        for name in self.detectors:
            mine = [j for j in jobs if j.detector == name]
            ran = [j for j in mine if j.status in ("done", "failed")]
            self.log(
                f"detector {name}: {sum(j.status == 'done' for j in mine)} done, "
                f"{sum(j.status == 'skipped' for j in mine)} skipped, {sum(j.status == 'failed' for j in mine)} failed; "
                f"wall-clock summed over its runs {sum(j.seconds for j in ran):.0f} s"
                + (f" (mean {sum(j.seconds for j in ran) / len(ran):.0f} s per run)" if ran else "")
            )
        failed = [j for j in jobs if j.status == "failed"]
        self.log(f"sweep finished in {elapsed:.0f} s: {len(failed)} failed of {len(jobs)} runs")
        for job in failed:
            self.log(f"failed: {job.label}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=4, help="subprocesses at a time (default 4)")
    parser.add_argument("--threads", type=int, default=1, help="numeric threads per subprocess (default 1)")
    parser.add_argument("--detectors", nargs="+", default=list(CLASSICAL_DETECTORS), choices=CLASSICAL_DETECTORS)
    parser.add_argument("--dry-run", action="store_true", help="list the runs that would start")
    args = parser.parse_args(argv)
    jobs = ClassicalSweep(tuple(args.detectors), args.workers, args.threads).run(dry_run=args.dry_run)
    if any(j.status == "failed" for j in jobs):
        sys.exit(1)


if __name__ == "__main__":
    main()
