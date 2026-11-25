"""Runs the grid T = 64 / 48 / 32 (or any of 64 / 48 / 32 / 20 / 10 / 5) x seeds 0-3 of one encoder, skipping every run that is already finished.

A run is finished when its folder (`config.run_path`) holds `summary.json` and `test_metrics.json`; it is skipped, so
a rerun of the grid is safe. A run with `summary.json` but no `test_metrics.json` (training done, evaluation missing)
is only evaluated. Every other run is trained and evaluated (`train.py --evaluate`) in its own process. All output is
shown and appended to `runs/<encoder>/logs/mae_all_<encoder>.log` (`config.logs_path`). The grid stops at the first
failing run (exit code 1).

Usage (repo root):
    .venv-train/bin/python -m task.masked_autoencoder.run_all --encoder mlp
    .venv-train/bin/python -m task.masked_autoencoder.run_all --encoder resnet
    .venv-train/bin/python -m task.masked_autoencoder.run_all --encoder resnet --windows 32 --seeds 0 1 --dry-run
    .venv-train/bin/python -m task.masked_autoencoder.run_all --encoder mlp --windows 20,10,5
    WINDOWS=20,10,5 ENCODER=resnet npm run mae:all

`--windows` takes numbers separated by spaces and / or commas; the default is the main grid 64 / 48 / 32 (`WINDOWS`).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from task.masked_autoencoder.config import (
    ENCODERS,
    PROTOCOLS,
    RUNS_DIR,
    SEEDS,
    WINDOWS,
    is_finished,
    logs_path,
    run_path,
)


class GridRunner:
    """Trains / evaluates the missing runs of one encoder's grid and logs everything to one file.

    Attributes:
        encoder: "mlp" or "resnet".
        windows, seeds: the grid.
        runs_dir: root of the run tree.
        device: passed to train.py / evaluate.py.
        log_path: the grid log.
    """

    def __init__(
        self,
        encoder: str,
        windows: tuple[int, ...] = WINDOWS,
        seeds: tuple[int, ...] = SEEDS,
        runs_dir: str = RUNS_DIR,
        device: str = "auto",
    ) -> None:
        self.encoder = encoder
        self.windows = windows
        self.seeds = seeds
        self.runs_dir = runs_dir
        self.device = device
        self.log_path = logs_path(encoder, runs_dir) / f"mae_all_{encoder}.log"

    def action(self, window: int, seed: int) -> str:
        """ "skip" (finished), "evaluate" (trained, not evaluated) or "train" (train + evaluate)."""
        run_dir = run_path(self.encoder, window, seed, self.runs_dir)
        if is_finished(run_dir):
            return "skip"
        if (run_dir / "summary.json").is_file():
            return "evaluate"
        return "train"

    def command(self, window: int, seed: int, action: str) -> list[str]:
        common = ["--window", str(window), "--encoder", self.encoder, "--seed", str(seed), "--runs-dir", self.runs_dir]
        if action == "evaluate":
            return [sys.executable, "-m", "task.masked_autoencoder.evaluate", *common, "--device", self.device]
        return [sys.executable, "-m", "task.masked_autoencoder.train", *common, "--device", self.device, "--evaluate"]

    def plan(self) -> list[tuple[int, int, str]]:
        return [(w, s, self.action(w, s)) for w in self.windows for s in self.seeds]

    def run(self, dry_run: bool = False) -> int:
        """Runs the plan; returns 0, or 1 at the first failing run. A dry run writes no log."""
        if not dry_run:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(os.devnull if dry_run else self.log_path, "a") as log:

            def say(line: str) -> None:
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()

            plan = self.plan()
            todo = sum(a != "skip" for _, _, a in plan)
            say(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} grid {self.encoder}: {len(plan)} runs, {todo} to do")
            for window, seed, action in plan:
                run_dir = run_path(self.encoder, window, seed, self.runs_dir)
                if action == "skip":
                    say(f"skip {run_dir} (summary.json + test_metrics.json exist)")
                    continue
                cmd = self.command(window, seed, action)
                say(f"{action} {run_dir}: {' '.join(cmd[1:])}")
                if dry_run:
                    continue
                code = self.stream(cmd, say)
                if code != 0:
                    say(f"FAILED {run_dir} (exit code {code}); grid stopped")
                    return 1
            say(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} grid {self.encoder}: done")
        return 0

    @staticmethod
    def stream(cmd: list[str], say) -> int:
        """Runs `cmd`, passing every output line to `say`; returns the exit code."""
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            say(line.rstrip("\n"))
        return proc.wait()


def parse_windows(values: list[str]) -> tuple[int, ...]:
    """`["20,10", "5"]` -> (20, 10, 5); every value must be a known window (`PROTOCOLS`)."""
    out = []
    for v in values:
        for part in v.replace(",", " ").split():
            w = int(part)
            if w not in PROTOCOLS:
                raise ValueError(f"window must be one of {sorted(PROTOCOLS)}, got {w}")
            out.append(w)
    if not out:
        raise ValueError("no window given")
    return tuple(out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--encoder", choices=ENCODERS, default="mlp")
    p.add_argument(
        "--windows",
        nargs="+",
        default=[",".join(map(str, WINDOWS))],
        help=f"windows, space- or comma-separated, from {sorted(PROTOCOLS)} (default {','.join(map(str, WINDOWS))})",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--runs-dir", default=RUNS_DIR)
    p.add_argument("--device", default="auto")
    p.add_argument("--dry-run", action="store_true", help="print the plan (skip / evaluate / train) only, write no log")
    a = p.parse_args(argv)
    try:
        windows = parse_windows(a.windows)
    except ValueError as e:
        p.error(str(e))
    runner = GridRunner(a.encoder, windows, tuple(a.seeds), str(Path(a.runs_dir)), a.device)
    sys.exit(runner.run(a.dry_run))


if __name__ == "__main__":
    main()
