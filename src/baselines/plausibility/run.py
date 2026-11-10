"""Runs the plausibility-check baseline on VeReMi-Extension subsets and writes results next to the paper's numbers.

Implements tracker 4.5 (reproduce the VeReMi-Extension paper's Table III). Reads the raw dataset (read only,
RULE 2), parses every receiver file once, scores every received BSM at every receiver under each configuration in
`config.VARIANTS`, and writes (default out dir `src/runs/baselines/plausibility/`):
`<subset>/results.json` for the headline variant (`catch`), `variants/<variant>/<subset>/results.json` for the
others (counts, scores, per-check fire rates, config, env, git commit, runtime), and `summary.json` (paper vs ours,
every variant side by side). Sent-message counts (ground-truth rows per sender label) are reported for comparison
with the paper's Table I.

Usage:
    .venv-train/bin/python -m src.baselines.plausibility.run --subset all --workers 10
    .venv-train/bin/python -m src.baselines.plausibility.run --subset GridSybil_0709 --max-files 50 --out-dir /tmp/x
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import multiprocessing
import platform
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.baselines.plausibility.config import HEADLINE_VARIANT, PAPER_TABLE_III, SUBSETS, VARIANTS, PlausibilityConfig
from src.baselines.plausibility.detector import ThresholdDetector
from src.baselines.plausibility.metrics import DetectionCounts
from src.baselines.plausibility.parser import DatasetLayout, RunIndex, TraceReader

_LOG = logging.getLogger(__name__)
DEFAULT_DATA_ROOT = Path("data/VeReMi-Dataset")
DEFAULT_OUT_DIR = Path("src/runs/baselines/plausibility")
SPEC_PATH = "docs/research-notes/veremi-plausibility-baseline-spec.md"
_GT_SENDER_RE = re.compile(rb'"sender":\s*(\d+)')
LABEL_RULE = "BSM misbehaving iff its sending vehicle's own trace file has attack code != 0 in the run (spec §3)"


@dataclasses.dataclass(frozen=True)
class Job:
    """A chunk of receiver files of one run, or (files empty) the run's ground-truth file, for one worker call.

    Attributes:
        subset: `<Family>_<group>` folder name.
        run_dir: run folder.
        files: receiver trace files of the run (empty: count the ground-truth rows instead).
        attack_code: vehicle id -> attack code of every vehicle of the run.
    """

    subset: str
    run_dir: Path
    files: tuple[Path, ...]
    attack_code: dict[int, int]


@dataclasses.dataclass
class JobResult:
    """What a worker returns: per-variant counts, sent-message counts and CPU seconds."""

    subset: str
    counts: dict[str, DetectionCounts]
    sent: dict[str, int]
    cpu_seconds: float


class JobRunner:
    """Scores the receivers of a `Job` under every variant (runs inside a worker process)."""

    def __init__(self, variants: dict[str, PlausibilityConfig]) -> None:
        self._variants = variants

    def __call__(self, job: Job) -> JobResult:
        """Processes one job."""
        start = time.process_time()
        if not job.files:
            sent = self._count_sent(job)
            return JobResult(job.subset, {}, sent, time.process_time() - start)
        reader = TraceReader(RunIndex(job.attack_code))  # codes come with the job: no directory scan per worker
        detectors = {name: ThresholdDetector(config) for name, config in self._variants.items()}
        counts = {name: DetectionCounts() for name in self._variants}
        for path in job.files:
            stream = reader.read(path)
            for name, detector in detectors.items():
                decision = detector.detect(stream)
                counts[name].add_messages(
                    stream.label,
                    decision.flagged,
                    decision.fired,
                    decision.evaluated,
                    self._variants[name].decision_checks,
                )
                counts[name].receivers += 1
        return JobResult(job.subset, counts, {}, time.process_time() - start)

    @staticmethod
    def _count_sent(job: Job) -> dict[str, int]:
        """Ground-truth rows (BSMs sent) by sender label."""
        sent = {"genuine": 0, "malicious": 0, "unknown_label": 0}
        for path in sorted(job.run_dir.glob("traceGroundTruthJSON-*.json")):
            with path.open("rb") as handle:
                for line in handle:
                    match = _GT_SENDER_RE.search(line)
                    if match is None:
                        continue
                    code = job.attack_code.get(int(match.group(1)))
                    key = "unknown_label" if code is None else ("malicious" if code else "genuine")
                    sent[key] += 1
        return sent


class BaselineRun:
    """Plans jobs over the subsets, runs them on a worker pool and writes the results."""

    def __init__(
        self,
        variants: dict[str, PlausibilityConfig],
        layout: DatasetLayout,
        out_dir: Path,
        workers: int,
        chunk_size: int = 40,
        max_files: int | None = None,
        headline: str = HEADLINE_VARIANT,
    ) -> None:
        if headline not in variants:
            raise ValueError(f"headline variant {headline!r} not in {sorted(variants)}")
        self._variants = variants
        self._layout = layout
        self._out_dir = out_dir
        self._workers = workers
        self._chunk_size = chunk_size
        self._max_files = max_files
        self._headline = headline

    def plan(self, subsets: Sequence[str]) -> list[Job]:
        """One ground-truth job per run plus receiver-file chunks (at most `max_files` files per subset)."""
        jobs: list[Job] = []
        for subset in subsets:
            budget = self._max_files
            for run_dir in self._layout.runs(subset):
                if budget is not None and budget <= 0:
                    break
                index = RunIndex.scan(run_dir)
                files = index.files if budget is None else index.files[:budget]
                if budget is not None:
                    budget -= len(files)
                jobs.append(Job(subset, run_dir, (), index.attack_code))
                for start in range(0, len(files), self._chunk_size):
                    jobs.append(Job(subset, run_dir, tuple(files[start : start + self._chunk_size]), index.attack_code))
        return jobs

    def run(self, subsets: Sequence[str]) -> dict:
        """Runs all subsets; writes results.json files and summary.json; returns the summary."""
        wall_start = time.time()
        jobs = self.plan(subsets)
        _LOG.info(
            "%d jobs over %d subsets, %d variants, %d workers",
            len(jobs),
            len(subsets),
            len(self._variants),
            self._workers,
        )
        totals = {v: {s: DetectionCounts() for s in subsets} for v in self._variants}
        sent = {s: {"genuine": 0, "malicious": 0, "unknown_label": 0} for s in subsets}
        cpu = {s: 0.0 for s in subsets}
        runner = JobRunner(self._variants)

        def collect(result: JobResult) -> None:
            for variant, counts in result.counts.items():
                totals[variant][result.subset].merge(counts)
            for key, value in result.sent.items():
                sent[result.subset][key] += value
            cpu[result.subset] += result.cpu_seconds

        if self._workers > 1:
            with multiprocessing.get_context("spawn").Pool(self._workers) as pool:
                for done, result in enumerate(pool.imap_unordered(runner, jobs), start=1):
                    collect(result)
                    if done % 100 == 0 or done == len(jobs):
                        _LOG.info("%d / %d jobs (%.0f s)", done, len(jobs), time.time() - wall_start)
        else:
            for job in jobs:
                collect(runner(job))
        wall = time.time() - wall_start
        env = environment()
        rows: dict[str, dict] = {}
        for subset in subsets:
            rows[subset] = {"paper": PAPER_TABLE_III.get(subset), "sent_messages": sent[subset]}
            for variant, config in self._variants.items():
                result = self._subset_result(
                    subset, variant, config, totals[variant][subset], sent[subset], cpu, wall, env
                )
                folder = self._variant_dir(variant) / subset
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "results.json").write_text(json.dumps(result, indent=2))
                key = "ours" if variant == self._headline else f"ours_{variant}"
                rows[subset][key] = result["scores"]
                rows[subset][f"counts_{variant}"] = result["counts"]
            rows[subset]["received_messages"] = rows[subset][f"counts_{self._headline}"]
        summary = self._summary(rows, wall, env)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        (self._out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        _LOG.info("wrote %s (%.0f s wall)", self._out_dir / "summary.json", wall)
        return summary

    def _variant_dir(self, variant: str) -> Path:
        return self._out_dir if variant == self._headline else self._out_dir / "variants" / variant

    def _subset_result(
        self,
        subset: str,
        variant: str,
        config: PlausibilityConfig,
        counts: DetectionCounts,
        sent: dict[str, int],
        cpu: dict[str, float],
        wall: float,
        env: dict,
    ) -> dict:
        body = counts.to_dict()
        paper = PAPER_TABLE_III.get(subset)
        delta = {k: body["scores"][k] - paper[k] for k in paper} if paper else None
        return {
            "subset": subset,
            "variant": variant,
            "headline": variant == self._headline,
            "scores": body["scores"],
            "paper_table_iii": paper,
            "delta_ours_minus_paper": delta,
            "counts": body["counts"],
            "sizes": {
                "received_messages_labelled": body["messages"],
                "received_genuine": body["genuine"],
                "received_malicious": body["malicious"],
                "received_unknown_label": body["counts"]["unknown_label"],
                "receivers": body["receivers"],
                "sent_messages": sent,
            },
            "checks": body["checks"],
            "check_counts": body["check_counts"],
            "label_rule": LABEL_RULE,
            "decision_rule": f"ThresholdApp: flag iff any decision check value <= {config.threshold}",
            "missing_checks": "C10 position plausibility (needs the LuST map) and C0 proximity (post-paper) not computed",
            "config": config.to_dict(),
            "spec": SPEC_PATH,
            "max_files_per_subset": self._max_files,
            "runtime": {
                "cpu_seconds_this_subset_all_variants": round(cpu[subset], 1),
                "wall_seconds_whole_run": round(wall, 1),
                "workers": self._workers,
            },
            "env": env,
        }

    def _summary(self, rows: dict, wall: float, env: dict) -> dict:
        return {
            "description": "VeReMi-Extension Table III plausibility baseline: paper vs ours (per received BSM)",
            "headline_variant": self._headline,
            "variants": {name: config.to_dict() for name, config in self._variants.items()},
            "subsets": rows,
            "label_rule": LABEL_RULE,
            "spec": SPEC_PATH,
            "max_files_per_subset": self._max_files,
            "workers": self._workers,
            "wall_seconds": round(wall, 1),
            "env": env,
        }


def environment() -> dict:
    """Python / numpy versions, platform, git commit and dirty flag (RULE 4)."""

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subset", default="all", help="<Family>_<group> or all")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 2))
    parser.add_argument("--variants", default=",".join(VARIANTS), help=f"comma list of {sorted(VARIANTS)}")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-files", type=int, default=None, help="receiver files per subset (smoke runs)")
    parser.add_argument("--chunk-size", type=int, default=40)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Parses arguments, builds the run and executes it."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.out_dir.resolve().is_relative_to(Path("data").resolve()):
        raise ValueError(f"refusing to write under data/ (RULE 2): {args.out_dir}")
    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = sorted(set(names) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants {unknown}; known: {sorted(VARIANTS)}")
    variants = {name: VARIANTS[name] for name in names}
    headline = HEADLINE_VARIANT if HEADLINE_VARIANT in variants else names[0]
    subsets = list(SUBSETS) if args.subset == "all" else [args.subset]
    run = BaselineRun(
        variants, DatasetLayout(args.data_root), args.out_dir, args.workers, args.chunk_size, args.max_files, headline
    )
    summary = run.run(subsets)
    for subset, row in summary["subsets"].items():
        paper = row["paper"] or {}
        for key in [k for k in row if k.startswith("ours")]:
            ours = row[key]
            _LOG.info(
                "%-24s %-20s F1 %.4f acc %.4f prec %.4f rec %.4f | paper F1 %s",
                subset,
                key,
                ours["f1"],
                ours["accuracy"],
                ours["precision"],
                ours["recall"],
                paper.get("f1"),
            )


if __name__ == "__main__":
    main()
