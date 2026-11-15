"""Rebuilds the result tables from the score files of whatever runs exist.

    .venv-train/bin/python -m baselinepaper.experiments.aggregate [--jobs 6]

Writes `results/results.json` (every metric of every (regime, detector, score), paired differences) and
`results/main_table.md` (regime x detector x attack, with the length-only control and the length strata).
Nothing is read from a run folder except its config.json, summary.json status and score files.
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from baselinepaper.experiments import config as cfg
from baselinepaper.experiments.data import WindowCache
from baselinepaper.experiments.detectors.length import LENGTH_ONLY
from baselinepaper.experiments.evaluation import Evaluation, Evaluator, RunGroup, paired_difference

ATTACK_TITLES = {
    "test_gridsybil": "GridSybil",
    "test_datareplay": "DataReplay",
    "test_dosrandom": "DoSRandom",
    "test_dosdisruptive": "DoSDisruptive",
}
DIFFERENCE_METRICS = ("auroc", "length_matched_auroc")

_EVALUATOR: Evaluator | None = None


def _evaluate(job: tuple[cfg.ProtocolConfig, RunGroup]) -> Evaluation:
    """Worker: one evaluator per process (the bootstrap draws are the same in every process)."""
    global _EVALUATOR
    config, group = job
    if _EVALUATOR is None or _EVALUATOR.config != config:
        _EVALUATOR = Evaluator(config)
    return _EVALUATOR.evaluate(group)


class Aggregator:
    """Evaluates every finished run group and writes results.json + main_table.md."""

    def __init__(self, config: cfg.ProtocolConfig | None = None, jobs: int = 1) -> None:
        self.config = config or cfg.ProtocolConfig()
        self.jobs = jobs
        self.cache = WindowCache(self.config)
        self.results_dir = Path(self.config.results_dir)

    def evaluate_all(self, log=print) -> dict[str, Evaluation]:
        evaluator = Evaluator(self.config, self.cache)
        groups = evaluator.discover()
        log(f"{len(groups)} (regime, detector, score) groups with finished runs")
        if self.jobs > 1 and len(groups) > 1:
            with ProcessPoolExecutor(min(self.jobs, len(groups)), mp_context=get_context("spawn")) as pool:
                evaluations = list(pool.map(_evaluate, [(self.config, g) for g in groups]))
        else:
            evaluations = [evaluator.evaluate(g) for g in groups]
        return {g.key: e for g, e in zip(groups, evaluations)}

    def differences(self, evaluations: dict[str, Evaluation]) -> dict:
        """Paired differences on the shared bootstrap resamples: detector - length-only, regime - regime."""
        level = self.config.ci_level
        length = next((e for e in evaluations.values() if e.summary["detector"] == LENGTH_ONLY), None)
        versus_length, between_regimes = {}, {}
        for key, e in evaluations.items():
            if length is None or e is length:
                continue
            versus_length[key] = {
                attack: {m: paired_difference(e, length, f"{attack}/{m}", level) for m in DIFFERENCE_METRICS}
                for attack in cfg.ATTACK_TEST_SPLITS
            }
        by_detector: dict[tuple[str, str], dict[str, Evaluation]] = {}
        for e in evaluations.values():
            by_detector.setdefault((e.summary["detector"], e.summary["score"]), {})[e.summary["regime"]] = e
        for (detector, score), regimes in by_detector.items():
            present = [r for r in cfg.REGIMES if r in regimes]
            for first, second in itertools.combinations(present, 2):
                between_regimes[f"{detector}/{score}: {second} - {first}"] = {
                    "detector": detector,
                    "score": score,
                    "minuend_regime": second,
                    "subtrahend_regime": first,
                    "tasks": {
                        attack: {
                            m: paired_difference(regimes[second], regimes[first], f"{attack}/{m}", level)
                            for m in DIFFERENCE_METRICS
                        }
                        for attack in cfg.ATTACK_TEST_SPLITS
                    },
                }
        return {"detector_minus_length_only": versus_length, "regime_minus_regime": between_regimes}

    def run(self, log=print) -> dict:
        evaluations = self.evaluate_all(log)
        results = {
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "protocol": self.config.to_dict(),
            "cache": {
                "meta_sha256": self.cache.fingerprint(),
                "windows": self.cache.meta["windows"],
                "training_pools": {r: self.cache.regime_composition(r) for r in cfg.REGIMES},
                "validation_pools": {r: self.cache.regime_composition(r, "val") for r in cfg.REGIMES},
            },
            "entries": [e.summary for e in evaluations.values()],
            "paired_differences": self.differences(evaluations),
        }
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "results.json").write_text(json.dumps(results, indent=1))
        (self.results_dir / "main_table.md").write_text(TableWriter(results).render())
        log(f"written: {self.results_dir / 'results.json'}, {self.results_dir / 'main_table.md'}")
        return results


class TableWriter:
    """results.json -> main_table.md."""

    def __init__(self, results: dict) -> None:
        self.results = results
        self.protocol = results["protocol"]
        self.entries = results["entries"]
        self.strata = [(s["name"], s["label"]) for s in self.protocol["strata"]]

    @staticmethod
    def value(metric: dict | None, with_std: bool = True) -> str:
        if metric is None:
            return "-"
        text = f"{metric['mean']:.3f}"
        if with_std and metric.get("std") is not None:
            text += f" ± {metric['std']:.3f}"
        return text

    @staticmethod
    def ci(metric: dict | None) -> str:
        if metric is None or metric.get("ci") is None:
            return "-"
        return f"[{metric['ci'][0]:.3f}, {metric['ci'][1]:.3f}]"

    def subset(self, record: dict) -> str:
        """AUROC inside a stratum / sensitivity subset, or n/a with its counts."""
        if record["auroc"] is None:
            return f"n/a ({record['n_attack']} att. windows, {record['n_attack_vehicles']} vehicles)"
        return self.value(record["auroc"], with_std=False)

    @staticmethod
    def difference(d: dict | None) -> str:
        if d is None:
            return "-"
        text = f"{d['difference']:+.3f}"
        return text + (f" [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]" if d.get("ci") else "")

    @staticmethod
    def label(entry: dict) -> str:
        seeds = f"{len(entry['seeds'])} seed{'s' if len(entry['seeds']) != 1 else ''}"
        if not entry["is_stochastic"]:
            seeds = "deterministic"
        elif not entry["all_seeds_present"]:
            seeds += f", INCOMPLETE: expected {len(entry['expected_seeds'])}"
        return f"{entry['detector']} / {entry['score']} ({seeds})"

    def rows(self) -> list[tuple[str, dict]]:
        """(regime shown, entry): the length-only control first, then every regime's detectors."""
        controls = [e for e in self.entries if e["regime"] == cfg.CONTROL_REGIME]
        out = [("control", e) for e in controls]
        for regime in cfg.REGIMES:
            out += [(regime, e) for e in self.entries if e["regime"] == regime]
        return out

    def render(self) -> str:
        p, cache = self.protocol, self.results["cache"]
        lines = [
            "# Results under the one protocol (rebuilt from score files)",
            "",
            f"Generated {self.results['generated_utc']} by `python -m baselinepaper.experiments.aggregate`; do not edit.",
            "Every number comes from the score files of the finished runs; a missing detector means no finished run.",
            "",
            f"- Tasks: benign test ({cache['windows']['test_benign']:,} windows) vs each attack test split; unit = window.",
            "- `mean ± std` over seeds (sample std; deterministic detectors have one run). "
            f"{round(p['ci_level'] * 100)}% CI: stratified cluster bootstrap over test vehicle keys "
            f"({p['bootstrap_resamples']} resamples, benign and attack keys resampled separately) of the seed mean.",
            "- Length-matched AUROC: pair-weighted mean of the AUROCs within each exact number of real rows; "
            "0.5 for the length-only score by construction.",
            f"- Strata with fewer than {p['min_attack_windows']} attack windows or {p['min_attack_vehicles']} attack "
            "vehicles are n/a.",
            "- Length-only control: score = -(number of real rows), fixed, not trained; repeated for reference.",
            "",
            "Training pools: "
            + "; ".join(
                f"`{r}` {c['windows']:,} windows ({c['attack_share']:.1%} GridSybil)"
                for r, c in cache["training_pools"].items()
            ),
            "",
        ]
        versus_length = self.results["paired_differences"]["detector_minus_length_only"]
        for attack, title in ATTACK_TITLES.items():
            first = next((e["tasks"][attack] for e in self.entries), None)
            if first is None:
                continue
            lines += [
                f"## {title}",
                "",
                f"{first['n_attack']:,} attack windows ({first['n_attack_vehicles']} vehicles) vs {first['n_benign']:,} "
                f"benign ({first['n_benign_vehicles']} vehicles); attack share {first['attack_share']:.3f}.",
                "",
                "### AUROC and window length",
                "",
                "| Regime | Detector / score | AUROC | 95% CI | AUROC − length-only [CI] | Length-matched AUROC [CI] | "
                + " | ".join(label for _, label in self.strata)
                + " |",
                "|---|---|---|---|---|---|" + "---|" * len(self.strata),
            ]
            for regime, e in self.rows():
                t = e["tasks"][attack]
                key = f"{e['regime']}/{e['detector']}/{e['score']}"
                delta = versus_length.get(key, {}).get(attack, {}).get("auroc")
                matched = f"{self.value(t['length_matched_auroc'], False)} {self.ci(t['length_matched_auroc'])}"
                cells = [regime, self.label(e), self.value(t["auroc"]), self.ci(t["auroc"]), self.difference(delta)]
                cells += [matched, *(self.subset(t["strata"][name]) for name, _ in self.strata)]
                lines.append("| " + " | ".join(cells) + " |")
            ops = list(self.entries[0]["thresholds_per_seed"])
            lines += [
                "",
                "### Average precision and operating points (thresholds from validation scores; alarm when score > threshold)",
                "",
                "| Regime | Detector / score | AP | 95% CI | Primary threshold | "
                + " | ".join(f"recall / benign FPR @ {op}" for op in ops)
                + " |",
                "|---|---|---|---|---|" + "---|" * len(ops),
            ]
            for regime, e in self.rows():
                t = e["tasks"][attack]
                cells = [regime, self.label(e), self.value(t["average_precision"]), self.ci(t["average_precision"])]
                cells.append(e["primary_operating_point"])
                cells += [
                    f"{self.value(t['recall'][op], False)} / {self.value(e['benign_test_fpr'][op], False)}"
                    for op in ops
                ]
                lines.append("| " + " | ".join(cells) + " |")
            names = list(first["sensitivity"])
            lines += [
                "",
                "### Sensitivity (AUROC on subsets of the same score files)",
                "",
                "| Regime | Detector / score | " + " | ".join(f"{n} [CI]" for n in names) + " |",
                "|---|---|" + "---|" * len(names),
            ]
            for regime, e in self.rows():
                s = e["tasks"][attack]["sensitivity"]
                cells = [regime, self.label(e)]
                cells += [
                    f"{self.subset(s[n])} {self.ci(s[n]['auroc']) if s[n]['auroc'] else ''}".strip() for n in names
                ]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        between = self.results["paired_differences"]["regime_minus_regime"]
        if between:
            lines += [
                "## Regime − regime, same detector (paired bootstrap, AUROC difference [CI])",
                "",
                "| Detector / score | Difference | " + " | ".join(ATTACK_TITLES.values()) + " |",
                "|---|---|---|---|---|---|",
            ]
            for d in between.values():
                cells = [f"{d['detector']} / {d['score']}", f"{d['minuend_regime']} − {d['subtrahend_regime']}"]
                cells += [self.difference(d["tasks"][a]["auroc"]) for a in ATTACK_TITLES]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=4, help="processes (one run group each)")
    args = parser.parse_args(argv)
    Aggregator(jobs=args.jobs).run()


if __name__ == "__main__":
    main()
