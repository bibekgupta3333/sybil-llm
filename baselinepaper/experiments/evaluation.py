"""Metrics of the protocol, computed from score files only (protocol section 6).

This is the only module that turns splits into labels: a score from `test_benign` is a benign score, a score from
an attack test split is an attack score. Four binary tasks, benign test vs each attack test split; unit = window.

Per task and seed: AUROC, average precision (attack share stated), the length-matched AUROC, AUROC inside each
window-length stratum and for three sensitivity subsets, and attack recall / benign false-positive rate at
thresholds taken from validation scores only (alarm when score > threshold):

* `benign_val_p95`, `benign_val_p99`: percentiles of the scores on benign val;
* `own_val_p95`: 95th percentile of the scores on the regime's own validation pool (label-free).

The primary operating point is `own_val_p95` for the mix regimes (there `benign_val_*` is an oracle calibration)
and `benign_val_p95` otherwise.

Length-matched AUROC: the mean of the within-n AUROCs (n = number of real rows, 1..32) weighted by the number of
benign x attack pairs at that n; equivalently the share of same-length (attack, benign) pairs the score orders
correctly (ties count one half). The length-only score is 0.5 by construction.

Uncertainty: per-seed values, mean and sample standard deviation over seeds, and a 95% percentile interval from a
stratified cluster bootstrap: benign test vehicle keys and the attack split's vehicle keys are resampled separately
with replacement; in every resample the metric is computed per seed and averaged over seeds. The resample counts
depend only on `bootstrap_seed`, so every detector sees the same resamples and differences are paired.

Speed: an AUROC is a sum over (attack vehicle, benign vehicle) pairs of pairwise wins, so one matrix of wins per
vehicle pair gives every resample by two matrix products; nothing is re-sorted per resample.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from baselinepaper.experiments import config as cfg
from baselinepaper.experiments.data import WindowCache
from baselinepaper.experiments.run import score_file

BENIGN_VAL = "benign_val_p{:g}"
OWN_VAL = "own_val_p{:g}"
SENSITIVITY_GROUP = "group_{}_only"
SENSITIVITY_NO_FOLDER = "benign_without_{}"
SENSITIVITY_OWN_FAMILY = "benign_of_attack_family_folders"


# ---------------------------------------------------------------------------------------------- point metrics


def auroc(benign: np.ndarray, attack: np.ndarray) -> float:
    """P(attack score > benign score) + 0.5 P(equal); nan if a class is empty."""
    benign, attack = np.sort(np.asarray(benign, dtype=np.float64)), np.asarray(attack, dtype=np.float64)
    if len(benign) == 0 or len(attack) == 0:
        return float("nan")
    below = np.searchsorted(benign, attack, side="left")
    below_or_equal = np.searchsorted(benign, attack, side="right")
    return float((0.5 * (below + below_or_equal)).sum() / (len(benign) * len(attack)))


def average_precision(benign: np.ndarray, attack: np.ndarray) -> float:
    """Average precision with attack as the positive class (sklearn's definition, tied scores share a threshold)."""
    ranking = ScoreRanking(benign, attack)
    return float(ranking.average_precision(np.ones((1, 1)), np.zeros(len(attack) + len(benign), dtype=np.int64))[0])


def pair_wins(
    benign: np.ndarray,
    benign_cluster: np.ndarray,
    n_benign_clusters: int,
    attack: np.ndarray,
    attack_cluster: np.ndarray,
    n_attack_clusters: int,
) -> np.ndarray:
    """(attack clusters, benign clusters) matrix: number of (attack, benign) window pairs the score orders
    correctly (attack > benign), ties counted one half."""
    wins = np.zeros((n_attack_clusters, n_benign_clusters), dtype=np.float64)
    if len(benign) == 0 or len(attack) == 0:
        return wins
    order = np.argsort(attack_cluster, kind="stable")
    sorted_clusters = attack_cluster[order]
    present, starts = np.unique(sorted_clusters, return_index=True)
    ends = np.r_[starts[1:], len(order)]
    for c, lo, hi in zip(present, starts, ends):
        scores = np.sort(attack[order[lo:hi]])
        below = np.searchsorted(scores, benign, side="left")
        below_or_equal = np.searchsorted(scores, benign, side="right")
        # attack scores above a benign score: n - below_or_equal; equal to it: below_or_equal - below
        wins[c] = np.bincount(
            benign_cluster, weights=(hi - lo) - 0.5 * (below + below_or_equal), minlength=n_benign_clusters
        )
    return wins


class ScoreRanking:
    """Benign and attack scores of one task in one descending order, for (weighted) average precision."""

    def __init__(self, benign: np.ndarray, attack: np.ndarray) -> None:
        scores = np.r_[np.asarray(attack, dtype=np.float64), np.asarray(benign, dtype=np.float64)]
        self.order = np.argsort(-scores, kind="stable")
        self.is_attack = self.order < len(attack)
        ranked = scores[self.order]
        # last position of each run of equal scores: one threshold per distinct score
        self.threshold_ends = np.flatnonzero(np.r_[ranked[1:] != ranked[:-1], True])

    def average_precision(self, weights: np.ndarray, column: np.ndarray, chunk: int = 16) -> np.ndarray:
        """Weighted average precision for each row of `weights`.

        Args:
            weights: (R, K) weight of each of K columns (vehicle keys) in each of R resamples.
            column: (n_attack + n_benign,) column of each window, attack windows first, then benign windows.
            chunk: resamples handled at once (memory = chunk x windows).

        Returns:
            (R,) average precision; nan where the attack class has zero weight.
        """
        ranked_column = np.asarray(column)[self.order]
        out = np.full(len(weights), np.nan)
        for lo in range(0, len(weights), chunk):
            ranked = weights[lo : lo + chunk][:, ranked_column]
            true_pos = np.cumsum(np.where(self.is_attack, ranked, 0.0), axis=1)[:, self.threshold_ends]
            flagged = np.cumsum(ranked, axis=1)[:, self.threshold_ends]
            gained = np.diff(true_pos, axis=1, prepend=0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                precision = np.where(flagged > 0, true_pos / flagged, 0.0)
                out[lo : lo + chunk] = (gained * precision).sum(axis=1) / true_pos[:, -1]
        return out


# ---------------------------------------------------------------------------------------------- bootstrap


class VehicleBootstrap:
    """Fixed resample counts of the stratified cluster bootstrap.

    Attributes:
        benign_keys: sorted vehicle keys of benign test.
        attack_keys: {attack split: sorted vehicle keys}.
        benign_counts: (R, benign keys) how often each benign key is drawn in each resample.
        attack_counts: {attack split: (R, attack keys)}.
    """

    def __init__(self, config: cfg.ProtocolConfig, cache: WindowCache) -> None:
        self.resamples = config.bootstrap_resamples
        self.benign_keys = np.unique(cache.split(cfg.TEST_BENIGN).vehicle_key)
        self.benign_counts = self._draw(config.bootstrap_seed, 0, len(self.benign_keys))
        self.attack_keys, self.attack_counts = {}, {}
        for i, split in enumerate(cfg.ATTACK_TEST_SPLITS, start=1):
            self.attack_keys[split] = np.unique(cache.split(split).vehicle_key)
            self.attack_counts[split] = self._draw(config.bootstrap_seed, i, len(self.attack_keys[split]))

    def _draw(self, seed: int, stream: int, n_clusters: int) -> np.ndarray:
        rng = np.random.default_rng([seed, stream])
        return rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=self.resamples).astype(
            np.float64
        )


@dataclasses.dataclass(frozen=True)
class Subset:
    """A restriction of one task to some benign and some attack windows.

    Attributes:
        name: key in the results.
        benign, attack: boolean row selections of benign test / the attack split.
        n_benign, n_attack, n_attack_vehicles: sizes.
        reported: False if the subset is below the protocol's minimum attack windows / attack vehicles.
    """

    name: str
    benign: np.ndarray
    attack: np.ndarray
    n_benign: int
    n_attack: int
    n_attack_vehicles: int
    reported: bool

    def counts(self) -> dict:
        return {
            "reported": self.reported,
            "n_benign": self.n_benign,
            "n_attack": self.n_attack,
            "n_attack_vehicles": self.n_attack_vehicles,
        }


class Task:
    """Benign test vs one attack test split: bookkeeping that does not depend on the scores.

    Attributes:
        attack_split: name of the attack test split.
        benign_cluster, attack_cluster: index of each window's vehicle key in the bootstrap's key lists.
        benign_rows, attack_rows: real rows per window.
        strata, sensitivity: {name: Subset}.
    """

    def __init__(
        self, attack_split: str, config: cfg.ProtocolConfig, cache: WindowCache, bootstrap: VehicleBootstrap
    ) -> None:
        self.attack_split = attack_split
        self.config = config
        benign, attack = cache.split(cfg.TEST_BENIGN), cache.split(attack_split)
        self.benign_cluster = np.searchsorted(bootstrap.benign_keys, benign.vehicle_key)
        self.attack_cluster = np.searchsorted(bootstrap.attack_keys[attack_split], attack.vehicle_key)
        self.n_benign_clusters = len(bootstrap.benign_keys)
        self.n_attack_clusters = len(bootstrap.attack_keys[attack_split])
        self.benign_counts = bootstrap.benign_counts
        self.attack_counts = bootstrap.attack_counts[attack_split]
        self.benign_rows = benign.n_messages.astype(np.int64)
        self.attack_rows = attack.n_messages.astype(np.int64)
        self.everything = self._subset("all", np.ones(len(benign), bool), np.ones(len(attack), bool))
        self.strata = {
            s.name: self._subset(
                s.name,
                (self.benign_rows >= s.lo) & (self.benign_rows <= s.hi),
                (self.attack_rows >= s.lo) & (self.attack_rows <= s.hi),
            )
            for s in config.strata
        }
        group = config.sensitivity_group
        excluded = config.sensitivity_excluded_benign_scenario
        attack_scenarios = np.unique(attack.scenario)
        every_attack = np.ones(len(attack), bool)
        self.sensitivity = {
            SENSITIVITY_GROUP.format(group): self._subset(
                SENSITIVITY_GROUP.format(group), benign.group == group, attack.group == group
            ),
            SENSITIVITY_NO_FOLDER.format(excluded): self._subset(
                SENSITIVITY_NO_FOLDER.format(excluded), benign.scenario != excluded, every_attack
            ),
            SENSITIVITY_OWN_FAMILY: self._subset(
                SENSITIVITY_OWN_FAMILY, np.isin(benign.scenario, attack_scenarios), every_attack
            ),
        }

    def _subset(self, name: str, benign: np.ndarray, attack: np.ndarray) -> Subset:
        n_attack = int(attack.sum())
        vehicles = len(np.unique(self.attack_cluster[attack]))
        reported = (
            n_attack >= self.config.min_attack_windows
            and vehicles >= self.config.min_attack_vehicles
            and bool(benign.any())
        )
        return Subset(name, benign, attack, int(benign.sum()), n_attack, vehicles, reported)

    def cluster_sizes(self, subset: Subset) -> tuple[np.ndarray, np.ndarray]:
        """Windows per attack cluster and per benign cluster inside a subset."""
        return (
            np.bincount(self.attack_cluster[subset.attack], minlength=self.n_attack_clusters).astype(np.float64),
            np.bincount(self.benign_cluster[subset.benign], minlength=self.n_benign_clusters).astype(np.float64),
        )

    def auroc(self, benign: np.ndarray, attack: np.ndarray, subset: Subset) -> tuple[float, np.ndarray]:
        """AUROC inside a subset and its value in every bootstrap resample."""
        wins = pair_wins(
            benign[subset.benign],
            self.benign_cluster[subset.benign],
            self.n_benign_clusters,
            attack[subset.attack],
            self.attack_cluster[subset.attack],
            self.n_attack_clusters,
        )
        size_a, size_b = self.cluster_sizes(subset)
        return self._ratio(wins, size_a[:, None], size_b[:, None])

    def length_matched_auroc(self, benign: np.ndarray, attack: np.ndarray) -> tuple[float, np.ndarray, int]:
        """Pair-weighted mean of the within-n AUROCs, its bootstrap values, and the number of same-length pairs."""
        lengths = range(1, self.config.max_rows_matched + 1)
        wins = np.zeros((self.n_attack_clusters, self.n_benign_clusters))
        size_a = np.zeros((self.n_attack_clusters, len(lengths)))
        size_b = np.zeros((self.n_benign_clusters, len(lengths)))
        for j, n in enumerate(lengths):
            sel_b, sel_a = self.benign_rows == n, self.attack_rows == n
            if not sel_b.any() or not sel_a.any():
                continue
            wins += pair_wins(
                benign[sel_b],
                self.benign_cluster[sel_b],
                self.n_benign_clusters,
                attack[sel_a],
                self.attack_cluster[sel_a],
                self.n_attack_clusters,
            )
            size_a[:, j] = np.bincount(self.attack_cluster[sel_a], minlength=self.n_attack_clusters)
            size_b[:, j] = np.bincount(self.benign_cluster[sel_b], minlength=self.n_benign_clusters)
        point, replicates = self._ratio(wins, size_a, size_b)
        return point, replicates, int((size_a.sum(axis=0) * size_b.sum(axis=0)).sum())

    def _ratio(self, wins: np.ndarray, size_a: np.ndarray, size_b: np.ndarray) -> tuple[float, np.ndarray]:
        """wins / pairs, as a point value and per resample; `size_*` are (clusters, length cells)."""
        pairs = float((size_a.sum(axis=0) * size_b.sum(axis=0)).sum())
        point = float(wins.sum() / pairs) if pairs > 0 else float("nan")
        resampled_wins = ((self.attack_counts @ wins) * self.benign_counts).sum(axis=1)
        resampled_pairs = ((self.attack_counts @ size_a) * (self.benign_counts @ size_b)).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return point, np.where(resampled_pairs > 0, resampled_wins / resampled_pairs, np.nan)

    def average_precision(self, benign: np.ndarray, attack: np.ndarray) -> tuple[float, np.ndarray]:
        """Average precision and its value in every bootstrap resample."""
        ranking = ScoreRanking(benign, attack)
        point = ranking.average_precision(np.ones((1, 1)), np.zeros(len(attack) + len(benign), dtype=np.int64))[0]
        weights = np.concatenate([self.attack_counts, self.benign_counts], axis=1)
        column = np.r_[self.attack_cluster, self.n_attack_clusters + self.benign_cluster]
        return float(point), ranking.average_precision(weights, column)

    @staticmethod
    def rate_above(
        scores: np.ndarray, threshold: float, cluster: np.ndarray, counts: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Share of windows with score > threshold, and its value in every resample of the windows' clusters."""
        above = np.asarray(scores) > threshold
        per_cluster = np.bincount(cluster, weights=above, minlength=counts.shape[1])
        size = np.bincount(cluster, minlength=counts.shape[1]).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(above.mean()), (counts @ per_cluster) / (counts @ size)


# ---------------------------------------------------------------------------------------------- runs and results


@dataclasses.dataclass(frozen=True)
class RunGroup:
    """The finished runs of one (regime, detector, score).

    Attributes:
        regime, detector, score: what was run.
        is_stochastic: from the runs' config.json.
        seeds: seeds with a finished run, ascending.
        run_dirs: their folders, same order.
    """

    regime: str
    detector: str
    score: str
    is_stochastic: bool
    seeds: tuple[int, ...]
    run_dirs: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.regime}/{self.detector}/{self.score}"


@dataclasses.dataclass
class Evaluation:
    """Result of one run group.

    Attributes:
        summary: JSON-ready metrics (the entry of results.json).
        replicates: {"<attack split>/<metric>": (R,) seed-averaged bootstrap values}; used for paired differences.
    """

    summary: dict
    replicates: dict[str, np.ndarray]


class Evaluator:
    """Computes every metric of a run group from its score files and the cache's bookkeeping arrays."""

    def __init__(self, config: cfg.ProtocolConfig | None = None, cache: WindowCache | None = None) -> None:
        self.config = config or cfg.ProtocolConfig()
        self.cache = cache or WindowCache(self.config)
        self.bootstrap = VehicleBootstrap(self.config, self.cache)
        self._tasks: dict[str, Task] = {}

    def task(self, attack_split: str) -> Task:
        if attack_split not in self._tasks:
            self._tasks[attack_split] = Task(attack_split, self.config, self.cache, self.bootstrap)
        return self._tasks[attack_split]

    def discover(self) -> list[RunGroup]:
        """Every (regime, detector, score) with at least one finished run under `runs_dir`."""
        groups: dict[tuple[str, str, str], dict] = {}
        for summary_path in sorted(Path(self.config.runs_dir).glob("*/*/seed*/summary.json")):
            run_dir = summary_path.parent
            if json.loads(summary_path.read_text()).get("status") != "complete":
                continue
            run_config = json.loads((run_dir / "config.json").read_text())
            for score in run_config["score_names"]:
                if not all(score_file(run_dir, score, s).exists() for s in cfg.SCORED_SPLITS):
                    continue
                key = (run_config["regime"], run_config["detector"], score)
                group = groups.setdefault(key, {"is_stochastic": run_config["is_stochastic"], "runs": {}})
                group["runs"][int(run_config["seed"])] = str(run_dir)
        order = {r: i for i, r in enumerate((cfg.CONTROL_REGIME, *cfg.REGIMES))}
        out = []
        for (regime, detector, score), g in sorted(groups.items(), key=lambda kv: (order.get(kv[0][0], 99), kv[0])):
            seeds = tuple(sorted(g["runs"]))
            out.append(RunGroup(regime, detector, score, g["is_stochastic"], seeds, tuple(g["runs"][s] for s in seeds)))
        return out

    def _scores(self, run_dir: str, score: str, split: str) -> np.ndarray:
        values = np.load(score_file(Path(run_dir), score, split))
        if values.shape != (len(self.cache.split(split)),):
            raise ValueError(
                f"{run_dir}: {score}/{split} has {values.shape} scores, cache has {len(self.cache.split(split))}"
            )
        return values

    def thresholds(self, group: RunGroup, run_dir: str) -> dict[str, float]:
        """Alarm thresholds of one run, from its validation scores only."""
        val_benign = self._scores(run_dir, group.score, cfg.VAL_BENIGN)
        out = {BENIGN_VAL.format(p): float(np.percentile(val_benign, p)) for p in self.config.benign_val_percentiles}
        own = val_benign
        if group.regime in cfg.REGIMES:
            rows_b, rows_g = self.cache.regime_rows(group.regime, cfg.VAL_BENIGN, cfg.VAL_GRIDSYBIL)
            own = np.r_[val_benign[rows_b], self._scores(run_dir, group.score, cfg.VAL_GRIDSYBIL)[rows_g]]
        out[OWN_VAL.format(self.config.own_val_percentile)] = float(np.percentile(own, self.config.own_val_percentile))
        return out

    def primary_operating_point(self, regime: str) -> str:
        if regime in cfg.MIX_REGIMES:
            return OWN_VAL.format(self.config.own_val_percentile)
        return BENIGN_VAL.format(self.config.benign_val_percentiles[0])

    def summarise(self, per_seed: list[float], replicates: list[np.ndarray]) -> tuple[dict, np.ndarray]:
        """Per-seed values + per-seed bootstrap values -> metric record and the seed-averaged bootstrap values."""
        values = np.asarray(per_seed, dtype=np.float64)
        averaged = np.mean(np.stack(replicates), axis=0)
        return {**interval(averaged, self.config.ci_level), **seed_statistics(values)}, averaged

    def evaluate(self, group: RunGroup) -> Evaluation:
        """All metrics of one (regime, detector, score) over its finished seeds."""
        thresholds = [self.thresholds(group, d) for d in group.run_dirs]
        benign_scores = [self._scores(d, group.score, cfg.TEST_BENIGN) for d in group.run_dirs]
        expected = self.config.seeds if group.is_stochastic else self.config.seeds[:1]
        summary = {
            "regime": group.regime,
            "detector": group.detector,
            "score": group.score,
            "is_stochastic": group.is_stochastic,
            "seeds": list(group.seeds),
            "expected_seeds": list(expected),
            "all_seeds_present": tuple(group.seeds) == tuple(expected),
            "primary_operating_point": self.primary_operating_point(group.regime),
            "thresholds_per_seed": {k: [t[k] for t in thresholds] for k in thresholds[0]},
            "benign_test_fpr": {},
            "tasks": {},
        }
        replicates: dict[str, np.ndarray] = {}

        any_task = self.task(cfg.ATTACK_TEST_SPLITS[0])
        for op in thresholds[0]:
            pairs = [
                any_task.rate_above(s, t[op], any_task.benign_cluster, any_task.benign_counts)
                for s, t in zip(benign_scores, thresholds)
            ]
            summary["benign_test_fpr"][op], _ = self.summarise([p for p, _ in pairs], [r for _, r in pairs])

        for attack_split in cfg.ATTACK_TEST_SPLITS:
            task = self.task(attack_split)
            attack_scores = [self._scores(d, group.score, attack_split) for d in group.run_dirs]
            seeds = list(zip(benign_scores, attack_scores))
            record = {
                "n_benign": task.everything.n_benign,
                "n_attack": task.everything.n_attack,
                "n_benign_vehicles": task.n_benign_clusters,
                "n_attack_vehicles": task.n_attack_clusters,
                "attack_share": task.everything.n_attack / (task.everything.n_attack + task.everything.n_benign),
            }

            def add(name: str, results: list[tuple[float, np.ndarray]]) -> dict:
                metric, averaged = self.summarise([p for p, _ in results], [r for _, r in results])
                replicates[f"{attack_split}/{name}"] = averaged
                return metric

            record["auroc"] = add("auroc", [task.auroc(b, a, task.everything) for b, a in seeds])
            record["average_precision"] = add("average_precision", [task.average_precision(b, a) for b, a in seeds])
            matched = [task.length_matched_auroc(b, a) for b, a in seeds]
            record["length_matched_auroc"] = add("length_matched_auroc", [(p, r) for p, r, _ in matched])
            record["length_matched_auroc"]["same_length_pairs"] = matched[0][2]
            record["recall"] = {
                op: add(
                    f"recall/{op}",
                    [
                        task.rate_above(a, t[op], task.attack_cluster, task.attack_counts)
                        for (_, a), t in zip(seeds, thresholds)
                    ],
                )
                for op in thresholds[0]
            }
            for kind, subsets in (("strata", task.strata), ("sensitivity", task.sensitivity)):
                record[kind] = {}
                for name, subset in subsets.items():
                    record[kind][name] = subset.counts()
                    record[kind][name]["auroc"] = (
                        add(f"{kind}/{name}", [task.auroc(b, a, subset) for b, a in seeds]) if subset.reported else None
                    )
            summary["tasks"][attack_split] = record
        return Evaluation(summary, replicates)


def seed_statistics(values: np.ndarray) -> dict:
    """Per-seed values, their mean and sample standard deviation (None with one seed)."""
    return {
        "per_seed": [float(v) for v in values],
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else None,
    }


def interval(replicates: np.ndarray, level: float) -> dict:
    """Percentile interval of bootstrap values; resamples where the metric is undefined are counted and left out."""
    undefined = int(np.isnan(replicates).sum())
    if undefined == len(replicates):
        return {"ci": None, "ci_level": level, "undefined_resamples": undefined}
    lo, hi = np.nanpercentile(replicates, [50 * (1 - level), 100 - 50 * (1 - level)])
    return {"ci": [float(lo), float(hi)], "ci_level": level, "undefined_resamples": undefined}


def paired_difference(a: Evaluation, b: Evaluation, key: str, level: float) -> dict | None:
    """a - b for one "<attack split>/<metric>": difference of the seed means and its paired bootstrap interval."""
    if key not in a.replicates or key not in b.replicates:
        return None
    attack_split, metric = key.split("/", 1)
    mean_a = _metric(a.summary["tasks"][attack_split], metric)["mean"]
    mean_b = _metric(b.summary["tasks"][attack_split], metric)["mean"]
    return {"difference": mean_a - mean_b, **interval(a.replicates[key] - b.replicates[key], level)}


def _metric(task_record: dict, metric: str) -> dict:
    node = task_record
    for part in metric.split("/"):
        node = node[part]
    return node["auroc"] if "auroc" in node and "mean" not in node else node
