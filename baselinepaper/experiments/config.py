"""Fixed settings of the evaluation protocol (`baselinepaper/05-protocol/protocol.md`).

Everything a run or a table depends on is a field of `ProtocolConfig`; the defaults are the protocol. Tests build
a config with other paths, nothing else changes.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

SEQ_LEN = 32
N_FEATURES = 13

# Training regimes (protocol section 3). `CONTROL_REGIME` is the run-folder name of detectors that are not trained.
BENIGN_ONLY = "benign_only"
BENIGN_GRIDFOLDERS = "benign_gridfolders"
MIX_ALL = "mix_all"
MIX_GRIDFOLDERS = "mix_gridfolders"
SUPERVISED = "supervised"
REGIMES = (BENIGN_ONLY, BENIGN_GRIDFOLDERS, MIX_ALL, MIX_GRIDFOLDERS, SUPERVISED)
UNLABELLED_REGIMES = (BENIGN_ONLY, BENIGN_GRIDFOLDERS, MIX_ALL, MIX_GRIDFOLDERS)
# Regimes whose training pool contains unlabelled GridSybil windows: no clean benign validation data is assumed,
# so their primary alarm threshold comes from the regime's own validation pool.
MIX_REGIMES = (MIX_ALL, MIX_GRIDFOLDERS)
# Regimes restricted to the two GridSybil scenario folders.
GRIDFOLDER_REGIMES = (BENIGN_GRIDFOLDERS, MIX_GRIDFOLDERS)
# Regimes whose detectors re-standardise the cached features with `WindowCache.regime_statistics`.
RESTANDARDISED_REGIMES = (BENIGN_GRIDFOLDERS, MIX_ALL, MIX_GRIDFOLDERS)
CONTROL_REGIME = "none"

# Cache splits. The split a window is in is its label: `*_benign` = benign, every other split = that attack.
TRAIN_BENIGN, VAL_BENIGN, TEST_BENIGN = "train_benign", "val_benign", "test_benign"
TRAIN_GRIDSYBIL, VAL_GRIDSYBIL = "train_gridsybil", "val_gridsybil"
ATTACK_TEST_SPLITS = ("test_gridsybil", "test_datareplay", "test_dosrandom", "test_dosdisruptive")
TEST_SPLITS = (TEST_BENIGN, *ATTACK_TEST_SPLITS)
ALL_SPLITS = (TRAIN_BENIGN, VAL_BENIGN, TRAIN_GRIDSYBIL, VAL_GRIDSYBIL, *TEST_SPLITS)
# Splits every run scores. The two validation splits are scored separately: benign val gives the benign-calibrated
# thresholds, benign val + GridSybil val give the label-free threshold of a regime's own validation pool.
SCORED_SPLITS = (VAL_BENIGN, VAL_GRIDSYBIL, *TEST_SPLITS)

# Cache split -> split folder of the T32_benign encoder input.
BENIGN_INPUT_SPLITS = {
    TRAIN_BENIGN: "train",
    VAL_BENIGN: "val",
    TEST_BENIGN: "test",
    "test_gridsybil": "test_gridsybil",
    "test_datareplay": "test_datareplay",
    "test_dosrandom": "test_dosrandom",
    "test_dosdisruptive": "test_dosdisruptive",
}
# Attack code (`label` in the T32_benign info files) expected in each cache split read from T32_benign.
EXPECTED_ATTACK_CODE = {
    TRAIN_BENIGN: 0,
    VAL_BENIGN: 0,
    TEST_BENIGN: 0,
    "test_gridsybil": 16,
    "test_datareplay": 17,
    "test_dosrandom": 18,
    "test_dosdisruptive": 19,
}
GRIDSYBIL_CODE = 16


@dataclasses.dataclass(frozen=True)
class LengthStratum:
    """Windows with `lo <= real rows <= hi`.

    Attributes:
        name: key in results (`rows_1_3`).
        label: column header (`1-3 rows`).
        lo, hi: inclusive bounds on the number of real rows.
    """

    name: str
    label: str
    lo: int
    hi: int


@dataclasses.dataclass(frozen=True)
class ProtocolConfig:
    """The protocol's constants.

    Attributes:
        benign_input_dir: `all/T32_benign` encoder input (read only): the split, the normalisation, the test sets.
        grid_input_dir: `all/T32_benign_grid` encoder input (read only): source of the train / val GridSybil windows.
        cache_dir: decoded arrays (.npy, gitignored).
        runs_dir: one folder per run, `<regime>/<detector>/seed<k>/` (gitignored).
        results_dir: aggregated tables (tracked).
        seq_len, n_features: window shape.
        gridsybil_scenarios: the two GridSybil scenario folders (`mix_gridfolders`).
        seeds: seeds of every stochastic detector; deterministic detectors use `seeds[0]` only.
        benign_val_percentiles: percentiles of the benign validation scores used as alarm thresholds (alarm when
            score > threshold); the first is the primary operating point outside the mix regimes.
        own_val_percentile: percentile of the scores on the regime's own validation pool (label-free threshold);
            the primary operating point of the mix regimes.
        strata: window-length strata for the within-stratum AUROC.
        max_rows_matched: the length-matched AUROC pools the within-n comparisons of n = 1 .. max_rows_matched.
        min_attack_windows, min_attack_vehicles: a stratum or sensitivity subset with fewer attack windows or
            attack vehicles is reported as n/a (with its counts).
        sensitivity_group: time window of the "one group only" sensitivity row.
        sensitivity_excluded_benign_scenario: benign test folder left out in the F14 sensitivity row.
        bootstrap_resamples: resamples of the stratified cluster bootstrap (benign test vehicle keys and attack
            test vehicle keys resampled separately).
        bootstrap_seed: seed of the resampling (the same draws for every detector, so differences are paired).
        ci_level: two-sided confidence level of the bootstrap interval.
        pool_seed: seed of the fixed permutation that pools benign and GridSybil windows of a training /
            validation set (window order must not reveal the class).
        max_rederive_abs_diff: tolerance (normalised units) of the check that the re-normalised GridSybil test
            windows equal `T32_benign/test_gridsybil`.
        workers: processes that decode the JSON shards when the cache is built.
        min_free_gb: free disk space required before the cache is built.
    """

    benign_input_dir: str = "src/data/encoder_input/all/T32_benign"
    grid_input_dir: str = "src/data/encoder_input/all/T32_benign_grid"
    cache_dir: str = "baselinepaper/experiments/cache"
    runs_dir: str = "baselinepaper/experiments/runs"
    results_dir: str = "baselinepaper/experiments/results"
    seq_len: int = SEQ_LEN
    n_features: int = N_FEATURES
    gridsybil_scenarios: tuple[str, ...] = ("GridSybil_0709", "GridSybil_1416")
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    benign_val_percentiles: tuple[float, ...] = (95.0, 99.0)
    own_val_percentile: float = 95.0
    strata: tuple[LengthStratum, ...] = (
        LengthStratum("rows_1_3", "1-3 rows", 1, 3),
        LengthStratum("rows_ge_4", ">= 4 rows", 4, SEQ_LEN),
        LengthStratum("rows_1", "1 row", 1, 1),
        LengthStratum("rows_2", "2 rows", 2, 2),
        LengthStratum("rows_3", "3 rows", 3, 3),
        LengthStratum("rows_4_16", "4-16 rows", 4, 16),
        LengthStratum("rows_ge_17", ">= 17 rows", 17, SEQ_LEN),
    )
    max_rows_matched: int = SEQ_LEN
    min_attack_windows: int = 200
    min_attack_vehicles: int = 20
    sensitivity_group: str = "1416"
    sensitivity_excluded_benign_scenario: str = "GridSybil_0709"
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 0
    ci_level: float = 0.95
    pool_seed: int = 0
    max_rederive_abs_diff: float = 1e-3
    workers: int = 8
    min_free_gb: float = 8.0

    def run_dir(self, regime: str, detector: str, seed: int) -> Path:
        """`<runs_dir>/<regime>/<detector>/seed<k>`."""
        return Path(self.runs_dir) / regime / detector / f"seed{seed}"

    def to_dict(self) -> dict:
        """JSON-ready settings."""
        return dataclasses.asdict(self)
