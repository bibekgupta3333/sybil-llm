# `baselinepaper/experiments/` — every detector under one protocol

Code of `baselinepaper/05-protocol/protocol.md` (plus the changes of `06-reviews/protocol-review-methods.md`):
one input, one split, one normalisation, one test set, one evaluation for every detector. Run everything from the
repo root with `.venv-train/bin/python`. Side experiments for the paper, not thesis results (RULE 1); nothing under
`data/`, `src/` or `task/` is written; no split is changed (RULE 3).

## Layout

| File | What it is |
|---|---|
| `config.py` | `ProtocolConfig` (frozen): paths, regimes, seeds 0–4, thresholds, length strata, bootstrap; split names |
| `data.py` | `WindowCache`: builds / loads the cache, `training_set(regime)`, `validation_set(regime)`, `regime_statistics(regime)` |
| `verify_data.py` | read-only checks of the cache → `cache/verify_report.json` |
| `detectors/base.py` | `Detector` (the interface) + registry (`register`, `get_detector`, `available`) |
| `detectors/length.py` | length-only control, score = −(number of real rows) |
| `run.py` | CLI: one detector × regime × seed → `runs/<regime>/<detector>/seed<k>/` |
| `evaluation.py` | every metric, from score files only (the only module that knows which split is which class) |
| `aggregate.py` | rebuilds `results/results.json` + `results/main_table.md` from whatever runs exist |
| `tests/` | pytest, CPU, synthetic inputs (9 tests) |
| `cache/`, `runs/` | gitignored (2.75 GB cache; score files) · `results/` is tracked |

## Commands

```bash
.venv-train/bin/python -m baselinepaper.experiments.data            # build the cache (~20 s, needs 8 GB free; --force)
.venv-train/bin/python -m baselinepaper.experiments.verify_data     # checks a–e, exit 1 on failure
.venv-train/bin/python -m baselinepaper.experiments.run --detector length_only --regime none --seed 0
.venv-train/bin/python -m baselinepaper.experiments.run --detector <name> --regime <regime> --seed <0-4> [--device auto|cpu|mps|cuda] [--force]
.venv-train/bin/python -m baselinepaper.experiments.aggregate --jobs 6
.venv-train/bin/python -m pytest baselinepaper/experiments/tests
```

A finished run (summary.json + every score file) is skipped unless `--force`. Deterministic detectors accept seed 0
only. `aggregate` costs about 10 s per (regime, detector, score, seed) on one core; `--jobs` runs groups in parallel.

## Data (`data.py`)

Cache splits, each `cache/<split>/` with `x.npy` (N, 32, 13) float32, `mask.npy` (N, 32) bool, `n_messages.npy`,
`vehicle_key.npy`, `scenario.npy` (source folder; `Split.family`, `Split.group` derive from it),
`link_window_index.npy`, `window_id.npy` (bytes) — same row order, memory-mapped on load. **No label array is
stored**: the split is the class.

| Split | Windows | Vehicles | 1–3-row share | Source |
|---|---|---|---|---|
| `train_benign` | 502,773 | 4,206 | 13.7% | `T32_benign/train` |
| `val_benign` | 65,313 | 524 | 13.9% | `T32_benign/val` |
| `test_benign` | 60,142 | 525 | 13.9% | `T32_benign/test` |
| `test_gridsybil` | 28,087 | 172 | 20.4% | `T32_benign/test_gridsybil` |
| `test_datareplay` | 76,283 | 173 | 94.9% | `T32_benign/test_datareplay` |
| `test_dosrandom` | 199,834 | 173 | 99.7% | `T32_benign/test_dosrandom` |
| `test_dosdisruptive` | 199,826 | 173 | 99.7% | `T32_benign/test_dosdisruptive` |
| `train_gridsybil` | 229,080 | 1,382 | 18.5% | `T32_benign_grid`, vehicle key in T32_benign's train list |
| `val_gridsybil` | 28,374 | 172 | 19.1% | `T32_benign_grid`, vehicle key in T32_benign's val list |

GridSybil train / val windows: every GridSybil window (attack code 16) of the three `T32_benign_grid` splits is
assigned by its vehicle key to the **T32_benign** split, un-normalised with the grid input's mean / std and
re-normalised with T32_benign's. One rule beyond protocol §2: the `log_dtau` of a link's first message is "missing"
and stored as 0 in both inputs; it stays 0 (an affine map would move it to (mean_grid − mean_benign) / std_benign).
The test-vehicle windows derived the same way are kept as `rederived_test_gridsybil` for check (a) only.

Regimes (`training_set(regime)`; `validation_set(regime)` has the same composition from the val vehicles):

| Regime | Training pool | Windows | GridSybil share | Val pool (windows, share) | Returned type |
|---|---|---|---|---|---|
| `benign_only` | benign train, 8 folders | 502,773 | 0 | 65,313, 0 | `WindowSet` |
| `benign_gridfolders` | benign train of the 2 GridSybil folders (control) | 125,749 | 0 | 15,893, 0 | `WindowSet` |
| `mix_all` | benign train (8 folders) + GridSybil train | 731,853 | **31.3%** | 93,687, 30.3% | `WindowSet` |
| `mix_gridfolders` | benign + GridSybil train of the 2 GridSybil folders | 354,829 | **64.6%** | 44,267, 64.1% | `WindowSet` |
| `supervised` | as `mix_all`, labelled | 731,853 | 31.3% | 93,687, 30.3% | `LabelledWindowSet` |

`WindowSet` has `x`, `mask` and nothing else; only `supervised` returns a `LabelledWindowSet` with `labels` (int8,
0 benign / 1 GridSybil). Pooled sets are in one fixed random order (`pool_seed = 0`), so position does not reveal
the class.

Label-free re-standardisation: `regime_statistics(regime)` returns the per-feature mean / std of the real rows of
the regime's training pool **in the cached units** (missing `log_dtau` cells left out, as in the encoder inputs;
stored in `cache/regime_statistics.json`). The runner passes it to detectors of `benign_gridfolders`, `mix_all`,
`mix_gridfolders`; `benign_only` and `supervised` keep the cached normalisation (`benign_only` statistics are
0 ± 1 with exactly the metadata's `n_values`, which re-checks the cache). For `mix_gridfolders` the pool's `range`
has std 3.67 and `log_dtau` std 2.01 in cached units.

## Verification (`verify_data.py`, 2026-11-10, all checks pass; `cache/verify_report.json`)

- **a.** GridSybil windows of the test vehicles re-derived from `T32_benign_grid` vs `T32_benign/test_gridsybil`:
  28,087 = 28,087 windows, identical id set, identical masks, max abs difference **1.9e-4** normalised units
  (feature `range`; mean 4.1e-5 on real rows; tolerance 1e-3). The difference is the double 4-decimal rounding
  times the std ratio of the two inputs.
- **b.** Vehicle keys train / val / test = 5,588 / 696 / 750 (benign + GridSybil + attack test splits); keys in
  two parts: 0 / 0 / 0.
- **c.** Every key of every split is in the T32_benign metadata list of its part (GridSybil train 1,382 keys,
  val 172: 0 outside); GridSybil windows come only from the two GridSybil folders; all 285,541 GridSybil windows of
  the grid input are assigned (229,080 + 28,374 + 28,087).
- **d.** Windows and messages of the seven T32_benign splits equal its metadata; counts and shares as in the tables
  above.
- **e.** Padding values ≠ 0: 0 in every split (0 in the sources too); all masks are non-empty prefix masks;
  `n_messages` = mask rows; window ids unique within and across splits; missing `log_dtau` cells are 0.

Benign test and attack test vehicle keys are disjoint (0 shared keys in each of the four tasks).

## The detector interface (`detectors/base.py`)

Add one file `detectors/<something>.py`; every module of the package is imported by the registry, no other edit.

```python
from baselinepaper.experiments import config as cfg
from baselinepaper.experiments.detectors.base import Detector, register

@register
class MyDetector(Detector):
    name = "my_detector"                    # run-folder name: lower case, digits, underscores
    regimes = cfg.UNLABELLED_REGIMES        # or (cfg.SUPERVISED,)
    is_stochastic = True                    # False = one run, seed 0
    score_names = ("reconstruction",)       # letters, digits, hyphens; one set of score files per name
    reused_modules = ("task.autoencoder_benign_t32",)   # packages outside this one whose .py files get hashed

    def hyperparameters(self) -> dict: ...  # fixed settings, written to config.json before fitting
    def fit(self, train, val, seed: int) -> None: ...
    def score(self, x, mask) -> dict[str, np.ndarray]: ...   # {score name: (N,) finite}, higher = more anomalous
    def history(self) -> list[dict]: ...    # optional, per epoch → history.json
    def fit_info(self) -> dict: ...         # optional → summary.json (parameters, best epoch, subsample size)
    def save(self, run_dir: Path) -> None: ...   # optional artefacts
```

- The runner constructs `cls(regime=..., device=..., statistics=...)` (do not change the constructor signature;
  call `super().__init__` if you override it) and sets `self.regime`, `self.device` (resolved: cpu / mps / cuda),
  `self.statistics`.
- `fit(train, val, seed)`: `train` / `val` are `WindowSet`s (`.x` (N, 32, 13) float32, `.mask` (N, 32) bool) in the
  cached normalisation; `LabelledWindowSet`s (`.labels`) only for `supervised`. `val` is the only data for early
  stopping / model selection / calibration. `x` may be a read-only memory map (`benign_only`): copy before writing.
- **Call `self.standardise(x, mask)` on every input, in `fit` and in `score`.** It applies the regime's
  training-pool statistics for `benign_gridfolders` / `mix_*` and returns `x` unchanged otherwise.
- `score(x, mask)` gets one whole split (up to 199,834 windows; batch internally) and nothing else — no split name,
  no vehicle key, no label.
- Seed everything from `seed`. Record subsample sizes in `fit_info()`.

Run folder: `config.json` (detector, regime, seed, hyperparameters, pools, statistics used, the protocol),
`env.json` (python, numpy / scipy / scikit-learn / torch versions, device, git commit + dirty flag, cache
fingerprint, sha256 of every .py of this package and of `reused_modules`), `history.json`,
`scores_<score>_<split>.npy` for `val_benign`, `val_gridsybil` and the five test splits (float64, cache row
order), `summary.json` (status, timings, `fit_info`, score quantiles — no metric).

## Evaluation (`evaluation.py`, `aggregate.py`)

Four tasks, benign test vs each attack test split, unit = window. Per (regime, detector, score):

- AUROC; average precision (attack share stated); **length-matched AUROC** = pair-weighted mean of the AUROCs
  within each exact number of real rows (1..32), 0.5 for the length-only score by construction.
- AUROC inside the strata 1–3, ≥ 4, n = 1, 2, 3, 4–16, ≥ 17. A stratum with < 200 attack windows or < 20 attack
  vehicles is `n/a` with its counts.
- Operating points from validation scores only, alarm when score **>** threshold: `benign_val_p95`,
  `benign_val_p99`, `own_val_p95` (95th percentile of the regime's own val pool, label-free). Attack recall per
  task, benign test FPR. Primary: `own_val_p95` for `mix_*` (there `benign_val_*` is the oracle calibration),
  `benign_val_p95` otherwise.
- Sensitivity (same score files): group 1416 only; benign test without the `GridSybil_0709` folder (F14); benign
  test restricted to the attack's own family folders.
- Per-seed values, mean ± sample std over seeds, 95% percentile interval of the seed mean from a stratified
  cluster bootstrap (1,000 resamples; benign vehicle keys and attack vehicle keys resampled separately). The
  resamples depend only on `bootstrap_seed`, so differences are paired: `results.json` has detector − length-only
  and regime − regime (same detector) for AUROC and length-matched AUROC.
- The bootstrap never re-sorts: an AUROC is a sum of pairwise wins over (attack vehicle, benign vehicle) pairs, so
  one 173 × 525 matrix per subset gives all resamples by two matrix products. Checked in the tests against
  `sklearn` with `sample_weight` (AUROC and average precision, with ties).

## Length-only control

Score = −`n_messages` (shorter = more anomalous): the rule of the earlier studies
(`task/autoencoder_benign_t32/autoencoder_benign_T32.ipynb`, `s = -n_messages`), fixed before any score of this
protocol existed, never flipped on labels. Run 2026-11-10 (`runs/none/length_only/seed0`):

| Attack | AUROC [95% CI] | inventory (one-class T32 study) | AP | 1–3 rows | ≥ 4 rows | group 1416 only |
|---|---|---|---|---|---|---|
| GridSybil | 0.559 [0.538, 0.579] | 0.559 | 0.367 | 0.521 | 0.538 | 0.580 |
| DataReplay | 0.952 [0.948, 0.955] | 0.952 | 0.934 | 0.711 | 0.927 | 0.954 |
| DoSRandom | 0.967 [0.965, 0.970] | 0.967 | 0.981 | 0.769 | n/a (523 windows, 13 vehicles) | 0.971 |
| DoSDisruptive | 0.967 [0.965, 0.970] | 0.967 | 0.981 | 0.769 | n/a (524 windows, 14 vehicles) | 0.971 |

Identical to `01-our-studies/inventory.md` (same windows, same rule). Its operating points are degenerate: more
than 5% of benign val windows have one row, so every percentile threshold is −1 and no window scores above it
(recall 0, FPR 0); the control is read through its AUROC.

## Notes for the paper

- The ≥ 4-row stratum of the two DoS attacks is `n/a` under the 20-vehicle rule (13 / 14 vehicles).
- `validation_set` of the unlabelled mix regimes contains GridSybil windows (same composition as training), so
  "benign val loss" model selection of protocol §4 does not apply there; benign val is used only for the oracle
  thresholds.
- F14, F1, F13 limitations are carried over unchanged.
