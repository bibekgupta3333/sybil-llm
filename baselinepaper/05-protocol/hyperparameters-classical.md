# Hyperparameters of the classical detectors — frozen before any test score

Written 2026-11-10, before any run of these detectors scored a validation or test split. Part of protocol §4
("hyperparameters are listed before any test score is computed"); the neural detectors have their own file. Nothing
here was chosen by looking at a validation or test detection metric, and nothing is tuned: values are the installed
library's defaults, named below, or one a-priori value where the library has no default. Changes after this point
go to protocol §9 with the reason.

Code: `baselinepaper/experiments/detectors/{window_statistics,pca,iforest,ocsvm,knn,statslr}.py`. Every run writes
these values to its `config.json` (`hyperparameters`).

## Environment

| Package | Version |
|---|---|
| scikit-learn | 1.9.1 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| Python | `.venv-train` (3.14) |

Device: CPU. Threads do not change any result below (no estimator uses a thread-dependent random stream).

## Shared input: window statistics

- Input of every classical detector: `self.standardise(x, mask)` (the regime's training-pool statistics for
  `benign_gridfolders`, `mix_all`, `mix_gridfolders`; the cached normalisation otherwise), then per feature over
  the **real rows** of the window: mean, standard deviation, minimum, maximum, last − first. 13 features × 5 = 65
  values, in that block order (13 means, 13 stds, 13 mins, 13 maxs, 13 last − first).
- Standard deviation: population (divide by n, `ddof = 0`), so a one-row window has std 0 and last − first 0.
- No number-of-rows column (the earlier `task/encoder` baseline had one; the protocol lists 65 values).
- `StandardScaler` (defaults: `with_mean=True`, `with_std=True`) fitted on the statistics of the **whole** training
  pool of the regime, applied to every scored window. A column with zero training variance is centred and left
  unscaled (the scaler's behaviour).
- Computed in float64, in chunks of 32,768 windows.

## Subsample (one size for every method that does not scale)

- **20,000 training windows**, drawn without replacement from the regime's training pool with
  `numpy.random.default_rng(seed).choice(n, 20000, replace=False)` (sorted), after the scaler is fitted on the whole
  pool. If a pool has fewer windows, all are used (does not happen: smallest pool 125,749).
- Used by: `ocsvm` (training set) and `knn` (reference set). The same seed gives the same subsample for both.
- Not used by `pca`, `iforest`, `statslr` (they are fitted on every training window).

## Detectors

| Detector | Regimes | Estimator and parameters | Score (higher = more anomalous) | Seeds |
|---|---|---|---|---|
| `pca` | 4 unsupervised | `PCA(n_components=0.95, svd_solver="full", whiten=False)`: the smallest number of components whose explained training variance is ≥ 95% (rule fixed a priori); other parameters default (`tol=0.0`, `iterated_power="auto"`, `n_oversamples=10` — unused by the full solver) | `reconstruction`: mean over the 65 scaled statistics of (value − reconstruction)² | deterministic, seed 0 only |
| `iforest` | 4 unsupervised | `IsolationForest(n_estimators=100, max_samples="auto", contamination="auto", max_features=1.0, bootstrap=False, random_state=seed)`; `max_samples="auto"` = 256 windows per tree; fitted on every training window; `n_jobs=1` | `isolation`: −`decision_function` | stochastic, seeds 0–4 |
| `ocsvm` | 4 unsupervised | `OneClassSVM(kernel="rbf", gamma="scale", nu=0.5, tol=1e-3, shrinking=True, max_iter=-1)` (`degree`, `coef0` unused by RBF) on the 20,000-window subsample; `cache_size=1000` MB (library default 200; kernel-cache memory only, the optimum is the same) | `decision`: −`decision_function` | stochastic (subsample), seeds 0–4 |
| `knn` | 4 unsupervised | `NearestNeighbors(n_neighbors=5, metric="minkowski", p=2)` (Euclidean; k = 5 is the library default and the a-priori value), `algorithm="brute"` (exact search; `"auto"` picks brute force at 65 dimensions anyway), reference set = the 20,000-window subsample, `n_jobs=1` | `distance`: mean Euclidean distance to the 5 nearest reference windows | stochastic (subsample), seeds 0–4 |
| `statslr` | `supervised` | `LogisticRegression(C=1.0, l1_ratio=0.0 (L2), solver="lbfgs", tol=1e-4, fit_intercept=True, class_weight=None)`, fitted on every training window with its label; `max_iter=1000` (library default 100; raised a priori so the solver can converge on 731,853 windows — an optimisation cap, not a model choice; the iterations used and whether it converged are recorded in `summary.json`) | `probability`: P(GridSybil) | deterministic, seed 0 only |

The 4 unsupervised regimes are `benign_only`, `benign_gridfolders`, `mix_all`, `mix_gridfolders`.

- No validation data is used by any classical detector: there is no model selection, early stopping or calibration
  (`val` is ignored in `fit`). Alarm thresholds come from the validation score files in the evaluation module, as
  for every detector.
- `statslr.score` takes x and mask only.
- `iforest`: `decision_function` = `score_samples` − `offset_`, and `contamination="auto"` fixes `offset_ = −0.5`,
  so the ranking is that of the isolation score.
- `knn`: scored windows are never in the reference set (val / test vehicles are disjoint from train), so no
  self-match is removed.
- The first seed-dependent choice of a stochastic detector is seeded by the run's seed alone.

## Runs

| Detector | Regimes | Seeds | Runs |
|---|---|---|---|
| `pca` | 4 | 0 | 4 |
| `iforest` | 4 | 0–4 | 20 |
| `ocsvm` | 4 | 0–4 | 20 |
| `knn` | 4 | 0–4 | 20 |
| `statslr` | 1 | 0 | 1 |
| total | | | 65 |

`baselinepaper/experiments/sweep_classical.py` runs them through `run.py` (CPU, at most 4 single-threaded worker
processes), skips finished runs and logs to `runs/sweep_classical.log`.
