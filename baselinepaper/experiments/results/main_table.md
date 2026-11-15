# Results under the one protocol (rebuilt from score files)

Generated 2026-11-10T17:03:31+00:00 by `python -m baselinepaper.experiments.aggregate`; do not edit.
Every number comes from the score files of the finished runs; a missing detector means no finished run.

- Tasks: benign test (60,142 windows) vs each attack test split; unit = window.
- `mean ± std` over seeds (sample std; deterministic detectors have one run). 95% CI: stratified cluster bootstrap over test vehicle keys (1000 resamples, benign and attack keys resampled separately) of the seed mean.
- Length-matched AUROC: pair-weighted mean of the AUROCs within each exact number of real rows; 0.5 for the length-only score by construction.
- Strata with fewer than 200 attack windows or 20 attack vehicles are n/a.
- Length-only control: score = -(number of real rows), fixed, not trained; repeated for reference.

Training pools: `benign_only` 502,773 windows (0.0% GridSybil); `benign_gridfolders` 125,749 windows (0.0% GridSybil); `mix_all` 731,853 windows (31.3% GridSybil); `mix_gridfolders` 354,829 windows (64.6% GridSybil); `supervised` 731,853 windows (31.3% GridSybil)

## GridSybil

28,087 attack windows (172 vehicles) vs 60,142 benign (525 vehicles); attack share 0.318.

### AUROC and window length

| Regime | Detector / score | AUROC | 95% CI | AUROC − length-only [CI] | Length-matched AUROC [CI] | 1-3 rows | >= 4 rows | 1 row | 2 rows | 3 rows | 4-16 rows | >= 17 rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control | length_only / length (deterministic) | 0.559 | [0.538, 0.579] | - | 0.500 [0.500, 0.500] | 0.521 | 0.538 | 0.500 | 0.500 | 0.500 | 0.538 | 0.488 |

### Average precision and operating points (thresholds from validation scores; alarm when score > threshold)

| Regime | Detector / score | AP | 95% CI | Primary threshold | recall / benign FPR @ benign_val_p95 | recall / benign FPR @ benign_val_p99 | recall / benign FPR @ own_val_p95 |
|---|---|---|---|---|---|---|---|
| control | length_only / length (deterministic) | 0.367 | [0.343, 0.394] | benign_val_p95 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |

### Sensitivity (AUROC on subsets of the same score files)

| Regime | Detector / score | group_1416_only [CI] | benign_without_GridSybil_0709 [CI] | benign_of_attack_family_folders [CI] |
|---|---|---|---|---|
| control | length_only / length (deterministic) | 0.580 [0.550, 0.611] | 0.561 [0.541, 0.582] | 0.560 [0.539, 0.581] |

## DataReplay

76,283 attack windows (173 vehicles) vs 60,142 benign (525 vehicles); attack share 0.559.

### AUROC and window length

| Regime | Detector / score | AUROC | 95% CI | AUROC − length-only [CI] | Length-matched AUROC [CI] | 1-3 rows | >= 4 rows | 1 row | 2 rows | 3 rows | 4-16 rows | >= 17 rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control | length_only / length (deterministic) | 0.952 | [0.948, 0.955] | - | 0.500 [0.500, 0.500] | 0.711 | 0.927 | 0.500 | 0.500 | 0.500 | 0.843 | n/a (3 att. windows, 2 vehicles) |

### Average precision and operating points (thresholds from validation scores; alarm when score > threshold)

| Regime | Detector / score | AP | 95% CI | Primary threshold | recall / benign FPR @ benign_val_p95 | recall / benign FPR @ benign_val_p99 | recall / benign FPR @ own_val_p95 |
|---|---|---|---|---|---|---|---|
| control | length_only / length (deterministic) | 0.934 | [0.922, 0.942] | benign_val_p95 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |

### Sensitivity (AUROC on subsets of the same score files)

| Regime | Detector / score | group_1416_only [CI] | benign_without_GridSybil_0709 [CI] | benign_of_attack_family_folders [CI] |
|---|---|---|---|---|
| control | length_only / length (deterministic) | 0.954 [0.945, 0.961] | 0.951 [0.947, 0.955] | 0.952 [0.948, 0.955] |

## DoSRandom

199,834 attack windows (173 vehicles) vs 60,142 benign (525 vehicles); attack share 0.769.

### AUROC and window length

| Regime | Detector / score | AUROC | 95% CI | AUROC − length-only [CI] | Length-matched AUROC [CI] | 1-3 rows | >= 4 rows | 1 row | 2 rows | 3 rows | 4-16 rows | >= 17 rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control | length_only / length (deterministic) | 0.967 | [0.965, 0.970] | - | 0.500 [0.500, 0.500] | 0.769 | n/a (523 att. windows, 13 vehicles) | 0.500 | 0.500 | 0.500 | n/a (523 att. windows, 13 vehicles) | n/a (0 att. windows, 0 vehicles) |

### Average precision and operating points (thresholds from validation scores; alarm when score > threshold)

| Regime | Detector / score | AP | 95% CI | Primary threshold | recall / benign FPR @ benign_val_p95 | recall / benign FPR @ benign_val_p99 | recall / benign FPR @ own_val_p95 |
|---|---|---|---|---|---|---|---|
| control | length_only / length (deterministic) | 0.981 | [0.979, 0.983] | benign_val_p95 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |

### Sensitivity (AUROC on subsets of the same score files)

| Regime | Detector / score | group_1416_only [CI] | benign_without_GridSybil_0709 [CI] | benign_of_attack_family_folders [CI] |
|---|---|---|---|---|
| control | length_only / length (deterministic) | 0.971 [0.964, 0.976] | 0.967 [0.965, 0.970] | 0.967 [0.964, 0.970] |

## DoSDisruptive

199,826 attack windows (173 vehicles) vs 60,142 benign (525 vehicles); attack share 0.769.

### AUROC and window length

| Regime | Detector / score | AUROC | 95% CI | AUROC − length-only [CI] | Length-matched AUROC [CI] | 1-3 rows | >= 4 rows | 1 row | 2 rows | 3 rows | 4-16 rows | >= 17 rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control | length_only / length (deterministic) | 0.967 | [0.965, 0.970] | - | 0.500 [0.500, 0.500] | 0.769 | n/a (524 att. windows, 14 vehicles) | 0.500 | 0.500 | 0.500 | n/a (524 att. windows, 14 vehicles) | n/a (0 att. windows, 0 vehicles) |

### Average precision and operating points (thresholds from validation scores; alarm when score > threshold)

| Regime | Detector / score | AP | 95% CI | Primary threshold | recall / benign FPR @ benign_val_p95 | recall / benign FPR @ benign_val_p99 | recall / benign FPR @ own_val_p95 |
|---|---|---|---|---|---|---|---|
| control | length_only / length (deterministic) | 0.981 | [0.979, 0.983] | benign_val_p95 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |

### Sensitivity (AUROC on subsets of the same score files)

| Regime | Detector / score | group_1416_only [CI] | benign_without_GridSybil_0709 [CI] | benign_of_attack_family_folders [CI] |
|---|---|---|---|---|
| control | length_only / length (deterministic) | 0.971 [0.965, 0.976] | 0.967 [0.965, 0.970] | 0.967 [0.964, 0.969] |
