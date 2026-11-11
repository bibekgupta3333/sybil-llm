# `task/autoencoder_benign_t48/`: one-class autoencoder on the T = 48 benign windows (benign + GridSybil input)

The one-class protocol of `task/autoencoder_benign_t32` at T = 48: the ConvAutoencoder of `task/autoencoder` (imported;
input 14 × 48 = 13 features + mask, z = 16, masked MSE on real rows, 71,741 parameters) trained on **benign windows
only** from `src/data/encoder_input/benign_gridsybil/T48_benign/`, using the 80 / 10 / 10 split by sender vehicle fixed
on disk (no split change). Train fits the weights; benign val drives early stopping, the LR schedule and the alarm
thresholds (val p95 / p99); `train.py` never opens `test` or `test_gridsybil`. Scores: reconstruction error and
Mahalanobis distance of z (fitted on benign train z), per window and per track (max / mean). `evaluate.py` scores val
and both test splits, writes the per-window score files, and only then reads labels.

**A side experiment, one seed. Not a thesis result and not the thesis TimesNet model** (RULE 1: numbers as produced).

**Run (repo root):**

- `.venv-train/bin/python -m task.autoencoder_benign_t48.train --seed 0` → `runs/seed0/` (decoded shards cached in `cache/`)
- `.venv-train/bin/python -m task.autoencoder_benign_t48.evaluate --seed 0` → `runs/seed0/scores/` + `test_metrics.json`
  (`--from-scores` rebuilds the metrics from the score files and `summary.json` alone)
- `.venv-train/bin/python -m pytest task/autoencoder_benign_t48/tests` (CPU, synthetic shards)
- `.venv-train/bin/python scripts/run_notebook.py task/autoencoder_benign_t48/autoencoder_benign_T48.ipynb --timeout 3600`

Run folder (gitignored): config.json, env.json, history.jsonl, train.log, curves.png, best / last.pt, mahalanobis.npz,
summary.json (training facts, val thresholds), test_metrics.json, and `scores/<split>_{ids,n_messages,reconstruction,
mahalanobis}.npy` for val, test and test_gridsybil (row-aligned with the window ids).

**Settings:** identical to `task/autoencoder_benign_t32` (Adam 1e-3, weight decay 1e-5, batch 256, grad clip 1.0,
ReduceLROnPlateau 0.5 / patience 5, early stopping patience 10 with relative min_delta 1e-4, max 300 epochs, z = 16,
seed 0; the training loop is the sibling's `Trainer`, subclassed). Differences: `seq_len` 48 instead of 32 (hence
71,741 instead of 63,293 parameters), the input folder, and one attack test split instead of four.

**Results (`seed0`, one seed):** best benign val masked MSE 0.1477 at epoch 143, early stopping at epoch 153 after 6 LR
halvings; 21.0 min wall-clock on MPS, of which about 13 min are three stalled epochs (147, 149, 153) on a shared
machine (median epoch 2.6 s). Test = 13,405 benign windows (`test`) vs 27,064 GridSybil windows (`test_gridsybil`) of
the test vehicles, labels read only after scoring; positive = GridSybil, chance for PR-AUC 0.669. All numbers are in
`runs/seed0/test_metrics.json` (training facts in `summary.json`).

| GridSybil vs benign test | AUROC | PR-AUC | recall / benign FPR at val p95 | recall / benign FPR at val p99 |
|---|---|---|---|---|
| reconstruction error, window | **0.689** | 0.829 | 0.243 / 0.047 | 0.154 / 0.008 |
| reconstruction error, track max | 0.678 | 0.815 | 0.213 / 0.048 | 0.122 / 0.008 |
| reconstruction error, track mean | 0.681 | 0.818 | 0.224 / 0.049 | 0.124 / 0.009 |
| Mahalanobis on z, window | 0.437 | 0.655 | 0.062 / 0.044 | 0.017 / 0.009 |
| Mahalanobis on z, track max | 0.430 | 0.643 | 0.058 / 0.044 | 0.016 / 0.009 |
| Mahalanobis on z, track mean | 0.427 | 0.641 | 0.057 / 0.044 | 0.013 / 0.008 |
| window length alone (reference) | 0.530 | 0.697 | | |

| window AUROC by length | 1-3 rows (1,846 benign / 5,003 GridSybil) | ≥ 4 rows (11,559 / 22,061) |
|---|---|---|
| reconstruction error | 0.599 | 0.720 |
| Mahalanobis on z | 0.406 | 0.453 |

Length alone uses the fixed rule score = −n_messages (shorter = more anomalous), set before any test label; the sibling
package uses the same rule, so its convention gives the same 0.530.

**Reading:** the reconstruction error separates part of GridSybil (errors in log-Δτ and range) and beats window length
alone; per-track aggregation does not help; Mahalanobis on z is below chance (GridSybil windows sit nearer the centre
of the benign z than benign test windows; the sign is not flipped, that would be a choice made on test labels). The
T = 32 sibling reports 0.736 on GridSybil, but on a different input (benign of all 8 folders, other split, benign test
set and normalisation), so the gap is not attributable to T. Caveats: one seed, GridSybil only, track ids hold the true
sender (F13). Details: notebook section 7.
