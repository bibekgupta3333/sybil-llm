# `task/autoencoder_t64/`: unlabelled autoencoder on the T = 64 benign + GridSybil windows

The 1D-conv autoencoder of `task/encoder` (14 × 64 in, z = 16, **80,189 parameters**), trained on **all** train
windows of `src/data/encoder_input/benign_gridsybil/T64/` with benign and GridSybil pooled and labels never read:
the protocol of `task/autoencoder` (T = 48) at T = 64. It fills the gap next to the two T = 64 models of
`task/encoder` (benign-only autoencoder, supervised encoder).

**One seed, a side experiment. Not the thesis TimesNet model and not a thesis result (RULE 1).**

**Why a package of its own:** `task/encoder` has no setting that trains its autoencoder on all windows (`ae` mode
selects the benign windows with the labels, and its validation set, thresholds and Mahalanobis fit are benign-only).
This package adds no model and no split logic: it imports `task/encoder`'s `ConvAutoencoder`, `WindowStore` (and
reads its decoded cache `task/encoder/runs/cache/`, read-only), `VehicleSplit` and `Mahalanobis`, and
`task/autoencoder_overlap`'s `TrackScores`. `task/encoder` is unchanged.

**Protocol**

- Split: 90 / 10 train / test by sender vehicle on disk (D8′; no split change, RULE 3). Validation = 10% of the train
  sender vehicles from `task.encoder.data.VehicleSplit(0.1, seed 0)` over all train windows: the 518 validation
  vehicles (33,268 windows) of `task/encoder`'s supervised mode, checked identical (set and window ids). Its
  benign-only `ae` mode draws from benign vehicles only (362), a different set.
- Training (`train.py`): fit windows 304,733 (4,660 vehicles), inputs x + mask only, masked MSE on real rows, Adam
  1e-3, batch 256, ReduceLROnPlateau (0.5, patience 5), early stopping (patience 10, `min_delta` 0), max 300 epochs.
  It opens the train split only and reads no label (`label_reads: []`, `splits_opened: ["train"]` in `summary.json`).
- Validation (mixed, unlabelled): early stopping, LR schedule, thresholds = p95 / p99 of the validation scores.
- Scores: reconstruction error per window; Mahalanobis distance of z to a Gaussian fitted on the z of the fit windows
  (ridge 1e-6). Per track (one link, 94.5% single-window): max and mean.
- Test (`evaluate.py`): every test window is scored and the score files written, then the labels are read.
  GridSybil = positive, alarm = `score >= threshold`.

**Run (repo root)**

```
.venv-train/bin/python -m task.autoencoder_t64.train --seed 0          # runs/seed0/, about 18 min on MPS
.venv-train/bin/python -m task.autoencoder_t64.evaluate --run task/autoencoder_t64/runs/seed0   # test_*.npy + results.json
.venv-train/bin/python -m pytest task/autoencoder_t64/tests             # 13 tests, synthetic shards, CPU
.venv-train/bin/python scripts/run_notebook.py task/autoencoder_t64/autoencoder_T64.ipynb --timeout 1800
```

**Files:** `config.py` (AutoencoderT64Config) · `trainer.py` (Trainer: fit, early stopping, LR schedule, best
checkpoint, scoring) · `scoring.py` (track ids, ScoreFiles; re-exports Mahalanobis, TrackScores) · `train.py`
(Experiment, ValidationThresholds; CLI) · `evaluate.py` (Evaluation, Metrics; CLI) · `autoencoder_T64.ipynb` ·
`tests/` · `runs/seed0/` (gitignored): `config.json`, `env.json`, `history.jsonl`, `train.log`, `loss_curve.png`,
`best.pt`, `mahalanobis.npz`, `summary.json` (training facts, validation thresholds), `val_vehicles.npy`,
`results.json` (every test metric below) and the per-window score files `val_*.npy` / `test_*.npy` (`ids`,
`n_messages`, `recon_error`, `recon_error_feat`, `mahalanobis`; row k of each file belongs to `<split>_ids.npy[k]`).

## Results (`seed0`, one seed, no confidence interval)

Training: best validation masked MSE **0.1452** at epoch 161, early stopping at epoch 171, 7 LR halvings; 6.4 s per
epoch on MPS (about 18 min of computing; recorded wall-clock 124 min because epochs 40–47 stalled while the machine
was busy or asleep). GridSybil share of the training windows: **64.6%** of the fit windows (196,923 of 304,733;
train 64.5%, validation 63.1%).

Test: 38,426 windows (12,741 benign, 25,685 GridSybil; PR-AUC chance level 0.668). All numbers from
`runs/seed0/results.json`.

| Score (window level) | AUROC | PR-AUC | val p95: recall / benign FPR | val p99: recall / benign FPR | AUROC 1–3 rows | AUROC ≥ 4 rows |
|---|---|---|---|---|---|---|
| Reconstruction error | **0.482** | 0.660 | 0.045 / 0.049 | 0.010 / 0.007 | 0.361 | 0.512 |
| Mahalanobis on z | **0.352** | 0.599 | 0.047 / 0.057 | 0.006 / 0.004 | 0.317 | 0.362 |
| Length alone (−n_messages) | 0.535 | 0.699 | — | — | 0.505 | 0.516 |

Windows: 6,720 with 1–3 rows, 31,706 with ≥ 4 rows. Precision / F1 at val p95: reconstruction 0.651 / 0.084,
Mahalanobis 0.623 / 0.087.

| Per track (35,957 tracks, 0 mixed-label) | AUROC max | AUROC mean | PR-AUC max / mean | val p95 (max): recall / benign FPR |
|---|---|---|---|---|
| Reconstruction error | 0.481 | 0.480 | 0.656 / 0.654 | 0.045 / 0.049 |
| Mahalanobis on z | 0.340 | 0.335 | 0.586 / 0.580 | 0.047 / 0.067 |

Reconstruction AUROC in `task/encoder`'s length bins (≤ 3 / 4–16 / 17–48 / 49–64 rows): 0.361 / 0.473 / 0.558 /
0.515. Per-feature error AUROC: range 0.652, claimed_pos_y 0.624, log_dtau 0.609, …, claimed_vel_x 0.270,
claimed_vel_y 0.271 (GridSybil is rebuilt better on claimed velocity).

**Length alone.** The rule is fixed before any label is read: score = −n_messages (shorter = more anomalous).
`task/encoder` reports the same number (0.535) for the same direction, but it picked the direction on the test
labels (+n gives 0.465), and inside its length bins it reports the better direction per bin.

**Context only: the T = 64 benign-only autoencoder of `task/encoder`** (same architecture and test windows; from
`task/encoder/runs/evaluation/results.json`). Its validation set and thresholds are benign-only, its patience is
5 / 3 instead of 10 / 5 and it has 3 seeds, so this is not a controlled comparison.

| Model | Reconstruction AUROC | PR-AUC | val p95: recall / benign FPR | Mahalanobis AUROC |
|---|---|---|---|---|
| This run: unlabelled, benign + GridSybil (seed 0) | 0.482 | 0.660 | 0.045 / 0.049 | 0.352 |
| `task/encoder` benign-only AE, seed 0 | 0.655 | 0.801 | 0.194 / 0.050 | 0.333 |
| `task/encoder` benign-only AE, mean ± std of 3 seeds | 0.655 ± 0.002 | 0.802 | 0.199 / 0.052 | 0.319 ± 0.016 |

**Reading.** The autoencoder learned to reconstruct, but neither score separates GridSybil from benign: both are
below the length rule, reconstruction error is at chance and the Mahalanobis distance is below chance. GridSybil is
about two thirds of the unlabelled training data, so the model and the Gaussian treat it as the normal case. The
T = 48 unlabelled run (`task/autoencoder`, AUROC 0.484) shows the same.

**Limits.** One seed; no confidence interval; GridSybil only; windows of one vehicle are not independent; track ids
contain the true sender (F13).
