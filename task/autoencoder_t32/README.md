# `task/autoencoder_t32/`: unlabelled autoencoder on the T = 32 benign + GridSybil windows

The unlabelled cell of the baseline grid at T = 32: the `task/autoencoder_benign_t32` ConvAutoencoder (imported
unchanged; input 14 × 32 = 13 features + mask, z = 16, masked MSE on real rows, 63,293 parameters; same Adam /
ReduceLROnPlateau / early-stopping settings) trained on **all train windows pooled, benign and GridSybil mixed, no
label read** — the protocol of `task/autoencoder_overlap` — on `src/data/encoder_input/all/T32_benign_grid/` (benign of
all 8 folders + GridSybil; on-disk 85 / 7.5 / 7.5 split by vehicle key, unchanged). Train fits the weights; val
(mixed, unlabelled) drives early stopping, the LR schedule and the alarm thresholds (val p95 / p99); the Mahalanobis
model is fitted on train z; `train.py` never opens `test`. Scores: reconstruction error and Mahalanobis distance of z,
per window and per track (max / mean). **One seed, a side experiment, not a thesis result (RULE 1).** Not the thesis
TimesNet model.

**Labels are unread in training:** `data.py` (`UnlabelledWindows`, attributes x, mask, ids, n_messages only) opens the
feature shards alone; `train.py` installs `LabelFileGuard` (an audit hook that raises on opening any `*_info.json`) in
its process and in every decoding worker; the cache holds x / mask / ids only. Labels live in `labels.py`, imported
only by `evaluate.py` and the notebook, and are opened after `scores/test_*.npy` is written.

**Run (repo root):**

- `.venv-train/bin/python -m task.autoencoder_t32.train --seed 0` → `runs/seed0/` (config, env, history, train.log,
  curves.png, best / last.pt, mahalanobis.npz, `scores/val_*.npy`, summary.json); feature shards cached in `cache/`
- `.venv-train/bin/python -m task.autoencoder_t32.evaluate --run task/autoencoder_t32/runs/seed0` →
  `scores/test_*.npy`, then `evaluation.json` (every number below)
- `.venv-train/bin/python -m pytest task/autoencoder_t32/tests` (17 tests, synthetic shards, CPU)
- `.venv-train/bin/python scripts/run_notebook.py task/autoencoder_t32/autoencoder_T32.ipynb --timeout 3600`
  (about 10 s with the cache)

Score files: `runs/seed0/scores/<split>_{ids,reconstruction,mahalanobis,n_messages,z}.npy`, all in the order of
`<split>_ids.npy`.

**Results (`seed0`, one seed, no CI):** train GridSybil share 0.313 (val 0.300, test 0.323; counted after scoring).
Best val masked MSE 0.1078 at epoch 206, early stopping at 216 after 9 LR halvings; 152 min wall-clock on MPS shared
with two other jobs (5 stalled epochs took 104 min; median epoch 10.5 s). Mahalanobis: cond 46.9, no shrinkage. Test,
GridSybil vs benign, labels read after scoring, thresholds from val:

| | all test windows (67,100; GridSybil 0.323) | GridSybil folders only (33,614; GridSybil 0.644) |
|---|---|---|
| reconstruction: window AUROC / PR-AUC | **0.610** / 0.436 | **0.613** / 0.741 |
| reconstruction: recall / benign FPR at val p95 | 0.108 / 0.040 | 0.108 / 0.040 |
| reconstruction: recall / benign FPR at val p99 | 0.027 / 0.007 | 0.027 / 0.006 |
| reconstruction: AUROC 1–3 rows / ≥ 4 rows | 0.496 / 0.649 | 0.491 / 0.654 |
| reconstruction: AUROC 0709 / 1416 | 0.611 / 0.605 | 0.614 / 0.606 |
| reconstruction: track max / mean AUROC | 0.595 / 0.599 (54,102 tracks) | 0.599 / 0.604 (27,348 tracks) |
| Mahalanobis: window AUROC / PR-AUC | **0.398** / 0.331 | **0.396** / 0.629 |
| Mahalanobis: recall / benign FPR at val p95 | 0.097 / 0.053 | 0.097 / 0.053 |
| Mahalanobis: recall / benign FPR at val p99 | 0.045 / 0.003 | 0.045 / 0.003 |
| Mahalanobis: AUROC 1–3 rows / ≥ 4 rows | 0.385 / 0.414 | 0.382 / 0.413 |
| Mahalanobis: AUROC 0709 / 1416 | 0.391 / 0.422 | 0.389 / 0.424 |
| Mahalanobis: track max / mean AUROC | 0.362 / 0.362 | 0.361 / 0.361 |
| length alone, fixed rule score = −n_messages | 0.560 | 0.564 |

Benign FPR at val p95 per source folder (reconstruction): 0.033–0.035 in the four 0709 folders, 0.063–0.065 in the
four 1416 folders (Mahalanobis: 0.052–0.054 everywhere); it follows the time window, not the family. The length rule
is fixed in advance (shorter = more anomalous, as in `task/autoencoder_benign_t32`); the `task/autoencoder_st_t32`
convention (best direction, max(a, 1 − a)) gives the same 0.560 / 0.564 here. No mixed-label track.

**Context (one seed each, not a ranking):** `task/autoencoder_st_t32` (GridSybil labels, same data, split and test
windows) AUROC 0.926, its reconstruction error alone 0.598; `task/autoencoder_benign_t32` (benign-only training, **a
different split and test set**: `all/T32_benign`, 80 / 10 / 10) 0.736, Mahalanobis 0.510. Both read from the result
cells of the siblings' executed notebooks (their `summary.json` hold training / val fields only).

**Reading:** with 31% GridSybil left unlabelled in the training pool, the reconstruction error is a weak detector
(0.610 vs 0.560 for length alone), all of it from windows with ≥ 4 rows; Mahalanobis on z is below chance because
GridSybil sits nearer the centre of z than benign traffic (as in `task/autoencoder_overlap`; the sign is not flipped).
Restricting to the GridSybil folders changes the AUROCs by ≤ 0.003, so the repeated benign copies (F1) do not drive
the ranking. Caveats: one seed; F14 (GridSybil_0709 near-copies may straddle the split, not measured); F1 (window
counts not independent); F13 (track ids hold the true sender, aggregation only); GridSybil only. Details: notebook
section 7.
