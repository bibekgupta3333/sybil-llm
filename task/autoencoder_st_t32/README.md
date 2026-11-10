# `task/autoencoder_st_t32/`: supervised autoencoder on the T = 32 windows (all/T32_benign_grid)

The `task/autoencoder_benign_t32` ConvAutoencoder (input 14 × 32 = 13 features + mask, z = 16, 63,293 parameters) plus
a classifier head on z (Linear 16→32 → ReLU → Dropout 0.2 → Linear 32→1, 577 parameters; 63,870 total), trained on
`src/data/encoder_input/all/T32_benign_grid/`: benign windows from all 8 scenario folders + GridSybil attack windows,
train / val / test = 85 / 7.5 / 7.5 by vehicle key `<group>:<vehicle>`, fixed on disk (⚠ RULE 3 new split). Loss =
BCEWithLogits with `pos_weight` = n_benign / n_gridsybil of train (2.198) + 1.0 · masked MSE. Inputs are x + mask
only. Val drives the LR schedule, early stopping on val AUROC (patience 10), `best.pt` and both alarm thresholds (max
F1 on val; benign FPR 5% on val); `train.py` never reads test. Not the thesis TimesNet model.

**Run (repo root):**

- `npm run aest32:train` trains seed 0 and writes `runs/seed0/` (about 6 min on MPS; `--resume` continues from `last.pt`)
- `npm run aest32:test` runs the pytest suite (synthetic shards, CPU)
- `npm run aest32:notebook` executes `autoencoder_st_T32.ipynb` (reads the newest run; seconds with `cache/`)

**Results (`seed0`, one seed, no CI):** val AUROC 0.894 at epoch 1, best 0.9220 at epoch 17; early stop at epoch 27
after 1 LR halving, 5.6 min on MPS. Test (67,100 windows / 54,102 tracks, GridSybil share 0.323, no mixed-label track;
labels read after scoring): AUROC **0.926**, PR-AUC 0.925; at the val max-F1 threshold (0.596) precision 0.972, recall
0.819, F1 0.889, benign FPR 0.011; at the val benign-FPR-5% threshold (0.296) recall 0.839, F1 0.865, benign FPR 0.048.
Per track (max / mean) AUROC 0.925 / 0.927. Length alone 0.560; 1-3-row windows 0.915, ≥ 4 rows 0.924. Per period
0709 / 1416: 0.927 / 0.925. Benign FPR at max-F1 is 0.010-0.015 in every folder (DataReplay / DoS benign windows are
not flagged more often; expected, the same benign traffic recurs in every family, F1). Reconstruction error alone: 0.598.

**Reading:** about the level of `task/autoencoder_st` on T48_tvt (0.929; different setup, context only), far above the
one-class T32 autoencoder on GridSybil (0.736). The plausibility baseline's GridSybil F1 0.727 / 0.692 is per message,
not comparable. Caveats: one seed; F1 (benign traffic repeats across the 4 families, so extra folders add copies);
F14 (GridSybil_0709 near-copies may straddle the split); GridSybil only, says nothing about unseen attacks; track ids
hold the true sender (F13).
