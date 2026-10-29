# `task/autoencoder_st/`: supervised autoencoder on the T = 48 windows (T48_tvt)

The `task/autoencoder_overlap` ConvAutoencoder (input 14 × 48 = 13 features + mask, z = 16, 71,741 parameters) plus a
classifier head on z (Linear 16→32 → ReLU → Dropout 0.2 → Linear 32→1, 577 parameters; 72,318 total), trained on
`src/data/encoder_input/benign_gridsybil/T48_tvt/` (no overlap; train / val / test = 80 / 10 / 10 by sender vehicle,
fixed on disk). Loss = BCEWithLogits with `pos_weight` = n_benign / n_gridsybil of train (0.5557) + λ · masked MSE on
real rows (λ = `recon_weight` = 1). Inputs are x + mask only. Val drives the LR schedule, early stopping on val AUROC
(min_delta 1e-4 absolute, patience 10), `best.pt` and both alarm thresholds (max F1 on val; benign FPR 5% on val);
`train.py` never reads test. Not the thesis TimesNet model.

**Run (repo root):**

- `npm run aest:train` trains seed 0 and writes `runs/seed0/` (about 6 min on MPS; `--resume` continues from `last.pt`)
- `npm run aest:test` runs the pytest suite (synthetic shards, CPU)
- `npm run aest:notebook` executes `autoencoder_st_T48.ipynb` (reads the newest run, about 15 s)

**Files:** `config.py` · `data.py` · `model.py` (ConvAutoencoder + ClassifierHead) · `metrics.py` (metrics, val
thresholds, TrackScores) · `trainer.py` · `train.py` (CLI) · notebook · `tests/` · `runs/<run_id>/` (gitignored).

**Results (`seed0`, one seed, no CI):** best val AUROC 0.9268 at epoch 44; early stopping at epoch 54 after 3 LR
halvings, 5.5 min on MPS. Test (40,469 windows / 35,957 tracks, GridSybil share 0.669, no mixed-label track; labels
read after scoring): AUROC **0.929**, PR-AUC 0.971; at the val max-F1 threshold (0.351) precision 0.971, recall 0.839,
F1 0.900, benign FPR 0.051; at the val benign-FPR-5% threshold (0.393) recall 0.833, F1 0.898, benign FPR 0.044.
Per track (max / mean): AUROC 0.928 / 0.929. Length alone: 0.530; windows with 1-3 real rows: 0.917.

**Reading:** with labels, the same small autoencoder separates benign from GridSybil (the unsupervised versions reach
0.484), at about the level of the `task/encoder` supervised T64 model (0.928, 3 seeds; different setup, context only).
The reconstruction error itself still does not detect GridSybil (AUROC 0.478); the head reads the class from z. Val BCE
plateaued from epoch 20 while train BCE kept falling. Trained and tested on GridSybil only: says nothing about unseen
attacks. Track ids hold the true sender (privileged grouping, F13).
