# `task/autoencoder_overlap/`: unsupervised autoencoder on the T = 48 stride-24 overlap windows

The `task/autoencoder` ConvAutoencoder (input 14 × 48 = 13 features + mask, z = 16, masked MSE on real rows, 71,741
parameters) trained on `src/data/encoder_input/benign_gridsybil/T48_tvt_s24/`: links longer than 48 messages are cut
into windows at stride 24 (+ one end-aligned window), and the train / val / test split (80 / 10 / 10 by sender
vehicle) is fixed on disk. Benign and GridSybil train windows are both treated as normal; labels are never read in
training, model selection or thresholds. Val drives early stopping (min_delta 1e-4 relative, patience 10), the LR
schedule and the alarm thresholds (val p95 / p99); test is used only for the final evaluation. Not the thesis
TimesNet model.

**Run (repo root):**

- `npm run aeo:train` trains seed 0 until early stopping and writes `runs/seed0/` (about 15 min on MPS)
- `npm run aeo:test` runs the pytest suite (synthetic shards, CPU)
- `npm run aeo:notebook` executes `autoencoder_overlap_T48.ipynb` (reads the newest run, about 15 s)

**Files:** `config.py` · `data.py` (WindowDataset, `track_id`) · `model.py` · `trainer.py` (Trainer, EarlyStopping,
gradient norm + LR per epoch) · `scoring.py` (Mahalanobis on z fitted on train, TrackScores: max / mean per link,
quantile thresholds) · `train.py` (CLI) · `autoencoder_overlap_T48.ipynb` · `tests/` · `runs/<run_id>/` (gitignored:
config.json, env.json, history.jsonl, train.log, loss_curve.png, best.pt, mahalanobis.npz, summary.json).

**Results (`seed0`, one seed):** best val masked MSE 0.1427 at epoch 139, early stopping at epoch 149 after 8 LR
halvings, 14 min on MPS. Test: 42,653 windows / 35,957 tracks (GridSybil share 0.673 / 0.665; no track with mixed
labels), labels read only after scoring. Reconstruction error: AUROC **0.484** per window (PR-AUC 0.664), 0.482 /
0.481 per track (max / mean); at val p95 recall 0.043, benign FPR 0.050, F1 0.080. Mahalanobis on z: AUROC **0.337**
per window, 0.307 / 0.302 per track; at val p95 recall 0.044, benign FPR 0.068.

**Reading:** neither score detects GridSybil. GridSybil is 64.5% of the train windows, so its z sits at the centre of
the fitted Gaussian and benign windows are the outliers (Mahalanobis below chance). Overlap did not change the
reconstruction AUROC of the no-overlap run (0.484; different cut and train / val vehicles, so context only).
