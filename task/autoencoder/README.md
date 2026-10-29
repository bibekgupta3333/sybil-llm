# `task/autoencoder/`: unsupervised autoencoder on the T = 48 benign + GridSybil windows

A small 1D-conv autoencoder (encoder + decoder, no classifier head) that rebuilds 48 × 13 message windows. It is
trained on **all** train windows of `src/data/encoder_input/benign_gridsybil/T48/` (benign and GridSybil alike); labels
are never used for training or model selection. 10% of the train sender vehicles (seed 0) are held out as validation
for early stopping and the LR schedule. Not the thesis TimesNet model.

**Run (repo root):**

- `npm run ae48:train` trains seed 0 until early stopping and writes `runs/seed0/` (about 15 min on MPS)
- `npm run ae48:test` runs the pytest suite (synthetic shards, CPU)
- `npm run ae48:notebook` executes `autoencoder_T48.ipynb` (reads the newest run, about 15 s)

**Files:** `config.py` (AutoencoderConfig) · `data.py` (WindowDataset, vehicle split) · `model.py` (ConvAutoencoder:
input 14 × 48 = 13 features + mask, z = 16, masked MSE on real rows) · `trainer.py` (Trainer: fit, early stopping,
ReduceLROnPlateau, best checkpoint, scoring) · `train.py` (CLI) · `autoencoder_T48.ipynb` (input stats, model,
training record, reconstructions, test, analysis) · `tests/` · `runs/<run_id>/` (gitignored: config.json, env.json,
history.jsonl, train.log, loss_curve.png, best.pt, summary.json).

**Results (`seed0`, one seed):** 71,741 parameters; best validation masked MSE 0.140 at epoch 170, early stopping at
epoch 180 after 9 LR halvings, 15 min on MPS. On the test split (13,405 benign, 27,064 GridSybil), with labels read
only after scoring: AUROC **0.484**, PR-AUC 0.661 (chance 0.669), length-only AUROC 0.530. At the 95th percentile of
validation errors: benign FPR 0.049, recall 0.046, precision 0.654, F1 0.087.

**Reading:** the autoencoder learned to reconstruct, but its error does not separate GridSybil from benign. Training
treated GridSybil (64% of train windows) as normal, so ghosts are rebuilt about as well as real cars (better on
claimed velocity, worse on range and claimed position). The one-class T = 64 run in `task/encoder` (benign-only
training) reached AUROC 0.655; T and training set differ, so this is context, not a fair comparison.
