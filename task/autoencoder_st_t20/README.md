# `task/autoencoder_st_t20/`: supervised autoencoder on the T = 20 windows (all/T20_benign_grid)

The `task/autoencoder_st_t32` supervised baseline at T = 20: the ConvAutoencoder (input 14 × 20 = 13 features + mask,
z = 16) plus the same classifier head on z (Linear 16→32 → ReLU → Dropout 0.2 → Linear 32→1, 577 parameters), trained on
`src/data/encoder_input/all/T20_benign_grid/` (benign of all 8 folders + GridSybil; 85 / 7.5 / 7.5 by vehicle key,
the all/T32_benign_grid split, fixed on disk; no split change). Loss = BCEWithLogits (`pos_weight` = n_benign /
n_gridsybil of train) + 1.0 · masked MSE. Inputs are x + mask only. Val drives the LR schedule, early stopping on val
AUROC, `best.pt` and both alarm thresholds (max F1 on val; benign FPR 5% on val); `train.py` never reads test; the test
evaluation is in the notebook, as for T = 32. **A side experiment, not a thesis result, not the thesis TimesNet model.**
No run yet.

This package also holds the supervised model on the depth-configurable ConvAutoencoder, and the parametrised
`train.Experiment` and Trainer that the T = 10 / 5 supervised packages subclass.

**What differs from T = 32:** only the input folder, `seq_len` = 20 and `n_down` = 2: encoder strides 1 / 2 / 2, decoder ConvTranspose k4 s2 / k4 s2 — exactly the T = 32 layers; latent length 5.
**57,534 trainable parameters** = autoencoder 56,957 + head 577 (T = 32: 63,870). Every other setting is
`task/autoencoder_st_t32`'s (config subclassed; checked by a test): Adam 1e-3, weight decay 1e-5, batch 256, max 100
epochs, grad clip 1.0, early stopping patience 10 (min_delta 1e-4 absolute AUROC), ReduceLROnPlateau 0.5 / 5 on val
AUROC, target FPR 0.05, optional `--log-feature-error`. Additions: `--workers` on the command line; `train.py` refuses
windows whose length differs from `seq_len`; the notebook's length-group table uses groups 1–3 / 4–(T−1) / T / 4–T.

**Run (repo root):**

- `npm run aest20:train` (seed 0; `SEED=k` for seed k) → `runs/seed<k>/` (config, env, history, train.log, curves.png,
  best / last.pt, summary.json; gitignored); decoded shards cached in `cache/`. Same as
  `.venv-train/bin/python -m task.autoencoder_st_t20.train --seed <k>` (`--resume`, `--log-feature-error`, `--workers N`).
- `npm run aest20:test`: pytest, synthetic shards, CPU. `npm run aest20:notebook`: executes `autoencoder_st_T20.ipynb`
  on the newest run (analysis cell is a placeholder until a run exists).
- `SEED=k npm run hf:download:baselines:st-t20`: fetch `baselines/autoencoder_st_t20/seed<k>` from the private HF repo.
