# `task/autoencoder_benign_t20/`: one-class autoencoder on the T = 20 benign windows of all 8 folders

The `task/autoencoder_benign_t32` one-class baseline at T = 20: ConvAutoencoder (input 14 × 20 = 13 features + mask,
z = 16, masked MSE on real rows) trained on **benign windows only** from `src/data/encoder_input/all/T20_benign/`
(80 / 10 / 10 by sender vehicle over all 8 folders, the all/T32_benign split, fixed on disk; no split change). Train fits
the weights; benign val drives early stopping, the LR schedule and the alarm thresholds (val p95 / p99); `train.py`
never opens `test` or `test_<attack>`. Scores: reconstruction error and Mahalanobis distance of z (fitted on benign
train z), per window and per track (max / mean); the test evaluation (four attack splits vs benign test, length-only
reference) is in the notebook, as for T = 32. **A side experiment, not a thesis result, not the thesis TimesNet model.**
No run yet.

This package also holds the depth-configurable `ConvAutoencoder` (`model.py`, setting `n_down`) and the
parametrised `train.Experiment` that the T = 10 / 5 one-class packages subclass.

**What differs from T = 32:** only the input folder, `seq_len` = 20 and the model depth `n_down` = 2 (number of
stride-2 stages; T = 32 has 2, which needs T divisible by 4): encoder strides 1 / 2 / 2, decoder ConvTranspose k4 s2 / k4 s2 — exactly the T = 32 layers. Channels (32 / 64 / 64), kernels, BatchNorm + ReLU,
z = 16 and the loss are unchanged; latent length 20 / 2^2 = 5. **56,957 trainable parameters** (T = 32: 63,293). Every
training setting is the T = 32 config's (subclassed, not copied; checked by a test): Adam 1e-3, weight decay 1e-5, batch
256, grad clip 1.0, ReduceLROnPlateau 0.5 / patience 5, early stopping patience 10 with relative min_delta 1e-4, max 300
epochs, thresholds p95 / p99, Mahalanobis (cond > 1e6 → shrinkage 0.01), optional `--log-feature-error`. Additions:
`--workers` on the command line, and `train.py` refuses windows whose length differs from `seq_len`.

**Run (repo root):**

- `npm run aeb20:train` (seed 0; `SEED=k npm run aeb20:train` for seed k) → `runs/seed<k>/` (config, env, history,
  train.log, curves.png, best / last.pt, mahalanobis.npz, summary.json; gitignored); decoded shards cached in `cache/`
  (gitignored). Same as `.venv-train/bin/python -m task.autoencoder_benign_t20.train --seed <k>` (`--resume`,
  `--log-feature-error`, `--workers N`, `--max-shards N --epochs N --device cpu` for a quick check).
- `npm run aeb20:test`: pytest, synthetic shards, CPU.
- `npm run aeb20:notebook`: executes `autoencoder_benign_T20.ipynb` on the newest run (its analysis cell is a
  placeholder until a run exists).
- `SEED=k npm run hf:download:baselines:benign-t20`: fetch `baselines/autoencoder_benign_t20/seed<k>` from the private
  HF model repo (once uploaded).
