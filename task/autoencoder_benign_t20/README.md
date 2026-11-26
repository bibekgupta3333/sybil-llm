# `task/autoencoder_benign_t20/`: one-class autoencoder on the T = 20 benign windows of all 8 folders

The `task/autoencoder_benign_t32` one-class baseline at T = 20: ConvAutoencoder (input 14 × 20 = 13 features + mask,
z = 16, masked MSE on real rows) trained on **benign windows only** from `src/data/encoder_input/all/T20_benign/`
(80 / 10 / 10 by sender vehicle over all 8 folders, the all/T32_benign split, fixed on disk; no split change). Train fits
the weights; benign val drives early stopping, the LR schedule and the alarm thresholds (val p95 / p99); `train.py`
never opens `test` or `test_<attack>`. Scores: reconstruction error and Mahalanobis distance of z (fitted on benign
train z), per window and per track (max / mean); the test evaluation (four attack splits vs benign test, length-only
reference) is in the notebook, as for T = 32. **A side experiment, not a thesis result, not the thesis TimesNet model.**
Seeds 0–3 trained and evaluated (results below).

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
  HF model repo.

## Results (seeds 0–3)

Test window AUROC vs benign test (`all/T20_benign` `test`), mean ± sample sd over seeds 0–3, from
`runs/seed<k>/test_results.json` (written by the package notebook executed per seed, copy
`runs/seed<k>/autoencoder_benign_T20.executed.ipynb`; values rounded to 3 dp in the file, means taken over those).
Length only = fixed rule −n_messages (identical for every seed). Recall at the benign val p95 threshold (benign test FPR).

| Attack | Reconstruction | Mahalanobis | Length only | Recon recall @ val p95 (benign FPR) |
|---|---|---|---|---|
| GridSybil | 0.756 ± 0.010 | 0.540 ± 0.029 | 0.547 | 0.354 ± 0.019 (0.049 ± 0.002) |
| DataReplay | 0.400 ± 0.010 | 0.449 ± 0.037 | 0.951 | 0.060 ± 0.008 (0.049 ± 0.002) |
| DoSRandom | 0.978 ± 0.004 | 0.826 ± 0.026 | 0.967 | 0.899 ± 0.018 (0.049 ± 0.002) |
| DoSDisruptive | 0.622 ± 0.011 | 0.622 ± 0.023 | 0.967 | 0.345 ± 0.024 (0.049 ± 0.002) |

GridSybil reconstruction per seed: 0.754 / 0.771 / 0.749 / 0.752; best epochs 232 / 262 / 214 / 190; 56,957 parameters.
DataReplay / DoS rows: every score is at or below length only except DoSRandom reconstruction, which is only
slightly above it (F11: most DataReplay / DoS windows have 1–3 rows); not length-matched here.

Side experiment, not a thesis result, not the thesis TimesNet model; seeds 0–3 on EC2 (g4dn T4); independently judged 2026-11-26 (56 PASS + 4 PASS-WITH-NOTES, 0 FAIL).
