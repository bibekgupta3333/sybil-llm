# `task/autoencoder_t20/`: unlabelled autoencoder on the T = 20 benign + GridSybil windows

The `task/autoencoder_t32` unlabelled cell of the baseline grid at T = 20: the ConvAutoencoder (input 14 × 20, z = 16,
masked MSE on real rows) trained on **all train windows pooled, benign and GridSybil mixed, no label read**, on
`src/data/encoder_input/all/T20_benign_grid/` (benign of all 8 folders + GridSybil; 85 / 7.5 / 7.5 by vehicle key,
the all/T32_benign_grid split, fixed on disk; no split change). Train fits the weights; val (mixed, unlabelled) drives
early stopping, the LR schedule and the alarm thresholds (val p95 / p99); the Mahalanobis model is fitted on train z;
`train.py` never opens `test`. **Labels are unread in training**: the reader opens feature shards only, the command line
installs `LabelFileGuard` (audit hook that raises on any `*_info.json`) in the process and every decoding worker, the
cache holds x / mask / ids only; labels are opened by `evaluate.py` after `scores/test_*.npy` is written. **A side
experiment, not a thesis result, not the thesis TimesNet model.** No run yet.

This package also holds the parametrised `train.Experiment`, `evaluate.TestEvaluation` and Trainer that the T = 10 / 5
unlabelled packages subclass; the model is `task/autoencoder_benign_t20`'s depth-configurable ConvAutoencoder.

**What differs from T = 32:** only the input folder, `seq_len` = 20 and `n_down` = 2: encoder strides 1 / 2 / 2, decoder ConvTranspose k4 s2 / k4 s2 — exactly the T = 32 layers; latent length 5.
**56,957 trainable parameters** (T = 32: 63,293). Every other setting is `task/autoencoder_t32`'s (config subclassed; checked
by a test). Additions: `train.py` / `evaluate.py` refuse windows whose length differs from `seq_len`; the context block
of `evaluation.json` reads the T = 20 siblings' notebooks (`autoencoder_st_t20`, `autoencoder_benign_t20`) and is empty
until they are executed.

**Run (repo root):**

- `npm run ae20:train` (seed 0; `SEED=k` for seed k) → `runs/seed<k>/` (config, env, history, train.log, curves.png,
  best / last.pt, mahalanobis.npz, `scores/val_*.npy`, summary.json; gitignored); feature shards cached in `cache/`.
- `npm run ae20:eval` (`SEED=k`) → `scores/test_{ids,reconstruction,mahalanobis,n_messages,z}.npy`, then
  `evaluation.json`.
- `npm run ae20:test`: pytest, synthetic shards, CPU. `npm run ae20:notebook`: executes `autoencoder_T20.ipynb` on the
  newest run with an `evaluation.json` (analysis cell is a placeholder until a run exists).
- `SEED=k npm run hf:download:baselines:ae20`: fetch `baselines/autoencoder_t20/seed<k>` from the private HF repo.
