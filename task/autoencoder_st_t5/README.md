# `task/autoencoder_st_t5/`: supervised autoencoder on the T = 5 windows (all/T5_benign_grid)

The `task/autoencoder_st_t32` supervised baseline at T = 5: the ConvAutoencoder (input 14 × 5 = 13 features + mask,
z = 16) plus the same classifier head on z (Linear 16→32 → ReLU → Dropout 0.2 → Linear 32→1, 577 parameters), trained on
`src/data/encoder_input/all/T5_benign_grid/` (benign of all 8 folders + GridSybil; 85 / 7.5 / 7.5 by vehicle key,
the all/T32_benign_grid split, fixed on disk; no split change). Loss = BCEWithLogits (`pos_weight` = n_benign /
n_gridsybil of train) + 1.0 · masked MSE. Inputs are x + mask only. Val drives the LR schedule, early stopping on val
AUROC, `best.pt` and both alarm thresholds (max F1 on val; benign FPR 5% on val); `train.py` never reads test; the test
evaluation is in the notebook, as for T = 32. **A side experiment, not a thesis result, not the thesis TimesNet model.**
Seeds 0–3 trained and evaluated (results below).

Thin package: config (subclass of `task/autoencoder_st_t20`'s), Trainer / Experiment subclasses that only set the
config class, plot title and parameter count; model, reader, metrics and the training loop are imported.

**What differs from T = 32:** only the input folder, `seq_len` = 5 and `n_down` = 0: encoder strides 1 / 1 / 1, decoder ConvTranspose k3 s1 / k3 s1 (no down- or upsampling); latent length 5.
**51,390 trainable parameters** = autoencoder 50,813 + head 577 (T = 32: 63,870). Every other setting is
`task/autoencoder_st_t32`'s (config subclassed; checked by a test): Adam 1e-3, weight decay 1e-5, batch 256, max 100
epochs, grad clip 1.0, early stopping patience 10 (min_delta 1e-4 absolute AUROC), ReduceLROnPlateau 0.5 / 5 on val
AUROC, target FPR 0.05, optional `--log-feature-error`. Additions: `--workers` on the command line; `train.py` refuses
windows whose length differs from `seq_len`; the notebook's length-group table uses groups 1–3 / 4–(T−1) / T / 4–T.

**Run (repo root):**

- `npm run aest5:train` (seed 0; `SEED=k` for seed k) → `runs/seed<k>/` (config, env, history, train.log, curves.png,
  best / last.pt, summary.json; gitignored); decoded shards cached in `cache/`. Same as
  `.venv-train/bin/python -m task.autoencoder_st_t5.train --seed <k>` (`--resume`, `--log-feature-error`, `--workers N`).
- `npm run aest5:test`: pytest, synthetic shards, CPU. `npm run aest5:notebook`: executes `autoencoder_st_T5.ipynb`
  on the newest run (analysis cell is a placeholder until a run exists).
- `SEED=k npm run hf:download:baselines:st-t5`: fetch `baselines/autoencoder_st_t5/seed<k>` from the private HF repo.

## Results (seeds 0–3)

GridSybil vs benign, test (`all/T5_benign_grid` `test`), window level, mean ± sample sd over seeds 0–3, from
`runs/seed<k>/test_results.json` (written by the package notebook executed per seed, copy
`runs/seed<k>/autoencoder_st_T5.executed.ipynb`; 4 dp in the file).

| Metric | Value | Per seed |
|---|---|---|
| Head AUROC | 0.9092 ± 0.0015 | 0.9094 / 0.9076 / 0.9112 / 0.9085 |
| Head PR-AUC | 0.9044 ± 0.0021 | 0.9048 / 0.9021 / 0.9071 / 0.9035 |
| F1 at val max-F1 θ | 0.8680 ± 0.0062 | 0.8696 / 0.8605 / 0.8753 / 0.8665 |
| Benign FPR at val max-F1 θ | 0.0131 ± 0.0019 | 0.0123 / 0.0158 / 0.0115 / 0.0127 |
| val max-F1 θ | 0.5708 ± 0.0339 | 0.5715 / 0.6094 / 0.5756 / 0.5268 |
| F1 / recall / benign FPR at val FPR-5% θ | 0.8461 ± 0.0048 / 0.8164 ± 0.0057 / 0.0479 ± 0.0027 | — |
| Reconstruction error alone (AUROC) | 0.6258 ± 0.0049 | 0.6257 / 0.6217 / 0.6327 / 0.6229 |
| Length only (−n_messages) | 0.5167 | — |

Best epochs 38 / 19 / 44 / 24; 51,390 parameters.

Side experiment, not a thesis result, not the thesis TimesNet model; seeds 0–3 on EC2 (g4dn T4); independently judged 2026-11-26 (56 PASS + 4 PASS-WITH-NOTES, 0 FAIL).
