# `task/autoencoder_st_t20/`: supervised autoencoder on the T = 20 windows (all/T20_benign_grid)

The `task/autoencoder_st_t32` supervised baseline at T = 20: the ConvAutoencoder (input 14 × 20 = 13 features + mask,
z = 16) plus the same classifier head on z (Linear 16→32 → ReLU → Dropout 0.2 → Linear 32→1, 577 parameters), trained on
`src/data/encoder_input/all/T20_benign_grid/` (benign of all 8 folders + GridSybil; 85 / 7.5 / 7.5 by vehicle key,
the all/T32_benign_grid split, fixed on disk; no split change). Loss = BCEWithLogits (`pos_weight` = n_benign /
n_gridsybil of train) + 1.0 · masked MSE. Inputs are x + mask only. Val drives the LR schedule, early stopping on val
AUROC, `best.pt` and both alarm thresholds (max F1 on val; benign FPR 5% on val); `train.py` never reads test; the test
evaluation is in the notebook, as for T = 32. **A side experiment, not a thesis result, not the thesis TimesNet model.**
Seeds 0–3 trained and evaluated (results below).

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

## Results (seeds 0–3)

GridSybil vs benign, test (`all/T20_benign_grid` `test`), window level, mean ± sample sd over seeds 0–3, from
`runs/seed<k>/test_results.json` (written by the package notebook executed per seed, copy
`runs/seed<k>/autoencoder_st_T20.executed.ipynb`; 4 dp in the file).

| Metric | Value | Per seed |
|---|---|---|
| Head AUROC | 0.9243 ± 0.0006 | 0.9240 / 0.9236 / 0.9246 / 0.9249 |
| Head PR-AUC | 0.9202 ± 0.0002 | 0.9201 / 0.9199 / 0.9203 / 0.9203 |
| F1 at val max-F1 θ | 0.8814 ± 0.0013 | 0.8830 / 0.8814 / 0.8813 / 0.8799 |
| Benign FPR at val max-F1 θ | 0.0116 ± 0.0015 | 0.0102 / 0.0128 / 0.0104 / 0.0130 |
| val max-F1 θ | 0.6269 ± 0.0450 | 0.6620 / 0.5619 / 0.6325 / 0.6510 |
| F1 / recall / benign FPR at val FPR-5% θ | 0.8569 ± 0.0010 / 0.8295 ± 0.0023 / 0.0488 ± 0.0013 | — |
| Reconstruction error alone (AUROC) | 0.6057 ± 0.0041 | 0.6054 / 0.6080 / 0.6093 / 0.6000 |
| Length only (−n_messages) | 0.5481 | — |

Best epochs 15 / 17 / 9 / 8; 57,534 parameters.

Side experiment, not a thesis result, not the thesis TimesNet model; seeds 0–3 on EC2 (g4dn T4); independently judged 2026-11-26 (56 PASS + 4 PASS-WITH-NOTES, 0 FAIL).
