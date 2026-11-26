# `task/autoencoder_t10/`: unlabelled autoencoder on the T = 10 benign + GridSybil windows

The `task/autoencoder_t32` unlabelled cell of the baseline grid at T = 10: the ConvAutoencoder (input 14 × 10, z = 16,
masked MSE on real rows) trained on **all train windows pooled, benign and GridSybil mixed, no label read**, on
`src/data/encoder_input/all/T10_benign_grid/` (benign of all 8 folders + GridSybil; 85 / 7.5 / 7.5 by vehicle key,
the all/T32_benign_grid split, fixed on disk; no split change). Train fits the weights; val (mixed, unlabelled) drives
early stopping, the LR schedule and the alarm thresholds (val p95 / p99); the Mahalanobis model is fitted on train z;
`train.py` never opens `test`. **Labels are unread in training**: the reader opens feature shards only, the command line
installs `LabelFileGuard` (audit hook that raises on any `*_info.json`) in the process and every decoding worker, the
cache holds x / mask / ids only; labels are opened by `evaluate.py` after `scores/test_*.npy` is written. **A side
experiment, not a thesis result, not the thesis TimesNet model.** Seeds 0–3 trained and evaluated (results below).

Thin package: config (subclass of `task/autoencoder_t20`'s), Trainer / Experiment / TestEvaluation subclasses that only
set the config class, plot title and parameter count; model, reader, labels, scoring and loops are imported.

**What differs from T = 32:** only the input folder, `seq_len` = 10 and `n_down` = 1: encoder strides 1 / 2 / 1, decoder ConvTranspose k3 s1 (length kept) / k4 s2; latent length 5.
**52,861 trainable parameters** (T = 32: 63,293). Every other setting is `task/autoencoder_t32`'s (config subclassed; checked
by a test). Additions: `train.py` / `evaluate.py` refuse windows whose length differs from `seq_len`; the context block
of `evaluation.json` reads the T = 10 siblings' notebooks (`autoencoder_st_t10`, `autoencoder_benign_t10`) and is empty
until they are executed.

**Run (repo root):**

- `npm run ae10:train` (seed 0; `SEED=k` for seed k) → `runs/seed<k>/` (config, env, history, train.log, curves.png,
  best / last.pt, mahalanobis.npz, `scores/val_*.npy`, summary.json; gitignored); feature shards cached in `cache/`.
- `npm run ae10:eval` (`SEED=k`) → `scores/test_{ids,reconstruction,mahalanobis,n_messages,z}.npy`, then
  `evaluation.json`.
- `npm run ae10:test`: pytest, synthetic shards, CPU. `npm run ae10:notebook`: executes `autoencoder_T10.ipynb` on the
  newest run with an `evaluation.json` (analysis cell is a placeholder until a run exists).
- `SEED=k npm run hf:download:baselines:ae10`: fetch `baselines/autoencoder_t10/seed<k>` from the private HF repo.

## Results (seeds 0–3)

GridSybil vs benign, test (`all/T10_benign_grid` `test`: 133,741 windows, GridSybil share
0.303; train share 0.301), window level, mean ± sample sd over seeds 0–3, from
`runs/seed<k>/evaluation.json` (`test_all`). Thresholds = mixed val p95.

| Score | AUROC | Recall @ val p95 (benign FPR) | Per seed |
|---|---|---|---|
| Reconstruction | 0.636 ± 0.003 | 0.103 ± 0.001 (0.041 ± 0.001) | 0.6346 / 0.6346 / 0.6349 / 0.6397 |
| Mahalanobis on z | 0.530 ± 0.020 | 0.087 ± 0.006 (0.044 ± 0.001) | 0.5246 / 0.5546 / 0.5354 / 0.5063 |
| Length only (−n_messages) | 0.530 | — | — |

Side experiment, not a thesis result, not the thesis TimesNet model; seeds 0–3 on EC2 (g4dn T4); independently judged 2026-11-26 (56 PASS + 4 PASS-WITH-NOTES, 0 FAIL).
