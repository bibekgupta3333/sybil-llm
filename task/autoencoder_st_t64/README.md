# `task/autoencoder_st_t64/`: supervised autoencoder on the T = 64 windows (benign_gridsybil/T64)

The T = 64 cell of the supervised-autoencoder row (`task/autoencoder_st` T48_tvt, `task/autoencoder_st_t32`,
`task/autoencoder_st_t{20,10,5}`): the `task/encoder` ConvAutoencoder (input 14 × 64 = 13 features + mask, z = 16,
80,189 parameters, encoder **and** decoder) plus the `task/autoencoder_st` classifier head on z (Linear 16→32 → ReLU →
Dropout 0.2 → Linear 32→1, 577 parameters): **80,766 trainable parameters**. Loss = BCEWithLogits (`pos_weight` =
n_benign / n_gridsybil of train) + 1.0 · masked MSE on real rows. Inputs are x + mask only. Unlike `task/encoder`
supervised mode (encoder + head, no decoder, no reconstruction loss), this is a true supervised autoencoder.
**A side experiment, not a thesis result, not the thesis TimesNet model.**

**Data and split (no split change).** `src/data/encoder_input/benign_gridsybil/T64/` (benign + GridSybil of the two
GridSybil folders), read through `task/encoder`'s npy cache (`task/encoder/runs/cache/`, opened read-only). On disk:
90 / 10 train / test by sender vehicle, no val. Val = 10% of the train sender vehicles drawn in code with
`task.encoder.data.VehicleSplit(0.1, split_seed 0)` over all train windows: the val vehicles of `task/encoder`
supervised mode and of `task/autoencoder_t64` (train 304,733 / val 33,268 windows, 4,660 / 518 vehicles; same for
every seed, `val_vehicles.npy` saved per run). Test = on-disk test (38,426 windows, 574 vehicles, GridSybil share
0.668). Val drives the LR schedule, early stopping on val AUROC, `best.pt` and both alarm thresholds (max F1 on val;
benign FPR 5% on val). `train.py` opens only the train split (`summary.json` → `splits_opened`, `label_reads`).

**Settings** = `task/autoencoder_st` (its `SupervisedAEConfig` is subclassed): Adam 1e-3, weight decay 1e-5, batch
256, max 100 epochs, grad clip 1.0, early stopping patience 10 (min_delta 1e-4 absolute AUROC), ReduceLROnPlateau 0.5 /
5 on val AUROC, target FPR 0.05, dropout 0.2, recon weight 1. The training loop is `task.autoencoder_st.trainer.Trainer`
(subclassed only for `from_checkpoint` and the curve title). Nothing in the existing packages was changed.

**Code:** `config.py` (settings), `data.py` (cache → train / val / test), `model.py`, `trainer.py`, `train.py`,
`evaluate.py` (scores test with `best.pt`, then reads labels; writes `test_results.json` + `test_scores.npz`),
`autoencoder_st_T64.ipynb` (input, model, training record, val thresholds re-check, test; calls `evaluate`),
`tests/` (CPU, synthetic cache: parameter count, shapes, padding invariance, val draw = `task/encoder`'s, training
never opens test).

**Run (repo root):**

- `SEED=k npm run aest64:train` → `runs/seed<k>/` (config, env, history, train.log, curves.png, best / last.pt,
  val_vehicles.npy, summary.json; gitignored). `--resume` supported.
- `SEED=k npm run aest64:notebook`: copies the notebook to `runs/seed<k>/autoencoder_st_T64.executed.ipynb` and
  executes it there with `AEST64_SEED=k` (writes `test_results.json`, `test_scores.npz`). `SEED=k npm run aest64:eval`:
  the same evaluation without the notebook.
- `npm run aest64:test`: pytest. `SEED=k npm run hf:download:baselines:st-t64`: fetch
  `baselines/autoencoder_st_t64/seed<k>` from the private HF repo.

## Results (seeds 0–3)

GridSybil vs benign, test, window level, mean ± sample sd over seeds 0–3 (EC2 g4dn T4, CUDA), from
`runs/seed<k>/test_results.json`. Not yet independently judged.

| Metric | Value | Per seed 0 / 1 / 2 / 3 |
|---|---|---|
| Head AUROC | **0.9279 ± 0.0021** | 0.9307 / 0.9279 / 0.9255 / 0.9274 |
| Head PR-AUC (chance 0.668) | 0.9704 ± 0.0009 | 0.9715 / 0.9705 / 0.9694 / 0.9701 |
| val max-F1 θ | 0.334 ± 0.038 | 0.299 / 0.334 / 0.388 / 0.316 |
| F1 / recall / benign FPR at val max-F1 θ | 0.8947 ± 0.0029 / 0.8250 ± 0.0045 / 0.0387 ± 0.0019 | F1 0.8975 / 0.8967 / 0.8912 / 0.8934 |
| F1 / recall / benign FPR at val FPR-5% θ | 0.8954 ± 0.0023 / 0.8300 ± 0.0039 / 0.0480 ± 0.0018 | FPR 0.0501 / 0.0459 / 0.0488 / 0.0471 |
| Track AUROC max / mean | 0.9266 ± 0.0023 / 0.9270 ± 0.0024 | — |
| AUROC on 1–3-row / ≥ 4-row windows | 0.917 / 0.928 | — |
| Reconstruction error alone (AUROC) | 0.4812 ± 0.0041 | — |
| Length only (best direction = fixed −n_messages) | 0.5348 | — |
| Best val AUROC | 0.9294 ± 0.0010 | — |

Best epochs 49 / 45 / 40 / 47 (early stopping at 59 / 55 / 50 / 57); training 8.3 / 7.2 / 6.3 / 7.1 min on the T4.
Context only (different inputs and splits): `task/autoencoder_st` T48_tvt 0.9278 ± 0.0019, `task/autoencoder_st_t32`
0.9289 ± 0.0018, `task/encoder` supervised mode (encoder + head, no decoder, same T64 data) 0.928 ± 0.002. Adding the
decoder and reconstruction loss does not change the head AUROC at T = 64; the reconstruction error itself is not a
GridSybil score here (0.48, as for the unlabelled T64 autoencoder, 0.482).

Note: on CUDA, the first eval batch (4,096 windows) logs a cuDNN workspace "allocation failed with OOM ... 17 GB"
warning; PyTorch falls back to another algorithm and the run continues (numbers unaffected; the notebook re-scores
val from `best.pt` and matches the run).
