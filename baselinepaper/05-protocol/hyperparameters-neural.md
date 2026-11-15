# Hyperparameters of the neural detectors — frozen before any test score

Written 2026-11-10, **before any run of `convae`, `lstmae` or `supae` scored a test split** (protocol §4). From here
on the values are fixed; a change goes to `protocol.md` §9 with the reason. Code:
`baselinepaper/experiments/detectors/{neural_common,conv_autoencoder,lstm_autoencoder,supervised_autoencoder}.py`.

What was looked at before freezing: training loss and validation loss / validation AUROC of 2–4 epochs per detector
(seed 0, `benign_gridfolders`, `mix_all`, `supervised`), to measure seconds per epoch. No test split was opened, no
run was scored, no value below was chosen from those numbers except the `lstmae` epoch cap (a wall-clock decision).

## 1. Shared by the three detectors

| Setting | Value | Source |
|---|---|---|
| Input | x (32 × 13) after `Detector.standardise` + prefix mask (32); padding rows zeroed, mask as 14th channel | protocol §2, §4; the `task/` models |
| Optimiser | Adam, lr 1e-3, weight decay 1e-5 (betas / eps: PyTorch defaults) | `task/autoencoder_benign_t32/config.py`, `task/autoencoder_st_t32/config.py` (identical) |
| Batch size | 256, one random permutation of the training pool per epoch, last batch kept | same |
| Gradient clipping | global L2 norm 1.0 | same |
| LR schedule | `ReduceLROnPlateau`, factor 0.5, patience 5, on the monitored validation value | same |
| Early stopping | patience 10 epochs on the monitored validation value; the best-epoch weights are restored | same |
| Evaluation batch | 4,096 (validation and scoring) | same |
| Seeds | 0–4; the seed sets python / numpy / torch generators before the model is built (initial weights, dropout) and a separate `torch.Generator` for the batch order. No subsampling anywhere | protocol §5; the `task/` trainers |
| Device | Apple MPS (`--device auto`), torch 2.14.1; recorded per run in `env.json` and `summary.json` | this Mac |
| Training loop | `neural_common.NeuralTrainer`: the loop of the two `task/` trainers without their checkpoints / plots / logs; the two `EarlyStopping` classes, `masked_mse`, `window_errors`, `seed_everything`, `Mahalanobis` and the model classes are imported, not copied. Tests check that it gives the same per-epoch validation values and scores as the `task/` trainers on CPU | — |

Monitored validation value = computed on the regime's validation pool only (protocol §3: same composition as the
training pool; for the mix regimes it contains unlabelled GridSybil windows).

## 2. `convae` — convolutional autoencoder (`task.autoencoder_benign_t32.model.ConvAutoencoder`)

| Setting | Value | Source |
|---|---|---|
| Architecture | Conv1d 14→32 (k3, s1) → 64 (k3, s2) → 64 (k3, s2), each BatchNorm + ReLU; 512 → Linear → z; Linear 16→512 + ReLU; ConvTranspose1d 64→64 (k4, s2) → 32 (k4, s2), each BatchNorm + ReLU; Conv1d 32→13 (k3) | imported class, unchanged |
| Latent size | 16 | `AutoencoderConfig.latent_dim` |
| Parameters | **63,293** | `TOTAL_PARAMS`; asserted when the model is built |
| Loss | masked MSE over real rows | imported `masked_mse` |
| Model selection | lowest validation loss (pooled over the real rows of the regime's validation pool) | protocol §4 |
| Early-stopping `min_delta` | 1e-4 **relative** decrease of the validation loss | `AutoencoderConfig.min_delta` |
| `max_epochs` | 300 | `AutoencoderConfig.max_epochs` (not lowered) |
| Regimes | `benign_only`, `benign_gridfolders`, `mix_all`, `mix_gridfolders` | protocol §4 |
| Score `recon` | masked reconstruction error of the window (mean over features of the per-feature MSE over real rows) | imported `window_errors` |
| Score `maha` | Mahalanobis distance of z; mean and sample covariance of z of **every training window of the regime** (best-epoch weights, eval mode); covariance shrunk towards a scaled identity with weight 0.01 only if its condition number exceeds 1e6; Hermitian pseudo-inverse | imported `Mahalanobis`; `mahalanobis_max_cond`, `mahalanobis_shrinkage` of `AutoencoderConfig` |

## 3. `lstmae` — LSTM autoencoder of matched size (`lstm_autoencoder.LSTMAutoencoder`)

New here; a recurrent reference on the same input, not a replication of a published model. Nothing was tuned: one
architecture, chosen to match the latent size and the parameter count of `convae`.

| Setting | Value | Source |
|---|---|---|
| Encoder | LSTM 14→80, 1 layer, unidirectional, over the 32 rows; hidden state **at the last real row** (what a packed sequence returns) → Linear 80→16 = z | this file |
| Decoder | z repeated at each of the 32 rows → LSTM 16→80, 1 layer → Linear 80→13 | this file |
| Latent size | 16 | = `convae` |
| Parameters | **64,429** (encoder LSTM 30,720 + Linear 1,296 + decoder LSTM 31,360 + Linear 1,053) = `convae` + 1.8% | hidden size 80 is the size that lands closest to 63,293; asserted when the model is built |
| Dropout / normalisation layers | none | — |
| Loss, model selection, `min_delta`, optimiser, batch size, LR schedule, patience, gradient clipping | exactly those of `convae` (§1, §2) | so that the comparison is about the architecture |
| `max_epochs` | **100 (cap introduced here)** | wall-clock budget, see §5 |
| Regimes | the same four | protocol §4 |
| Score `recon` | masked reconstruction error of the window, as `convae` | imported `window_errors` |

Padding: both LSTMs run forwards in time and the decoder never reads x, so z and the reconstruction of every real
row are independent of the padding rows (tested).

## 4. `supae` — autoencoder + classifier head (`task.autoencoder_st_t32.model.SupervisedAutoencoder`)

| Setting | Value | Source |
|---|---|---|
| Architecture | the `convae` autoencoder + head on z: Linear 16→32 → ReLU → Dropout 0.2 → Linear 32→1 | imported class, unchanged |
| Parameters | **63,870** (63,293 + 577) | `TOTAL_PARAMS`; asserted |
| Loss | `BCEWithLogits(pos_weight)` + 1.0 × masked MSE | `SupervisedAEConfig.recon_weight` |
| `pos_weight` | benign / GridSybil count of the regime's **training labels** (502,773 / 229,080 = 2.195 for this pool) | rule of `task/autoencoder_st_t32/train.py` |
| Model selection | highest validation AUROC, benign vs GridSybil windows of the validation pool | protocol §4 |
| Early-stopping `min_delta` | 1e-4 **absolute** increase of the validation AUROC; LR schedule in mode "max" on it | `SupervisedAEConfig.min_delta` |
| `max_epochs` | 100 | `SupervisedAEConfig.max_epochs` |
| Regime | `supervised` | protocol §4 |
| Score `prob` | sigmoid of the logit = P(GridSybil) | — |
| Score `recon` | masked reconstruction error of the window | secondary score |

Labels are read in `fit` only (training loss, `pos_weight`, validation AUROC); `score` receives x and mask.

## 5. Wall-clock budget and the `lstmae` cap

Measured 2026-11-10 on this Mac (M4 Pro, MPS; four CPU-bound jobs of the classical detectors were running at the
same time), training + validation only:

| Detector | s / epoch, `benign_gridfolders` (125,749) | s / epoch, `mix_all` / `supervised` (731,853) | scaled: `benign_only` (502,773) | scaled: `mix_gridfolders` (354,829) |
|---|---|---|---|---|
| `convae` | 1.7 | 11.1 | ≈ 7.2 | ≈ 5.1 |
| `lstmae` | 2.0 | 11.9 | ≈ 8.2 | ≈ 5.8 |
| `supae` | — | 12.6 | — | — |

- `convae`, one seed, four regimes: ≈ 25 s per epoch of all four. Its settings are not touched. If every run stops
  near the 161 epochs of the earlier `benign_only` seed-0 run: ≈ 67 min per seed, **≈ 5.6 h** for 5 seeds. If every
  run used all 300 epochs: ≈ 10.5 h (upper bound).
- `supae`: the earlier run on a similar pool stopped after 27 epochs → ≈ 6–7 min per run, **≈ 0.6 h** for 5 seeds
  (upper bound at 100 epochs: 1.75 h).
- `lstmae`: ≈ 28 s per epoch of all four regimes. With the `convae` bound of 300 epochs it could take up to 11.7 h on
  its own. **Cap: `max_epochs = 100`** → at most 47 min per seed, **≤ 3.9 h** for 5 seeds. Reason: the sweep (45
  runs) should fit in roughly 10 h; this is the only setting in which `lstmae` differs from `convae` besides the
  architecture. Consequence, to be stated with the results: an `lstmae` run whose `stop_reason` is `max_epochs` was
  stopped by the budget, not by convergence (`summary.json` → `fit_info.stop_reason`, `history.json`).
- Expected total of the sweep: ≈ 5.6 + 3.9 + 0.6 h + scoring ≈ **10–10.5 h**; upper bound ≈ 16.5 h if every `convae`
  and `supae` run uses its full epoch budget.

## 6. Reproducibility

- CPU: same seed → identical training history and scores (tested).
- MPS: every generator is seeded and two repeated 3-epoch `lstmae` runs gave the same validation losses to the
  five printed decimals, but bit-identical results on MPS are **not guaranteed** (Metal kernels, no deterministic
  mode), and MPS results differ from CPU results in the last digits. A re-run of the sweep is expected to reproduce
  the scores closely, not bit for bit; the seed-to-seed standard deviation of the protocol covers this.
- `torch.mps.synchronize()` is called every 50 training batches (it waits for queued GPU work; without it the LSTM
  step slows down about 15×). It does not change any value: the validation losses with and without it were equal.
- Each run records the device, package versions, git commit and the sha256 of this package and of the two `task/`
  packages (`env.json`), the hyperparameters (`config.json`), the per-epoch history (`history.json`), the best-epoch
  weights (`best.pt`) and, for `convae`, the fitted Mahalanobis model (`mahalanobis.npz`).

## 7. Prior exposure (protocol §4)

`convae` and `supae` were developed earlier and seed-0 test results of related runs are known (the benign-only
autoencoder on this test set; the supervised model on another split). Their settings are taken as they were.
`lstmae` has had no exposure to any test result.
