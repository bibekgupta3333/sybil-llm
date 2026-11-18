# task/masked_autoencoder — one-class masked autoencoder (pytorch-widedeep encoder-decoder pretraining)

Side experiment, not the thesis TimesNet model. A masking-based autoencoder that follows pytorch-widedeep's
self-supervised pretraining (`EncoderDecoderTrainer(encoder=TabMlp(...), masked_prob=0.2)`, docs example
"15_Self_Supervised_Pretraning_pt1"), re-implemented in plain PyTorch (the library is **not** installed: it would
downgrade numpy in `.venv-train`). It is trained one-class (benign windows only) on exactly the windows of the existing
one-class ConvAE cells, at T = 64, 48 and 32, so its numbers sit next to theirs.

## Method

- **Input**: each window (T x 13, prefix row mask) is flattened to 13 T continuous columns; the T row-mask values are
  appended as T extra input columns (the ConvAE's mask channel). Padding cells are zero.
- **Obfuscation** (`RandomObfuscator`): every training step hides each real cell with probability `masked_prob = 0.2`
  (Bernoulli, drawn on the CPU from the seeded torch generator); hidden cells are set to 0 in the input. Padding cells
  are never hidden.
- **Encoder / decoder**: `mlp` = TabMlp [14 T -> 64 -> 16] (Linear -> ReLU -> Dropout 0.1 per layer) with the mirrored
  TabMlpDecoder [16 -> 64 -> 13 T]; `resnet` = TabResnet DenseResnet blocks [64, 32, 16] (dropout 0.1) with the mirrored
  TabResnetDecoder [16, 32, 64] + Linear(64, 13 T, bias = False). z = the encoder output (16).
- **Loss** (`EncoderDecoderLoss`): mean over windows of the sum over hidden cells of (pred − true)² / var_f, divided by
  the window's number of hidden cells; var_f = batch variance of feature f over real cells.
- **Training**: AdamW (widedeep's default: lr 1e-3, weight decay 0.01), batch 256, ReduceLROnPlateau (factor 0.5) and
  early stopping on the benign val masked-reconstruction MSE, with the sibling's patience: T64 patience 5 / plateau 3 /
  min_delta 0 (task/encoder), T48 and T32 patience 10 / plateau 5 / relative min_delta 1e-4; max 300 epochs. The seed
  sets weights, batch order and training masks.
- **Scores** (per window, real cells only, computed and saved before any label is read):
  - `masked_recon`: the T x 13 cells are partitioned once into 5 equal groups by a fixed RNG (`EVAL_MASK_SEED`
    20261117, the same for every window, run and seed); in draw k group k is hidden; every real cell is hidden exactly
    once, and the score is the mean squared error of the real cells reconstructed while hidden (5 draws, 20% each);
  - `full_recon`: reconstruction MSE with nothing hidden (widedeep's eval mode);
  - `mahalanobis`: distance of z (clean input) to a Gaussian fitted on benign train z.
- **Thresholds**: benign val p95 / p99 per window and per track (max / mean over a link's windows).

## Data protocol (same windows as the one-class ConvAE cells)

| T | Sibling (imported) | Train | Val | Test |
|---|---|---|---|---|
| 64 | `task/encoder` mode ae (`WindowStore`, `VehicleSplit`, cache `task/encoder/runs/cache`) | benign train windows (labels read once, to select them, as in the sibling) | 10% of their sender vehicles, split seed 0 | on-disk `test`: benign + GridSybil, separated by label after scoring |
| 48 | `task/autoencoder_benign_t48` (`WindowDataset`, its cache) | on-disk `train` (benign) | on-disk `val` | `test` + `test_gridsybil` |
| 32 | `task/autoencoder_benign_t32` (`WindowDataset`, its cache) | on-disk `train` (benign) | on-disk `val` | `test` + `test_gridsybil` / `_datareplay` / `_dosrandom` / `_dosdisruptive` |

Mahalanobis (`task.autoencoder_benign_t32.scoring.Mahalanobis`, shrinkage when cond > 1e6), `TrackScores`,
`quantile_thresholds`, `EarlyStopping` / `RunLog` and the metric code (`task.autoencoder_benign_t48.evaluate.
DetectionMetrics`) are imported. Note: at T64 the sibling used a ridge-1e-6 Mahalanobis; here the shrinkage version is
used at every T.

## widedeep correspondence (pytorch_widedeep 1.7.0 wheel, read only)

| Here | widedeep source |
|---|---|
| `RandomObfuscator` | `models/tabular/self_supervised/_random_obfuscator.py` L14-33 |
| training / eval forward | `models/tabular/self_supervised/encoder_decoder_model.py` L60-70 (eval: mask of ones) |
| mirrored decoder | same file, L96-130 |
| `EncoderDecoderLoss` | `losses.py` L908-970 |
| `dense_layer`, `MLP` | `models/tabular/mlp/_layers.py` L7-22, L52-89 |
| TabMlp / TabMlpDecoder defaults | `models/tabular/mlp/tab_mlp.py` L158-163, L196, L223-301 |
| `BasicBlock`, `DenseResnet` | `models/tabular/resnet/_layers.py` L8-55, L58-90 |
| TabResnet / TabResnetDecoder | `models/tabular/resnet/tab_resnet.py` L189-190, L239, L284-427 |
| optimizer AdamW, train / eval step | `self_supervised_training/_base_encoder_decoder_trainer.py` L59-63; `encoder_decoder_trainer.py` L217-243 |

## Deviations from widedeep

1. **The encoder sees the masked input.** In widedeep 1.7.0 the TabMlp / TabResnet path draws the mask and computes
   `masked_x`, but then calls `self.decoder(self.encoder(X))` on the clean input (`encoder_decoder_model.py` L64-65);
   only the TabNet path uses the masked input. Taken literally, the "masked" model is a plain autoencoder whose loss is
   restricted to random cells. Here the encoder gets the masked input (`feed_masked_input = True`); the literal
   behaviour is kept behind `feed_masked_input = False` (not run).
2. **Linear TabMlpDecoder output.** widedeep's MLP puts ReLU + dropout after every layer including the reconstruction
   layer (`mlp/_layers.py` L20-21), which cannot output the negative values of z-scored features. Here the last decoder
   layer is a plain Linear (`linear_mlp_output = True`). The TabResnetDecoder already ends in a plain Linear.
3. **Padding.** Continuous-only input, flattened windows with a row mask appended as input columns; mask columns are
   never hidden or reconstructed; padding cells are never hidden, never in the loss, never scored.
4. **Loss variance per feature, not per column.** widedeep divides by the batch variance of each column; with 13 T
   flattened columns a late time step can have no real cell in a batch, so the variance is pooled per feature over the
   batch's real cells (identical to widedeep when T = 1; tested).
5. **Batch 256, early stopping and LR schedule** (widedeep: batch 32, fixed `n_epochs`, no schedule by default), to
   match the sibling ConvAEs. Validation loss = deterministic masked-reconstruction MSE (widedeep's eval loss uses an
   all-ones mask on the clean input).
6. **Model size.** widedeep's defaults (TabMlp [200, 100], TabResnet [200, 100, 100]) are replaced by small widths with
   z = 16, like the ConvAEs. Parameters: TabMlp 113,616 / 85,760 / 57,904 and TabResnet 128,224 / 100,576 / 72,928 at
   T = 64 / 48 / 32 (one-class ConvAEs: 80,189 / 71,741 / 63,293).

## Commands (repo root)

```bash
npm run mae:test                                  # CPU tests (synthetic windows)
WINDOW=32 ENCODER=mlp SEED=0 npm run mae:train    # train + evaluate one run
WINDOW=32 ENCODER=mlp SEED=0 npm run mae:eval     # re-score / re-evaluate one run
npm run mae:all                                   # TabMlp, T = 64 / 48 / 32, seeds 0-3 (ENCODER=resnet for TabResnet)
npm run mae:plots                                 # runs/<run>/loss_curve.png for every finished run
npm run mae:notebook                              # execute masked_autoencoder.ipynb (reads the run files only)
.venv-train/bin/python -m task.masked_autoencoder.evaluate --run-dir task/masked_autoencoder/runs/T32-mlp-seed0 --from-scores
```

Run folder `runs/T<w>-<encoder>-seed<k>/` (gitignored): config.json, env.json, history.jsonl, train.log, best.pt,
mahalanobis.npz, loss_curve.png, summary.json (val thresholds, label reads), `scores/<split>.npz` (val and every test split),
test_metrics.json (per attack vs benign test: AUROC, PR-AUC, recall / benign FPR at val p95 / p99, AUROC on 1-3-row and
>= 4-row windows, per-track max / mean, length-only AUROC with score = −n_messages).

## Results (TabMlp, seeds 0-3)

12 finished runs `runs/T{64,48,32}-mlp-seed{0,1,2,3}/` (MPS, 2026-11-17; every run stopped early). Analysis, all tables,
loss curves and ROC curves: notebook [`masked_autoencoder.ipynb`](masked_autoencoder.ipynb) (`npm run mae:notebook`; reads
the run files only, recomputes every AUROC from the saved score files and asserts it equal to `test_metrics.json`).
Loss curve of each run: `runs/<run>/loss_curve.png` (`npm run mae:plots`; written automatically by `train.py` for new
runs). Training minimises the variance-normalised `EncoderDecoderLoss`; early stopping, the LR schedule and all test
scores use the plain MSE on the z-scored features, so the training curve sits above the val curves by construction.

Window AUROC vs benign test, mean ± sample sd over seeds 0-3 (one-class ConvAE on the same windows, seeds 0-3, from
`baselinepaper/01-our-studies/inventory.md`):

| Case | masked_recon | full_recon | mahalanobis | length only | length-matched masked_recon | ConvAE recon |
|---|---|---|---|---|---|---|
| T64 GridSybil | 0.531 ± 0.002 | 0.539 ± 0.003 | 0.398 ± 0.018 | 0.535 | 0.510 ± 0.002 | 0.657 ± 0.004 |
| T48 GridSybil | 0.607 ± 0.001 | 0.603 ± 0.002 | 0.403 ± 0.016 | 0.530 | 0.589 ± 0.001 | 0.682 ± 0.006 |
| T32 GridSybil | 0.657 ± 0.002 | 0.650 ± 0.003 | 0.408 ± 0.017 | 0.559 | 0.631 ± 0.001 | 0.734 ± 0.003 |
| T32 DataReplay | 0.693 ± 0.003 * | 0.540 ± 0.010 | 0.269 ± 0.036 | 0.952 | 0.490 ± 0.009 | 0.387 ± 0.010 |
| T32 DoSRandom | 0.988 ± 0.001 | 0.976 ± 0.002 | 0.394 ± 0.073 | 0.967 | 0.970 ± 0.001 | 0.967 ± 0.005 |
| T32 DoSDisruptive | 0.851 ± 0.006 | 0.732 ± 0.007 | 0.222 ± 0.032 | 0.967 | 0.679 ± 0.005 | 0.598 ± 0.012 |

Length-matched = AUROC inside each exact window length (real rows), weighted by the attack windows at that length.

- GridSybil: below the ConvAE by 0.08-0.13 at every T; at T64 at the length-only level. Recall at val p95 0.09 / 0.17 /
  0.21 at benign test FPR 0.045-0.052.
- \* The masked-reconstruction score partly measures window length (a 1-row window has no time neighbours): on benign
  test windows alone it separates 1-3-row from longer windows with AUROC 0.688-0.712 (full_recon 0.537-0.545). The
  pooled T32 DataReplay 0.693 is this length effect (70.2% of DataReplay windows have 1 row vs 5.2% of benign; length-
  matched 0.490) and **must not be reported as detection**; it is also far below length only (0.952). The same effect
  inflates DoSDisruptive (pooled 0.851, length-matched 0.679).
- DoS >= 4-row AUROC 1.000 rests on only 523 / 524 attack windows.
- Mahalanobis on z is below chance in every case and the least stable across seeds.
- Training: best epochs 41-68 / 80-110 / 43-58 at T = 64 / 48 / 32, 0.7-5.4 min per run; 68-85% of the val decrease by
  epoch 10, 1-3% after the first LR halving.

Side experiment, one-class (benign windows selected with labels, no label in the loss); not the thesis TimesNet model.
