# `task/encoder/`: a simple baseline encoder for benign vs GridSybil

This is a small, self-contained baseline. It is **not** the thesis TimesNet model. The question it answers: can a small
1D-conv encoder, trained either one-class (a reconstruction autoencoder on benign windows only) or supervised (labels
used), tell benign windows from GridSybil windows in the prepared benign + GridSybil input? It is compared with two
shortcut baselines that use no learned encoder.

**Beginner walk-through:** [`explained.html`](explained.html) (open in a browser; self-contained, light / dark,
animated) explains the data, both models, the losses, training, thresholds and the results below, with every number
taken from the run files.

**Everything in one notebook:** [`benign_vs_gridsybil_autoencoder.ipynb`](benign_vs_gridsybil_autoencoder.ipynb)
repeats the same method and settings (one seed, 0) in short, explained cells, with no imports from this folder: data,
vehicle split, autoencoder, masked loss, training, thresholds, metrics, supervised and shortcut baselines, length
groups, and a comparison with the 3-seed results. Run it with `npm run enc:notebook` (about 10 minutes on MPS);
outputs go to `runs/notebook-seed0/`.

- **Data (read-only):** `src/data/encoder_input/benign_gridsybil/T64/`, 64 × 13 windows + mask, with the existing 90 / 10
  train / test split by sender vehicle (D8′). Train has 120,073 benign and 217,928 GridSybil windows; test has 12,741
  benign and 25,685 GridSybil windows from 574 sender vehicles. The split is not changed (RULE 3).
- **Validation:** 10% of the train **sender vehicles** are held out (seed 0). Fit and validation share no vehicle.
  For the AE this uses benign vehicles only (fit 107,924 / val 12,149 windows). For the supervised model it uses all
  train vehicles. `sender` / `receiver` are never model inputs. `sender` is used only for this split and as the
  cluster key of the bootstrap.
- **Test labels** are read only after every test score exists.

## How to run (from the repo root)

```bash
npm run enc:baselines          # length-only + stats-LR baselines -> task/encoder/runs/baselines/  (~30 s)
npm run enc:ae                 # one-class AE, seeds 0-2          -> task/encoder/runs/ae-seed{0,1,2}/
npm run enc:supervised         # supervised upper bound, seeds 0-2 -> task/encoder/runs/supervised-seed{0,1,2}/ (~1 min each)
# The reported AE runs used a 300-epoch cap (the default is 50), one seed at a time (~4 min each on MPS):
.venv-train/bin/python -m task.encoder.train --mode ae --seed 0 --epochs 300
npm run enc:eval               # every finished run under task/encoder/runs/ -> task/encoder/runs/evaluation/
npm run enc:test               # pytest, synthetic data
```

`enc:eval` with no arguments evaluates every `task/encoder/runs/*/` that has a `scores_test.npz`. It skips
`baselines/`, `evaluation/`, `*-check` and folders starting with `_`. To pick runs explicitly:
`.venv-train/bin/python -m task.encoder.evaluate --runs task/encoder/runs/ae-seed0 …`.

| File | Job |
|---|---|
| `config.py`, `data.py`, `model.py`, `train.py` | settings, T64 shard reader + vehicle fit / val split, `ConvAutoencoder` / `SupervisedHead`, training CLI (`--mode ae\|supervised --seed N`) |
| `baselines.py` | length-only and per-window-statistics + logistic-regression baselines (reads the T64 shards directly) |
| `evaluate.py` | metrics, vehicle bootstrap, length strata, figures → `runs/evaluation/results.json` + PNGs |
| `tests/` | pytest on synthetic shards and synthetic score files |

Every run folder has `config.json` (with the seed), `env.json`, `history.jsonl` (epoch losses), `summary.json`,
`best.pt`, `scores_test.npz` and `scores_val.npz`. All outputs are gitignored (`/task/encoder/runs/`).

## Method

- **One-class autoencoder (the main model).** The input is 13 features + the mask channel (14 × 64), with padding rows
  zeroed. Encoder: Conv1d 32 (k3, s1) → 64 (k3, s2) → 64 (k3, s2), each with BatchNorm + ReLU, then flatten (64 × 16) →
  Linear → z (16). The decoder mirrors it (Linear → ConvTranspose1d ×2 → Conv1d → 13 × 64). That is **80,189
  parameters**. Loss = MSE on real rows only. Training: Adam (lr 1e-3, weight decay 1e-5), batch 256,
  ReduceLROnPlateau (×0.5 after 3 epochs), early stopping with patience 5 on the validation benign loss. Seeds 0–2.
  The 50-epoch cap in the brief stopped the first runs while validation loss was still falling (AUROC 0.645 at
  50 epochs). They were retrained with a 300-epoch cap and stopped early at 121 / 103 / 123 epochs (best val loss
  0.1209 / 0.1214 / 0.1231). Only the retrained runs are reported below.
  - Score 1 = masked reconstruction MSE per window (also per feature).
  - Score 2 = Mahalanobis distance of z to the benign-fit z distribution.
  - Thresholds = the 95th / 99th percentile of the **validation-benign** scores. No attack data is used for them.
- **Supervised upper bound.** The same encoder (37,233 parameters with the head Linear 16 → 32 → 1) trained with BCE
  (pos_weight = benign / GridSybil) on all train windows. It stopped early after 19 / 16 / 16 epochs (best val loss
  0.1970 / 0.1949 / 0.1929). Its probability is thresholded at 0.5. The Mahalanobis distance of its z to the benign z
  is reported as well. That encoder has seen labels, so this score is not one-class.
- **Shortcut baselines.**
  - (a) Length only: score = ±n_messages. The direction (−n, shorter = GridSybil) was picked on the test labels, which
    can only flatter this baseline.
  - (b) Statistics: mean / std / min / max of each feature over real rows + n_messages (53 numbers), standardised →
    logistic regression (C = 1, class_weight balanced, seed 0), fit on train, scored on test.
- **Metrics on test (GridSybil = positive).** AUROC and PR-AUC, mean ± std over the 3 seeds. A 95% CI of AUROC from
  1,000 bootstrap resamples of the 574 test sender vehicles; it reflects test-set sampling only, not seed variance.
  Precision / recall / F1 / FPR + confusion matrices at the thresholds. AUROC per length stratum. Per-feature
  reconstruction error.

## Results (test split, `runs/evaluation/results.json`)

GridSybil is 66.8% of test windows, so a constant score has PR-AUC 0.668.

| Method | AUROC (mean ± std, 3 seeds) | 95% vehicle-bootstrap CI (seed 0) | PR-AUC | At threshold: precision / recall / F1 / FPR |
|---|---|---|---|---|
| AE reconstruction error (one-class) | **0.655 ± 0.002** | 0.630 – 0.677 | 0.802 | p95: 0.885 / 0.199 / 0.325 / 0.052 · p99: 0.951 / 0.105 / 0.190 / 0.011 |
| AE latent Mahalanobis (one-class) | 0.319 ± 0.016 | 0.317 – 0.352 | 0.589 | p95: 0.633 / 0.045 / 0.084 / 0.052 |
| Supervised encoder, latent Mahalanobis | 0.787 ± 0.028 | 0.753 – 0.777 | 0.899 | p95: 0.952 / 0.502 / 0.656 / 0.050 |
| Supervised encoder + head (upper bound) | **0.928 ± 0.002** | 0.921 – 0.932 | 0.970 | 0.5: 0.987 / 0.816 / 0.893 / 0.023 |
| Length only (−n_messages) | 0.535 | 0.511 – 0.557 | 0.699 | — |
| Stats + logistic regression | **0.855** | 0.846 – 0.865 | 0.926 | 0.5: 0.865 / 0.832 / 0.849 / 0.261 |

AUROC per length stratum (mean over seeds). The length-only column is the AUROC of n_messages inside the stratum, in
its better direction.

| n_messages | windows (benign / GridSybil) | length only | AE recon | AE Mahalanobis | stats LR | supervised |
|---|---|---|---|---|---|---|
| ≤ 3 | 6,720 (1,822 / 4,898) | 0.505 | 0.552 | 0.300 | 0.681 | 0.916 |
| 4–16 | 15,865 (5,236 / 10,629) | 0.536 | 0.651 | 0.256 | 0.851 | 0.924 |
| 17–48 | 11,800 (4,452 / 7,348) | 0.507 | 0.710 | 0.309 | 0.872 | 0.922 |
| 49–64 | 4,041 (1,231 / 2,810) | 0.527 | 0.752 | 0.419 | 0.927 | 0.949 |

**Per-feature reconstruction error** (mean masked MSE, benign → GridSybil; per-feature AUROC):

| Feature | Error (benign → GridSybil) | Per-feature AUROC |
|---|---|---|
| range | 0.016 → 0.115 | 0.707 |
| claimed_pos_y | 0.011 → 0.090 | 0.695 |
| log_dtau | 0.205 → 0.983 | 0.640 |
| bearing | 0.123 → 0.286 | 0.635 |
| claimed_pos_x | 0.013 → 0.071 | 0.608 |
| velocities, accelerations, receiver position | — | 0.50 – 0.61 |

GridSybil error is higher on every feature.

Figures in `runs/evaluation/`:

- `error_histograms.png`: AE scores, benign vs GridSybil, with the thresholds.
- `roc_curves.png`
- `strata_auroc.png`
- `per_feature_error.png`
- `z_pca.png`: PCA of z, AE vs supervised.

## Interpretation

- **The one-class AE is a weak detector.** It is above chance and above the length shortcut (0.655 vs 0.535, with
  non-overlapping CIs), and its advantage grows with window length (0.55 at n ≤ 3 → 0.75 at n ≥ 49). It is clearly
  **below the simple-statistics baseline** (0.855), which beats it in every stratum. At the 95th-percentile benign
  threshold it flags only 20% of GridSybil windows (FPR 5.2%). What the AE learned about "normal" motion separates the
  classes less well than 53 hand-made window statistics plus a linear model. The stats LR does use labels, so it is
  not a one-class method; it is still a much cheaper model.
- **The AE's latent Mahalanobis score is inverted** (AUROC 0.32 in every seed and stratum). GridSybil windows map
  *closer* to the centre of the benign z distribution than benign windows do. The score direction was fixed before
  testing and is reported as it came out. Flipping it after seeing test labels would give about 0.68, but that would
  be tuned on test, so it is not a result. A 16-d bottleneck trained only to reconstruct benign data does not place
  GridSybil windows far from benign ones.
- **With labels, the same small encoder works well.** AUROC 0.928 and F1 0.893 at 0.5, and it beats the stats LR in
  every stratum, most of all on the 1–3-message windows (0.92 vs 0.68). Even its z, scored one-class against benign z,
  reaches 0.787. The architecture can represent the difference, but benign-only reconstruction does not find it.
- **Length is not the shortcut here** (0.535 overall, 0.49–0.54 within strata). This differs from the all-attack T24
  input, where length dominated the DataReplay / DoS results. Which features carry the signal is not analysed.
  Absolute positions (`claimed_pos_*`, `rx_pos_*`) are inputs (D3) and are among the features with the largest error
  gaps, so part of what the stats LR and the supervised model use may be location rather than motion consistency. This
  is not tested here.
- **Caveats.**
  - One data subset (benign + GridSybil, both time windows pooled), one architecture size, no hyperparameter search.
  - The bootstrap CIs cover test-vehicle sampling only. For the supervised Mahalanobis score the seed-to-seed spread
    (± 0.028) is larger than any one seed's CI.
  - These are baseline numbers for this folder's task, not thesis results for the TimesNet SSL encoder.
