# Evaluation protocol (E1–E5) — written before any run

Draft 2026-11-10, fixed before results exist. Changes after the first result must be listed in §9 with the reason
(RULE 1). Code: `baselinepaper/experiments/` (Python package, run from the repo root).

## 1. Question

Under one leak-controlled protocol, how well do detectors separate each Sybil attack from benign traffic, as a
function of what the training data is assumed to be (benign only, unlabelled mix, labelled)?

## 2. Data (one input, one split, one test set)

- Windows: receiver-side pseudonym links cut into T = 32 rows × 13 features + prefix mask, from
  `src/data/encoder_input/all/T32_benign/` (verified 2026-10-29). Its 80 / 10 / 10 split by vehicle key
  `<group>:<vehicle>` (seed 0) and its normalisation (mean / std of benign train real rows) are used unchanged for
  every detector. **No new split** (RULE 3: nothing changes; noted in `docs/plan/research-plan.md`).
- Splits read as they are: `train`, `val`, `test` (benign) and `test_gridsybil`, `test_datareplay`,
  `test_dosrandom`, `test_dosdisruptive` (attack windows of the test vehicles).
- Added for training with attacks: the GridSybil windows of the **train** and **val** vehicles of that same split.
  They are taken from `src/data/encoder_input/all/T32_benign_grid/` (same crops, other split and normalisation):
  selected by vehicle key, un-normalised with that input's statistics, re-normalised with T32_benign's. Check: the
  GridSybil windows of the test vehicles derived this way must equal `T32_benign/test_gridsybil` (same ids, values
  within rounding).
- Known limitations carried over, stated in the paper: F14 (GridSybil_0709 near-copies may straddle the split,
  unmeasured), F1 (benign traffic repeated in each family; handled in the intervals by resampling vehicle keys), F13.

## 3. Training regimes

| Regime | Training windows | Labels used |
|---|---|---|
| `benign_only` | benign train, all 8 folders | to select benign; benign val for model selection and threshold |
| `benign_gridfolders` | benign train of the two GridSybil folders only (control for `mix_gridfolders`) | as above |
| `mix_all` | benign train (all 8 folders) + GridSybil train, pooled | none |
| `mix_gridfolders` | benign + GridSybil train of the two GridSybil folders (their natural composition) | none |
| `supervised` | benign train + GridSybil train, labelled | yes; GridSybil is the only attack seen |

- The mix regimes are label-free end to end: inputs are re-standardised with the mean / std of their own training
  pool, model selection uses a val pool of the same composition, and the primary operating point is the 95th
  percentile of that mixed val pool. The benign-val threshold is reported for them as an "oracle calibration" column
  only. A benign-normalised, benign-early-stopped variant of the mix runs is not run.
- `mix_all` and `mix_gridfolders` contain the same GridSybil windows and differ in how much benign traffic is added
  (the benign traffic repeats in every family, F1). They are two confounded points, not a dose-response curve (that
  is E6). Their attack shares are computed from counts before any run and reported.
- DataReplay and DoS are never in training. The supervised result on them is "transfer from GridSybil", read within
  length strata; it supports no general unseen-attack claim (E8).

## 4. Detectors

Every detector outputs one or more scores per window, higher = more anomalous, from x and mask only. **Every
detector can see the window length** (the mask is an input channel of the autoencoders; a one-row window has zero
spread in every statistic), so length is controlled in the evaluation (§6), not by the features.

| Detector | Regimes | Score |
|---|---|---|
| Convolutional autoencoder (the `task/autoencoder_benign_t32` model, 63,293 parameters) | the four unsupervised regimes | masked reconstruction error; Mahalanobis distance on z (fitted on train z) |
| LSTM autoencoder of matched size on our input (latent 16; not a replication of a published model) | same four | masked reconstruction error |
| PCA reconstruction, on window statistics | same four | reconstruction error |
| Isolation Forest, on window statistics | same four | negated decision score |
| One-class SVM (RBF), on window statistics | same four | negated decision score |
| kNN distance, on window statistics | same four | mean distance to the k nearest training windows |
| Autoencoder + classifier head (the `task/autoencoder_st_t32` model) | supervised | P(attack) |
| Window statistics + logistic regression | supervised | P(attack) |
| Length only (control) | none | −(number of real rows): shorter = more anomalous, fixed here, never chosen on labels |

Window statistics: per feature over the real rows, mean, std, min, max, last − first (65 values).

Hyperparameters are listed in `hyperparameters-neural.md` and `hyperparameters-classical.md` next to this file **before** any test score is computed: library
defaults where one exists, named; one fixed subsample size for methods that do not scale. Nothing is tuned on test.
Model selection uses the regime's validation pool only (val loss for unsupervised neural models, val AUROC benign vs
GridSybil for supervised).

**Prior exposure, disclosed:** the convolutional autoencoder and the supervised model were developed earlier, and
seed-0 results of the benign-only autoencoder on this same test set are already known (window AUROC 0.736 / 0.391 /
0.973 / 0.611 for GridSybil / DataReplay / DoSRandom / DoSDisruptive). Their architecture and training settings are
frozen as they were; the reference detectors have had no such exposure.

## 5. Seeds

Seeds 0–4 for every stochastic detector (initialisation, batch order, subsamples). Deterministic detectors are run
once and marked as such. Five seeds give a rough standard deviation only.

## 6. Evaluation

- Four binary tasks: benign test vs each attack test split. Unit = window; the split is sender-held-out (receiver
  position / velocity features come from receivers that also hear train senders).
- Metrics: AUROC, average precision with the attack share stated, and attack recall + benign test false-positive
  rate at the regime's primary operating point (§3; 95th percentile of benign val for the benign and supervised
  regimes) and at the 99th percentile of benign val.
- **Length control.** Strata by exact number of real rows n = 1, 2, 3, then 4–16 and ≥ 17 (plus the coarse 1–3 and
  ≥ 4). For every detector and for the length-only score: AUROC inside each stratum, and a **length-matched AUROC**
  (mean of the within-n AUROCs for n = 1…32, weighted by the number of benign × attack pairs at that n). A stratum
  with fewer than 200 attack windows or 20 attack vehicles is not reported; this minimum applies to reported strata,
  not to the per-n terms of the length-matched AUROC (a term needs one benign and one attack window).
- **Headline per attack.** GridSybil: pooled AUROC, with the length-matched AUROC next to it. DataReplay and DoS
  (95–100% of windows have 1–3 rows; a short pseudonym life is itself the observable signature of these attacks):
  the length-matched AUROC and the paired difference to length-only; pooled AUROC is secondary.
- **Uncertainty.** Mean and standard deviation over seeds. 95% interval from a cluster bootstrap over test vehicle
  keys, benign and attack keys resampled separately, 1,000 resamples shared by all detectors; in each resample the
  metric is computed per seed and averaged. It covers test sampling only (one split, fixed thresholds). Paired
  intervals are reported for the primary comparisons, fixed here:
  1. each detector − length-only, per attack;
  2. `benign_only` − `mix_all`, same detector, GridSybil;
  3. `benign_gridfolders` − `mix_gridfolders`, same detector, GridSybil;
  4. `supervised` autoencoder + head − each `benign_only` detector, GridSybil (all reported; no "best" is picked on test);
  5. convolutional autoencoder − each reference detector, `benign_only`, GridSybil.
- **Sensitivity rows** (from the score files, no retraining): group 1416 only; benign test windows of the
  GridSybil_0709 folder excluded (F14); benign test restricted to the attack's own family folders.
- Test labels are read only by the evaluation module, after scores are written.

## 7. Records per run (E5)

`runs/<regime>/<detector>/seed<k>/`: `config.json`, `env.json` (packages, device, git commit, dirty flag, sha256 of
every source file of the package), training history, `scores_<split>.npy` for val and every test split (aligned with
the cached window ids), `summary.json`. Aggregated tables are rebuilt from the score files only.

## 8. Out of scope here

Attack-share sweep (E6), random-split comparison and F14 measurement (E7), leave-one-attack-out training (E8),
sequence-unit ablation (E9), window-level plausibility score (E10), 0709 ↔ 1416 transfer.

## 8a. How results will be read (fixed before results; from `../06-reviews/protocol-review-framing.md`)

Window AUROC, GridSybil unless stated.

| Rule | Outcome | Consequence for the paper |
|---|---|---|
| R1 | any benign-only detector ≥ 0.90 | "reconstruction fails" is dropped; the paper becomes a benchmark of detectors |
| R2 | all benign-only detectors in 0.60–0.80, spread ≤ 0.10 | lead message as planned: the training-data assumption matters more than the detector |
| R3 | best benign-only detector in 0.80–0.90 | "partial and model-dependent" |
| R4 | every detector ≤ length-only + 0.02 on an attack | that attack is "not evaluable at this unit"; no detection claim |
| R5 | DataReplay > 0.60 on windows of ≥ 4 rows | the "in-distribution by construction" claim is dropped |
| R6 | `mix_gridfolders` ≥ 0.65 | the attack-majority finding leaves the abstract |
| R7 | seed spread ≥ a difference between two detectors | no ranking of those two |
| Kill | everything within 0.05 of length-only, or the GridSybil derivation check (§2) fails | no paper; thesis appendix |

The gap to published one-class numbers (0.93–0.99) is reported as an observation; no cause ("inflated") is claimed
without E7 or E9.

## 9. Deviations after first results

- 2026-11-10, before any run: §3–§6 rewritten after `../06-reviews/protocol-review-methods.md` (label-free mix regimes, `benign_gridfolders` control, length score fixed as −n, exact-length strata and length-matched AUROC, stratified paired bootstrap, sensitivity rows, hyperparameter table, disclosure of prior test exposure) and §8a added from `../06-reviews/protocol-review-framing.md`. No result existed at that time.
- 2026-11-10, before any detector result was computed: comparison 4 changed from "best benign-only detector" to every benign-only detector (picking the best on test is a selection effect); the stratum minimum clarified; hyperparameter file names corrected. Where §8a and the framing review differ in detail, §8a is authoritative.
