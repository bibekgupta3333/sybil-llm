# Inventory of the baseline studies already run (internal, sourced)

Written 2026-11-10. Read-only inventory: nothing was trained, scored or re-evaluated for this file. Every number is
copied from a file on disk; the source is given per block. Machine-readable copy: [`results.json`](results.json)
(same numbers at full stored precision, each with its path).

**Updated 2026-11-11:** three studies run after this inventory was written are added as **F**, **G**, **H** (one
seed each; numbers copied from their test-metrics JSON files, nothing re-run). They fill the cells that were
missing: benign-only at T = 48, unlabelled mix at T = 64, unlabelled mix at T = 32.

**Updated 2026-11-16 (seed-3 reruns):** one extra seed (3) for five grid cells: S0 AE, C, D, F, G. Run on EC2 g4dn
(Tesla T4, CUDA, torch 2.14.1+cu126, git 0c104ef, `git_dirty: false`), same config as the earlier runs except the seed.
Each run was checked by an independent CPU recompute from `best.pt` (judge verdict; S0 / C / D / F PASS-WITH-NOTES,
G PASS). Seed-0 numbers below are unchanged; each section gets a "Seed 3" block, and a summary table follows the
cross-study table. Run folders (gitignored, on the EC2 host): `task/encoder/runs/ae-seed3`,
`task/autoencoder_st/runs/seed3`, `task/autoencoder_benign_t32/runs/seed3`, `task/autoencoder_benign_t48/runs/seed3`,
`task/autoencoder_t64/runs/seed3`; copies with logs, judge verdict and SHA256SUMS in the private HF model repo
`bibekgupta3333/roadfm-lite-timesnet` under `baselines/<study>/seed3` (`baselines/encoder/ae-seed3`). A, E and H
seed 3 are pending (their inputs are not on HF yet). Not an exact reproduction of seed 0 (different device and seed).

**Updated 2026-11-17 (four seeds per cell):** seeds 1 and 2 for all eight grid cells (S0 AE, A, C, D, E, F, G, H),
seed 3 for A, E, H, and an extra seed 4 for S0 AE, so every cell now has seeds 0–3 (S0 AE: 0–4). Seed 0 = the Mac
(MPS) runs above; seeds 1–3 / 4 = EC2 g4dn (Tesla T4, CUDA, torch 2.14.1+cu126, git 6b08c39, `git_dirty: true` =
`package.json` npm lines only; `git diff 0c104ef 6b08c39` is empty for `task/` and `src/data/encoder_input/`). Same
configs, only the seed changed. A, E and H seeds 1–3 trained on inputs rebuilt on EC2 from the raw HF data
(`benign_gridsybil/T48`, `all/T32_benign_grid`), verified equal to the Mac builds (same window ids and counts,
values within 1e-4 rounding); no split change. Every run was judged by an independent CPU recompute from `best.pt`
(own data reader): all PASS or PASS-WITH-NOTES, no FAIL. Sources per run: HF `bibekgupta3333/roadfm-lite-timesnet`
`baselines/<package>/seed<k>/` (run files, console logs, `judge/verdict.md`, `judge/recompute.{py,log,json}`,
SHA256SUMS; `baselines/encoder/ae-seed4`) and `task/<package>/runs/seed<k>/summary.json` (training record). Each
section gets a "Seeds 1–2" / "Seeds 1–3" block below; the 4-seed summary table follows the seed-3 table. Mean ± sd =
sample sd over the per-seed values (n = 4), computed from the sourced per-seed numbers (seed 0 at the precision
stored in this file). Machine-readable: `seed_reruns` and `seed_summary` per study in `results.json`.

**Updated 2026-11-18:** (1) the "Four seeds per cell" table and `seed_summary` were checked against the run files now
on the Mac (`task/*/runs/seed{1,2,3}/`, `task/encoder/runs/{evaluation,ae-seed3,ae-seed4}/`): every per-seed value
equals its file within 1e-6 (rounded seed-0 notebook values aside) and every mean / sd recomputes to the stored
value; one rounded value in `results.json` replaced (H GridSybil-folders seed 0). (2) New section **M**
(`task/masked_autoencoder/`, one-class masked autoencoder, TabMlp and TabResnet seeds 0–3 at T = 64 / 48 / 32
complete; TabResnet T32 filled later on 2026-11-18). (3) New section **T** (TimesNet one-class `model-t32b-seed0`, context only, one
seed). (4) M and T rows added to the cross-study, other-attack and four-seed tables and to the comparability notes.
Length-matched AUROCs of M and T were recomputed for this file from their score files; every other new number is
copied from a run file and was re-derived from the score files (|Δ| < 1e-6).

**Updated 2026-11-26 (small windows):** new section **W** — the one-class, unlabelled and supervised ConvAE and the
masked AE (TabMlp / TabResnet) at T = 20 / 10 / 5, seeds 0–3 each (60 runs, EC2), on new inputs `all/T{20,10,5}_benign{,_grid}`
that reuse the all/T32 vehicle split (no RULE 3 change), with a T64 → T5 GridSybil overview per baseline. Not yet
judged. Machine-readable: `small_window_baselines` in `results.json`; nothing else in this file changed.

**Where the numbers live (important for the paper):**

- `task/encoder/` stores its test metrics in a file: `task/encoder/runs/evaluation/results.json`.
- The five earlier `task/autoencoder*` studies (A–E) store **only training facts and val thresholds** in `runs/seed0/summary.json`.
  Their **test metrics exist only as stored notebook cell outputs** (rounded to 3–4 decimals) and as README text. No
  `scores_test.npz` or test-metrics JSON is written. A paper needs these re-emitted to a file by an evaluation script.
- F, G, H (added 2026-11-11) **do** write test metrics to a JSON file and per-window score files (`.npy`) in
  `runs/seed0/`: `task/autoencoder_benign_t48/runs/seed0/test_metrics.json`,
  `task/autoencoder_t64/runs/seed0/results.json`, `task/autoencoder_t32/runs/seed0/evaluation.json`.
- `src/baselines/plausibility/` stores everything in `src/runs/baselines/plausibility/summary.json`.
- M (added 2026-11-18) writes `test_metrics.json` and `scores/<split>.npz` per run in
  `task/masked_autoencoder/runs/<encoder>/T<w>-seed<k>/`; T writes `summary.json` and `scores/<split>.npz` in
  `src/runs/evaluation/all/T32_benign/model-t32b-seed0[-random-init]/`.

Study labels used below: **S0** = `task/encoder`, **A** = `task/autoencoder`, **B** = `task/autoencoder_overlap`,
**C** = `task/autoencoder_st`, **D** = `task/autoencoder_benign_t32`, **E** = `task/autoencoder_st_t32`,
**F** = `task/autoencoder_benign_t48`, **G** = `task/autoencoder_t64`, **H** = `task/autoencoder_t32`,
**P** = `src/baselines/plausibility` (A–E as in `task/autoencoder-study-slides.html`), **M** = `task/masked_autoencoder`
(added 2026-11-18), **T** = TimesNet `model-t32b-seed0` (`src/model/all_t32_benign`, context only).

Common to S0 and A–H (M and T differ: see their sections) (sources: each `runs/*/config.json`, `env.json`, `trainer.py` / `train.py`, dataset
`metadata.json`):

- **Sample** = one window: the messages one receiver heard from one sender pseudonym (a link), cut to T rows at
  1 Hz, padded to T, with a prefix mask of real rows. Model input = 13 features + mask channel = 14 × T.
- **13 features** (`src/data/encoder_input/*/metadata.json` → `features`): `claimed_pos_x/y`, `rx_pos_x/y`,
  `claimed_vel_x/y`, `rx_vel_x/y`, `claimed_acl_x/y`, `range`, `bearing`, `log_dtau`. Absolute positions are inputs.
- **Optimizer** Adam, lr 1e-3, weight decay 1e-5, batch 256, ReduceLROnPlateau factor 0.5.
- **Hardware / software**: Apple Silicon Mac, `macOS-26.4-arm64`, device `mps`, Python 3.14.5, torch 2.14.1, numpy
  2.5.3 (sklearn 1.9.1 where used). Every `env.json` has `git_dirty: true` (code state is not exactly the recorded
  commit).
- **Positive class** = attack. Test labels are read after scoring (stated in every README / notebook).
- None of these is the thesis TimesNet model; all READMEs say "not a thesis result".

---

## S0. `task/encoder/` — one-class AE vs supervised encoder vs shortcut baselines, T = 64

Sources: `task/encoder/README.md`; `task/encoder/runs/{ae,supervised}-seed{0,1,2}/{config,env,summary}.json`;
`task/encoder/runs/baselines/summary.json`; `task/encoder/runs/evaluation/results.json`;
`task/encoder/runs/notebook-seed0/results.json`.

| Item | Value |
|---|---|
| Purpose | Can a small 1D-conv encoder, trained one-class (benign-only reconstruction) or supervised, separate benign from GridSybil windows; compared with two shortcut baselines |
| Dataset | `src/data/encoder_input/benign_gridsybil/T64/` (benign + GridSybil, the 2 GridSybil folders, 0709 + 1416 pooled) |
| Split | on-disk 90 / 10 train / test by sender vehicle (D8′, seed 0, stratified). Train 338,001 windows (benign 120,073 / GridSybil 217,928); test 38,426 (benign 12,741 / GridSybil 25,685; GridSybil share 66.8%), 574 test sender vehicles |
| Validation | not on disk: 10% of train sender vehicles held out in code (`val_frac` 0.1, `split_seed` 0). AE: benign vehicles only, fit 107,924 / val 12,149 windows (3,263 / 362 vehicles). Supervised: all train vehicles, fit 304,733 / val 33,268 (4,660 / 518 vehicles) |
| Input | 14 × 64 |
| AE architecture | Conv1d 32 (k3, s1) → 64 (k3, s2) → 64 (k3, s2), BatchNorm + ReLU, flatten 64 × 16 → Linear → z = 16; mirrored decoder. **80,189 params** |
| Supervised model | same encoder + head Linear 16 → 32 → 1, **no decoder**. **37,233 params** |
| Loss | AE: MSE on real rows. Supervised: BCE, pos_weight = benign / GridSybil |
| Training | max epochs 300 (AE; reported runs) / 50 (supervised); early stopping patience 5 on val loss; plateau patience 3; no gradient clipping |
| Val used for | early stopping, LR schedule; AE thresholds = p95 / p99 of **val-benign** scores. Supervised threshold fixed at 0.5 (not tuned) |
| Scores | AE reconstruction error (masked MSE); Mahalanobis of z to the benign-fit z (ridge 1e-6); supervised P(GridSybil); Mahalanobis of the supervised z to benign z |
| Seeds | **3** (0, 1, 2) for AE and supervised; baselines deterministic (AE seeds 3, 4 added 2026-11-16 / 17, blocks below) |
| Epochs / best val | AE: 121 / 103 / 123 epochs, best val loss 0.12086 / 0.12138 / 0.12312. Supervised: 19 / 16 / 16, best val loss 0.19700 / 0.19488 / 0.19286 |
| Runtime | AE 249.9 / 211.9 / 248.7 s; supervised 59.8 / 49.5 / 48.3 s; baselines 28.3 s (MPS / CPU) |

**Test results** (`runs/evaluation/results.json`; mean ± std over 3 seeds, n = 3; CI = 1,000 vehicle-bootstrap
resamples of the 574 test sender vehicles, per seed):

| Method | AUROC per seed (0 / 1 / 2) | AUROC mean ± std | 95% CI (seed 0) | PR-AUC mean | At threshold (mean over seeds): precision / recall / F1 / FPR |
|---|---|---|---|---|---|
| AE reconstruction (one-class) | 0.6546 / 0.6580 / 0.6536 | 0.655 ± 0.002 | 0.630 – 0.677 | 0.802 | val p95: 0.885 / 0.199 / 0.325 / 0.052 · val p99: 0.951 / 0.105 / 0.190 / 0.011 |
| AE latent Mahalanobis (one-class) | 0.3334 / 0.3023 / 0.3208 | 0.319 ± 0.016 | 0.317 – 0.352 | 0.589 | val p95: 0.633 / 0.045 / 0.084 / 0.052 · val p99: 0.789 / 0.019 / 0.036 / 0.010 |
| Supervised encoder, latent Mahalanobis | 0.7648 / 0.7785 / 0.8184 | 0.787 ± 0.028 | 0.753 – 0.777 | 0.899 | val p95: 0.952 / 0.502 / 0.656 / 0.050 · val p99: 0.986 / 0.382 / 0.548 / 0.011 |
| Supervised encoder + head | 0.9259 / 0.9271 / 0.9295 | 0.928 ± 0.002 | 0.921 – 0.932 | 0.970 | fixed 0.5: 0.987 / 0.816 / 0.893 / 0.023 |
| Length only (−n_messages) | 0.5348 | — | 0.511 – 0.557 | 0.699 | — |
| 53 window statistics + logistic regression | 0.8549 | — | 0.846 – 0.865 | 0.926 | fixed 0.5: 0.865 / 0.832 / 0.849 / 0.261 |

PR-AUC chance level = 0.668 (GridSybil share). Stats-LR train AUROC 0.8511 (`baselines/summary.json`).

**AUROC by window length** (mean over seeds; `results.json` → `methods.*.summary.strata_auroc`, `strata`):

| n_messages | windows (benign / GridSybil) | AE recon | AE Mahalanobis | sup. Mahalanobis | supervised | stats LR | length only (README: best direction per stratum) |
|---|---|---|---|---|---|---|---|
| ≤ 3 | 6,720 (1,822 / 4,898) | 0.552 | 0.300 | 0.776 | 0.916 | 0.681 | 0.505 |
| 4–16 | 15,865 (5,236 / 10,629) | 0.651 | 0.256 | 0.849 | 0.924 | 0.851 | 0.536 |
| 17–48 | 11,800 (4,452 / 7,348) | 0.710 | 0.309 | 0.863 | 0.922 | 0.872 | 0.507 |
| 49–64 | 4,041 (1,231 / 2,810) | 0.752 | 0.419 | 0.905 | 0.949 | 0.927 | 0.527 |

**Per-feature AE reconstruction-error AUROC** (mean of 3 seeds): range 0.707, claimed_pos_y 0.695, log_dtau 0.640,
bearing 0.635, rx_pos_y 0.614, claimed_pos_x 0.608, claimed_acl_y 0.592, rx_vel_y 0.558, rx_pos_x 0.549,
claimed_vel_y 0.540, claimed_acl_x 0.537, rx_vel_x 0.530, claimed_vel_x 0.501.

**Not reported in S0:** per-track scores; per-folder / per-period breakdown; recall at a fixed FPR for the
supervised head.

**Caveats** (README + files): the length-only direction (−n) was picked on test labels; the AE Mahalanobis score is
inverted in every seed (direction fixed before testing, reported as is); the first AE runs with a 50-epoch cap hit
`max_epochs` (kept in `runs/_old_50ep/`; README quotes AUROC 0.645 for them) and were replaced by 300-epoch-cap runs
(a protocol change after seeing results); no hyperparameter search; std is over 3 seeds only; CI covers
test-vehicle sampling only. `runs/notebook-seed0/` is a one-notebook replication of seed 0 (AE 0.6546, supervised
0.9259: identical to the package runs).

**Seed 3 (AE only, 2026-11-16, EC2 T4)** — sources: `task/encoder/runs/ae-seed3/{config,env,summary}.json`;
`task/encoder/runs/evaluation/results.json` as written on EC2 (`config.runs` = `[ae-seed3]`); judge verdict
`baselines/encoder/ae-seed3/judge/verdict.md` (HF). 130 epochs, best val loss 0.12044 at epoch 125, early stopping
(patience 5), 281.5 s wall, CUDA. Test: reconstruction AUROC **0.6615** (vehicle-bootstrap CI 0.637 – 0.683), PR-AUC
0.807; val p95 precision / recall / F1 / FPR 0.890 / 0.205 / 0.333 / 0.051; Mahalanobis AUROC 0.3332, PR-AUC 0.598.
Reconstruction AUROC by length bin ≤ 3 / 4–16 / 17–48 / 49–64: 0.555 / 0.660 / 0.717 / 0.755. Length-only 0.5348 and
stats-LR 0.8549 unchanged (deterministic). Pooled over seeds 0–3 (judge, for information only): reconstruction
0.6569 ± 0.0036, Mahalanobis 0.3224 ± 0.0147. Judge: PASS-WITH-NOTES. **Caution:** the EC2
`runs/evaluation/results.json` holds seed 3 only, so its "mean ± std" has n = 1; it is not the 3-seed table above
(that file is on the Mac).

**Seed 4 (AE only, 2026-11-17, EC2 T4)** — sources: `task/encoder/runs/ae-seed4/summary.json`;
`task/encoder/runs/evaluation_ae_seed34/results.json` (per_seed seed 4); judge verdict + recompute
`baselines/encoder/ae-seed4/judge/` (HF). 114 epochs, best val loss 0.12119 at epoch 109, early stopping (patience 5),
184.1 s wall, CUDA. Test: reconstruction AUROC **0.6532**, PR-AUC 0.801; val p95 precision / recall / F1 / FPR 0.888
/ 0.197 / 0.323 / 0.050; Mahalanobis AUROC 0.3269, PR-AUC 0.594. Reconstruction AUROC by length bin ≤ 3 / 4–16 /
17–48 / 49–64: 0.533 / 0.650 / 0.713 / 0.755. Judge: PASS-WITH-NOTES. Pooled per-seed values: seeds 0–3
reconstruction 0.6569 ± 0.0036, Mahalanobis 0.3224 ± 0.0146; seeds 0–4 0.6562 ± 0.0035 / 0.3233 ± 0.0128 (seeds 0–2
MPS, 3–4 CUDA). **Caution:** `evaluation_ae_seed34/` (on HF under `ae-seed4`) is a 2-seed evaluation (seeds 3 + 4,
n = 2: 0.657 ± 0.006 / 0.330 ± 0.004), not the 0–2 table and not a 5-seed mean.

---

## A. `task/autoencoder/` — unsupervised AE, benign + GridSybil both "normal", T = 48

Sources: `task/autoencoder/README.md`; `task/autoencoder/runs/seed0/{config,env,summary}.json`;
`task/autoencoder/autoencoder_T48.ipynb` (stored outputs, cells 4, 16–30).

| Item | Value |
|---|---|
| Purpose | Does plain reconstruction error flag GridSybil when the AE is trained on all train windows without labels |
| Dataset | `src/data/encoder_input/benign_gridsybil/T48/` (same messages, vehicle split and normalisation as T64) |
| Split | on-disk 90 / 10 by sender vehicle. Train 355,382 windows (benign 126,474 / GridSybil 228,908); test 40,469 (benign 13,405 / GridSybil 27,064; share 0.669) |
| Validation | 10% of train sender vehicles held out in code (seed 0): fit 320,168 / val 35,214 windows (518 of 5,178 vehicles) |
| Input / model | 14 × 48; ConvAutoencoder (3 conv + BatchNorm, z = 16, mirrored decoder), **71,741 params** |
| Loss | masked MSE on real rows |
| Training | max 300 epochs, early stopping patience 10 (`min_delta` 0.0), plateau patience 5, no gradient clipping |
| Val used for | early stopping, LR schedule, threshold = p95 of **all** val errors (val holds both classes; labels unused) |
| Score | reconstruction error per window only (no Mahalanobis, no track level) |
| Seeds | **1** (seed 0) in this table; seeds 1–3 added 2026-11-17 (block at the end of this section) |
| Training record | best val masked MSE 0.13979 at epoch 170; stop at 180; 9 LR halvings; 911.9 s train / 918.0 s wall, MPS |

**Test** (notebook cells 22–30): AUROC **0.484**, PR-AUC 0.661 (chance 0.669). At val p95 (θ 0.3724): precision
0.654, recall 0.046, F1 0.087, benign FPR 0.049 (TN 12,742 / FP 663 / FN 25,810 / TP 1,254). Length-only AUROC
0.530 (fewer messages = more suspicious).

AUROC by length: ≤ 3 rows 0.378 (6,849 windows), 4–16 0.473 (16,468), 17–32 0.554 (9,431), 33–48 0.537 (7,721).

Per-feature error AUROC: range 0.650, claimed_pos_y 0.623, log_dtau 0.617, claimed_pos_x 0.576, bearing 0.546, …,
claimed_vel_x 0.280, claimed_vel_y 0.269 (GridSybil rebuilt better on claimed velocity).

**Caveats:** GridSybil is 64% of train windows and is treated as normal; one seed, no CI; `runs/seed0_featlog/` is a
re-run of the same seed (identical best epoch and val loss), not a second seed.

**Seeds 1–3 (2026-11-17, EC2 T4, on the EC2 rebuild of T48)** — sources: `runs/seed{1,2,3}/summary.json`; executed
notebook copies `runs/seed{k}/autoencoder_T48.executed.ipynb` (the only place the test metrics live); judge verdict +
recompute `baselines/autoencoder/seed{k}/judge/` (HF). Val vehicles are drawn with the run seed, so val differs across
seeds (518 vehicles each; 34,975 / 33,185 / 34,881 windows); test fixed.

| Seed | best val MSE (epoch) / stop / LR halvings | AUROC | PR-AUC | θ val p95 | recall / F1 / benign FPR @ θ | AUROC ≤ 3 / 4–16 / 17–32 / 33–48 rows | Judge |
|---|---|---|---|---|---|---|---|
| 1 | 0.13049 (169) / 179 / 9 | 0.4859 | 0.6619 | 0.3744 | 0.0457 / 0.0853 / 0.0517 | 0.388 / 0.478 / 0.551 / 0.533 | PASS-WITH-NOTES |
| 2 | 0.13924 (157) / 167 / 6 | 0.4850 | 0.6621 | 0.3777 | 0.0425 / 0.0798 / 0.0454 | 0.370 / 0.476 / 0.556 / 0.537 | PASS-WITH-NOTES |
| 3 | 0.12886 (152) / 162 / 7 | 0.4856 | 0.6618 | 0.3694 | 0.0488 / 0.0907 / 0.0531 | 0.380 / 0.476 / 0.552 / 0.538 | PASS-WITH-NOTES |

Length-only 0.5304 in every seed. Seeds 0–3: AUROC 0.485 ± 0.001. Judge notes: the executed copies' markdown prose
describes the Mac seed-0 run (quote code-cell outputs only); test metrics exist only in the notebooks (no JSON, no
score files).

---

## B. `task/autoencoder_overlap/` — unsupervised AE on stride-24 overlap windows, T = 48

Sources: `task/autoencoder_overlap/README.md`; `runs/seed0/{config,env,summary}.json`;
`autoencoder_overlap_T48.ipynb` (cells 4, 8, 22–38).

| Item | Value |
|---|---|
| Purpose | A with 50% window overlap, an on-disk val split, a second score (Mahalanobis on z) and per-track aggregation |
| Dataset | `src/data/encoder_input/benign_gridsybil/T48_tvt_s24/` (links > 48 messages cut at stride 24 + one end-aligned window) |
| Split | on-disk **80 / 10 / 10** by sender vehicle (test vehicles = the T48 / T64 test vehicles). Windows train 331,215 (benign 117,551 / GridSybil 213,664), val 41,440 (14,172 / 27,268), test 42,653 (13,964 / 28,689; share 0.673). Tracks 283,212 / 35,028 / 35,957. Sender vehicles 4,603 / 575 / 574 |
| Overlap | 6,980,356 unique messages, 8,845,352 message rows (× 1.267); 90.7–91.3% of tracks have one window |
| Input / model | 14 × 48; same ConvAutoencoder, **71,741 params** |
| Loss | masked MSE |
| Training | max 300, early stopping patience 10 with `min_delta` 1e-4 (relative), plateau patience 5, no gradient clipping |
| Val used for | early stopping, LR schedule, thresholds p95 / p99 of val scores (both classes in val, labels unused) |
| Scores | reconstruction error; Mahalanobis on z fitted on **train z** (benign + GridSybil; cond 38.6, shrinkage 0); per window and per track (max / mean) |
| Seeds | **1** (seed 0) |
| Training record | best val 0.14267 at epoch 139; stop at 149; 8 LR halvings; 841.6 s train / 850.5 s wall, MPS |

**Test** (notebook cell 26 / 38; chance PR-AUC 0.673 window, 0.665 track):

| Score | Level | AUROC | PR-AUC | val p95: precision / recall / F1 / benign FPR | val p99: recall / benign FPR |
|---|---|---|---|---|---|
| reconstruction | window | **0.484** | 0.664 | 0.637 / 0.043 / 0.080 / 0.050 | 0.009 / 0.008 |
| reconstruction | track max | 0.482 | 0.655 | 0.630 / 0.044 / 0.082 / 0.051 | 0.009 / 0.008 |
| reconstruction | track mean | 0.481 | 0.654 | 0.622 / 0.044 / 0.082 / 0.053 | 0.008 / 0.009 |
| Mahalanobis | window | **0.337** | 0.595 | 0.573 / 0.044 / 0.082 / 0.068 | 0.009 / 0.002 |
| Mahalanobis | track max | 0.307 | 0.565 | 0.499 / 0.039 / 0.072 / 0.077 | 0.010 / 0.011 |
| Mahalanobis | track mean | 0.302 | 0.559 | 0.461 / 0.036 / 0.066 / 0.083 | 0.008 / 0.011 |

No track with mixed labels. **Not found in this study:** length-only AUROC and AUROC by window length (not
computed in the notebook).

**Caveats:** overlapping windows of one link share messages (window-level counts are not independent); one seed;
track ids hold the true sender (F13, privileged grouping).

---

## C. `task/autoencoder_st/` — supervised AE (reconstruction + classifier head), T = 48

Sources: `task/autoencoder_st/README.md`; `runs/seed0/{config,env,summary}.json`; `autoencoder_st_T48.ipynb`
(cells 4, 8, 14–36).

| Item | Value |
|---|---|
| Purpose | Same autoencoder with labels: a head on z trained jointly with reconstruction |
| Dataset | `src/data/encoder_input/benign_gridsybil/T48_tvt/` (no overlap) |
| Split | on-disk 80 / 10 / 10 by sender vehicle. Windows train 316,050 (benign 112,890 / GridSybil 203,160), val 39,332 (13,584 / 25,748), test 40,469 (13,405 / 27,064; share 0.669). Test 35,957 tracks, 574 sender vehicles |
| Input / model | 14 × 48; ConvAutoencoder 71,741 + head Linear 16 → 32 → ReLU → Dropout 0.2 → 32 → 1 (577) = **72,318 params** |
| Loss | BCEWithLogits (pos_weight 0.5557 = 112,890 / 203,160) + 1.0 · masked MSE |
| Training | max 100 epochs, grad clip 1.0, early stopping on **val AUROC** (patience 10, min_delta 1e-4 absolute), plateau patience 5 |
| Val used for | LR schedule, early stopping, `best.pt`, both thresholds (max F1 on val; benign FPR 5% on val) |
| Score | P(GridSybil) per window; per track max / mean; reconstruction error reported as a secondary score |
| Seeds | **1** (seed 0) in this table; seeds 1–3 added 2026-11-17 (block at the end of this section) |
| Training record | best val AUROC 0.9268 at epoch 44 (val PR-AUC 0.9684, val F1 0.8921); stop at 54; 3 LR halvings; 329.5 s train / 338.8 s wall, MPS |

**Test** (notebook cell 20 / 36; chance PR-AUC 0.669):

| Level | AUROC | PR-AUC | val max-F1 θ: precision / recall / F1 / bal. acc. / benign FPR | val FPR-5% θ: precision / recall / F1 / benign FPR | fixed 0.5: recall / F1 / benign FPR |
|---|---|---|---|---|---|
| window | **0.9289** | 0.9710 | (θ 0.3507) 0.9708 / 0.8387 / 0.9000 / 0.8939 / 0.0509 | (θ 0.3932) 0.9745 / 0.8334 / 0.8984 / 0.0440 | 0.8267 / 0.8979 / 0.0299 |
| track max | 0.9277 | 0.9702 | (θ 0.3507) 0.9673 / 0.8403 / 0.8994 / 0.8920 / 0.0563 | (θ 0.4235) 0.9744 / 0.8325 / 0.8978 / 0.0434 | 0.8277 / 0.8975 / 0.0333 |
| track mean | 0.9286 | 0.9704 | (θ 0.3507) 0.9703 / 0.8389 / 0.8998 / 0.8940 / 0.0509 | (θ 0.3940) 0.9744 / 0.8332 / 0.8983 / 0.0435 | 0.8253 / 0.8969 / 0.0300 |

Confusion (window): max-F1 TN 12,723 / FP 682 / FN 4,365 / TP 22,699; FPR-5% TN 12,815 / FP 590 / FN 4,509 /
TP 22,555.

Length-only AUROC (best direction) 0.5304. AUROC by real rows: 1–3 0.9170 (6,849 windows), 4–16 0.9252 (16,468),
17–47 0.9249 (12,497), 48 0.9523 (4,655). Reconstruction error alone: AUROC 0.4780.

**Not reported:** per-folder / per-period breakdown.

**Caveats:** one seed, no CI; trained and tested on GridSybil only (nothing about unseen attacks); track ids hold the
true sender (F13); val BCE plateaued from epoch 20 while train BCE kept falling (README).

**Seed 3 (2026-11-16, EC2 T4)** — sources: `runs/seed3/{config,env,summary}.json`;
`runs/seed3/autoencoder_st_T48.executed.ipynb` (cells 20, 28, 30, 36); judge verdict
`baselines/autoencoder_st/seed3/judge/verdict.md` (HF). Best val AUROC 0.9244 at epoch 43; stop at 53; 2 LR halvings;
460.5 s train / 531.5 s wall, CUDA. Test window: AUROC **0.9253**, PR-AUC 0.9695; val max-F1 θ 0.2556: precision /
recall / F1 / benign FPR 0.9562 / 0.8406 / 0.8947 / 0.0778; val FPR-5% θ 0.3222: 0.9717 / 0.8278 / 0.8940 / 0.0487.
Track max / mean AUROC 0.9245 / 0.9250. Length-only 0.5304; reconstruction error alone 0.4814. Judge: PASS-WITH-NOTES
(benign FPR at the val max-F1 θ is 0.078 vs 0.051 for seed 0 because the val max-F1 θ fell lower; the FPR-5% θ gives
0.049 on test).

**Seeds 1–2 (2026-11-17, EC2 T4)** — sources: `runs/seed{1,2}/summary.json`; executed notebook copies (RUN pinned to
the seed folder) `runs/seed{k}/autoencoder_st_T48.executed.ipynb`; judge verdict + recompute
`baselines/autoencoder_st/seed{k}/judge/` (HF). Test window, judge recompute = notebook:

| Seed | best val AUROC (epoch) / stop | AUROC | PR-AUC | val max-F1 θ: P / R / F1 / benign FPR | val FPR-5% θ: P / R / F1 / benign FPR | track max / mean | Judge |
|---|---|---|---|---|---|---|---|
| 1 | 0.9249 (39) / 49 | 0.9296 | 0.9712 | (0.2373) 0.9628 / 0.8433 / 0.8991 / 0.0657 | (0.2803) 0.9746 / 0.8333 / 0.8984 / 0.0438 | 0.9279 / 0.9287 | PASS-WITH-NOTES |
| 2 | 0.9245 (37) / 47 | 0.9273 | 0.9703 | (0.3538) 0.9778 / 0.8269 / 0.8960 / 0.0379 | (0.3513) 0.9772 / 0.8273 / 0.8960 / 0.0390 | 0.9260 / 0.9266 | PASS-WITH-NOTES |

Seeds 0–3: AUROC 0.9278 ± 0.0019. **Threshold drift (judge):** the val max-F1 θ swings 0.24–0.35 across seeds and
moves the test benign FPR 0.038–0.078; the val FPR-5% θ is stable (test FPR 0.039–0.049, F1 0.8967 ± 0.0021, recall
0.8304 ± 0.0034 over seeds 0–3). Report the FPR-5% operating point as the headline and max-F1 with its spread. The
notebooks' context-only task/encoder row shows the README fallback (0.928) because
`task/encoder/runs/evaluation/results.json` was set aside during execution (restored, sha256 equal).

---

## D. `task/autoencoder_benign_t32/` — one-class AE (benign only), all 8 folders, T = 32, four attacks tested

Sources: `task/autoencoder_benign_t32/README.md`; `runs/seed0/{config,env,summary}.json`;
`autoencoder_benign_T32.ipynb` (cells 4, 10, 16, 20–43); `src/data/encoder_input/all/T32_benign/metadata.json`.

| Item | Value |
|---|---|
| Purpose | One-class (genuine-only) reconstruction baseline, evaluated against all four Sybil attack types |
| Dataset | `src/data/encoder_input/all/T32_benign/` (benign of all 8 folders; attack windows only of the test vehicles) |
| Split | on-disk **80 / 10 / 10** by vehicle key `<group>:<vehicle>`, strata (group, class of the key), seed 0 (a new split, RULE 3, 2026-10-29). Benign windows train 502,773 / val 65,313 / test 60,142 (4,206 / 524 / 525 vehicles; 401,986 / 52,108 / 48,188 tracks). Attack test splits: GridSybil 28,087, DataReplay 76,283, DoSRandom 199,834, DoSDisruptive 199,826 windows (172 / 173 / 173 / 173 vehicles; 225 attack test vehicles in total) |
| Short windows | 1–3-row share: benign 13.7–13.9%, GridSybil 20.4%, DataReplay 94.9%, DoSRandom 99.7%, DoSDisruptive 99.7% (F11) |
| Input / model | 14 × 32; ConvAutoencoder with BatchNorm, z = 16, **63,293 params** |
| Loss | masked MSE |
| Training | max 300, grad clip 1.0, early stopping patience 10 (`min_delta` 1e-4 relative), plateau patience 5; `train.py` never opens the test splits |
| Val used for | early stopping, LR schedule, thresholds p95 / p99 of **benign val** scores |
| Scores | reconstruction error; Mahalanobis on z fitted on benign train z (cond 30.2, shrinkage 0); per window and per track (max / mean) |
| Seeds | **1** (seed 0) in this table; seeds 1–3 added 2026-11-17 (block at the end of this section) |
| Training record | best benign val masked MSE 0.10503 at epoch 151; stop at 161; 7 LR halvings; 1,165.2 s train / 1,168.4 s wall (19.4 min), MPS |

**Test, window level, each attack split vs the same 60,142 benign test windows** (notebook cells 27 / 43):

| Attack (windows) | Score | AUROC | AUROC ≥ 4 rows (n attack) | PR-AUC (chance) | recall @ val p95 | recall @ val p99 |
|---|---|---|---|---|---|---|
| GridSybil (28,087) | reconstruction | **0.736** | 0.778 (22,365) | 0.629 (0.318) | 0.298 | 0.179 |
| | Mahalanobis | 0.510 | 0.540 | 0.394 | 0.127 | 0.048 |
| | length only | 0.559 | — | 0.367 | — | — |
| DataReplay (76,283) | reconstruction | **0.391** | 0.349 (3,883) | 0.499 (0.559) | 0.046 | 0.026 |
| | Mahalanobis | 0.336 | 0.295 | 0.450 | 0.008 | 0.000 |
| | length only | 0.952 | — | 0.934 | — | — |
| DoSRandom (199,834) | reconstruction | **0.973** | 1.000 (523) | 0.991 (0.769) | 0.880 | 0.766 |
| | Mahalanobis | 0.837 | 0.999 | 0.920 | 0.223 | 0.021 |
| | length only | 0.967 | — | 0.981 | — | — |
| DoSDisruptive (199,826) | reconstruction | **0.611** | 1.000 (524) | 0.864 (0.769) | 0.292 | 0.192 |
| | Mahalanobis | 0.556 | 0.999 | 0.810 | 0.075 | 0.014 |
| | length only | 0.967 | — | 0.981 | — | — |

Benign test FPR: 0.048 at val p95, 0.009 at val p99 (reconstruction, window); Mahalanobis 0.048 / 0.010.

**Track level AUROC** (max / mean): GridSybil reconstruction 0.713 / 0.721, Mahalanobis 0.488 / 0.491; DataReplay
0.348 / 0.364, 0.294 / 0.316; DoSRandom 0.969 / 0.973, 0.812 / 0.833; DoSDisruptive 0.581 / 0.595, 0.520 / 0.543.
Recall at val p95 per track (reconstruction, max / mean): GridSybil 0.256 / 0.276, DataReplay 0.043 / 0.048,
DoSRandom 0.867 / 0.888, DoSDisruptive 0.278 / 0.300 (benign FPR 0.048 / 0.049).

Per-feature reconstruction-error AUROC, highest per attack: GridSybil range 0.768, claimed_pos_y 0.735; DataReplay
range 0.644, claimed_pos_y 0.639; DoSRandom range 0.945, claimed_pos_x 0.929; DoSDisruptive range 0.801,
claimed_pos_y 0.783. The ≥ 4-row DoS windows have median normalised log_dtau 12.135 (benign −0.134).

**Not reported:** per-folder / per-period (0709 vs 1416) breakdown; F1 / precision at the thresholds.

**Caveats:** one seed, no CI; F11 (DataReplay / DoS windows are almost all 1–3 rows; length alone matches or beats
the model there); F14 (GridSybil_0709 near-copies can straddle the split; **stated, not measured** for this split);
F1 (benign traffic appears once per attack family, ≈ 4×, so benign windows are not independent); track ids hold the
true sender (F13); 306 unknown-label GridSybil_0709 windows dropped when the input was built (`agent.md`).

**Seed 3 (2026-11-16, EC2 T4)** — sources: `runs/seed3/{config,env,summary}.json`;
`runs/seed3/autoencoder_benign_T32.executed.ipynb` (cells 27, 37, 43); judge verdict
`baselines/autoencoder_benign_t32/seed3/judge/verdict.md` (HF). Best benign val masked MSE 0.10820 at epoch 170; stop
at 180; 8 LR halvings; 1,913.8 s train / 1,999.0 s wall, CUDA; Mahalanobis cond 30.0. Test window AUROC
reconstruction / Mahalanobis (seed 0 in brackets): GridSybil **0.730** / 0.508 (0.736 / 0.510), ≥ 4 rows 0.775 / 0.539;
DataReplay 0.386 / 0.365 (0.391 / 0.336); DoSRandom 0.967 / 0.918 (0.973 / 0.837); DoSDisruptive 0.593 / 0.595
(0.611 / 0.556). GridSybil recall @ val p95 / p99 (reconstruction) 0.298 / 0.180, Mahalanobis @ p95 0.142; benign
test FPR @ val p95 0.048. Length-only unchanged (0.559 / 0.952 / 0.967 / 0.967). Judge: PASS-WITH-NOTES
(reconstruction within 0.02 AUROC of seed 0 everywhere; Mahalanobis on z is seed-sensitive outside GridSybil, e.g.
DoSRandom recall @ p95 0.555 vs 0.223; all values stay at or below the length-only reference, so only GridSybil beats
the length shortcut, as before).

**Seeds 1–2 (2026-11-17, EC2 T4)** — sources: `runs/seed{1,2}/summary.json`; executed notebook copies
`runs/seed{k}/autoencoder_benign_T32.executed.ipynb` (the only place the test metrics live); judge verdict + recompute
`baselines/autoencoder_benign_t32/seed{k}/judge/` (HF). Window AUROC reconstruction / Mahalanobis, each attack split vs
the 60,142 benign test windows:

| Seed | best val MSE (epoch) / stop / LR halvings | GridSybil | DataReplay | DoSRandom | DoSDisruptive | GridSybil recall @ p95 (recon) | DoSRandom recall @ p95 recon / Maha | Judge |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.10502 (181) / 191 / 8 | 0.732 / 0.487 | 0.375 / 0.327 | 0.960 / 0.874 | 0.584 / 0.578 | 0.300 | 0.830 / 0.316 | PASS-WITH-NOTES |
| 2 | 0.10898 (178) / 188 / 10 | 0.736 / 0.514 | 0.397 / 0.339 | 0.967 / 0.907 | 0.605 / 0.617 | 0.301 | 0.860 / 0.578 | PASS-WITH-NOTES |

GridSybil ≥ 4 rows (recon) 0.775 / 0.777; benign test FPR @ val p95 0.049 / 0.049. Seeds 0–3: GridSybil
reconstruction 0.7335 ± 0.0029, Mahalanobis 0.505 ± 0.012; DataReplay 0.387 ± 0.010; DoSRandom 0.967 ± 0.005 (Maha
0.884 ± 0.036, Maha recall @ p95 0.22–0.58); DoSDisruptive 0.598 ± 0.012 (Maha 0.587 ± 0.026). Judge notes:
Mahalanobis on z is seed-sensitive outside GridSybil (report its spread, not one seed); seed 2's per-feature mean
reconstruction errors on GridSybil are outlier-driven (do not quote mean errors over attack windows); every DataReplay
/ DoS score stays at or below length-only (0.952 / 0.967), so only GridSybil beats length.

---

## E. `task/autoencoder_st_t32/` — supervised AE, benign of all 8 folders + GridSybil, T = 32

Sources: `task/autoencoder_st_t32/README.md`; `runs/seed0/{config,env,summary}.json`; `autoencoder_st_T32.ipynb`
(cells 4, 6, 10, 12, 20–42); `src/data/encoder_input/all/T32_benign_grid/metadata.json`.

| Item | Value |
|---|---|
| Purpose | C at T = 32 with a larger benign side (all 8 folders) |
| Dataset | `src/data/encoder_input/all/T32_benign_grid/` (benign of 8 folders + GridSybil of the 2 GridSybil folders; DataReplay / DoS attack windows dropped) |
| Split | on-disk **85 / 7.5 / 7.5** by vehicle key, strata (group, class of the key), seed 0 (a new split, RULE 3, 2026-11-03); 7,511 keys → 6,383 / 564 / 564. Windows train 779,087 (benign 535,501 / GridSybil 243,586), val 67,582 (47,279 / 20,303), test 67,100 (45,448 / 21,652; share 0.323). Vehicles with windows 5,933 / 524 / 524; test 54,102 tracks |
| Input / model | 14 × 32; ConvAutoencoder 63,293 + head 577 = **63,870 params** |
| Loss | BCEWithLogits (pos_weight 2.1984 = 535,501 / 243,586) + 1.0 · masked MSE |
| Training | max 100, grad clip 1.0, early stopping on val AUROC (patience 10, min_delta 1e-4), plateau patience 5 |
| Val used for | LR schedule, early stopping, `best.pt`, both thresholds (val max F1; val benign FPR 5%) |
| Score | P(GridSybil) per window; per track max / mean |
| Seeds | **1** (seed 0) in this table; seeds 1–3 added 2026-11-17 (block at the end of this section) |
| Training record | val AUROC 0.8941 at epoch 1; best 0.9220 at epoch 17 (val PR-AUC 0.9151, val F1 0.8748); stop at 27; 1 LR halving; 335.2 s train / 350.3 s wall, MPS |

**Test** (notebook cell 24 / 42; chance PR-AUC 0.323 window, 0.329 track):

| Level | AUROC | PR-AUC | val max-F1 θ: precision / recall / F1 / bal. acc. / benign FPR | val FPR-5% θ: precision / recall / F1 / benign FPR | fixed 0.5: recall / F1 / benign FPR |
|---|---|---|---|---|---|
| window | **0.9263** | 0.9248 | (θ 0.5958) 0.9721 / 0.8192 / 0.8891 / 0.9040 / 0.0112 | (θ 0.2960) 0.8936 / 0.8389 / 0.8654 / 0.0476 | 0.8233 / 0.8838 / 0.0190 |
| track max | 0.9248 | 0.9240 | (θ 0.6435) 0.9716 / 0.8152 / 0.8865 / 0.9018 / 0.0117 | (θ 0.3373) 0.8936 / 0.8350 / 0.8633 / 0.0487 | 0.8227 / 0.8794 / 0.0237 |
| track mean | 0.9269 | 0.9255 | (θ 0.5259) 0.9682 / 0.8174 / 0.8865 / 0.9022 / 0.0131 | (θ 0.2857) 0.8912 / 0.8387 / 0.8641 / 0.0502 | 0.8190 / 0.8856 / 0.0150 |

Confusion (window): max-F1 TN 44,939 / FP 509 / FN 3,914 / TP 17,738; FPR-5% TN 43,285 / FP 2,163 / FN 3,489 /
TP 18,163.

Length-only AUROC (best direction) 0.5599. AUROC by real rows: 1–3 0.9149 (10,585 windows), 4–16 0.9225 (27,057),
17–31 0.9125 (15,860), 32 0.9300 (13,598), ≥ 4 rows 0.9240 (56,515). Reconstruction error alone: AUROC 0.5975.

Per period: 0709 AUROC 0.9267, PR-AUC 0.9242, F1 0.8879, benign FPR 0.0107 (52,961 windows); 1416 AUROC 0.9251,
PR-AUC 0.9270, F1 0.8936, benign FPR 0.0130 (14,139 windows), at the val max-F1 θ.

Benign FPR per folder at val max-F1: 0.0105–0.0149 (8 folders); at val FPR-5%: 0.0431–0.0536; AUROC of each
folder's benign vs same-period GridSybil: 0.9249–0.9268.

**Caveats:** one seed, no CI; GridSybil is the only attack in train and test; F1 (the extra six folders add mostly
copies of the same benign traffic; DoSRandom and DoSDisruptive benign counts are identical); F14 (**stated, not
measured** for this split); F13.

**Seeds 1–3 (2026-11-17, EC2 T4, on the EC2 rebuild of all/T32_benign_grid)** — sources: `runs/seed{1,2,3}/summary.json`;
executed notebook copies `runs/seed{k}/autoencoder_st_T32.executed.ipynb` (the only place the test metrics live); judge
verdict + recompute `baselines/autoencoder_st_t32/seed{k}/judge/` (HF). Test window, judge recompute = notebook:

| Seed | best val AUROC (epoch) / stop | AUROC | PR-AUC | val max-F1 θ: P / R / F1 / benign FPR | val FPR-5% θ: R / F1 / benign FPR | track max / mean | recon alone | Judge |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.9210 (14) / 24 | 0.9294 | 0.9267 | (0.7415) 0.9794 / 0.8166 / 0.8906 / 0.0082 | (0.3398) 0.8395 / 0.8635 / 0.0500 | 0.9274 / 0.9294 | 0.5955 | PASS-WITH-NOTES |
| 2 | 0.9221 (22) / 32 | 0.9302 | 0.9275 | (0.6978) 0.9742 / 0.8200 / 0.8904 / 0.0104 | (0.3326) 0.8435 / 0.8662 / 0.0496 | 0.9281 / 0.9305 | 0.6000 | PASS-WITH-NOTES |
| 3 | 0.9219 (18) / 28 | 0.9298 | 0.9274 | (0.6891) 0.9806 / 0.8190 / 0.8926 / 0.0077 | (0.3176) 0.8430 / 0.8634 / 0.0523 | 0.9286 / 0.9305 | 0.5966 | PASS-WITH-NOTES |

Length-only 0.5599 in every seed. Seeds 0–3: AUROC 0.9289 ± 0.0018, PR-AUC 0.9266 ± 0.0013, F1 at val max-F1 0.8907
± 0.0014 (benign FPR 0.0094 ± 0.0017), recall at val FPR-5% 0.8412 ± 0.0023 (benign FPR 0.0499 ± 0.0019). Judge
notes: the val max-F1 θ moved from 0.596 (seed 0) to 0.69–0.74 but sits on a flat F1 plateau (test F1 0.889–0.893);
seed 0 used the Mac copy of the input, seeds 1–3 the EC2 rebuild (counts identical, not byte for byte).

---

## F. `task/autoencoder_benign_t48/` — one-class AE (benign only), benign + GridSybil folders, T = 48

Sources: `task/autoencoder_benign_t48/runs/seed0/{config,env,summary}.json`;
`task/autoencoder_benign_t48/runs/seed0/test_metrics.json` (all test numbers below).

| Item | Value |
|---|---|
| Purpose | D's one-class protocol on the benign + GridSybil folders at T = 48 (the missing benign-only T = 48 cell) |
| Dataset | `src/data/encoder_input/benign_gridsybil/T48_benign/` (benign subset of the T48_tvt split + `test_gridsybil`) |
| Split | on-disk 80 / 10 / 10 by sender vehicle (T48_tvt's). Benign windows train 112,890 / val 13,584 / test 13,405 (train 3,223 / val 402 / test 401 vehicles); `test_gridsybil` 27,064 windows (173 vehicles); GridSybil share of the test pair 0.669; 0 test vehicles in train or val |
| Input / model | 14 × 48; ConvAutoencoder, **71,741 params** |
| Loss | masked MSE |
| Training | max 300, grad clip 1.0, early stopping patience 10 (`min_delta` 1e-4), plateau patience 5 |
| Val used for | early stopping, LR schedule, thresholds p95 / p99 of **benign val** scores |
| Scores | reconstruction error; Mahalanobis on z fitted on benign train z (cond 14.0, shrinkage 0); per window and per track |
| Seeds | **1** (seed 0) in this table; seeds 1–3 added 2026-11-17 (block at the end of this section) |
| Training record | best benign val masked MSE 0.14770 at epoch 143; stop at 153; 6 LR halvings; 1,259.4 s train / 1,265.4 s wall, MPS |

**Test, window level, GridSybil (27,064) vs benign test (13,405)** (`test_metrics.json`):

| Score | AUROC | AUROC 1–3 rows | AUROC ≥ 4 rows | PR-AUC (chance 0.669) | val p95: recall / benign FPR | val p99: recall / benign FPR |
|---|---|---|---|---|---|---|
| reconstruction | **0.689** | 0.599 | 0.720 | 0.829 | 0.243 / 0.047 | 0.154 / 0.008 |
| Mahalanobis | 0.437 | 0.406 | 0.453 | 0.655 | 0.062 / 0.044 | 0.017 / 0.009 |
| length only (−n_messages, fixed in advance) | 0.530 | — | — | 0.697 | — | — |

Track level (12,053 benign / 23,904 GridSybil tracks) AUROC max / mean: reconstruction 0.678 / 0.681, Mahalanobis
0.430 / 0.427. 1–3-row share: benign 13.8%, GridSybil 18.5%.

**Caveats:** one seed, no CI; normalisation fitted on benign train rows only (differs from T48 / T48_tvt); track ids
hold the true sender (F13); precision / F1 at the thresholds are not in the file.

**Seed 3 (2026-11-16, EC2 T4)** — sources: `runs/seed3/{config,env,summary}.json`; `runs/seed3/test_metrics.json`;
judge verdict `baselines/autoencoder_benign_t48/seed3/judge/verdict.md` (HF). Best benign val masked MSE 0.14998 at
epoch 126; stop at 136; 6 LR halvings; 392.2 s train / 426.9 s wall, CUDA; Mahalanobis cond 15.8. Test window:
reconstruction AUROC **0.685** (1–3 rows 0.590, ≥ 4 rows 0.716), PR-AUC 0.826, val p95 recall / benign FPR 0.236 /
0.045; Mahalanobis AUROC 0.454, PR-AUC 0.689, val p95 recall / FPR 0.111 / 0.045. Track max / mean: reconstruction
0.672 / 0.675, Mahalanobis 0.436 / 0.435. Length-only 0.530 (unchanged). Judge: PASS-WITH-NOTES (all headline metrics
within 0.02 of seed 0; Mahalanobis recall @ p95 0.111 vs 0.062, the latent score is seed-sensitive, AUROC still
below 0.5).

**Seeds 1–2 (2026-11-17, EC2 T4)** — sources: `runs/seed{1,2}/{summary,test_metrics}.json`; judge verdict + recompute
`baselines/autoencoder_benign_t48/seed{k}/judge/` (HF; recompute = test_metrics.json, max |Δ| ≤ 3e-8).

| Seed | best val MSE (epoch) / stop / LR halvings | recon AUROC | Maha AUROC | recall @ val p95 recon / Maha (benign FPR recon) | recon AUROC ≥ 4 rows | Judge |
|---|---|---|---|---|---|---|
| 1 | 0.15301 (96) / 106 / 6 | 0.6775 | 0.4312 | 0.235 / 0.060 (0.046) | 0.707 | PASS-WITH-NOTES |
| 2 | 0.15341 (107) / 117 / 5 | 0.6764 | 0.4629 | 0.238 / 0.101 (0.046) | 0.711 | PASS-WITH-NOTES |

Seeds 0–3: reconstruction **0.682 ± 0.006**, Mahalanobis 0.446 ± 0.015. Judge: seed 0's 0.689 is the best of four, so
quote the multi-seed 0.682 ± 0.006, not 0.689; Mahalanobis stays below chance and below length (0.530) in every seed.

---

## G. `task/autoencoder_t64/` — unlabelled AE (benign + GridSybil pooled), T = 64

Sources: `task/autoencoder_t64/runs/seed0/{config,env,summary}.json`; `task/autoencoder_t64/runs/seed0/results.json`
(all test numbers below).

| Item | Value |
|---|---|
| Purpose | A's unlabelled protocol at T = 64 on S0's data (the missing unlabelled-mix T = 64 cell) |
| Dataset | `src/data/encoder_input/benign_gridsybil/T64/` (S0's input, test windows and normalisation) |
| Split | on-disk 90 / 10 by sender vehicle. Train 338,001 windows (benign 120,073 / GridSybil 217,928; share 64.5%); test 38,426 (benign 12,741 / GridSybil 25,685; share 0.668) |
| Validation | 10% of train sender vehicles held out in code (S0's supervised-mode draw, seed 0): fit 304,733 windows (benign 107,810 / GridSybil 196,923; **GridSybil 64.6%**), val 33,268 (12,263 / 21,005); 4,660 / 518 vehicles |
| Input / model | 14 × 64; `task/encoder`'s ConvAutoencoder, **80,189 params** |
| Loss | masked MSE |
| Training | max 300, early stopping patience 10 (`min_delta` 0.0), plateau patience 5, no gradient clipping in the config; labels never read in training (`label_reads: []`) |
| Val used for | early stopping, LR schedule, thresholds p95 / p99 of **all** val scores (both classes, labels unused) |
| Scores | reconstruction error; Mahalanobis on z of the fit windows (both classes; ridge 1e-6); per window and per track |
| Seeds | **1** (seed 0) in this table; seeds 1–3 added 2026-11-17 (block at the end of this section) |
| Training record | best val masked MSE 0.14518 at epoch 161; stop at 171; 7 LR halvings; 7,460.2 s train / 7,465.3 s wall, MPS |

**Test, window level** (`results.json`; chance PR-AUC 0.668):

| Score | AUROC | AUROC 1–3 rows | AUROC ≥ 4 rows | PR-AUC | val p95: precision / recall / F1 / benign FPR | val p99: recall / benign FPR |
|---|---|---|---|---|---|---|
| reconstruction | **0.482** | 0.361 | 0.512 | 0.660 | 0.651 / 0.045 / 0.084 / 0.049 | 0.010 / 0.007 |
| Mahalanobis | 0.352 | 0.317 | 0.362 | 0.599 | 0.623 / 0.047 / 0.087 / 0.057 | 0.006 / 0.004 |
| length only (−n_messages, fixed before reading labels) | 0.535 | 0.505 | 0.516 | 0.699 | — | — |

AUROC by S0's length bins (reconstruction): ≤ 3 0.361, 4–16 0.473, 17–48 0.558, 49–64 0.515. Track level (35,957
tracks, 0 mixed-label) AUROC max / mean: reconstruction 0.481 / 0.480, Mahalanobis 0.340 / 0.335.

**Caveats:** GridSybil is the majority of the fit windows and is treated as normal; one seed, no CI; early-stopping
and plateau patience (10 / 5) differ from S0's (5 / 3); F13 for the track level.

**Seed 3 (2026-11-16, EC2 T4)** — sources: `runs/seed3/{config,env,summary}.json`; `runs/seed3/results.json`; judge
verdict `baselines/autoencoder_t64/seed3/judge/verdict.md` (HF). Best val masked MSE 0.14530 at epoch 117; stop at
127; 6 LR halvings; 624.7 s train / 681.2 s wall, CUDA; `label_reads: []`. Test window: reconstruction AUROC **0.480**
(1–3 rows 0.367, ≥ 4 rows 0.509), PR-AUC 0.659, val p95 precision / recall / F1 / benign FPR 0.648 / 0.045 / 0.084 /
0.049; Mahalanobis AUROC 0.337, PR-AUC 0.593, val p95 recall / FPR 0.044 / 0.057. Track max / mean: reconstruction
0.479 / 0.477, Mahalanobis 0.327 / 0.321. Length-only 0.535 (unchanged). Judge: PASS (every metric within 0.02 of
seed 0; reading unchanged).

**Seeds 1–2 (2026-11-17, EC2 T4)** — sources: `runs/seed{1,2}/{summary,results}.json`; judge verdict + recompute
`baselines/autoencoder_t64/seed{k}/judge/` (HF; recompute = results.json, max |Δ| ≤ 2.2e-7). Evaluate was re-run with
`task/encoder/runs/evaluation/results.json` set aside (its context lookup expects a seed-0 entry the EC2 file lacks;
`context_task_encoder_benign_only_T64` is null; file restored, sha256 equal); `label_reads: []`.

| Seed | best val MSE (epoch) / stop / LR halvings | recon AUROC | Maha AUROC | recall @ val p95 recon / Maha | recon AUROC 1–3 / ≥ 4 rows | Judge |
|---|---|---|---|---|---|---|
| 1 | 0.15114 (100) / 110 / 7 | 0.4810 | 0.3430 | 0.045 / 0.044 | 0.387 / 0.507 | PASS-WITH-NOTES |
| 2 | 0.14419 (114) / 124 / 5 | 0.4835 | 0.3282 | 0.043 / 0.033 | 0.390 / 0.510 | PASS-WITH-NOTES |

Seeds 0–3: reconstruction 0.482 ± 0.002, Mahalanobis 0.340 ± 0.010; both below the length rule (0.535) in every seed.
Judge notes: seed 1's best val MSE is ~4% above the other seeds (weaker optimum, test in line); seed 2's Mahalanobis is
0.024 below seed 0, within the spread of a below-chance score.

---

## H. `task/autoencoder_t32/` — unlabelled AE (benign of 8 folders + GridSybil pooled), T = 32

Sources: `task/autoencoder_t32/runs/seed0/{config,env,summary}.json`;
`task/autoencoder_t32/runs/seed0/evaluation.json` (all test numbers below).

| Item | Value |
|---|---|
| Purpose | The unlabelled protocol on E's input (the missing unlabelled-mix T = 32 cell) |
| Dataset | `src/data/encoder_input/all/T32_benign_grid/` (E's input, split, test windows and normalisation) |
| Split | on-disk 85 / 7.5 / 7.5 by vehicle key. Windows train 779,087 (benign 535,501 / GridSybil 243,586; **GridSybil 31.3%**), val 67,582 (47,279 / 20,303; 30.0%), test 67,100 (45,448 / 21,652; 32.3%); train 5,933 / val 524 vehicles |
| Input / model | 14 × 32; D's ConvAutoencoder, **63,293 params** |
| Loss | masked MSE |
| Training | max 300, grad clip 1.0, early stopping patience 10 (`min_delta` 1e-4), plateau patience 5; label files never opened in training (`labels_read: false`, guard in process and workers) |
| Val used for | early stopping, LR schedule, thresholds p95 / p99 of the **mixed** val scores (labels unused) |
| Scores | reconstruction error; Mahalanobis on z (cond 46.9, shrinkage 0); per window and per track |
| Seeds | **1** (seed 0) in this table; seeds 1–3 added 2026-11-17 (block at the end of this section) |
| Training record | best val masked MSE 0.10781 at epoch 206; stop at 216; 9 LR halvings; 9,137.7 s train / 9,157.8 s wall, MPS |

**Test, window level** (`evaluation.json`):

| Test set | Score | AUROC | AUROC 1–3 rows | AUROC ≥ 4 rows | PR-AUC (chance) | val p95: precision / recall / benign FPR | val p99: recall / benign FPR |
|---|---|---|---|---|---|---|---|
| all 8 folders (67,100) | reconstruction | **0.610** | 0.496 | 0.648 | 0.436 (0.323) | 0.560 / 0.108 / 0.040 | 0.027 / 0.007 |
| | Mahalanobis | 0.398 | 0.385 | 0.414 | 0.331 | 0.466 / 0.097 / 0.053 | 0.045 / 0.003 |
| | length only (−n_messages) | 0.560 | — | — | 0.373 | — | — |
| GridSybil folders only (33,614: benign 11,962 / GridSybil 21,652) | reconstruction | 0.613 | — | 0.654 | — | 0.828 / 0.108 / 0.040 | — |
| | Mahalanobis | 0.396 | — | — | — | — | — |
| | length only | 0.564 | — | — | — | — | — |

Per period (all folders, reconstruction): 0709 0.611, 1416 0.605. Track level (54,102 tracks, 0 mixed-label) AUROC
max / mean: reconstruction 0.594 / 0.599, Mahalanobis 0.362 / 0.362.

**Caveats:** one seed, no CI; F1 (benign repeated per family), F14 (**stated, not measured** for this split), F13;
F1 score at the thresholds is not in the file.

**Seeds 1–3 (2026-11-17, EC2 T4, on the EC2 rebuild of all/T32_benign_grid)** — sources:
`runs/seed{1,2,3}/{summary,evaluation}.json`; judge verdict + recompute `baselines/autoencoder_t32/seed{k}/judge/` (HF;
recompute = evaluation.json over 42 values per seed, max |Δ| ≤ 4.3e-7; label-file guard on in process and workers).

| Seed | best val MSE (epoch) / stop / LR halvings | recon AUROC | Maha AUROC | recon 1–3 / ≥ 4 rows | recall / benign FPR @ val p95 (recon) | track max / mean (recon) | GridSybil folders recon / Maha | Judge |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.10635 (223) / 233 / 8 | 0.6171 | 0.4295 | 0.514 / 0.651 | 0.107 / 0.040 | 0.601 / 0.606 | 0.620 / 0.428 | PASS-WITH-NOTES |
| 2 | 0.11075 (197) / 207 / 9 | 0.6070 | 0.4056 | 0.502 / 0.644 | 0.102 / 0.041 | 0.590 / 0.596 | 0.609 / 0.403 | PASS-WITH-NOTES |
| 3 | 0.10540 (210) / 220 / 8 | 0.6129 | 0.4501 | 0.522 / 0.647 | 0.108 / 0.042 | 0.596 / 0.602 | 0.614 / 0.447 | PASS-WITH-NOTES |

Length-only 0.5599 (GridSybil folders 0.5636) in every seed. Seeds 0–3: reconstruction 0.612 ± 0.004 (GridSybil
folders 0.614 ± 0.004), Mahalanobis 0.421 ± 0.024 (below chance in every seed, the least stable score).
Reconstruction is ~0.05 above length and at chance on 1–3-row windows. Judge notes: `labels_read: false` in
summary.json is a hard-coded literal (the evidence is the label-file guard, tested); `history.jsonl` minimum can sit
1–2 epochs after `best_epoch` (relative min_delta, by design).

---

## M. `task/masked_autoencoder/` — one-class masked autoencoder (widedeep-style TabMlp / TabResnet), T = 64 / 48 / 32

Added 2026-11-18. Sources: `task/masked_autoencoder/README.md` (method, deviations, protocol); the executed notebook
`task/masked_autoencoder/masked_autoencoder.ipynb`; per run `task/masked_autoencoder/runs/<encoder>/T<w>-seed<k>/`
`{config,env,summary,test_metrics}.json` and `scores/<split>.npz` (`<encoder>` = `mlp` TabMlp or `resnet` TabResnet).
Pooled, length-only, 1–3-row and ≥ 4-row AUROCs are copied from `test_metrics.json`; for this inventory every one of
them was recomputed from the score files with sklearn and equals the file (|Δ| < 1e-6, 20 finished runs). The
length-matched AUROC, the benign short-vs-long AUROC and the 1-row shares are **not** in the run files: they were
recomputed here from `scores/*.npz` with the notebook's method (`length_matched_auroc`, section 5b; T64 labels read
through `task.masked_autoencoder.data.make_source`, as the notebook does). Machine-readable: `masked_autoencoder` in
`results.json`.

| Item | Value |
|---|---|
| Purpose | A second model family for the one-class question: a masking-based autoencoder after pytorch-widedeep's self-supervised `EncoderDecoderTrainer(encoder=TabMlp / TabResnet, masked_prob=0.2)`, trained one-class on **exactly the windows of the one-class ConvAE cells** (S0 AE at T64, F at T48, D at T32), so its numbers sit next to theirs |
| Method | each window (T × 13, prefix row mask) flattened to 13 T columns + T row-mask columns (padding cells 0); `RandomObfuscator`: every step hides each real cell with p = 0.2 (set to 0); encoder **TabMlp** [14 T → 64 → 16] (Linear → ReLU → Dropout 0.1) with mirrored decoder [16 → 64 → 13 T], or **TabResnet** DenseResnet blocks [64, 32, 16] (dropout 0.1) with mirrored decoder + Linear(64, 13 T); z = 16. Loss = widedeep `EncoderDecoderLoss`: per window, mean over hidden cells of (pred − true)² / var_f |
| Deviations from widedeep 1.7.0 (README "Deviations") | (1) the encoder sees the masked input (widedeep's TabMlp / TabResnet path computes `masked_x` but encodes the clean input); (2) TabMlp decoder ends in a plain Linear (widedeep's ReLU output cannot produce negative z-scores); (3) mask columns never hidden or reconstructed, padding never hidden / in the loss / scored; (4) loss variance pooled per feature over real cells, not per flattened column; (5) batch 256, early stopping and ReduceLROnPlateau (widedeep: batch 32, fixed epochs, no schedule); (6) small widths with z = 16 (widedeep defaults [200, 100] / [200, 100, 100]). Library not installed (it would downgrade numpy); re-implemented in PyTorch |
| Data per T | T64: `benign_gridsybil/T64` via `task/encoder` mode ae — benign train windows, val = 10% of their sender vehicles (split seed 0): fit 107,924 (3,263 vehicles) / val 12,149 (362 vehicles); test = on-disk mixed test, 12,741 benign + 25,685 GridSybil (labels read after scoring). T48: `benign_gridsybil/T48_benign` on-disk train 112,890 (3,223 vehicles) / val 13,584 (402) / test 13,405 benign + `test_gridsybil` 27,064. T32: `all/T32_benign` on-disk train 502,773 (4,206) / val 65,313 (524) / test 60,142 benign + `test_gridsybil` 28,087 / `test_datareplay` 76,283 / `test_dosrandom` 199,834 / `test_dosdisruptive` 199,826. Test vehicles 574 / 574 / 750, 0 in train or val (`test_metrics.json` → `label_checks`) |
| Params | TabMlp **113,616 / 85,760 / 57,904**; TabResnet **128,224 / 100,576 / 72,928** at T = 64 / 48 / 32 (one-class ConvAE 80,189 / 71,741 / 63,293) |
| Training | AdamW (widedeep default) lr 1e-3, weight decay 0.01, batch 256, max 300 epochs, no grad clip; early stopping + ReduceLROnPlateau (factor 0.5) on the benign val masked-reconstruction MSE with the sibling's patience: T64 5 / 3 / `min_delta` 0, T48 and T32 10 / 5 / relative 1e-4. The seed sets weights, batch order and training masks. **One configuration, not tuned** |
| Val used for | early stopping, LR schedule, thresholds p95 / p99 of **benign val** scores (window, and track max / mean) |
| Scores (per window, real cells only, saved before any label is read) | `masked_recon`: cells partitioned once into 5 fixed groups (`eval_mask_seed` 20261117, same for every run), each group hidden in turn, MSE of each real cell reconstructed while hidden; `full_recon`: MSE with nothing hidden; `mahalanobis`: z of the clean input vs a Gaussian fitted on benign train z (shrinkage only if cond > 1e6: never needed, cond 56–1,229 TabMlp, 12–32 TabResnet (T32: 14.0–18.8), shrinkage 0). Per track: max / mean over a link's windows |
| Labels | training reads no label at T48 / T32 (`summary.json` → `label_reads` = []); at T64 the train labels are read once to select the benign windows (as S0); one-class = benign windows selected with labels, no label in the loss |
| Seeds | 0–3 per T and encoder. **TabMlp: 12 / 12 runs complete. TabResnet: 12 / 12 runs complete** (T32 seeds 0–3 finished later on 2026-11-18; every run stopped by early stopping) |
| Hardware / software | Mac MPS, torch 2.14.1 (`env.json`); TabMlp git cff8c53, TabResnet git 4250b0e, both `git_dirty: true` |

**Training record** (`summary.json`; every run stopped by early stopping):

| Encoder | T | best epoch (seeds 0 / 1 / 2 / 3) | epochs run | LR halvings | train minutes | best val masked MSE |
|---|---|---|---|---|---|---|
| TabMlp | 64 | 63 / 41 / 46 / 68 | 68 / 46 / 51 / 73 | 3 / 2 / 3 / 4 | 0.9 / 0.7 / 0.7 / 1.0 | 0.2917 / 0.2926 / 0.2910 / 0.2913 |
| TabMlp | 48 | 101 / 80 / 110 / 84 | 111 / 90 / 120 / 94 | 5 / 5 / 5 / 3 | 1.6 / 1.3 / 1.7 / 1.3 | 0.3469 / 0.3457 / 0.3427 / 0.3468 |
| TabMlp | 32 | 43 / 58 / 45 / 48 | 53 / 68 / 55 / 58 | 3 / 4 / 3 / 3 | 4.2 / 5.4 / 4.4 / 4.6 | 0.3248 / 0.3192 / 0.3267 / 0.3259 |
| TabResnet | 64 | 161 / 129 / 174 / 164 | 166 / 134 / 179 / 169 | 6 / 6 / 5 / 5 | 6.6 / 5.3 / 6.9 / 6.6 | 0.2247 / 0.2330 / 0.2257 / 0.2234 |
| TabResnet | 48 | 198 / 209 / 214 / 193 | 208 / 219 / 224 / 203 | 7 / 7 / 5 / 8 | 9.3 / 8.6 / 8.8 / 8.6 | 0.2804 / 0.2791 / 0.2749 / 0.2823 |
| TabResnet | 32 | 138 / 108 / 154 / 148 | 148 / 118 / 164 / 158 | 7 / 5 / 7 / 5 | 29.2 / 22.0 / 33.9 / 32.8 | 0.2462 / 0.2491 / 0.2482 / 0.2452 |

**TabMlp, GridSybil vs benign test, window level, seeds 0–3** (mean ± sample sd, n = 4; other columns are means;
length-only = −n_messages, fixed in advance):

| T | Score | Seeds 0 / 1 / 2 / 3 | Mean ± sd | Length-matched (mean ± sd) | ≥ 4 rows | 1–3 rows | PR-AUC (chance) | val p95: recall / benign FPR | val p99: recall / benign FPR | Track max / mean | Length-only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | masked recon | 0.5329 / 0.5316 / 0.5285 / 0.5311 | **0.531 ± 0.002** | 0.510 ± 0.002 | 0.550 | 0.398 | 0.707 (0.668) | 0.093 / 0.048 | 0.027 / 0.010 | 0.521 / 0.522 | 0.535 |
| 64 | full recon | 0.5400 / 0.5407 / 0.5342 / 0.5402 | **0.539 ± 0.003** | 0.530 ± 0.003 | 0.558 | 0.448 | 0.722 (0.668) | 0.112 / 0.047 | 0.051 / 0.011 | 0.529 / 0.530 | 0.535 |
| 64 | Mahalanobis | 0.3774 / 0.4129 / 0.4128 / 0.3893 | **0.398 ± 0.018** | 0.403 ± 0.019 | 0.412 | 0.374 | 0.634 (0.668) | 0.065 / 0.055 | 0.038 / 0.011 | 0.385 / 0.385 | 0.535 |
| 48 | masked recon | 0.6060 / 0.6073 / 0.6089 / 0.6072 | **0.607 ± 0.001** | 0.589 ± 0.001 | 0.619 | 0.499 | 0.773 (0.669) | 0.172 / 0.045 | 0.076 / 0.008 | 0.594 / 0.595 | 0.530 |
| 48 | full recon | 0.6006 / 0.6034 / 0.6048 / 0.6014 | **0.603 ± 0.002** | 0.595 ± 0.002 | 0.618 | 0.526 | 0.772 (0.669) | 0.174 / 0.045 | 0.084 / 0.009 | 0.589 / 0.590 | 0.530 |
| 48 | Mahalanobis | 0.4204 / 0.3846 / 0.4105 / 0.3948 | **0.403 ± 0.016** | 0.408 ± 0.019 | 0.414 | 0.382 | 0.641 (0.669) | 0.072 / 0.052 | 0.038 / 0.009 | 0.385 / 0.383 | 0.530 |
| 32 | masked recon | 0.6556 / 0.6589 / 0.6567 / 0.6552 | **0.657 ± 0.002** | 0.631 ± 0.001 | 0.674 | 0.529 | 0.514 (0.318) | 0.210 / 0.052 | 0.087 / 0.009 | 0.630 / 0.638 | 0.559 |
| 32 | full recon | 0.6479 / 0.6545 / 0.6511 / 0.6474 | **0.650 ± 0.003** | 0.638 ± 0.003 | 0.669 | 0.563 | 0.515 (0.318) | 0.214 / 0.051 | 0.094 / 0.008 | 0.623 / 0.630 | 0.559 |
| 32 | Mahalanobis | 0.3957 / 0.3912 / 0.4219 / 0.4241 | **0.408 ± 0.017** | 0.415 ± 0.019 | 0.424 | 0.373 | 0.316 (0.318) | 0.077 / 0.053 | 0.048 / 0.011 | 0.382 / 0.382 | 0.559 |

**TabMlp, T = 32, the four attacks vs the same benign test (60,142 windows), seeds 0–3** (mean ± sd; recall / FPR
means; 1-row share = windows with 1 real row, attack / benign test):

| Attack (attack windows; 1-row share attack / benign) | Score | Pooled AUROC | Length-matched | ≥ 4 rows (attack windows) | 1–3 rows | val p95: recall / benign FPR | Length-only |
|---|---|---|---|---|---|---|---|
| GridSybil (28,087; 0.084 / 0.052) | masked recon | 0.657 ± 0.002 | 0.631 ± 0.001 | 0.674 ± 0.002 (22,365) | 0.529 | 0.210 / 0.052 | 0.559 |
| GridSybil | full recon | 0.650 ± 0.003 | 0.638 ± 0.003 | 0.669 ± 0.003 | 0.563 | 0.214 / 0.051 | 0.559 |
| GridSybil | Mahalanobis | 0.408 ± 0.017 | 0.415 ± 0.019 | 0.424 ± 0.020 | 0.373 | 0.077 / 0.053 | 0.559 |
| DataReplay (76,283; 0.702 / 0.052) | masked recon | 0.693 ± 0.003 | **0.490 ± 0.009** | 0.359 ± 0.004 (3,883) | 0.557 | 0.137 / 0.052 | 0.952 |
| DataReplay | full recon | 0.540 ± 0.010 | 0.491 ± 0.011 | 0.301 ± 0.002 | 0.521 | 0.091 / 0.051 | 0.952 |
| DataReplay | Mahalanobis | 0.269 ± 0.036 | 0.437 ± 0.030 | 0.415 ± 0.014 | 0.327 | 0.009 / 0.053 | 0.952 |
| DoSRandom (199,834; 0.888 / 0.052) | masked recon | 0.988 ± 0.001 | 0.970 ± 0.001 | 1.000 ± 0.000 (523) | 0.969 | 0.942 / 0.052 | 0.967 |
| DoSRandom | full recon | 0.976 ± 0.002 | 0.971 ± 0.002 | 1.000 ± 0.000 | 0.959 | 0.884 / 0.051 | 0.967 |
| DoSRandom | Mahalanobis | 0.394 ± 0.073 | 0.656 ± 0.118 | 0.715 ± 0.172 | 0.472 | 0.020 / 0.053 | 0.967 |
| DoSDisruptive (199,826; 0.888 / 0.052) | masked recon | 0.851 ± 0.006 | **0.679 ± 0.005** | 1.000 ± 0.000 (524) | 0.746 | 0.448 / 0.052 | 0.967 |
| DoSDisruptive | full recon | 0.732 ± 0.007 | 0.683 ± 0.003 | 1.000 ± 0.000 | 0.704 | 0.367 / 0.051 | 0.967 |
| DoSDisruptive | Mahalanobis | 0.222 ± 0.032 | 0.447 ± 0.070 | 0.269 ± 0.050 | 0.288 | 0.002 / 0.053 | 0.967 |

**Length confound (TabMlp).** `masked_recon` partly measures window length: a 1-row window has no time neighbours,
so a hidden cell must be rebuilt from the other 12 features of the same message. On **benign test windows alone**,
`masked_recon` separates 1–3-row from ≥ 4-row windows with AUROC 0.712 ± 0.009 / 0.698 ± 0.004 / 0.688 ± 0.004 at
T = 64 / 48 / 32 (`full_recon` 0.545 / 0.537 / 0.541; Mahalanobis 0.286 / 0.300 / 0.413). Consequences: the pooled
T32 DataReplay 0.693 is this length effect (70.2% of DataReplay windows have 1 row vs 5.2% of benign; length-matched
0.490, ≥ 4 rows 0.359) and **must not be reported as detection**; it is also far below length only (0.952).
DoSDisruptive falls from 0.851 pooled to 0.679 length-matched, below length only (0.967). DoSRandom stays separable
inside every length (length-matched 0.970 ± 0.001, ≥ 4-row AUROC 1.000 on only 523 attack windows), the only one of
DataReplay / DoS where a reconstruction score is not below length only. On GridSybil the TabMlp reconstruction
scores lose 0.008–0.026 under length matching (T32 masked 0.657 → 0.631); Mahalanobis does not drop. The length-matched AUROC covers 100% of attack windows in every case (every attack length
also has benign windows).

**TabResnet (status 2026-11-18: T64, T48, T32 complete).** GridSybil vs benign test, window level, seeds 0–3 (same
columns as the TabMlp table; from `runs/resnet/T{64,48,32}-seed{0..3}/test_metrics.json`, length-matched recomputed
from `scores/*.npz` as above; the T64 / T48 rows re-checked against the run files, unchanged):

| T | Score | Seeds 0 / 1 / 2 / 3 | Mean ± sd | Length-matched (mean ± sd) | ≥ 4 rows | 1–3 rows | PR-AUC (chance) | val p95: recall / benign FPR | val p99: recall / benign FPR | Track max / mean | Length-only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | masked recon | 0.5947 / 0.5897 / 0.5959 / 0.5978 | **0.595 ± 0.003** | 0.578 ± 0.004 | 0.604 | 0.507 | 0.754 (0.668) | 0.140 / 0.050 | 0.056 / 0.009 | 0.586 / 0.587 | 0.535 |
| 64 | full recon | 0.6044 / 0.6010 / 0.6068 / 0.6078 | **0.605 ± 0.003** | 0.609 ± 0.003 | 0.608 | 0.629 | 0.765 (0.668) | 0.155 / 0.052 | 0.078 / 0.010 | 0.596 / 0.597 | 0.535 |
| 64 | Mahalanobis | 0.3660 / 0.3598 / 0.3649 / 0.3809 | **0.368 ± 0.009** | 0.347 ± 0.012 | 0.385 | 0.298 | 0.618 (0.668) | 0.059 / 0.054 | 0.025 / 0.013 | 0.355 / 0.352 | 0.535 |
| 48 | masked recon | 0.6493 / 0.6508 / 0.6518 / 0.6480 | **0.650 ± 0.002** | 0.640 ± 0.002 | 0.654 | 0.596 | 0.802 (0.669) | 0.208 / 0.046 | 0.110 / 0.009 | 0.639 / 0.640 | 0.530 |
| 48 | full recon | 0.6472 / 0.6488 / 0.6478 / 0.6466 | **0.648 ± 0.001** | 0.650 ± 0.001 | 0.654 | 0.641 | 0.798 (0.669) | 0.195 / 0.045 | 0.101 / 0.009 | 0.636 / 0.637 | 0.530 |
| 48 | Mahalanobis | 0.3718 / 0.3875 / 0.3814 / 0.3834 | **0.381 ± 0.007** | 0.366 ± 0.006 | 0.398 | 0.318 | 0.641 (0.669) | 0.083 / 0.052 | 0.052 / 0.011 | 0.358 / 0.353 | 0.530 |
| 32 | masked recon | 0.6954 / 0.6964 / 0.6950 / 0.6978 | **0.696 ± 0.001** | 0.678 ± 0.001 | 0.706 | 0.619 | 0.566 (0.318) | 0.250 / 0.051 | 0.127 / 0.009 | 0.674 / 0.681 | 0.559 |
| 32 | full recon | 0.6839 / 0.6845 / 0.6840 / 0.6887 | **0.685 ± 0.002** | 0.688 ± 0.004 | 0.699 | 0.664 | 0.553 (0.318) | 0.233 / 0.049 | 0.116 / 0.008 | 0.660 / 0.667 | 0.559 |
| 32 | Mahalanobis | 0.3757 / 0.3740 / 0.3820 / 0.3806 | **0.378 ± 0.004** | 0.387 ± 0.003 | 0.402 | 0.320 | 0.330 (0.318) | 0.100 / 0.049 | 0.073 / 0.010 | 0.347 / 0.344 | 0.559 |

**TabResnet, T = 32, the four attacks vs the same benign test (60,142 windows), seeds 0–3** (same columns as the
TabMlp four-attack table; ≥ 4-row and 1–3-row columns are means except where ± is given):

| Attack (attack windows; 1-row share attack / benign) | Score | Pooled AUROC | Length-matched | ≥ 4 rows (attack windows) | 1–3 rows | val p95: recall / benign FPR | Length-only |
|---|---|---|---|---|---|---|---|
| GridSybil (28,087; 0.084 / 0.052) | masked recon | 0.696 ± 0.001 | 0.678 ± 0.001 | 0.706 ± 0.001 (22,365) | 0.619 | 0.250 / 0.051 | 0.559 |
| GridSybil | full recon | 0.685 ± 0.002 | 0.688 ± 0.004 | 0.699 ± 0.002 | 0.664 | 0.233 / 0.049 | 0.559 |
| GridSybil | Mahalanobis | 0.378 ± 0.004 | 0.387 ± 0.003 | 0.402 ± 0.008 | 0.320 | 0.100 / 0.049 | 0.559 |
| DataReplay (76,283; 0.702 / 0.052) | masked recon | 0.695 ± 0.008 | **0.559 ± 0.007** | 0.375 ± 0.012 (3,883) | 0.605 | 0.142 / 0.051 | 0.952 |
| DataReplay | full recon | 0.460 ± 0.011 | 0.579 ± 0.006 | 0.305 ± 0.009 | 0.549 | 0.053 / 0.049 | 0.952 |
| DataReplay | Mahalanobis | 0.271 ± 0.023 | 0.474 ± 0.023 | 0.313 ± 0.034 | 0.443 | 0.012 / 0.049 | 0.952 |
| DoSRandom (199,834; 0.888 / 0.052) | masked recon | 0.994 ± 0.000 | 0.988 ± 0.000 | 1.000 ± 0.000 (523) | 0.984 | 0.971 / 0.051 | 0.967 |
| DoSRandom | full recon | 0.964 ± 0.002 | 0.985 ± 0.002 | 1.000 ± 0.000 | 0.957 | 0.825 / 0.049 | 0.967 |
| DoSRandom | Mahalanobis | 0.629 ± 0.165 | 0.827 ± 0.089 | 0.762 ± 0.064 | 0.795 | 0.193 / 0.049 | 0.967 |
| DoSDisruptive (199,826; 0.888 / 0.052) | masked recon | 0.859 ± 0.004 | **0.734 ± 0.001** | 1.000 ± 0.000 (524) | 0.784 | 0.452 / 0.051 | 0.967 |
| DoSDisruptive | full recon | 0.672 ± 0.009 | 0.770 ± 0.007 | 1.000 ± 0.000 | 0.722 | 0.311 / 0.049 | 0.967 |
| DoSDisruptive | Mahalanobis | 0.281 ± 0.033 | 0.488 ± 0.031 | 0.332 ± 0.178 | 0.445 | 0.032 / 0.049 | 0.967 |

Reading (TabResnet T32): as with TabMlp, DataReplay is not detected — pooled masked 0.695 is far below length only
(0.952) and length-matched 0.559 is close to chance; DoSDisruptive falls from 0.859 to 0.734, below length only;
DoSRandom stays separable inside every length (length-matched 0.988 ± 0.000, slightly above TabMlp's 0.970).
Mahalanobis on DoSRandom is seed-sensitive (0.443–0.809 pooled). Benign short-vs-long AUROC, TabResnet:
`masked_recon` 0.657 ± 0.006 / 0.659 ± 0.013 / 0.630 ± 0.005 (T64 / T48 / T32), `full_recon` 0.381 ± 0.006 /
0.436 ± 0.011 / 0.417 ± 0.011 (the full-reconstruction error is *lower* on short windows here, so its length-matched
AUROC is ≥ pooled), Mahalanobis 0.209 ± 0.049 / 0.179 ± 0.004 / 0.279 ± 0.024.

**Against the one-class ConvAE on the same windows** (ConvAE seeds 0–3 from the "Four seeds per cell" table):
GridSybil reconstruction 0.657 ± 0.004 / 0.682 ± 0.006 / 0.734 ± 0.003 at T = 64 / 48 / 32 vs TabMlp masked
0.531 / 0.607 / 0.657 (−0.126 / −0.075 / −0.077) and TabResnet masked 0.595 / 0.650 / 0.696 (−0.062 / −0.032 /
−0.038): TabResnet is above TabMlp at every T and below the ConvAE at every T.
At T64 TabMlp sits at the length-only level (0.535). Mahalanobis on z is below chance in every masked-AE run (as in
the ConvAE). The ConvAE length-matched AUROC was not computed here (D has no per-window score files).

**Caveats:** one configuration (no tuning of p, widths, z); Mac MPS only, no EC2 replicate; the T64 Mahalanobis uses
shrinkage-when-ill-conditioned (never triggered) where S0 used a ridge 1e-6; length-matched and short-vs-long numbers
are computed for this inventory, not by the package's evaluation script (the notebook computes the same method); track
ids hold the true sender (F13).

---

## T. TimesNet one-class `model-t32b-seed0` (context only, one seed)

Added 2026-11-18. Sources: training `src/runs/pretraining/all/T32_benign/model-t32b-seed0/summary.json`; evaluation
`src/runs/evaluation/all/T32_benign/model-t32b-seed0/summary.json` and its random-init control
`src/runs/evaluation/all/T32_benign/model-t32b-seed0-random-init/summary.json` (pooled, ≥ 4-row, recall / FPR at val
p95: `metrics.test_<attack>.window.<score>.{all,n_ge_4}`); analysis `…/model-t32b-seed0/ANALYSIS.md`. The pooled and
≥ 4-row AUROCs were recomputed here from `scores/*.npz` and equal the summary (|Δ| < 1e-6); the **length-matched
AUROC is not in the run files**: recomputed here from `scores/*.npz` (`n_real_rows`) with the masked-AE notebook's
method. Machine-readable: `timesnet_t32b_seed0` in `results.json`. Not a baseline-grid cell and not the thesis's
fine-tuned detector; listed because it shares D's and M's T32 split and test windows.

| Item | Value |
|---|---|
| Model / training | TimesNet encoder (d = 128, d_ff = 64, 4 blocks) + heads, **2,301,312** encoder / **2,402,461** total params; one-class self-supervised pretraining on the benign train windows of `all/T32_benign` (masked reconstruction + P1–P3 injected-violation heads + NT-Xent); 20 epochs, 37,220 updates, best epoch 20 (val joint 0.31416, benign val), stop `max_epochs`, 29,449.7 s wall; EC2 Tesla T4, CUDA, torch 2.14.1+cu126, git 0c104ef, `git_dirty: false` |
| Data | `all/T32_benign`: train 502,773 benign windows; benign val 65,313 (thresholds); test 60,142 benign + the four attack splits — **identical windows to D and M (T32)** |
| Scores | recon = masked MSE on hidden real rows, 4 passes (windows with < 2 real rows get the benign val median 0.0623: filled, not measured; 5.2% of benign test, 8.4% GridSybil, 70.2% DataReplay, 88.8% DoS); Mahalanobis on z (fit on the 502,773 train windows, cond 1.8e15, shrinkage 0.01); kNN = 1 − mean cosine similarity to the K nearest of a 200,000-window benign train bank, K 5 / 10 / 20 |
| Control | random-init: the same architecture untrained, scored identically |
| Seeds | **1** (seed 0); seeds 1, 2 pending (`agent.md` §6) |

Window AUROC, attack vs benign test (pooled → length-matched), ≥ 4-row AUROC, recall / benign FPR at val p95:

| Attack | Score | Pretrained pooled → length-matched | ≥ 4 rows | val p95 recall / benign FPR | Random-init pooled → length-matched | ≥ 4 rows | val p95 recall / benign FPR |
|---|---|---|---|---|---|---|---|
| GridSybil | recon | 0.741 → 0.734 | 0.772 | 0.310 / 0.051 | 0.553 → 0.541 | 0.569 | 0.170 / 0.050 |
| GridSybil | Mahalanobis | **0.880 → 0.846** | 0.892 | 0.657 / 0.051 | 0.673 → 0.642 | 0.676 | 0.198 / 0.051 |
| GridSybil | kNN K5 | **0.886 → 0.875** | 0.897 | 0.717 / 0.052 | **0.839 → 0.836** | 0.845 | 0.451 / 0.051 |
| GridSybil | kNN K10 | 0.885 → 0.871 | 0.895 | 0.711 / 0.051 | 0.831 → 0.828 | 0.838 | 0.420 / 0.051 |
| GridSybil | kNN K20 | 0.884 → 0.866 | 0.893 | 0.705 / 0.049 | 0.821 → 0.817 | 0.829 | 0.392 / 0.050 |
| GridSybil | length only | 0.559 | 0.538 | — | 0.559 | 0.538 | — |
| DataReplay | recon | 0.486 → 0.481 | 0.430 | 0.027 / 0.051 | 0.466 → 0.460 | 0.393 | 0.018 / 0.050 |
| DataReplay | Mahalanobis | 0.946 → 0.579 | 0.877 | 0.611 / 0.051 | 0.944 → 0.620 | 0.896 | 0.633 / 0.051 |
| DataReplay | kNN K5 | 0.840 → 0.777 | 0.863 | 0.326 / 0.052 | 0.867 → 0.780 | 0.871 | 0.361 / 0.051 |
| DataReplay | length only | 0.952 | 0.927 | — | 0.952 | 0.927 | — |
| DoSRandom | recon | 0.550 → 0.555 | 1.000 | 0.112 / 0.051 | 0.537 → 0.555 | 1.000 | 0.111 / 0.050 |
| DoSRandom | Mahalanobis | 0.960 → 0.523 | 1.000 | 0.638 / 0.051 | 0.995 → 0.951 | 0.999 | 0.993 / 0.051 |
| DoSRandom | kNN K5 | 0.971 → 0.966 | 1.000 | 0.844 / 0.052 | 0.993 → 0.989 | 0.993 | 0.980 / 0.051 |
| DoSRandom | length only | 0.967 | 0.978 | — | 0.967 | 0.978 | — |
| DoSDisruptive | recon | 0.549 → 0.555 | 0.999 | 0.111 / 0.051 | 0.535 → 0.552 | 0.997 | 0.107 / 0.050 |
| DoSDisruptive | Mahalanobis | 0.954 → 0.477 | 1.000 | 0.590 / 0.051 | 0.982 → 0.764 | 0.998 | 0.898 / 0.051 |
| DoSDisruptive | kNN K5 | 0.859 → 0.829 | 0.999 | 0.409 / 0.052 | 0.903 → 0.848 | 0.973 | 0.509 / 0.051 |
| DoSDisruptive | length only | 0.967 | 0.978 | — | 0.967 | 0.978 | — |

(kNN K10 / K20 for DataReplay / DoS: `results.json` → `timesnet_t32b_seed0`.) Length-only recall at val p95 is 0: the
val p95 of −n_real_rows is −1, and no window scores above it (stated in the evaluation protocol).

Reading (one seed): on GridSybil the pretrained encoder's kNN / Mahalanobis gain over random-init survives length
matching (kNN K5 0.875 vs 0.836; Mahalanobis 0.846 vs 0.642), and both are above every reconstruction score of D and
M on the same windows. For DataReplay / DoS the pooled Mahalanobis 0.95–0.96 is mostly length (length-matched
0.48–0.58), and random-init is at or above pretrained on every DataReplay / DoS Mahalanobis and kNN length-matched AUROC, so
there is no evidence of a pretraining gain there (F11).

---

## P. `src/baselines/plausibility/` — VeReMi-Extension paper Table III detector, re-implemented

Sources: `src/baselines/plausibility/README.md`; `src/runs/baselines/plausibility/summary.json`; spec
`docs/research-notes/veremi-plausibility-baseline-spec.md`.

| Item | Value |
|---|---|
| Purpose | Offline re-implementation of the paper's "simple detection algorithm" (Kamel et al., ICC 2020, §V / Table III) as a literature baseline |
| Data | raw `data/VeReMi-Dataset/`, all 8 subsets, **no split and no training** (fixed rules on every received BSM at every receiver) |
| Unit | one received message (not a window) |
| Method | F2MD plausibility / consistency checks + ThresholdApp: flag when any check ≤ 0.28125. Variants `catch` (headline), `legacy`, `catch_accel301` |
| Label | misbehaving iff the sender's own trace file has attack code ≠ 0 |
| Not computed | map check C10 (needs the LuST road network) and proximity C0 |
| Seeds | none (deterministic), one run |
| Runtime | 431.1 s, 10 workers, CPU |

**Results, headline variant `catch`** (`summary.json` → `subsets.*`; paper values = its `paper` block):

| Subset | Ours Acc / P / R / F1 | Paper Acc / P / R / F1 | ΔF1 | `legacy` F1 | received labelled messages (TP + FP + TN + FN) |
|---|---|---|---|---|---|
| GridSybil_0709 | 0.734 / 0.997 / 0.572 / **0.727** | 0.820 / 0.997 / 0.590 / 0.742 | −0.015 | 0.743 | 5,439,447 |
| GridSybil_1416 | 0.694 / 0.997 / 0.530 / **0.692** | 0.800 / 0.997 / 0.584 / 0.737 | −0.045 | 0.715 | 1,493,482 |
| DataReplaySybil_0709 | 0.799 / 0.915 / 0.353 / **0.510** | 0.801 / 0.920 / 0.353 / 0.510 | −0.000 | 0.662 | 2,931,107 |
| DataReplaySybil_1416 | 0.796 / 0.910 / 0.381 / **0.537** | 0.795 / 0.892 / 0.371 / 0.524 | +0.013 | 0.702 | 763,999 |
| DoSRandomSybil_0709 | 0.908 / 0.997 / 0.799 / **0.887** | 0.990 / 0.999 / 0.978 / 0.989 | −0.101 | 0.888 | 3,792,694 |
| DoSRandomSybil_1416 | 0.902 / 0.997 / 0.796 / **0.885** | 0.989 / 0.999 / 0.977 / 0.988 | −0.103 | 0.886 | 1,000,191 |
| DoSDisruptiveSybil_0709 | 0.766 / 0.980 / 0.497 / **0.660** | 0.770 / 0.982 / 0.501 / 0.664 | −0.004 | 0.713 | 3,792,694 |
| DoSDisruptiveSybil_1416 | 0.778 / 0.986 / 0.539 / **0.697** | 0.779 / 0.988 / 0.532 / 0.691 | +0.006 | 0.742 | 1,000,191 |

(Message totals are sums of the stored confusion counts; GridSybil_0709's equals `received_messages_labelled` in
`GridSybil_0709/results.json`.) 6 / 8 subsets within 0.05 F1 of the paper.

**Caveats** (README): DoSRandom recall 0.80 vs 0.98 — history checks run on 1.6% of attack messages, most likely the
missing map check; GridSybil accuracy differs because local data has 62–65% attack messages vs 44–48% implied by the
paper (GridSybil_0709 is a 2025 re-simulation, F14); which check variant the paper used is inferred, not known; no
ROC / AUROC exists (the detector outputs a binary decision at one threshold).

---

## W. Small-window baselines, T = 20 / 10 / 5 (one-class, unlabelled, supervised ConvAE + masked AE)

Added 2026-11-26. Five baselines × three window lengths × seeds 0–3 = 60 runs, EC2 g4dn (Tesla T4, CUDA), 2026-11-25 → 26,
all finished, all stopped by early stopping. **Not yet independently judged.** Sources: per run
`task/<package>/runs/seed<k>/test_results.json` (one-class, supervised: written by the package notebook executed with
RUN pinned to that seed, copy `*.executed.ipynb` in the run folder; one-class values rounded to 3 dp, supervised 4 dp),
`task/autoencoder_t<T>/runs/seed<k>/evaluation.json` (unlabelled, full precision), `task/masked_autoencoder/runs/<enc>/T<w>-seed<k>/test_metrics.json`
(masked AE, full precision); `summary.json` everywhere. Means / sample sd (n = 4) computed for this file from those
values. Machine-readable: `small_window_baselines` in `results.json`. Private HF `bibekgupta3333/roadfm-lite-timesnet`:
`baselines/<package>/seed<k>`, `baselines/masked_autoencoder/<enc>/T<w>-seed<k>`.

| Item | Value |
|---|---|
| Inputs | `src/data/encoder_input/all/T{20,10,5}_benign` (one-class, masked AE) and `all/T{20,10,5}_benign_grid` (unlabelled, supervised), builders `src/pipeline/all/encoder_input_T{20,10,5}_benign{,_grid}.ipynb` (copies of the T32 builders: `seq_len` = crop stride = T, buckets (T,), plain JSON), npm `pipeline:encoder-input:all-t{20,10,5}-benign{,-grid}`; on private HF dataset `bibekgupta3333/roadfm-lite-encoder-input` at `all/T<T>_<kind>` |
| Split | **the all/T32_benign / all/T32_benign_grid vehicle split** (same vehicle keys per split, every link in the same split with the same message count, normalisation identical; 55 / 35 checks PASS per input, `metadata.json` + build report); **no RULE 3 change** |
| Windows (benign inputs) train / val / test / test_gridsybil | T20 640,176 / 82,897 / 76,038 / 34,181; T10 1,033,475 / 133,974 / 123,097 / 52,655; T5 1,843,237 / 238,931 / 219,722 / 91,436 (test_datareplay 76,283 / 76,338 / 78,040; test_dosrandom 199,834, test_dosdisruptive 199,826 at every T) |
| Windows (benign_grid inputs) train / val / test | T20 983,719 / 84,969 / 83,826; T10 1,575,074 / 136,109 / 133,741; T5 2,796,619 / 241,196 / 236,997 (GridSybil share of test 0.314 / 0.303 / 0.297) |
| Packages | one-class ConvAE `task/autoencoder_benign_t{20,10,5}`, unlabelled ConvAE `task/autoencoder_t{20,10,5}`, supervised ConvAE `task/autoencoder_st_t{20,10,5}` (each subclasses its T = 32 sibling's config; only input folder, `seq_len` and the depth `n_down` = 2 / 1 / 0 change); masked AE `task/masked_autoencoder` with `--window 20|10|5` (`SMALL_WINDOWS`) |
| Params | ConvAE 56,957 / 52,861 / 50,813 (+ head 577 supervised: 57,534 / 53,438 / 51,390); masked AE TabMlp 37,012 / 19,602 / 10,897, TabResnet 52,192 / 34,912 / 26,272 at T = 20 / 10 / 5 |
| Protocol | the T = 32 cell of each baseline (D, E, H, M): benign val (one-class, masked AE), mixed unlabelled val (unlabelled) or labelled val (supervised) for early stopping, LR schedule and thresholds; test never read in training; length only = −n_messages |

**One-class ConvAE** (`task/autoencoder_benign_t<T>`), window AUROC vs benign test, mean ± sd, seeds 0–3:

| T | Attack | Reconstruction | Mahalanobis | Length only | Recon recall @ val p95 (benign FPR) |
|---|---|---|---|---|---|
| 20 | GridSybil | 0.756 ± 0.010 | 0.540 ± 0.029 | 0.547 | 0.354 ± 0.019 (0.049 ± 0.002) |
| 20 | DataReplay | 0.400 ± 0.010 | 0.449 ± 0.037 | 0.951 | 0.060 ± 0.008 (0.049 ± 0.002) |
| 20 | DoSRandom | 0.978 ± 0.004 | 0.826 ± 0.026 | 0.967 | 0.899 ± 0.018 (0.049 ± 0.002) |
| 20 | DoSDisruptive | 0.622 ± 0.011 | 0.622 ± 0.023 | 0.967 | 0.345 ± 0.024 (0.049 ± 0.002) |
| 10 | GridSybil | 0.759 ± 0.002 | 0.616 ± 0.018 | 0.530 | 0.367 ± 0.004 (0.051 ± 0.001) |
| 10 | DataReplay | 0.566 ± 0.019 | 0.689 ± 0.043 | 0.951 | 0.073 ± 0.003 (0.051 ± 0.001) |
| 10 | DoSRandom | 0.986 ± 0.001 | 0.896 ± 0.026 | 0.968 | 0.931 ± 0.006 (0.051 ± 0.001) |
| 10 | DoSDisruptive | 0.748 ± 0.007 | 0.790 ± 0.032 | 0.968 | 0.391 ± 0.012 (0.051 ± 0.001) |
| 5 | GridSybil | 0.782 ± 0.002 | 0.608 ± 0.013 | 0.517 | 0.393 ± 0.007 (0.051 ± 0.001) |
| 5 | DataReplay | 0.664 ± 0.010 | 0.754 ± 0.055 | 0.941 | 0.077 ± 0.006 (0.051 ± 0.001) |
| 5 | DoSRandom | 0.991 ± 0.002 | 0.936 ± 0.024 | 0.969 | 0.952 ± 0.012 (0.051 ± 0.001) |
| 5 | DoSDisruptive | 0.792 ± 0.009 | 0.878 ± 0.034 | 0.969 | 0.392 ± 0.019 (0.051 ± 0.001) |

**Unlabelled ConvAE** (`task/autoencoder_t<T>`), GridSybil vs benign, `test_all`, mean ± sd:

| T | Reconstruction | Mahalanobis | Length only | Recon recall @ val p95 (benign FPR) |
|---|---|---|---|---|
| 20 | 0.619 ± 0.001 | 0.468 ± 0.016 | 0.548 | 0.104 ± 0.003 (0.041 ± 0.001) |
| 10 | 0.636 ± 0.003 | 0.530 ± 0.020 | 0.530 | 0.103 ± 0.001 (0.041 ± 0.001) |
| 5 | 0.643 ± 0.002 | 0.580 ± 0.002 | 0.517 | 0.103 ± 0.002 (0.040 ± 0.001) |

**Supervised ConvAE** (`task/autoencoder_st_t<T>`), GridSybil vs benign, window, mean ± sd:

| T | Head AUROC | F1 at val max-F1 θ | Benign FPR at val max-F1 θ | F1 / benign FPR at val FPR-5% θ | Recon alone | Length only |
|---|---|---|---|---|---|---|
| 20 | 0.9243 ± 0.0006 | 0.8814 ± 0.0013 | 0.0116 ± 0.0015 | 0.8569 ± 0.0010 / 0.0488 ± 0.0013 | 0.6057 ± 0.0041 | 0.5481 |
| 10 | 0.9159 ± 0.0011 | 0.8738 ± 0.0026 | 0.0132 ± 0.0014 | 0.8509 ± 0.0038 / 0.0473 ± 0.0026 | 0.6247 ± 0.0047 | 0.5302 |
| 5 | 0.9092 ± 0.0015 | 0.8680 ± 0.0062 | 0.0131 ± 0.0019 | 0.8461 ± 0.0048 / 0.0479 ± 0.0027 | 0.6258 ± 0.0049 | 0.5167 |

**Masked AE** (`task/masked_autoencoder`), window AUROC vs benign test, masked recon / full recon / Mahalanobis, mean ± sd (length-matched not computed):

| T | Attack | TabMlp masked | TabMlp full | TabMlp Maha | TabResnet masked | TabResnet full | TabResnet Maha | Length only |
|---|---|---|---|---|---|---|---|---|
| 20 | GridSybil | 0.667 ± 0.002 | 0.659 ± 0.003 | 0.423 ± 0.017 | 0.707 ± 0.001 | 0.698 ± 0.001 | 0.416 ± 0.010 | 0.547 |
| 20 | DataReplay | 0.728 ± 0.006 | 0.553 ± 0.017 | 0.318 ± 0.060 | 0.717 ± 0.010 | 0.488 ± 0.006 | 0.421 ± 0.060 | 0.951 |
| 20 | DoSRandom | 0.990 ± 0.000 | 0.975 ± 0.003 | 0.533 ± 0.153 | 0.992 ± 0.000 | 0.963 ± 0.002 | 0.806 ± 0.160 | 0.967 |
| 20 | DoSDisruptive | 0.874 ± 0.007 | 0.743 ± 0.015 | 0.327 ± 0.044 | 0.859 ± 0.009 | 0.683 ± 0.008 | 0.512 ± 0.085 | 0.967 |
| 10 | GridSybil | 0.668 ± 0.002 | 0.665 ± 0.004 | 0.472 ± 0.007 | 0.693 ± 0.006 | 0.694 ± 0.007 | 0.487 ± 0.019 | 0.530 |
| 10 | DataReplay | 0.733 ± 0.002 | 0.589 ± 0.014 | 0.435 ± 0.033 | 0.761 ± 0.007 | 0.576 ± 0.030 | 0.411 ± 0.043 | 0.951 |
| 10 | DoSRandom | 0.990 ± 0.000 | 0.973 ± 0.005 | 0.727 ± 0.069 | 0.993 ± 0.000 | 0.970 ± 0.001 | 0.697 ± 0.106 | 0.968 |
| 10 | DoSDisruptive | 0.882 ± 0.005 | 0.761 ± 0.014 | 0.463 ± 0.042 | 0.903 ± 0.005 | 0.750 ± 0.018 | 0.399 ± 0.077 | 0.968 |
| 5 | GridSybil | 0.653 ± 0.003 | 0.656 ± 0.002 | 0.509 ± 0.029 | 0.685 ± 0.001 | 0.691 ± 0.003 | 0.538 ± 0.013 | 0.517 |
| 5 | DataReplay | 0.750 ± 0.008 | 0.640 ± 0.015 | 0.550 ± 0.104 | 0.749 ± 0.008 | 0.597 ± 0.019 | 0.705 ± 0.052 | 0.941 |
| 5 | DoSRandom | 0.990 ± 0.001 | 0.974 ± 0.005 | 0.790 ± 0.213 | 0.993 ± 0.000 | 0.972 ± 0.003 | 0.958 ± 0.015 | 0.969 |
| 5 | DoSDisruptive | 0.887 ± 0.009 | 0.788 ± 0.018 | 0.675 ± 0.087 | 0.886 ± 0.005 | 0.760 ± 0.017 | 0.820 ± 0.045 | 0.969 |

**GridSybil overview, T = 64 → 5** (window AUROC, mean ± sd over seeds 0–3; T64 / T48 / T32 from the "Four seeds per cell"
table and section M of this file). T64 / T48 cells use the `benign_gridsybil` inputs (GridSybil share of test ≈ 0.67,
574 test vehicles); T32 → T5 use the all-folder split (share ≈ 0.29–0.32, the same test vehicles at every T), so only
the T32 → T5 part of each row is on one test population.

| Baseline | T64 | T48 | T32 | T20 | T10 | T5 |
|---|---|---|---|---|---|---|
| One-class ConvAE recon | 0.6569 ± 0.0036 (S0) | 0.682 ± 0.006 (F) | 0.7335 ± 0.0029 (D) | 0.756 ± 0.010 | 0.759 ± 0.002 | 0.782 ± 0.002 |
| Unlabelled ConvAE recon | 0.482 ± 0.002 (G) | 0.485 ± 0.001 (A) | 0.612 ± 0.004 (H) | 0.619 ± 0.001 | 0.636 ± 0.003 | 0.643 ± 0.002 |
| Supervised ConvAE head | — (no T64 supervised ConvAE; S0 supervised encoder + head, no decoder: 0.928 ± 0.002, seeds 0–2) | 0.9278 ± 0.0019 (C, T48_tvt) | 0.9289 ± 0.0018 (E) | 0.9243 ± 0.0006 | 0.9159 ± 0.0011 | 0.9092 ± 0.0015 |
| Masked AE TabMlp masked recon | 0.531 ± 0.002 | 0.607 ± 0.001 | 0.657 ± 0.002 | 0.667 ± 0.002 | 0.668 ± 0.002 | 0.653 ± 0.003 |
| Masked AE TabResnet masked recon | 0.595 ± 0.003 | 0.650 ± 0.002 | 0.696 ± 0.001 | 0.707 ± 0.001 | 0.693 ± 0.006 | 0.685 ± 0.001 |
| Length only (−n_messages; one-class test) | 0.535 | 0.530 | 0.559 | 0.547 | 0.530 | 0.517 |

**Readings (side experiment, not a thesis result, RULE 1; judged: all PASS or PASS-WITH-NOTES):** one-class and unlabelled ConvAE
reconstruction rise as T shrinks (one-class 0.734 → 0.756 / 0.759 / 0.782, unlabelled 0.612 → 0.619 / 0.636 / 0.643);
the masked AE peaks near T20 (TabMlp 0.667, TabResnet 0.707) and stays below the one-class ConvAE at every small T;
the supervised head drops slowly (0.929 → 0.924 / 0.916 / 0.909). Unlabelled Mahalanobis rises from below chance
(T32 0.421) to 0.580 at T5. DataReplay / DoS: every score is below length only except DoSRandom reconstruction
(one-class ConvAE 0.978 / 0.986 / 0.991 vs length only 0.967 / 0.968 / 0.969; masked AE 0.99), as at T32; DataReplay /
DoSDisruptive pooled masked-AE AUROCs carry the window-length effect (≈ 95% / 99.7% of their test windows have 1–3
rows vs ≈ 14% of benign; DataReplay ≥ 4-row masked AUROC 0.37–0.50). Shorter windows also mean more, shorter test
windows per vehicle (e.g. GridSybil test windows 28,087 at T32 vs 91,436 at T5), so window-level AUROCs at different T
are not the same unit of decision.

---

## Cross-study comparison table

GridSybil vs benign, test split, window level unless stated. One row per reported score.

| Study | Training signal | Input (T) | Test set (windows; GridSybil share) | Params | Seeds | AUROC | PR-AUC (chance) | Recall @ ≈ 5% benign FPR (actual FPR) | F1 (threshold) | Track AUROC max / mean | Length-only AUROC | AUROC 1–3 rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 AE recon | one-class (benign) | T64 (64) | 38,426; 0.668 | 80,189 | 3 | 0.655 ± 0.002 | 0.802 (0.668) | 0.199 (0.052) | 0.325 (val-benign p95) | not found | 0.535 | 0.552 |
| S0 AE Mahalanobis | one-class (benign) | T64 | same | 80,189 | 3 | 0.319 ± 0.016 | 0.589 | 0.045 (0.052) | 0.084 | not found | 0.535 | 0.300 |
| S0 supervised head | labels (BCE, no decoder) | T64 | same | 37,233 | 3 | 0.928 ± 0.002 | 0.970 | not found (fixed 0.5: recall 0.816 at FPR 0.023) | 0.893 (0.5) | not found | 0.535 | 0.916 |
| S0 supervised z Mahalanobis | labels in encoder | T64 | same | 37,233 | 3 | 0.787 ± 0.028 | 0.899 | 0.502 (0.050) | 0.656 | not found | 0.535 | 0.776 |
| S0 stats + LR | labels (53 statistics) | T64 | same | — | 1 (det.) | 0.855 | 0.926 | not found (0.5: recall 0.832 at FPR 0.261) | 0.849 (0.5) | not found | 0.535 | 0.681 |
| A recon | unsupervised (both classes "normal") | T48 (48) | 40,469; 0.669 | 71,741 | 1 | 0.484 | 0.661 (0.669) | 0.046 (0.049) | 0.087 (val p95, all val) | not found | 0.530 | 0.378 |
| B recon | unsupervised, overlap | T48_tvt_s24 (48) | 42,653; 0.673 | 71,741 | 1 | 0.484 | 0.664 (0.673) | 0.043 (0.050) | 0.080 | 0.482 / 0.481 | not found | not found |
| B Mahalanobis | unsupervised, overlap | T48_tvt_s24 | same | 71,741 | 1 | 0.337 | 0.595 | 0.044 (0.068) | 0.082 | 0.307 / 0.302 | not found | not found |
| C head | labels (BCE + MSE) | T48_tvt (48) | 40,469; 0.669 | 72,318 | 1 | 0.929 | 0.971 (0.669) | 0.833 (0.044) | 0.900 (val max-F1) | 0.928 / 0.929 | 0.530 | 0.917 |
| C recon alone | — | T48_tvt | same | 72,318 | 1 | 0.478 | not found | not found | not found | not found | 0.530 | not found |
| D recon | one-class (benign, 8 folders) | all/T32_benign (32) | 60,142 benign + 28,087 GridSybil; 0.318 | 63,293 | 1 | 0.736 | 0.629 (0.318) | 0.298 (0.048) | not found | 0.713 / 0.721 | 0.559 | not found (≥ 4 rows: 0.778) |
| D Mahalanobis | one-class | all/T32_benign | same | 63,293 | 1 | 0.510 | 0.394 | 0.127 (0.048) | not found | 0.488 / 0.491 | 0.559 | not found (≥ 4 rows: 0.540) |
| E head | labels (BCE + MSE) | all/T32_benign_grid (32) | 67,100; 0.323 | 63,870 | 1 | 0.926 | 0.925 (0.323) | 0.839 (0.048) | 0.889 (val max-F1) | 0.925 / 0.927 | 0.560 | 0.915 |
| E recon alone | — | all/T32_benign_grid | same | 63,870 | 1 | 0.598 | not found | not found | not found | not found | 0.560 | not found |
| F recon | one-class (benign) | T48_benign (48) | 13,405 benign + 27,064 GridSybil; 0.669 | 71,741 | 1 | 0.689 | 0.829 (0.669) | 0.243 (0.047) | not found | 0.678 / 0.681 | 0.530 | 0.599 (≥ 4 rows: 0.720) |
| F Mahalanobis | one-class | T48_benign | same | 71,741 | 1 | 0.437 | 0.655 | 0.062 (0.044) | not found | 0.430 / 0.427 | 0.530 | 0.406 (≥ 4 rows: 0.453) |
| G recon | unlabelled (both classes "normal"; GridSybil 64.6% of fit) | T64 (64) | 38,426; 0.668 | 80,189 | 1 | 0.482 | 0.660 (0.668) | 0.045 (0.049) | 0.084 (val p95, all val) | 0.481 / 0.480 | 0.535 | 0.361 (≥ 4 rows: 0.512) |
| G Mahalanobis | unlabelled | T64 | same | 80,189 | 1 | 0.352 | 0.599 | 0.047 (0.057) | 0.087 | 0.340 / 0.335 | 0.535 | 0.317 (≥ 4 rows: 0.362) |
| H recon | unlabelled (GridSybil 31.3% of train) | all/T32_benign_grid (32) | 67,100; 0.323 | 63,293 | 1 | 0.610 (GridSybil folders only: 0.613) | 0.436 (0.323) | 0.108 (0.040) | not found | 0.594 / 0.599 | 0.560 | 0.496 (≥ 4 rows: 0.648) |
| H Mahalanobis | unlabelled | all/T32_benign_grid | same | 63,293 | 1 | 0.398 | 0.331 | 0.097 (0.053) | not found | 0.362 / 0.362 | 0.560 | 0.385 (≥ 4 rows: 0.414) |
| M TabMlp masked recon | one-class (benign) | T64 (64), S0 AE's windows | 38,426; 0.668 | 113,616 | 4 | 0.531 ± 0.002 (length-matched 0.510) | 0.707 (0.668) | 0.093 (0.048) | not computed | 0.521 / 0.522 | 0.535 | 0.398 (≥ 4 rows: 0.550) |
| M TabMlp full recon | one-class | T64 | same | 113,616 | 4 | 0.539 ± 0.003 (0.530) | 0.722 | 0.112 (0.047) | not computed | 0.529 / 0.530 | 0.535 | 0.448 (≥ 4 rows: 0.558) |
| M TabMlp Mahalanobis | one-class | T64 | same | 113,616 | 4 | 0.398 ± 0.018 (0.403) | 0.634 | 0.065 (0.055) | not computed | 0.385 / 0.385 | 0.535 | 0.374 (≥ 4 rows: 0.412) |
| M TabMlp masked recon | one-class | T48_benign (48), F's windows | 13,405 benign + 27,064 GridSybil; 0.669 | 85,760 | 4 | 0.607 ± 0.001 (0.589) | 0.773 (0.669) | 0.172 (0.045) | not computed | 0.594 / 0.595 | 0.530 | 0.499 (≥ 4 rows: 0.619) |
| M TabMlp full recon | one-class | T48_benign | same | 85,760 | 4 | 0.603 ± 0.002 (0.595) | 0.772 | 0.174 (0.045) | not computed | 0.589 / 0.590 | 0.530 | 0.526 (≥ 4 rows: 0.618) |
| M TabMlp Mahalanobis | one-class | T48_benign | same | 85,760 | 4 | 0.403 ± 0.016 (0.408) | 0.641 | 0.072 (0.052) | not computed | 0.385 / 0.383 | 0.530 | 0.382 (≥ 4 rows: 0.414) |
| M TabMlp masked recon | one-class | all/T32_benign (32), D's windows | 60,142 benign + 28,087 GridSybil; 0.318 | 57,904 | 4 | 0.657 ± 0.002 (0.631) | 0.514 (0.318) | 0.210 (0.052) | not computed | 0.630 / 0.638 | 0.559 | 0.529 (≥ 4 rows: 0.674) |
| M TabMlp full recon | one-class | all/T32_benign | same | 57,904 | 4 | 0.650 ± 0.003 (0.638) | 0.515 | 0.214 (0.051) | not computed | 0.623 / 0.630 | 0.559 | 0.563 (≥ 4 rows: 0.669) |
| M TabMlp Mahalanobis | one-class | all/T32_benign | same | 57,904 | 4 | 0.408 ± 0.017 (0.415) | 0.316 | 0.077 (0.053) | not computed | 0.382 / 0.382 | 0.559 | 0.373 (≥ 4 rows: 0.424) |
| M TabResnet masked recon | one-class | T64 | 38,426; 0.668 | 128,224 | 4 | 0.595 ± 0.003 (0.578) | 0.754 | 0.140 (0.050) | not computed | 0.586 / 0.587 | 0.535 | 0.507 (≥ 4 rows: 0.604) |
| M TabResnet full recon | one-class | T64 | same | 128,224 | 4 | 0.605 ± 0.003 (0.609) | 0.765 | 0.155 (0.052) | not computed | 0.596 / 0.597 | 0.535 | 0.629 (≥ 4 rows: 0.608) |
| M TabResnet masked recon | one-class | T48_benign | 13,405 + 27,064; 0.669 | 100,576 | 4 | 0.650 ± 0.002 (0.640) | 0.802 | 0.208 (0.046) | not computed | 0.639 / 0.640 | 0.530 | 0.596 (≥ 4 rows: 0.654) |
| M TabResnet full recon | one-class | T48_benign | same | 100,576 | 4 | 0.648 ± 0.001 (0.650) | 0.798 | 0.195 (0.045) | not computed | 0.636 / 0.637 | 0.530 | 0.641 (≥ 4 rows: 0.654) |
| M TabResnet masked recon | one-class | all/T32_benign | 60,142 + 28,087; 0.318 | 72,928 | 4 | 0.696 ± 0.001 (0.678) | 0.566 (0.318) | 0.250 (0.051) | not computed | 0.674 / 0.681 | 0.559 | 0.619 (≥ 4 rows: 0.706) |
| M TabResnet full recon | one-class | all/T32_benign | same | 72,928 | 4 | 0.685 ± 0.002 (0.688) | 0.553 | 0.233 (0.049) | not computed | 0.660 / 0.667 | 0.559 | 0.664 (≥ 4 rows: 0.699) |
| M TabResnet Mahalanobis | one-class | all/T32_benign | same | 72,928 | 4 | 0.378 ± 0.004 (0.387) | 0.330 | 0.100 (0.049) | not computed | 0.347 / 0.344 | 0.559 | 0.320 (≥ 4 rows: 0.402) |
| *Context:* T TimesNet kNN K5 | one-class SSL pretraining (benign) | all/T32_benign (32), D's windows | 60,142 + 28,087; 0.318 | 2,402,461 | 1 | 0.886 (length-matched 0.875; random-init 0.839 / 0.836) | 0.871 (0.318) | 0.717 (0.052) | not computed | 0.880 / 0.884 | 0.559 | 0.788 (≥ 4 rows: 0.897) |
| *Context:* T TimesNet Mahalanobis | one-class SSL | all/T32_benign | same | 2,402,461 | 1 | 0.880 (0.846; random-init 0.673 / 0.642) | 0.843 | 0.657 (0.051) | not computed | 0.873 / 0.877 | 0.559 | 0.686 (≥ 4 rows: 0.892) |
| *Context:* T TimesNet recon | one-class SSL | all/T32_benign | same | 2,402,461 | 1 | 0.741 (0.734; < 2-row windows filled) | 0.615 | 0.310 (0.051) | not computed | 0.719 / 0.727 | 0.559 | 0.601 (≥ 4 rows: 0.772) |
| P plausibility (`catch`) | none (rules) | raw messages | all received messages, 0709 / 1416 | — | — | not found (binary output) | not found | not found | 0.727 / 0.692 per message | — | — | — |

Other attacks (D and P; seed 0 of D. Added 2026-11-18: M TabMlp T32 and T as context, table below):

| Attack | D recon AUROC | D Mahalanobis AUROC | D length-only AUROC | D recall @ val p95 | P F1 per message (0709 / 1416) |
|---|---|---|---|---|---|
| DataReplay | 0.391 | 0.336 | 0.952 | 0.046 | 0.510 / 0.537 |
| DoSRandom | 0.973 | 0.837 | 0.967 | 0.880 | 0.887 / 0.885 |
| DoSDisruptive | 0.611 | 0.556 | 0.967 | 0.292 | 0.660 / 0.697 |

Same benign test (60,142 windows of `all/T32_benign`), window AUROC **pooled → length-matched** (length-matched =
AUROC inside each exact window length, weighted by attack windows; recomputed from score files, see sections M, T).
D: seeds 0–3 pooled (no score files, so no length-matched value). M: TabMlp masked recon (and full recon), seeds 0–3 mean ± sd.
T: TimesNet seed 0, pretrained / random-init. M TabResnet: masked recon, seeds 0–3 mean ± sd.

| Attack | Length-only | D recon (seeds 0–3) | M masked recon | M full recon | T kNN K5: pretrained / random-init | T Mahalanobis: pretrained / random-init | M TabResnet masked recon |
|---|---|---|---|---|---|---|---|
| DataReplay | 0.952 | 0.387 ± 0.010 | 0.693 ± 0.003 → **0.490 ± 0.009** | 0.540 ± 0.010 → 0.491 ± 0.011 | 0.840 → 0.777 / 0.867 → 0.780 | 0.946 → 0.579 / 0.944 → 0.620 | 0.695 ± 0.008 → **0.559 ± 0.007** |
| DoSRandom | 0.967 | 0.967 ± 0.005 | 0.988 ± 0.001 → 0.970 ± 0.001 | 0.976 ± 0.002 → 0.971 ± 0.002 | 0.971 → 0.966 / 0.993 → 0.989 | 0.960 → 0.523 / 0.995 → 0.951 | 0.994 ± 0.000 → 0.988 ± 0.000 |
| DoSDisruptive | 0.967 | 0.598 ± 0.012 | 0.851 ± 0.006 → **0.679 ± 0.005** | 0.732 ± 0.007 → 0.683 ± 0.003 | 0.859 → 0.829 / 0.903 → 0.848 | 0.954 → 0.477 / 0.982 → 0.764 | 0.859 ± 0.004 → **0.734 ± 0.001** |

**Seed-3 reruns (2026-11-16, EC2 T4)**, GridSybil vs benign, test, window AUROC; seed 0 (or S0's 3-seed mean) for
comparison. Sources: the "Seed 3" block of each section; judge verdicts under `baselines/<study>/seed3/judge/` (HF).

| Cell | Score | Seed 0 | Seed 3 | Judge |
|---|---|---|---|---|
| S0 one-class AE, T64 | recon / Mahalanobis | 0.655 ± 0.002 / 0.319 ± 0.016 (seeds 0–2) | 0.6615 / 0.3332 (seeds 0–3 pooled: 0.6569 ± 0.0036 / 0.3224 ± 0.0147) | PASS-WITH-NOTES |
| C supervised AE, T48_tvt | head | 0.9289 | 0.9253 | PASS-WITH-NOTES |
| D one-class AE, all/T32_benign | recon / Mahalanobis | 0.736 / 0.510 | 0.730 / 0.508 | PASS-WITH-NOTES |
| F one-class AE, T48_benign | recon / Mahalanobis | 0.689 / 0.437 | 0.685 / 0.454 | PASS-WITH-NOTES |
| G unlabelled AE, T64 | recon / Mahalanobis | 0.482 / 0.352 | 0.480 / 0.337 | PASS |
| A, E, H | — | — | pending on 2026-11-16; done 2026-11-17, see the 4-seed table below | — |

**Four seeds per cell (seeds 0–3, 2026-11-17)**, GridSybil vs benign, test, window AUROC, mean ± sample sd (n = 4;
S0 AE also n = 5). Seed 0 = Mac MPS, seeds 1–3 (4) = EC2 CUDA. Sources: the per-section seed blocks above; judge
verdicts + recomputes under `baselines/<package>/seed<k>/judge/` (HF); `seed_summary` in `results.json`. The columns
are still not directly comparable (different inputs, splits, test vehicles; see below).

| Cell | Training signal | Input | Headline score | Seeds 0 / 1 / 2 / 3 | Mean ± sd | Secondary (mean ± sd) | Judges |
|---|---|---|---|---|---|---|---|
| S0 AE | one-class | T64 | recon | 0.6546 / 0.6580 / 0.6536 / 0.6615 (seed 4: 0.6532) | 0.6569 ± 0.0036 (0–4: 0.6562 ± 0.0035) | Maha 0.3224 ± 0.0146 (0–4: 0.3233 ± 0.0128) | 3, 4: PASS-WITH-NOTES |
| A | unlabelled mix | T48 | recon | 0.484 / 0.4859 / 0.4850 / 0.4856 | 0.485 ± 0.001 | length-only 0.530 | PASS-WITH-NOTES ×3 |
| C | supervised | T48_tvt | head | 0.9289 / 0.9296 / 0.9273 / 0.9253 | 0.9278 ± 0.0019 | at val FPR-5% θ: F1 0.8967 ± 0.0021, recall 0.8304 ± 0.0034, benign FPR 0.039–0.049; at val max-F1 θ benign FPR 0.038–0.078 | PASS-WITH-NOTES ×3 |
| D | one-class | all/T32_benign | recon | 0.736 / 0.732 / 0.736 / 0.730 | 0.7335 ± 0.0029 | Maha 0.505 ± 0.012; DataReplay 0.387 ± 0.010, DoSRandom 0.967 ± 0.005, DoSDisruptive 0.598 ± 0.012 (all ≤ length-only 0.952 / 0.967) | PASS-WITH-NOTES ×3 |
| E | supervised | all/T32_benign_grid | head | 0.9263 / 0.9294 / 0.9302 / 0.9298 | 0.9289 ± 0.0018 | F1 at val max-F1 0.8907 ± 0.0014 (benign FPR 0.0094 ± 0.0017) | PASS-WITH-NOTES ×3 |
| F | one-class | T48_benign | recon | 0.6894 / 0.6775 / 0.6764 / 0.6845 | 0.682 ± 0.006 | Maha 0.446 ± 0.015 | PASS-WITH-NOTES ×3 |
| G | unlabelled mix | T64 | recon | 0.4819 / 0.4810 / 0.4835 / 0.4800 | 0.482 ± 0.002 | Maha 0.340 ± 0.010 | seed 3 PASS, 1–2 PASS-WITH-NOTES |
| H | unlabelled mix | all/T32_benign_grid | recon | 0.6102 / 0.6171 / 0.6070 / 0.6129 | 0.612 ± 0.004 | Maha 0.421 ± 0.024; GridSybil folders recon 0.614 ± 0.004 | PASS-WITH-NOTES ×3 |
| M TabMlp (added 2026-11-18) | one-class | T64 (S0 AE's windows) | masked recon | 0.5329 / 0.5316 / 0.5285 / 0.5311 | 0.531 ± 0.002 | full recon 0.539 ± 0.003; Maha 0.398 ± 0.018; length-matched masked 0.510 ± 0.002 | no judge; every AUROC recomputed from score files (Δ < 1e-6) |
| M TabMlp | one-class | T48_benign (F's windows) | masked recon | 0.6060 / 0.6073 / 0.6089 / 0.6072 | 0.607 ± 0.001 | full recon 0.603 ± 0.002; Maha 0.403 ± 0.016; length-matched masked 0.589 ± 0.001 | same |
| M TabMlp | one-class | all/T32_benign (D's windows) | masked recon | 0.6556 / 0.6589 / 0.6567 / 0.6552 | 0.657 ± 0.002 | full recon 0.650 ± 0.003; Maha 0.408 ± 0.017; length-matched masked 0.631 ± 0.001; DataReplay 0.693 → 0.490, DoSRandom 0.988 → 0.970, DoSDisruptive 0.851 → 0.679 (pooled → length-matched) | same |
| M TabResnet | one-class | T64 | masked recon | 0.5947 / 0.5897 / 0.5959 / 0.5978 | 0.595 ± 0.003 | full recon 0.605 ± 0.003; Maha 0.368 ± 0.009 | same |
| M TabResnet | one-class | T48_benign | masked recon | 0.6493 / 0.6508 / 0.6518 / 0.6480 | 0.650 ± 0.002 | full recon 0.648 ± 0.001; Maha 0.381 ± 0.007 | same |
| M TabResnet | one-class | all/T32_benign | masked recon | 0.6954 / 0.6964 / 0.6950 / 0.6978 | 0.696 ± 0.001 | full recon 0.685 ± 0.002; Maha 0.378 ± 0.004; length-matched masked 0.678 ± 0.001; DataReplay 0.695 → 0.559, DoSRandom 0.994 → 0.988, DoSDisruptive 0.859 → 0.734 (pooled → length-matched) | same |

Readings unchanged by the extra seeds: reconstruction is stable (sd ≤ 0.006); Mahalanobis on z is the least stable
score in every cell that has it and below chance in every seed of S0, F, G and H (D: 0.487–0.514); the single-seed F number 0.689 was the best
of four. Seed 0 of A, D (3 dp) and C, E (4 dp) are rounded notebook values.
M (masked AE, added 2026-11-18): all seeds on the Mac (MPS), no EC2 replicate and no judge; same reading — reconstruction
stable (sd ≤ 0.004), Mahalanobis below chance in every run and the least stable score. 2026-11-18 check: every
per-seed value of this table and of `seed_summary` was compared with the run files now on the Mac
(`task/*/runs/seed{1,2,3}/judge/recompute.json`, `test_metrics.json`, `results.json`, `evaluation.json`,
`task/encoder/runs/evaluation/results.json`, `ae-seed{3,4}/judge/recompute.json`): all equal within 1e-6 except the
rounded seed-0 notebook values; every mean / sd recomputes to the stored 4 dp. One rounded value was replaced in
`results.json` (H GridSybil-folders seed 0: 0.613 → 0.61311 from `task/autoencoder_t32/runs/seed0/evaluation.json`;
mean ± sd unchanged).

---

## What is comparable and what is not

**Test-set identity** (sources: `agent.md` §2 rows for T48 / T48_tvt / T48_tvt_s24; dataset `metadata.json`;
notebook counts):

| Group | Studies | What is shared | What differs |
|---|---|---|---|
| G1: the benign + GridSybil test vehicles (574 sender vehicles) | S0, A, B, C, F, G; M at T64 / T48 (added 2026-11-18) | the same 10% test vehicles | window cut and count: S0 and G T = 64 → the same 38,426 windows; F = 13,405 benign + 27,064 GridSybil windows at T = 48, the same counts as A and C (T48_benign windows = T48_tvt's of the same id, `agent.md` §2); A and C T = 48 → the same 40,469 windows (`agent.md`: T48_tvt test = T48 test, vehicles + window ids); B stride 24 → 42,653 windows over the same 35,957 tracks as C |
| G2: `all/T32_benign` test | D; M at T32; T (context) | D, M (T32) and T: the same 60,142 benign + attack test windows and split; D and M share the on-disk normalisation, T applies its own feature scaling at load time (evaluation `summary.json` → `checkpoint.scaling`; median / IQR for three features, D13) | 80 / 10 / 10 split over 7,511 vehicle keys; benign from 8 folders (525 vehicles) + attack windows of 225 test attack vehicles |
| G3: `all/T32_benign_grid` test | E, H | E and H: the same 67,100 test windows, split and normalisation | 85 / 7.5 / 7.5 split over the same keys → **different test vehicles from D** (524 vehicles) |
| G4: raw messages | P | — | no split; every received message of every subset, including the vehicles the learned models trained on |

**Normalisation** (dataset `metadata.json` → `normalisation`; e.g. the std of `range`): T64 and T48 share one set of
constants (fitted on the 90% train, both classes; range std 419.2); T48_tvt and T48_tvt_s24 share another (80%
train, both classes; 404.6); `all/T32_benign` is fitted on benign train rows only (108.2); `all/T32_benign_grid` on
benign + GridSybil train rows (304.5). Reconstruction errors (masked MSE in normalised units), thresholds and
per-feature errors are therefore **not on one scale** across these four groups. AUROC within a study is unaffected
by the scale, but the inputs the models saw differ.

**Rows that can sit in one table honestly:**

1. **S0 internal table** (AE recon, AE Mahalanobis, supervised head, supervised-z Mahalanobis, stats + LR, length
   only): identical test windows, identical normalisation, same fit / val rule. Three seeds for the neural rows.
   This is the only fully controlled comparison in the repo. Two footnotes are required: the supervised model has no
   decoder (it is not "the AE + labels"), and the length-only direction was chosen on test labels.
2. **A vs C** (unsupervised recon 0.484 vs supervised head 0.929; also C's recon-alone 0.478): identical 40,469 test
   windows and the same architecture body. Footnotes: different normalisation constants (T48 vs T48_tvt), different
   train / val vehicle sets (A: 90% minus 10% in code; C: 80 / 10 on disk), C adds grad clip and a 100-epoch cap, one
   seed each.
3. **B vs C at track level** (same 35,957 test tracks, same on-disk split and normalisation): B track AUROC 0.482 /
   0.481 (recon), 0.307 / 0.302 (Mahalanobis) vs C 0.928 / 0.929. Window-level B vs C is not the same set (42,653 vs
   40,469 windows).
4. **D's four attacks** against one benign test set: one model, one normalisation, one threshold. AUROC and recall at
   val p95 are comparable across attacks; PR-AUC is not (attack share 0.318 / 0.559 / 0.769 / 0.769).
5. **P's eight subsets** with each other and with the paper's Table III (same unit, same metric).
6. **M vs the one-class ConvAE at the same T** (added 2026-11-18): M at T64 / T48 / T32 uses exactly the train, val
   and test windows of S0 AE / F / D (imported data classes, caches and splits; T64 also the same 362 val vehicles),
   the same benign-val p95 / p99 threshold rule and the sibling's early-stopping / plateau patience; both have
   seeds 0–3. Footnotes: a different model family (flattened MLP / ResNet vs 1-D ConvAE) with a different input
   corruption (random cell masking) and optimiser (AdamW, weight decay 0.01 vs Adam 1e-5); **M is untuned** (one
   configuration from widedeep's defaults, scaled down); M ran on the Mac only (ConvAE seeds 1–3 on EC2 CUDA); M's
   T64 Mahalanobis uses shrinkage-when-ill-conditioned (not triggered) vs S0's ridge. The length-matched AUROC exists
   only for M (and T), not for the ConvAE.
7. **D vs M vs T at T32** (same 60,142 benign test windows and attack splits): one-class ConvAE, masked AE (4 seeds
   each) and TimesNet (context, **one seed**, EC2 CUDA, 2.4 M params vs 58–73 k, SSL losses beyond reconstruction,
   its own feature scaling, D13). T's recon score fills < 2-row windows with the benign val median, so its pooled recon
   AUROC is not a measured error on those windows; compare T on kNN / Mahalanobis, and length-matched.

**Rows that must not be ranked against each other:**

- **S0 (T64) vs A / B / C (T48):** same test vehicles but different window cut (38,426 vs 40,469 / 42,653 windows),
  different architecture size (80,189 vs 71,741), different early-stopping patience (5 vs 10) and, for the one-class
  question, a different training set (S0 AE = benign only; A / B = benign + GridSybil). The 0.655 vs 0.484 gap mixes
  "benign-only training" with T and protocol. Added 2026-11-11: **G vs S0 AE** is the closest pair (same T64 input,
  test windows, normalisation and architecture; unlabelled 0.482 vs benign-only 0.655 ± 0.002), but early-stopping /
  plateau patience (10 / 5 vs 5 / 3), the val vehicles (518 of both classes vs 362 benign) and seeds (1 vs 3)
  differ. **F vs A** (benign-only 0.689 vs unlabelled 0.484, T = 48) score the same test-window counts, with
  different normalisation constants, train / val vehicle sets and gradient clipping.
- **H vs A / B / G (unlabelled, GridSybil 31.3% vs 64–65% of training):** 0.610 vs 0.482–0.484 differ in T, benign
  source (8 folders vs 2), split, test vehicles, normalisation and test GridSybil share (0.323 vs 0.67); the contrast
  is suggestive only, not an attack-share effect.
- **H vs D (T = 32, unlabelled 0.610 vs benign-only 0.736):** different splits and test vehicles.
- **D vs S0 AE (both one-class):** 0.736 vs 0.655 differ in T (32 vs 64), benign source (8 folders vs 2), split, test
  vehicles, GridSybil share (0.318 vs 0.668), normalisation (benign-only vs both classes) and seeds (1 vs 3).
- **E vs C (both supervised):** 0.926 vs 0.929 differ in T, split ratio, test vehicles, benign source and class
  balance. PR-AUC (0.925 vs 0.971) and F1 are prevalence-dependent (GridSybil share 0.323 vs 0.669) and cannot be
  compared at all; benign FPR at max-F1 (0.011 vs 0.051) reflects a different operating point chosen on a different
  val balance.
- **D vs E:** different splits and test vehicles although both are T = 32 over all 8 folders.
- **P vs any learned model:** per received message vs per window; F1 at one fixed threshold vs AUROC; no train /
  test split; evaluated on all vehicles. The READMEs of E and the slides state this explicitly.
- **"Recall at 5% FPR" columns:** the threshold is the p95 of *benign* val scores in S0-AE and D, the p95 of *all*
  val scores (both classes) in A and B, and the benign-val 95th percentile of the head probability in C and E. The
  achieved benign test FPR is 0.044–0.052 everywhere except B Mahalanobis (0.068), so the column is roughly aligned,
  but the definitions differ.
- **Length-only AUROC:** direction chosen on test labels in S0, fixed in advance in A, D, F, G and H (−n_messages), "best direction" in C and
  E; absent in B.
- **Short-window bins** differ per study (≤ 3 / 4–16 / 17–48 / 49–64 in S0; ≤ 3 / 4–16 / 17–32 / 33–48 in A; 1–3 /
  4–16 / 17–47 / 48 in C; 1–3 / 4–16 / 17–31 / 32 in E; only "≥ 4 rows" in D). Only the 1–3 (≤ 3) bin lines up.
- **M vs any cell other than its ConvAE sibling:** inherits every mismatch of that sibling's row above (e.g. M T32 vs
  H: different split and test vehicles).
- **Length-matched AUROC** (M, T only) is not comparable with the pooled AUROC of other rows; for long-window
  GridSybil the two differ by ≤ 0.034, for DataReplay / DoS by up to 0.44 (T Mahalanobis DoSRandom 0.960 → 0.523).
- **Track level:** S0 and A have none. Track ids contain the true sender (F13), so every track-level number is a
  privileged grouping.

---

## What is missing for a paper-grade empirical study (factual)

**Seeds and uncertainty**

- Seeds per study (as written 2026-11-10): S0 = 3 (AE and supervised); A, B, C, D, E, F, G, H = **1 each** (seed 0). The
  `seed0_featlog` folders are re-runs of seed 0 with identical histories, not extra seeds. **Updated 2026-11-18:** since
  2026-11-17 every grid cell (S0 AE, A, C, D, E, F, G, H) has seeds 0–3 (S0 AE 0–4; seed 0 Mac, 1–3 EC2); B and the S0
  supervised model keep 1 / 3 seeds; M has seeds 0–3 per T and encoder (TabMlp and TabResnet complete); T has 1 seed.
- Confidence intervals: only S0 has any (vehicle bootstrap, 1,000 resamples, test-sampling only; not combined with
  seed variance). A–E: none. P: none (deterministic, but no interval over receivers / vehicles).
- No significance test between any two methods (no paired bootstrap, no DeLong).
- Windows of one vehicle are not independent, and with overlap (B) or repeated benign traffic (D, E; F1 ≈ 4×) the
  effective sample size is unknown; it is stated in the notebooks, not estimated.

**Test metrics not persisted**

- F, G, H persist test metrics and per-window scores (added 2026-11-11); so do M (`test_metrics.json`,
  `scores/<split>.npz`) and T (evaluation `summary.json`, `scores/<split>.npz`) (added 2026-11-18); the rest of this
  item applies to A–E (the EC2 seeds 1–3 of A–E have judge recompute JSONs, not per-window score files).
- A–E test metrics are stored only as notebook outputs (3–4 decimals) and README text; no per-window score files, so
  CIs, paired tests and re-binned breakdowns cannot be computed without re-scoring from `best.pt`.

**Attack coverage**

| Study | GridSybil | DataReplay | DoSRandom | DoSDisruptive |
|---|---|---|---|---|
| S0, A, B, C, E, F, G, H | yes | no | no | no |
| D | yes | yes | yes | yes |
| M (T32 only; T64 / T48: GridSybil only) | yes | yes | yes | yes |
| T (context, T32) | yes | yes | yes | yes |
| P | yes | yes | yes | yes |

- No supervised model is tested on an attack it was not trained on (no leave-one-attack-out), and no supervised
  multi-attack model exists.
- For DataReplay / DoS, 94.9–99.7% of windows hold 1–3 messages (F11), so the window unit itself is not settled for
  these attacks (`agent.md` §5: "DataReplay / DoS windows … receiver time window?" is open).
- No 0709 ↔ 1416 transfer run (train on one period, test on the other) for any study; only E reports a per-period
  breakdown of a pooled model, and only E reports per-folder numbers.

**Mismatched test sets and protocols** (details in the section above)

- Four different test sets (G1–G4), four normalisations, three T values, three split ratios.
- No one-class run under the same protocol as an unsupervised run. Since 2026-11-11 each window length has both
  (T64: S0-AE vs G, same test windows; T48: F vs A / B; T32: D vs H, different test sets), but training settings,
  val sets or normalisation still differ within each pair, and no pair is a paired comparison with an interval
  (2026-11-18: each of these cells now has seeds 0–3, so the "one seed" part no longer holds; the one-class masked AE M
  shares windows with S0 AE / F / D, but none of them shares a protocol with an unsupervised run).
- Early-stopping patience, `min_delta`, epoch cap and gradient clipping differ between studies; the S0 AE epoch cap
  was raised from 50 to 300 after the first runs.
- No hyperparameter search or sensitivity study anywhere (latent size, T, λ of the joint loss, thresholds); M was
  run in one configuration only (masking probability 0.2, widths, z = 16 fixed).
- Length control: only M and T report a length-matched AUROC (computed for this inventory from their score files);
  all other studies report length-only and 1–3 / ≥ 4-row splits only.
- Supervised operating points: S0 uses a fixed 0.5 threshold, C / E use val-chosen thresholds.

**Classical baselines in the repo**

- Present: length-only score and 53 window statistics + logistic regression (`task/encoder/baselines.py`, T64 only,
  supervised, no val-tuned threshold); the rule-based plausibility detector (P).
- **Not present** (searched `*.py` and `*.ipynb` outside the venvs for the class names): Isolation Forest, One-Class
  SVM, Local Outlier Factor, PCA reconstruction error, kNN distance, Gaussian / Mahalanobis on raw features or on
  the 53 statistics, Deep SVDD, LSTM / GRU autoencoder, VAE, Transformer or TimesNet autoencoder, random forest /
  gradient boosting on window statistics. PCA appears only for 2-D plots of z; `LogisticRegression` also appears in
  `scripts/evaluate_frozen_encoder.py` and an EDA notebook, which are not baseline studies.
- No unsupervised classical baseline exists at all, and no classical baseline exists on the T48 or T32 inputs.
- Added 2026-11-18 (neural, not classical): an MLP / ResNet masked autoencoder (M) and a TimesNet one-class SSL
  encoder with kNN / Mahalanobis scores (T, context, one seed). The classical one-protocol sweep under
  `baselinepaper/experiments/` is not inventoried here.

**Leakage: stated but not measured**

- **F14** (GridSybil_0709 is a re-simulation; near-copies of other 0709 cars can straddle a vehicle-key split): the
  only measurement is on the `all/T24` split (2.5–5.9% of GridSybil_0709 test benign identities with a near-copy in
  train within 1 m; `agent.md` §4). For `all/T32_benign` (D) and `all/T32_benign_grid` (E) the metadata says "Not
  measured or fixed here". The benign + GridSybil inputs (S0, A, B, C) are stated to be unaffected (one family).
- **F1** (the 4 families replay the same benign traffic): the 89.7% near-copy figure is for the legacy v1 split.
  For D and E the key-based split keeps copies in one split by construction, but no near-copy audit was run on these
  splits and the ≈ 4× benign duplication is not corrected in any metric.
- Whether absolute position features (`claimed_pos_*`, `rx_pos_*`) act as a location shortcut is raised in the S0
  README ("not tested here"); no ablation without position features exists.
- **F13**: track-level results use a grouping that contains the true sender.
- P: the paper's check variant (CaTCh vs legacy) is inferred from the match, and two checks (map, proximity) are
  missing.

**Other**

- No message-level score from any learned model, so no like-for-like comparison with P or with the paper.
- No inference-time / latency or model-size-vs-accuracy analysis; training time is recorded.
- All runs have `git_dirty: true` in `env.json` (F, G, H included).
- S0 std: computed over n = 3 (the file does not say whether population or sample std).

---

## Where sources disagree or need care

1. **Test vehicles, T48_tvt family.** `src/data/encoder_input/benign_gridsybil/T48_tvt/metadata.json` →
   `split.vehicles` = 4,605 / 575 / 575; `agent.md` §2 and the notebooks of B and C report 4,603 / 575 / 574 sender
   vehicles (vehicles that have windows), and S0's `results.json` has `n_sender_vehicles` 574. Both are on disk; the
   difference (2 train, 1 test vehicle) is presumably vehicles assigned to a split that contribute no window — not
   verified here.
2. **Split keys vs vehicles with windows, E.** Metadata: 6,383 / 564 / 564 keys; windows come from 5,933 / 524 / 524
   vehicles (`agent.md` reports both). Not a contradiction, but the two must not be mixed in a paper.
3. **S0 length-only by stratum.** The README table shows 0.505 / 0.536 / 0.507 / 0.527 (best direction per
   stratum, from `strata.*.length_auroc_best_direction`), while `methods.baseline:length.strata_auroc` in the same
   file holds 0.505 / 0.536 / 0.493 / 0.473 (fixed −n direction).
4. **S0 stats + LR AUROC.** `runs/baselines/summary.json` 0.85492 vs the notebook replication
   `runs/notebook-seed0/results.json` 0.85498 (4th-decimal difference, same method).
5. **S0 supervised-z Mahalanobis CI.** The README's CI 0.753 – 0.777 is seed 0's (AUROC 0.765) and does not contain
   the 3-seed mean 0.787 printed next to it; per-seed CIs are 0.753–0.777, 0.766–0.791, 0.807–0.830.
6. **`agent.md` §2** has no row for `task/encoder/` or `task/autoencoder/`; both appear only in changelog entries.
7. **`agent.md` vs run files for A–E and P:** every headline number checked (best epochs, val scores, test AUROC /
   PR-AUC / F1 / recall, thresholds, runtimes, P's F1 table) agrees with `summary.json`, the notebook outputs and the
   READMEs after rounding. No disagreement found.
8. **Run folders:** none missing. All of `task/*/runs/seed0/` (A–E), `task/encoder/runs/{ae,supervised}-seed{0,1,2}/`,
   `runs/baselines/`, `runs/evaluation/` and `src/runs/baselines/plausibility/` exist with config, env and summary.
9. **M README vs run folders (2026-11-18).** `task/masked_autoencoder/README.md` ("Results (TabResnet, seeds 0-3)")
   now lists all 12 finished TabResnet runs (T64 / T48 / T32, seeds 0-3), matching section M; its TabMlp table
   equals the numbers in section M (pooled, length-matched, length-only, ConvAE column).
10. **T recon score.** The TimesNet pooled recon AUROC includes windows with < 2 real rows whose score is the benign
    val median (filled, not measured; `summary.json` → `protocol.recon_fill`); its DataReplay / DoS recon rows are
    therefore mostly fill (70–89% of attack windows). The evaluation's own like-for-like subset with D is GridSybil
    ≥ 4 rows (`protocol.comparable_to_baseline`): T recon 0.772, D recon 0.778 (seed 0).
