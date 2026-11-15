# Gap analysis — baseline studies vs the literature

Written 2026-11-10. Inputs: `../01-our-studies/inventory.md` (our numbers, each traced to a file),
`../02-literature/papers.json` (76 unique papers: slice A = learning on VeReMi, B = reconstruction / one-class in
vehicular networks, C = methodology), and the three `notes_*.md`.

**Read this first.** Our numbers are side experiments, mostly one seed, and are not thesis results (RULE 1). On the
literature side, 22 of the 51 vehicular papers were read in full; the rest are abstract-level, so every "no paper
does X" below means "none of the papers we read or skimmed", and belongs in the paper as "to our knowledge".

## 1. What we have

| Regime | Study | Input | Seeds | GridSybil AUROC (test, window) | Length alone |
|---|---|---|---|---|---|
| Labels | `task/encoder` supervised head | T64 | 3 | 0.928 ± 0.002 | 0.535 |
| Labels | `task/autoencoder_st` | T48_tvt | 1 | 0.929 | 0.530 |
| Labels | `task/autoencoder_st_t32` | all/T32_benign_grid | 1 | 0.926 | 0.560 |
| Labels | `task/encoder` window statistics + logistic regression | T64 | 1 | 0.855 | 0.535 |
| One-class (benign only) | `task/encoder` autoencoder, reconstruction | T64 | 3 | 0.655 ± 0.002 | 0.535 |
| One-class (benign only) | `task/encoder` autoencoder, Mahalanobis | T64 | 3 | 0.319 ± 0.016 | 0.535 |
| One-class (benign only) | `task/autoencoder_benign_t32` | all/T32_benign | 1 | 0.736 | 0.559 |
| Unlabelled mix (64% attack) | `task/autoencoder` | T48 | 1 | 0.484 | 0.530 |
| Unlabelled mix (64% attack) | `task/autoencoder_overlap` (reconstruction / Mahalanobis) | T48_tvt_s24 | 1 | 0.484 / 0.337 | not found |
| Rules | `src/baselines/plausibility` | raw messages | — | F1 0.727 / 0.692 per message (no AUROC) | — |

Other attacks exist only in the one-class T32 study: DataReplay 0.391, DoSRandom 0.973, DoSDisruptive 0.611, against
length alone 0.952 / 0.967 / 0.967.

## 2. Where the literature stands

| Topic | What the papers do | Ours |
|---|---|---|
| Train / test split | No VeReMi paper in slice A holds out physical vehicles or scenarios; stated splits are mostly random over windows or sequences (A22, A23). Only PAMPOS (B03) states a split by sender. None mentions that the four attack folders replay the same traffic | Split by physical vehicle; cross-scenario reuse found (F1) |
| One-class results on Sybil | 0.93–0.99 reported; the four Sybil types are treated as easy (B01 recall 0.98–1.0, B03 F1 0.94–0.95, B06 F1 0.93 / 0.90) | GridSybil 0.66–0.74 AUROC; DataReplay below chance |
| Threshold | Tuned on labelled test data in B01, B15, B20; labels used in validation in B04, B07 | From validation only (benign percentiles or val max-F1) |
| What "unsupervised" means | Almost always benign-only training data selected with ground-truth labels. One mixed-data case (B15) lands near chance | Both settings run, kept apart |
| Shortcut control | None found | Length-only AUROC next to every result |
| Several detectors under one protocol | None for Sybil attacks; closest is VehiGAN (B04) on 35 non-Sybil attacks | Six models, but not yet under one protocol (§4) |
| Replay attacks | Known to defeat reconstruction on the CAN bus (B19 detection rate 0.57, B17 TPR 0.03); not reported for V2X Sybil | DataReplay 0.391 |
| Unseen attacks | One paper (A21): F-score 0.48–0.49 on unseen DoS Sybil variants | Not tested |
| Seeds, confidence intervals, fixed false-positive rate | Not reported in slice A | 3 seeds in one study, 1 in five |

## 3. Gaps we can fill (candidate contributions)

1. **A one-protocol comparison of training regimes for Sybil detection.** Rules, unlabelled mix, benign-only and
   labelled models on the same vehicle-held-out test set, per attack type. To our knowledge no such table exists for
   the VeReMi-Extension Sybil attacks.
2. **Reconstruction detection of Sybil attacks is much weaker than reported once thresholds come from validation and
   vehicles are held out,** and it is attack-dependent: partial on GridSybil, below chance on DataReplay. The cause
   for DataReplay is the threat model (a replayed genuine trajectory is in-distribution for any per-sender detector),
   which matches the CAN-bus replay results (B19) and the outlier-reconstruction literature (C04–C06).
3. **Attack-majority training data.** In the GridSybil folders the attack is about 64% of windows, so a model trained
   without labels learns the attack as normal (AUROC 0.48; Mahalanobis 0.34, below chance). The general contamination
   literature studies a small anomalous minority (C08, C09); we found no study of a majority-anomaly training set.
4. **The length shortcut.** Pseudonym change every 1–2 messages makes 95–100% of DataReplay / DoS windows 1–3 rows
   long, so window length alone reaches 0.95–0.97. Any sequence model's score on these attacks must be read against
   that. No paper in the sample reports this control.
5. **Dataset findings:** the cross-scenario traffic reuse, and the reproduction of the dataset paper's plausibility
   detector (6 of 8 subsets within 0.05 F1) as a non-learned reference.

Not new, to be cited and not claimed: contaminated autoencoders fitting anomalies (C01, C08, C09), autoencoders
reconstructing anomalies (C05, C06, C04), latent-distance scores needing clean data (C02, C07), the gain from labels
(C03, C10), shortcut learning (C18–C20), leakage as a class of error (C20, C23). The paper's claim is that these known
failure modes occur in Sybil detection on VeReMi-Extension, with their measured size.

## 4. Gaps in our own work (what a reviewer will ask)

| # | Weakness | Why it matters |
|---|---|---|
| W1 | The six studies use different T (32 / 48 / 64), splits, test vehicles and normalisation; only a few pairs share a test set | The central table cannot be printed honestly today |
| W2 | One seed in five of six studies; no confidence intervals outside `task/encoder` | Differences such as 0.655 vs 0.736 cannot be interpreted |
| W3 | Only one model family (a small convolutional autoencoder). No Isolation Forest, one-class SVM, PCA reconstruction, kNN, LSTM autoencoder, Deep SVDD | "Reconstruction fails" could just mean "our autoencoder is weak"; the literature's models are LSTM / CNN-LSTM autoencoders (B01, B06) |
| W4 | Our one-class numbers (0.66–0.74) are far below published ones (0.93–0.99) and we cannot yet say why | The paper's main claim. Candidate reasons, none tested: test-tuned thresholds, random splits, grouping messages by true sender id (PAMPOS needs ≥ 15 messages per sequence; A19 groups by sender id) versus our pseudonym links, engineered features |
| W5 | Leakage is argued, not measured in this setting: no random-split vs vehicle-split run of the same model; F14 near-copies unmeasured on both T32 splits; benign traffic repeated about 4× across folders | "Published numbers are inflated by the split" needs our own number |
| W6 | DataReplay / DoS covered by one study only; supervised models trained and tested on GridSybil only | No unseen-attack result, no 0709 ↔ 1416 transfer |
| W7 | Attack-majority result is two points (0% and 64%) | A sweep over the attack share would turn it into a finding |
| W8 | Test scores of five studies exist only as notebook output; every run has `git_dirty: true` | Reproducibility section of the paper |
| W9 | Plausibility baseline is per message (F1), learned models per window (AUROC) | Not directly comparable in one table |

## 5. Verdict

The findings are publishable as a short empirical paper if W1–W3 are fixed, and noticeably stronger with W4, W5 and
W7. Without W1–W3 the paper is a collection of single runs on different data and is unlikely to pass review. The
required work is listed with effort estimates in `../04-paper-plan/paper-plan.md`.

## 6. Things to check before they go into the paper

- A02 Table III values were taken from our own notes (HAL blocked the fetch): re-read from the paper.
- B01 / A09 (Alladi et al.): split and threshold details from full text; A09 was abstract-level.
- B03 (PAMPOS): how sequences are formed for DataReplay / DoS senders (by true sender id or by pseudonym).
- B06 numbers are internally inconsistent (aggregate F1 above every per-attack F1): cite with care or not at all.
- B13, A13, A16, A30: bibliographic only, content unread.
- Slice C details listed in `../02-literature/notes_C.md` §4.
