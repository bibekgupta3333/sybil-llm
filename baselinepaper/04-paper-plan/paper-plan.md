# Paper plan — empirical study from the baseline work

Written 2026-11-10. Based on `../03-gap-analysis/gap-analysis.md`. Venues and verified deadlines: `venues.md`.
Nothing here has been implemented; every enhancement waits on the user's OK (it is outside the adopted thesis plan,
RULE 7).

## 1. Working title and questions

**"When Does Reconstruction Detect a Sybil? An Empirical Study of Autoencoder Anomaly Detection on
VeReMi-Extension"**

- **Q1** How well does reconstruction-based detection work for each Sybil type when vehicles are held out and the
  threshold comes from validation?
- **Q2** How much does the training-data assumption matter: unlabelled mix, benign only, labelled?
- **Q3** How much of the apparent performance is explained by a trivial shortcut (window length) and by the split?
- **Q4** How do learned detectors compare with the dataset paper's plausibility checks?

Contribution wording: known failure modes of reconstruction-based anomaly detection, measured for the first time (to
our knowledge) on Sybil attacks under a leak-controlled protocol, plus an evaluation checklist for VeReMi-Extension.

## 2. Enhancements

Effort assumes the existing packages are reused; the models have about 64k parameters and train in 5–20 minutes on
the Mac.

| # | Enhancement | Fixes | Priority | Effort |
|---|---|---|---|---|
| E1 | One protocol: one input (the all-folder T = 32 data), one vehicle split, one test set with benign + all four attacks; rerun every regime on it | W1, W6 | must | 2–3 days |
| E2 | 5 seeds per model; confidence intervals by bootstrap over vehicles | W2 | must | 1 day + compute |
| E3 | Reference detectors on the same windows: Isolation Forest, one-class SVM, PCA reconstruction, kNN distance (on window statistics), and an LSTM autoencoder (the literature's model) | W3 | must | 2–3 days |
| E4 | Shortcut control in every table: length-only score, and AUROC within length strata (1–3 rows, ≥ 4 rows) | Q3 | must | 0.5 day (partly exists) |
| E5 | Save per-window test scores for every run; run from a clean commit | W8 | must | 0.5 day |
| E6 | Attack-share sweep: train the autoencoder with 0, 1, 5, 20, 50, 64% GridSybil windows | W7 | should | 1 day |
| E7 | Split experiment: same model with a random window split vs the vehicle split; measure F14 near-copies on the T32 split | W5 | should | 1–2 days |
| E8 | Unseen attacks: supervised model trained on GridSybil, tested on the other three; leave-one-attack-out | W6 | should | 1 day |
| E9 | Sequence-unit ablation: group messages by true sender id (privileged) vs by pseudonym link, same model | W4 | could | 2 days |
| E10 | Window-level score for the plausibility detector (share of flagged messages) so it enters the same table | W9 | could | 0.5 day |

E1 creates one more evaluation layout on existing data. If it needs a new split it is a RULE 3 change and gets
recorded in `docs/plan/research-plan.md`; reusing the `all/T32_benign` split (80 / 10 / 10) avoids that.

Minimum for submission: E1–E5. Recommended: add E6 and E7, because they turn the two most original observations
(attack-majority training, split inflation) into measured results. E9 is the only one that can explain the gap to
published one-class numbers, but it is the most work.

## 3. Outline (6 pages)

1. Introduction: published Sybil detection results on VeReMi-Extension are high; the protocols behind them vary.
2. Background: dataset, the four Sybil attacks, pseudonym change; related work (slices A, B) and the general
   anomaly-detection warnings (slice C).
3. Protocol: receiver-side pseudonym windows, vehicle split, thresholds from validation, length-only control, metrics.
4. Detectors: plausibility rules, reference detectors, autoencoders under three training regimes, supervised model.
5. Results: the regime × attack table (Q1, Q2), shortcut and split analysis (Q3), comparison with rules (Q4).
6. Discussion: why DataReplay is in-distribution; attack-majority data; checklist for future VeReMi work; limitations.

## 4. Schedule options

| Target | Deadline | What fits |
|---|---|---|
| USENIX VehicleSec '27 short paper | 15 Dec 2026 | E1–E5 by 30 Nov, E6–E7 by 7 Dec, writing in parallel. Tight. Best audience fit; not on IEEE Xplore |
| IEEE SoutheastCon 2027 | 30 or 31 Dec 2026 (official pages disagree) | E1–E8 comfortably. On Xplore, in Florida. Refuses work already posted as a preprint |
| VTC2027-Spring Recent Results | 1 Feb 2027 | Everything including E9 |

## 5. Decisions needed from the user / professor

1. Target venue (decides the page limit and how many enhancements fit).
2. OK to implement E1–E5 (and which of E6–E10).
3. Whether the paper may use the all-folder T = 32 split as is, with F14 stated as a limitation, or F14 is measured
   first (E7).
