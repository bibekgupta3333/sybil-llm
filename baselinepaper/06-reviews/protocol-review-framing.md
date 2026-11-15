# Protocol review — framing, contribution, venue

Written 2026-11-10, before any E1–E5 run. Two roles: thesis advisor (A), programme-committee chair (E). Methods are
reviewed in `protocol-review-methods.md`.

## 1. Verdict

**Go, conditional and time-boxed.** (A) Worth at most 12 experiment days, because the harness (one split, length
control, vehicle bootstrap, reference detectors) is reused for the thesis's S1.4 evaluation. Smallest publishable
version: E1–E5 + E6, questions Q1–Q3, one main table. (E) "Known failure modes in a new setting" alone is a weak
accept; it is defensible only with the regime × attack table under one protocol and the length control. Overreach to
remove:

- "for the first time": 22 of 51 vehicular papers were read in full. Write "we found no …".
- "evaluation checklist": one study does not earn one. Three recommendations in the discussion.
- Q4: per-message F1 vs per-window AUROC is not a comparison. Drop it unless E10 is done.

## 2. Lead message

*On VeReMi-Extension Sybil attacks with vehicles held out, what a detector is trained on (benign only, unlabelled mix,
labels) changes detection far more than which detector is used, and for the three pseudonym-rotating attacks window
length alone explains what any detector scores.*

Title: the current one promises an answer to "when" and names only autoencoders, while the table holds six detector
families. Proposed: **"Training Assumptions and a Length Shortcut: A Vehicle-Disjoint Evaluation of Sybil Anomaly
Detection on VeReMi-Extension"**. Keep the question title only if R2 fires.

## 3. Framing rules, fixed before results

Window AUROC, seed mean, vehicle-bootstrap 95% interval. Record the rule that fired in protocol §9.

| Rule | Outcome | The paper says |
|---|---|---|
| R1 | A benign-only detector reaches GridSybil ≥ 0.90 (lower bound ≥ 0.85) | "Reconstruction fails" is dead. Lead: a simple detector suffices under a leak-controlled protocol; our autoencoder is weak. Benchmark paper, new title |
| R2 | All benign-only detectors in 0.60–0.80 on GridSybil, spread ≤ 0.10 | §2 as written |
| R3 | Best benign-only detector in 0.80–0.90 | "Partial, model-dependent". No "fails", no regime-over-model claim |
| R4 | For an attack, every detector ≤ length-only + 0.02, or the ≥ 4-row interval is wider than 0.15 | "Not evaluable at the pseudonym-link unit"; no detection claim either way |
| R5 | DataReplay, ≥ 4 rows: any benign-only detector > 0.60 | Drop "in-distribution by construction, bounds every per-sender detector"; model-specific. If all intervals lie below 0.5: keep it, and state that inverting the score needs labels |
| R6 | `mix_gridfolders` autoencoder ≥ 0.65 on GridSybil | Attack-majority finding did not replicate: out of abstract and title, number reported |
| R7 | Seed standard deviation ≥ a difference | No ranking for that pair |
| Kill | Every detector and regime within 0.05 of length-only and inside each other's intervals; or the GridSybil derivation check (protocol §2) fails and is not fixed in 2 days | No paper; thesis appendix |

**Gap to published one-class numbers (0.93–0.99 vs 0.66–0.74): not claimable as a cause without E7 or E9.** Published
values are accuracy / F1 / recall at tuned thresholds on other sequence units; ours are AUROC. Allowed: two sentences
("lower under a stricter protocol") and a table of protocol differences. Not allowed: "inflated", "due to leakage".
E7's random-split run licenses one sentence about the split only.

## 4. Recommendations (ranked)

| # | Tag | Recommendation | Cost |
|---|---|---|---|
| 1 | MUST | E1–E5 as protocolled; freeze the table, then apply §3 | 6–8 days |
| 2 | MUST | Scope: Q1–Q3; main table = benign-only regime × 4 attacks; mix and supervised regimes for GridSybil only | 0 |
| 3 | MUST | Full-text check of B01 / A09, B03, B06 (gap analysis §6) before any sentence on published numbers | 1 day |
| 4 | MUST | Experiments stop 30 Nov; 2 days per week stay on `model-grid` / S1.4 | 0 |
| 5 | SHOULD | **E6 — best value of E6–E10**: the most original finding, a certain figure, extends the two mix regimes | 1 day |
| 6 | SHOULD | E7, random-split run only | 0.5–1 day |
| 7 | SHOULD | E10, or delete the plausibility row | 0.5 day |
| 8 | OPTIONAL | E7 F14 measurement (also informs the thesis's open F14 decision) | 1 day |
| 9 | OPTIONAL | E8 leave-one-attack-out; E9 → thesis or journal version | 1 / 2 days |

Carries the paper: GridSybil across regimes; the length-only row with the ≥ 4-row stratum; DataReplay. Filler:
Mahalanobis rows (one sentence), full-window DoS AUROC, average precision and recall beyond GridSybil, per-detector
rows for each mix regime.

## 5. Figures and tables

- **Table 1** protocol differences, B01 / B03 / B06 / A19 vs ours (split unit, threshold source, sequence unit).
- **Table 2** detectors × 4 attacks: full and ≥ 4-row AUROC, length-only row, intervals.
- **Table 3** GridSybil by regime, with recall and benign false-positive rate at benign-val p95.
- **Figure 1** window-length distribution per class (the shortcut, before any model).
- **Figure 2** GridSybil AUROC vs attack share in training (E6); without E6, regime bars.
- **Figure 3** score distributions: benign, GridSybil, DataReplay (benign-only autoencoder).

## 6. Venue

**IEEE SoutheastCon 2027, Cybersecurity track; submit by 20 Dec** (rolling review, decision within 20 days, early
papers may be reviewed twice). Xplore, Florida, 6 pages: it meets "the student needs a publication". No arXiv before
the decision. VehicleSec has the better audience, but 35 days with unrun experiments, a harder committee and no Xplore
make it the wrong first target; never both. A longer version with E9 can follow (VTC Recent Results, journal).

## 7. Thesis risk

TimesNet loses about three weeks while the `model-grid` upload, the LR-schedule fix and S1.4 are pending. Mitigation:
the pretrained encoder plugs into the harness as one more detector; recommendation 4; a missed 30 Nov gate shrinks the
paper, it does not extend the experiments.

## 8. Schedule

| Week | Work | Gate |
|---|---|---|
| 10–16 Nov | Harness, derivation check, run records, convolutional autoencoder × 3 regimes × 5 seeds | check passes |
| 17–23 Nov | Reference detectors, LSTM autoencoder, bootstrap, length strata | 23 Nov: Table 2 frozen, §3 applied |
| 24–30 Nov | E6, E7 random split, literature checks, sections 2–4 | 30 Nov: experiments stop |
| 1–7 Dec | E10 if time, figures, full draft | draft to professor |
| 8–14 Dec | Comments; tables rebuilt from a clean commit | numbers = score files |
| 15–20 Dec | Anonymise, page fit, submit | buffer to 30 Dec |
