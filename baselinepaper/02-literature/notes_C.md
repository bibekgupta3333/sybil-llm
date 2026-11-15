# Notes C — methodological / critical literature on reconstruction-based anomaly detection

Slice C of the literature search (general ML, time series, network intrusion). Records: `papers_C.json` (C01–C28).
All 28 papers were confirmed by web search or page fetch on 2026-11-10 (title, authors, year, venue, URL). Where a
detail is from memory of the paper and not from a page seen in this session, the record's `verified_via` says so —
read those passages in the paper before quoting. Our numbers below are one seed each (side experiments), not thesis
results.

## 1. Already known in general anomaly detection — cite, do not claim as new

| Our observation | Known as | Cite |
|---|---|---|
| Autoencoder trained on mixed benign + attack data is at chance (AUROC ≈ 0.48) | Unsupervised AD assumes the training set is (almost) all normal; an autoencoder fits whatever is frequent, including anomalies | C01, C08, C09, C11 |
| One-class AE scores DataReplay below chance (0.39); GridSybil only 0.74 | Autoencoders reconstruct anomalies well ("outlier reconstruction"); generative novelty scores can rank out-of-distribution data as more normal than the training data | C05, C06, C04, C28 |
| Mahalanobis distance on the latent does not help (below chance when contaminated) | Latent-distance scores (Deep SVDD centre, Mahalanobis) need clean, compact normal data | C02, C07, C01 |
| Supervised ≈ 0.93 vs label-free 0.48–0.74 | A few labels beat any unsupervised detector; supervision level matters more than the model | C03, C10, C12 |
| Detectability differs strongly per attack type | No detector is best for every anomaly type | C10, C15, C17 |
| DoS / DataReplay "detected" through window length (length alone 0.95–0.97) | Shortcut learning / spurious correlation; trivial benchmarks | C18, C19, C20, C13 |
| The usual split leaks (same traffic in all four scenarios) | Train / test non-independence leak; data snooping | C23, C20, C22 |
| 62–65% attack messages in the GridSybil folders | Unrealistic anomaly density; base-rate fallacy; spatial bias | C13, C20, C22 |
| Need for a rule-based and a one-feature baseline, threshold from validation | Simple baselines and sound metrics in time-series AD | C14, C15, C16 |
| Dataset defects change reported results | Audits of CICIDS2017 / 2018 and other NIDS benchmarks | C24, C25, C26, C21 |

Consequence for the wording: none of the four mechanisms (contamination, outlier reconstruction, shortcut, leakage) is
a new phenomenon. The paper should say "we show that these known failure modes occur in Sybil detection on
VeReMi-Extension and measure their size", with C20 as the checklist and C21 as the framing (abnormal ≠ malicious).

## 2. Appears specific to the vehicular / Sybil setting (candidate contributions; check against slices A and B)

- **Attack majority, not contamination.** The contamination literature treats anomalies as a small unlabeled minority
  (C08 assumes sparsity; C09's ratios not checked this session). In the VeReMi-Extension Sybil folders the attack is
  ≈ 64% of windows, so "normal" is redefined and Mahalanobis goes below chance because the attack sits at the centre
  of the latent. I found no general-AD paper that studies a majority-anomaly training set; state it as "to our
  knowledge".
- **A causal reason for the below-chance score.** In C04 / C28 the reason is model inductive bias on unrelated image
  sets. Here the attack itself is a replay of genuine trajectories, so the input window is by construction in the
  normal distribution; only cross-sender context (two pseudonyms with the same track) can reveal it. This is a
  property of the threat model, not a model defect, and it bounds every per-sender reconstruction detector.
- **The shortcut comes from the attack mechanism and the sequence unit.** Pseudonym rotation every 1–2 messages makes
  95–100% of DoS / DataReplay windows 1–3 rows long. Length is then a valid attack signature in this dataset but not a
  learned motion feature; the general shortcut literature (C18, C19) has no such case where the unit of analysis
  creates the shortcut.
- **The form of the leak.** Four attack scenarios replay one traffic simulation, so a within-scenario vehicle split
  still leaks across scenarios (89.7% of benign test identities have a near-copy in train in the old split), and one
  folder (GridSybil_0709) is a separate re-simulation. This is an instance of C23's taxonomy, but the dataset-specific
  finding and the fixed split are ours. Whether earlier VeReMi papers reported it is for slices A / B.
- **Reproduction of the dataset paper's rule-based detector** (6 of 8 subsets within 0.05 F1) as a non-learned
  reference point next to the autoencoders.

## 3. Gaps a reviewer may raise

- No learned one-class objective (Deep SVDD, C02), no robust / contamination-aware training (C08, C09), no few-label
  method (Deep SAD, C03). Either add one or list them as not tested.
- One seed per model; C14 / C16 / C20 all ask for variance and trivial baselines.
- Supervised 0.93 is on a seen attack only (GridSybil); it says nothing about unseen attacks.
- Overlapping windows (stride 24) are not independent samples; confidence intervals should be per track or vehicle.
- F14: GridSybil_0709 near-copies may still straddle the all-folder splits (stated, not measured).

## 4. Details still to check before citing

- C06: PMLR volume / pages. C07: peer-reviewed venue (if any) and author list on the PDF. C19, C26: DOI strings.
- C10: exact wording of the findings; C09: contamination ratios; C20: exact names of the ten pitfalls; C23: name of the
  leakage type; C11: year (2020 vs 2021) as printed by ACM; C25: authors' first names.
