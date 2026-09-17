# Class Balance — Windows and Senders, per Scenario Group (Task 1.4)

Source: `data/prepared_data/window_metadata.parquet` (285,926 rows, one per window) and
`data/prepared_data/config.json`. Computed directly from the committed prepared-data artifacts —
no estimation. `attack_label` is constant within every `sender_uid` (verified: 0 of 25,194 senders
carry more than one label), so a window-level count and a "how many attacker vehicles" count are
both well-defined and reported separately below.

## Window counts (a sender contributes many overlapping windows)

| Class | 0709 | 1416 | Total |
|---|---:|---:|---:|
| Benign | 99,873 | 43,052 | 142,925 |
| DataReplaySybil | 10,213 | 4,811 | 15,024 |
| DoSDisruptiveSybil | 22,094 | 10,318 | 32,412 |
| DoSRandomSybil | 22,094 | 10,318 | 32,412 |
| GridSybil | 41,277 | 21,876 | 63,153 |
| **Total** | **195,551** | **90,375** | **285,926** |

## Unique sender pseudonyms (distinct vehicles/identities)

| Class | 0709 | 1416 | Total |
|---|---:|---:|---:|
| Benign | 10,847 | 4,540 | 15,387 |
| DataReplaySybil | 1,152 | 492 | 1,644 |
| DoSDisruptiveSybil | 1,196 | 502 | 1,698 |
| DoSRandomSybil | 1,196 | 502 | 1,698 |
| GridSybil | 4,263 | 504 | 4,767 |
| **Total** | **18,654** | **6,540** | **25,194** |

Note the DoSDisruptiveSybil and DoSRandomSybil rows are identical in both tables — the two attack
variants produce the same sender/window counts in this dataset (same injection rate and duration,
differing only in payload construction), so any accuracy difference between them in later modeling
is a property of the *attack signal*, not of class size.

## Imbalance ratios

**Group size:** 0709 has 195,551 windows vs 1416's 90,375 — a **2.164 : 1** ratio, close to the "≈2×"
noted in the plan's known risks, now exact.

**Overall class imbalance (Benign : attack family), pooled across groups:**

| Pair | Ratio |
|---|---|
| Benign : DataReplaySybil | 9.51 : 1 (rarest attack class) |
| Benign : DoSDisruptiveSybil | 4.41 : 1 |
| Benign : DoSRandomSybil | 4.41 : 1 |
| Benign : GridSybil | 2.26 : 1 (least imbalanced) |

Attack-family size itself ranges 15,024 (DataReplaySybil) to 63,153 (GridSybil) — a **4.20 : 1**
spread among the attack classes alone, on top of the benign/attack imbalance.

**The imbalance is not stable across groups** — GridSybil is markedly more benign-skewed in 1416
(1.97 : 1) than in 0709 (2.42 : 1), and DataReplaySybil moves the other way (8.95 : 1 in 1416 vs
9.78 : 1 in 0709). A model or threshold tuned on one group's class balance will not transfer cleanly
to the other, which matters directly for the Phase 5 cross-scenario transfer protocol (0709 ↔ 1416).

## Implications for Phases 3 and 5

Five-way classification is imbalanced by up to 9.5:1 (Benign vs DataReplaySybil), so **accuracy is not
a safe headline metric** — a classifier that never predicts DataReplaySybil already clears 94.7% overall
accuracy pooled across classes. Report **macro-F1** and **per-class recall** (per Phase 6 acceptance
criteria), not plain accuracy, and treat DataReplaySybil's low recall as the hardest-class candidate to
investigate first in the Phase 6 error analysis. Given the ratios above, class-weighted loss (or
focal loss) is worth trying in Phase 3 fine-tuning specifically to lift DataReplaySybil and GridSybil-in-1416
recall — but any such weighting must be tuned on validation only, never on the group being tested for transfer.
