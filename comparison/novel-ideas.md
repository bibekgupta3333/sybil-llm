# Candidate multi-agent problems: novelty, feasibility, one recommendation

*2026-11-01 · built from the 109 verified papers in `papers.json` (ids A = Sybil / misbehaviour detection, B = road
network / traffic / cooperative perception, C = crowdsensing). **Proposal only:** every idea here is outside the
adopted thesis plan (RULE 7). Nothing is implemented; the user decides.*

## What we already have (the starting point)

- **Raw VeReMi-Extension, receiver view:** every received copy of every BSM at every receiver, with `messageID`,
  `sender`, `senderPseudo` and the receiver's own GNSS state (type-2 rows). One broadcast is heard by **3.7 to 5.4
  receivers on average** (pilot below), so the data already holds the multi-agent picture: many receivers judging the
  same message.
- **`src/baselines/plausibility/`:** F2MD checks + ThresholdApp per received copy, Table III reproduced for 6 / 8
  subsets within 0.05 F1. Its known gap: **DoSRandomSybil recall 0.80 vs 0.98** in the paper (no map check C10).
- **One-class / supervised autoencoders** (`task/autoencoder_*`), the `all/T32_benign` split by vehicle, prepared
  receiver-view data (`src/data/prepared_data/`).

## Pilot measurement that shapes the recommendation (read-only, scratchpad, not a result)

One run per family, group 1416, every received copy of up to 200,000 broadcasts. Check: F2MD-style range
plausibility only (claimed position more than 420 m from the receiver). "Cooperative" = a copy is also flagged when
**any other receiver of the same `messageID`** flagged it (verdicts shared by messageID, honest reporters).

| Run (1416) | Attack copies passing their own range check | Extra attack copies flagged by sharing | Extra benign copies flagged | Mean receivers per broadcast |
|---|---|---|---|---|
| DoSRandomSybil (54000–57600) | 20.3% | **+6.8 pp** | +0.18 pp | 3.8 |
| GridSybil (50400–54000) | 98.3% | +3.9 pp | +0.22 pp | 5.4 |
| DataReplaySybil (50400–54000) | 93.3% | +4.8 pp | +0.33 pp | 5.4 |

A second cooperative check, "silent neighbours" (most vehicles within 50 m of the claimed position did not receive
the message), was weak: it flagged 1.4–5.2% of attack broadcasts vs 0.75–0.86% of benign. One run, one check, honest
reporters only: these numbers say the idea is worth a week, not that it works.

---

## Ideas

Scores 1–5: **N** novelty, **F** feasibility in 7 days, **T** fit with the thesis, **P** publishability.

### I1. Cooperative plausibility fusion across receivers, with lying Sybil reporters (recommended)

**Problem.** Every VeReMi-Extension receiver judges a BSM alone. But a broadcast is heard by several receivers, and a
pseudonym by many over time. If receivers share their verdicts (as ETSI misbehaviour reports do) keyed by
`messageID` and pseudonym, how much detection do we gain per attack family, at what false-alarm cost, and does it
survive when the attackers, who are 30% of the vehicles and also receivers, submit false reports (accuse benign
vehicles, vouch for their own ghosts)?

**Why novel.** Closest work: A14 (global misbehaviour detection over reports, F2MD group: no Sybil-specific
evaluation, abstract gives no result), A36 (ADVENT, FL with time slicing on the VeReMi-Extension DoS family only,
benign reporters), A04 (RSU knowledge transfer, trust only between RSUs), A01 / A02 / A25 (FL shares model weights,
not verdicts), B09 MISO-V and B03 (multi-source consistency for collective perception, assume independent sources),
B07 / B08 (per-agent tests / consensus that Sybil identities can outvote), C01 / C31 / C36 (Sybil-resistant
aggregation of worker reports, non-vehicular). None measures multi-receiver verdict fusion on the receiver-view
VeReMi-Extension Sybil subsets, none lets the Sybil attackers poison the fusion, and none asks how much of the
paper-vs-local gap (e.g. the map check) cooperation recovers. **Honest caveat:** report fusion itself is an old idea
in trust / reputation work (A18, A22, C22); the novelty is the measured, leakage-safe evaluation on this public
dataset with an adaptive Sybil reporter, not the mechanism.

**What we would build.** A per-copy export from the plausibility baseline (check scores per received copy), a
`messageID → copies` and `pseudonym → copies` index, fusion rules (OR, k-of-n distinct reporters, mean score,
reporter-trust weighting learned from agreement), pseudonym-level propagation with time-to-flag, and a lying-reporter
model (attacker receivers flip their verdicts).

**Data.** Raw `data/VeReMi-Dataset/` (read only) through the existing baseline; no new datasets, no training.

**Metric and baseline to beat.** Paper-style per-copy accuracy / precision / recall / F1 per subset (Table II
definitions) plus per-pseudonym recall and time to first flag; baseline = our `catch` local detector (and the paper's
Table III numbers). Fusion parameters chosen on 0709, reported on 1416 (and the reverse).

**Risks.** Gains may be small where local checks already catch most (GridSybil precision 0.997, recall limited by
plausible ghosts); DataReplay copies are plausible everywhere, so sharing may add little; the "any receiver flagged"
rule inflates FPR with many receivers (k-of-n must control it); labels are the sender's file code (unmodified
attacker messages count as attacks, as in Table III); GridSybil_0709 is a 2025 re-simulation (F14, class balance
differs).

**Effort:** 6–7 days. **Scores:** N 3 · F 5 · T 3 · P 3.

### I2. Linking Sybil pseudonyms to one radio from co-reception sets

**Problem.** All ghosts of one attacker are sent by one radio, so their messages are heard by the **same set of
receivers at almost the same instants**, whatever position they claim. Can pseudonyms be grouped into physical
senders from co-reception fingerprints (Jaccard of hearer sets over time, send-time alignment), and does grouping
turn many weak per-ghost signals into one strong per-attacker signal?

**Why novel.** A19 (Fréchet + DBSCAN on reported trajectories, not on VeReMi), A09 (trajectory cliques but needs RSU
location proofs), A33 / A34 / A27 (graph traceability, datasets unclear), B35 / C18 (co-location proximity graphs,
need device attestations), C01 (account grouping by data similarity), C03 (needs an RSSI protocol). Passive
co-reception linking from existing logs is not in any of them.

**What we would build.** Per pseudonym, its hearer-set time series; pairwise similarity; clustering; evaluation
against the true `sender` (ARI, pairwise precision / recall), then detection by cluster size and internal
consistency.

**Data.** Raw receiver logs (read only). **Metric / baseline.** Linking ARI vs trajectory-only Fréchet clustering
(A19-style); detection F1 vs `catch`. **Risks.** Benign platoons share hearer sets too (false links); the true
sender is a privileged grouping (F13) and must be ground truth only, never an input; DoS pseudonyms live 1–2
messages, so their fingerprints are tiny. **Effort:** 6–8 days. **Scores:** N 4 · F 3 · T 4 · P 4.

### I3. Sybil-aware consensus: deduplicate identities before voting

**Problem.** Consensus defences for cooperative perception (B01 ROBOSAC, B08 CP-Guard, B09 MISO-V) and trust voting
(A13, A28) assume attackers are a minority of *identities*. A Sybil attacker breaks that. Test how many ghosts flip
a vote on VeReMi and whether voting over estimated physical senders (I2's clusters) restores it.

**Why novel.** B01 / B08 state the minority assumption; B03 notes Sybil voting as unaddressed. **What / data.**
Needs I2 first, then a simulated voting task (e.g. "is this pseudonym genuine?") over receivers. **Risks.** Depends
on I2's linking quality; the voting task on VeReMi is artificial (no perception data). **Effort:** 10+ days (I2 +
voting). **Scores:** N 4 · F 2 · T 3 · P 4.

### I4. Federated one-class detection across receivers (RSUs / vehicles as clients)

**Problem.** Train the `task/autoencoder_benign_t32` one-class model federated across receiver clients (non-IID by
area and time) instead of centrally, and measure the cost of federation and of 30% poisoning clients.

**Why novel.** Thin. A01 (VAE + GMM, Fed+ on VeReMi), A02 / A03 (DFL on VeReMi-Extension, poisoning), A25, A35,
A26 already cover FL misbehaviour detection and its poisoning. Only the leakage-safe vehicle split and the
trajectory-window input are new. **Data.** `all/T32_benign` (on disk). **Metric / baseline.** AUROC per attack vs
the centralised `seed0` (GridSybil 0.736). **Risks.** Our centralised model only beats the length shortcut on
GridSybil, so federation would study a weak detector. **Effort:** 5 days. **Scores:** N 2 · F 4 · T 3 · P 2.

### I5. GridSybil as a crowdsensing attack: Sybil-robust traffic-state estimation

**Problem.** GridSybil is literally "traffic congestion Sybil": ghosts fake a jam. Treat every pseudonym as a
crowdsensing worker reporting speed and presence per road cell, measure how much GridSybil distorts estimated
density / speed per cell, and test Sybil-resistant truth discovery (C01 grouping, C31 RCTD, C36 TDSSA) adapted to
motion data.

**Why novel.** C04 and B33 / B35 show Sybils distort traffic services but use other data and offer no detector on
V2X logs; C01 / C31 / C36 are non-vehicular. Bridging truth discovery to VeReMi-Extension is new. **Data.** Raw logs
(benign vehicles' type-2 rows give the true density). **Metric / baseline.** Per-cell density / speed error vs naive
averaging. **Risks.** Low fit with the thesis; "truth" per cell is easy to define for density, harder for speed.
**Effort:** 5–6 days. **Scores:** N 3 · F 4 · T 2 · P 3.

### I6. Trust that survives pseudonym churn

**Problem.** Per-identity trust / reputation (A18, A22, A24, C22, C24) accumulates evidence per identity; Sybil
churn (DoS: new pseudonym every 1–2 messages; DataReplay: per target) resets it. Measure on VeReMi-Extension the
evidence each scheme can accumulate before an identity disappears, and propose trust keyed by I2's linked senders.

**Why novel.** The churn problem is noted qualitatively (C22, C24) but not measured on this dataset. **Risks.** Partly
an analysis of pseudonym lifetimes we already know (F11); thin as a standalone paper. **Effort:** 3–4 days.
**Scores:** N 2 · F 5 · T 3 · P 2.

### I7. A panel of LLM agents that explain flagged tracks

**Problem.** Several LLM agents (per receiver, plus an aggregator) read flagged tracks as text and debate a verdict.

**Why novel.** A06 (MistralBSM, one fine-tuned LLM, selected VeReMi attacks), B27–B32 (LLM agents for traffic, trust
inputs). **Risks.** Cost and latency, accuracy hard to trust, leakage-prone evaluation (A06 does not discuss split by
vehicle), weak fit with the thesis. **Effort:** 7+ days. **Scores:** N 3 · F 2 · T 1 · P 2.

Multi-agent versions of I1, I2 + I3 and I7 (agents, protocol, diagrams): [`multi-agent-ideas.md`](multi-agent-ideas.md).

## Scorecard

| Idea | N | F (7 d) | T | P | Sum | Effort (days) |
|---|---|---|---|---|---|---|
| **I1 Cooperative plausibility fusion + lying reporters** | 3 | 5 | 3 | 3 | **14** | 6–7 |
| I2 Pseudonym linking by co-reception | 4 | 3 | 4 | 4 | **15** | 6–8 |
| I3 Sybil-aware consensus | 4 | 2 | 3 | 4 | 13 | 10+ |
| I4 Federated one-class across receivers | 2 | 4 | 3 | 2 | 11 | 5 |
| I5 Sybil-robust traffic-state truth discovery | 3 | 4 | 2 | 3 | 12 | 5–6 |
| I6 Trust under pseudonym churn | 2 | 5 | 3 | 2 | 12 | 3–4 |
| I7 LLM agent panel | 3 | 2 | 1 | 2 | 8 | 7+ |

## Recommendation: I1, with I2 as the follow-up

I2 has the higher total, but its 7-day feasibility is uncertain (benign platoons, tiny DoS fingerprints) and nothing
in our pilot supports it yet. **I1 is the one that fits a week:** no training, no new data, it reuses the verified
baseline, the pilot already shows a measurable gain on the family where we trail the paper most (DoSRandomSybil
+6.8 pp of attack copies at +0.18 pp benign, range check alone), and it produces the per-copy / per-pseudonym index
that I2 needs next. Novelty is modest: say so in any write-up.

### 7-day plan

| Day | Work | Output |
|---|---|---|
| 1 | Add a per-copy export option to the plausibility baseline (every check score + flag per received copy, run / receiver / messageID / pseudonym / sender code); run all 8 subsets | per-copy tables under gitignored `src/runs/` (raw data read only) |
| 2 | Index copies by messageID and by pseudonym; reproduce Table III from the export (Δ 0 against `summary.json`) | sanity check that the export equals the baseline |
| 3 | Honest fusion: OR, k-of-n distinct receivers, mean score; pseudonym propagation (flag a pseudonym after k reports, keep it flagged for τ s); choose k, τ on 0709, report 1416, and the reverse | per-subset metrics, ΔF1 vs `catch` |
| 4 | Lying reporters: attacker receivers (code ≠ 0) accuse benign senders and exonerate their own; fusion robust variants (distinct-reporter k-of-n, agreement-based reporter trust) | metrics vs fraction of lying reporters (0, 10, 30%) |
| 5 | Ablations: which checks to share (range only vs all), reports per second (communication cost), time to first flag per pseudonym | ablation table |
| 6 | Uncertainty: bootstrap by run and receiver; per-family breakdown; compare with paper Table III and the map-check gap | CIs, final table |
| 7 | Write-up: method, numbers, limitations (labels = sender file code; F14; one simulator) | short report + README table |

**Success criterion (fixed before running).** With k and τ chosen on 0709 and evaluated on 1416 (and reverse),
cooperative fusion raises per-copy F1 over `catch` by **≥ 0.05 on at least 2 of the 4 families** while benign
per-copy FPR rises by **≤ 0.5 pp**, and with 30% lying reporters the robust variant stays **at or above the local
`catch` F1** on every family. If it fails, report the null result (RULE 1).

Out of scope for the thesis plan (RULE 7): this needs the user's say-so before any implementation.
