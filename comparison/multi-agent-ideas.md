# Three multi-agent ideas (proposals)

*2026-11-04 · Proposal only. Nothing implemented, nothing trained, no result of these systems exists. Outside the adopted thesis plan (RULE 7) until the user approves. Every number quoted is a measured one from our repo with its source; pilot numbers are marked 'not a result'. Paper ids refer to comparison/papers.json. Scores 1-5 as in novel-ideas.md: N novelty, F feasibility in 7 days, T fit with the thesis, P publishability.*

Machine-readable twin: `multi-agent-ideas.json` (same content). Source ideas: `novel-ideas.md`.

| Idea | Kind of multi-agent system | Pitch | N | F | T | P | Sum | Effort (days) |
|---|---|---|---|---|---|---|---|---|
| **MA1** Cooperative receiver agents with lying Sybil reporters (from I1) | Cooperative multi-agent detection with adversarial agents (report fusion, trust) | Many receivers hear the same broadcast; let them vote on it, and make the vote survive attackers who lie. | 3 | 5 | 3 | 3 | **14** | 6–7 |
| **MA2** Sybil-aware consensus over linked identities (from I2, I3) | Consensus / voting multi-agent system made Sybil-resistant by identity linking | One radio, one vote: link ghost pseudonyms to their radio from who heard them, then count votes per radio. | 4 | 2 | 4 | 4 | **14** | 10–12 |
| **MA3** Agentic detection pipeline: tool-using LLM orchestrator with specialist agents (from I7) | Hierarchical tool-using agent system (LLM planner + specialist tool agents + critic) | Measured tools decide; an LLM only routes evidence and writes an auditable explanation. | 3 | 2 | 2 | 2 | **9** | 8–10 |

## Comparison and what to do first

The three ideas are three kinds of multi-agent system. MA1 is cooperative detection: many honest receivers and some lying ones share verdicts, and a fusion agent weighs them by learned trust. MA2 is Sybil-resistant consensus: linker agents turn pseudonyms back into radios so that any vote counts one radio once. MA3 is a hierarchical tool-using pipeline: an LLM planner routes evidence between specialist detectors and explains, while a fixed rule decides. MA1 is the cheapest and fits a week (scores 14, 6-7 days); MA2 is the most novel and closest to the thesis (14, 10-12 days) but its linking may fail in dense benign traffic; MA3 has the weakest fit and unmeasured cost (9, 8-10 days) and depends on the other two for half its tools.

**First: MA1.** Do MA1 first: no training or new data, it reuses the verified plausibility baseline, the pilot shows a signal on the family where we trail the paper most (DoSRandomSybil, not a result), and its per-copy index is exactly MA2's input. Then MA2. MA3 only if both exist and the user wants an explanation layer.

---

## MA1 — Cooperative receiver agents with lying Sybil reporters

*Cooperative multi-agent detection with adversarial agents (report fusion, trust). From I1.*

### 1. Intro

In VeReMi-Extension every receiver judges each BSM alone, with local plausibility checks. But one broadcast is heard by several receivers, and one pseudonym by many over time. If receivers share their verdicts, a ghost that fools one receiver may still be caught by another. The catch: attacker vehicles are receivers too, and they can lie in their reports.

The raw receiver-view logs already hold the multi-agent picture: every received copy at every receiver, keyed by messageID and pseudonym (3.7 to 5.4 receivers per broadcast in our pilot). Our local `catch` detector trails the paper most on DoSRandomSybil (F1 0.887 / 0.885 vs 0.989 / 0.988), the family where the pilot shows the largest sharing gain.

**Pitch.** Fuse local verdicts across receivers with k-of-n and agreement-learned trust, and test it against Sybil reporters who lie.

### 2. Why novel

| Paper | What it does | What it misses |
|---|---|---|
| A14 | Global misbehaviour detection: OBUs / RSUs send reports, a Misbehavior Authority analyses them (F2MD group). | No Sybil-specific evaluation; the abstract gives no result; report-level, so it depends on the local detectors' outputs. |
| A36 | ADVENT: vehicles report malicious nodes to neighbours, FL across vehicles, on VeReMi-Extension. | DoS family only; benign reporters; identities reported per pseudonym. |
| A04 | RSUs share knowledge; some RSUs are compromised; transfer only from trusted RSUs. | Trust only between RSUs; no per-message verdict fusion; no Sybil linking. |
| A18 | Agents share event reports, fused with subjective logic and reputation. | No real V2X data; event reports, not BSM kinematics. |
| B09 | MISO-V: cross-check several independent sources for collective perception. | Assumes independent sources; Sybil identities can fake independence. |
| B07 | MADE: per-agent tests with FPR control in collaborative perception. | Each identity tested independently, so Sybil identities can each pass. |
| C31 | RCTD: reputation-constrained truth discovery under Sybil workers. | Labelling tasks in crowdsourcing, not vehicular motion data. |

**Gap.** No paper measures multi-receiver verdict fusion on the receiver-view VeReMi-Extension Sybil subsets, lets the Sybil attackers poison the fusion, or asks how much of the paper-vs-local gap (the missing map check) cooperation recovers.

**Honest caveat.** Report fusion and reputation are old ideas (A18, A22, C22). The novelty is the measured, leakage-safe evaluation on a public dataset with an adaptive lying reporter, not the mechanism. Say so in any write-up.

### 3. Agents

| Agent | Runs on | Inputs | Outputs | Local state | Honest / adversarial | Note |
|---|---|---|---|---|---|---|
| **Sender (benign or Sybil ghost)** (`S`) | vehicle OBU | own GNSS state | BSM (messageID, pseudonym, claimed position / speed / heading, sendTime) | pseudonym | either | An attacker radio sends ghost BSMs under many pseudonyms. |
| **Receiver agent 1** (`R1`) | vehicle OBU | received BSM + own GNSS state + per-pseudonym history | local verdict: check scores, flag (ThresholdApp, theta 0.28125) | node table of recent pseudonyms (Kalman filters, last message) | honest | Runs `src/baselines/plausibility/` (catch checks) unchanged. |
| **Receiver agent 2** (`R2`) | vehicle OBU | same as R1 | local verdict | node table | honest | — |
| **Receiver agent 3 (roadside)** (`RSU`) | RSU | same as R1 | local verdict | node table | honest | Same checks; fixed position. |
| **Lying attacker receiver** (`RA`) | attacker vehicle OBU | received BSMs + knowledge of its own ghost pseudonyms | false reports: vouch for own ghosts, accuse benign senders | list of own ghosts, target benign pseudonyms | adversarial | Attacker vehicles in VeReMi-Extension also have receive logs, so they can be modelled as reporters. |
| **Fusion agent (misbehaviour authority)** (`F`) | RSU or cloud misbehaviour authority | reports (messageID, pseudonym, reporter id, score, flag, time) | fused verdict per copy and per pseudonym; flag list broadcast back | reporter trust table; per-pseudonym report counts; flag timers tau | honest | k-of-n distinct reporters, mean score, trust-weighted vote. |

### 4. Interaction protocol

| # | From | To | Content | When |
|---|---|---|---|---|
| 1 | S | R1, R2, RSU, RA | BSM broadcast: messageID m, pseudonym p, claimed state | every beacon (1 Hz) |
| 2 | R1 | R1 | Local plausibility checks on m: scores + flag (each honest receiver does this) | on receipt |
| 3 | R1 | F | Report (m, p, reporter, min score, flag, rcvTime) | each report period |
| 4 | R2 | F | Report from receiver 2 | each report period |
| 5 | RSU | F | Report from the RSU | each report period |
| 6 | RA | F | Lying report: 'p is genuine' for its own ghost, or 'benign q is fake' | each report period |
| 7 | F | F | Group by messageID; k-of-n distinct reporters, weighted by reporter trust -> fused verdict | end of report period |
| 8 | F | F | Update reporter trust by agreement with the fused verdict (the liar loses trust) | after each fused verdict |
| 9 | F | R1, R2, RSU, RA | Flag pseudonym p for tau seconds (receivers drop or down-weight its BSMs) | after k reports |

### 5. Working methodology

| Stage | Input | Operation | Output | Reuse |
|---|---|---|---|---|
| 1. Per-copy export | raw data/VeReMi-Dataset/ (read only), 8 subsets | Add an export option to the plausibility baseline: every check score and flag per received copy, with run, receiver, messageID, pseudonym, sender code | per-copy tables under gitignored src/runs/ | src/baselines/plausibility/ (parser.py, catch_checks.py, detector.py, run.py) |
| 2. Sanity check | per-copy tables | Recompute Table III from the export; must equal summary.json exactly | Delta 0 against summary.json | src/baselines/plausibility/metrics.py |
| 3. Index | per-copy tables | Index copies by messageID (who heard it) and by pseudonym (all its copies over time) | messageID -> copies, pseudonym -> copies | this index is also MA2's input |
| 4. Honest fusion | index + local verdicts | Rules: OR, k-of-n distinct receivers, mean score; pseudonym propagation (flag after k reports, keep tau s); choose k, tau on 0709 | fused verdict per copy and per pseudonym | — |
| 5. Lying reporters | fused pipeline + attacker receivers (code != 0) | Attacker receivers flip verdicts: vouch for own ghosts, accuse benign senders; 0 / 10 / 30% of reporters lie; robust variants: distinct-reporter k-of-n, agreement-learned trust | metrics vs fraction of liars | — |
| 6. Ablations and cost | all of the above | Share range check only vs all checks; reports per second; time to first flag per pseudonym | ablation table | — |
| 7. Metric and uncertainty | fused verdicts + labels (sender file code) | Per-copy accuracy / precision / recall / F1 per subset (Table II definitions), per-pseudonym recall, benign FPR; bootstrap by run and receiver | final table with CIs, Delta F1 vs `catch` | src/baselines/plausibility/metrics.py |

### 6. Algorithm

```python
# fusion agent, one report period
for m in messages_in_period:
    R = distinct_reporters(m)                # receivers that heard m
    w = {r: trust[r] for r in R}             # trust starts at 1.0
    yes = sum(w[r] for r in R if flag[r, m])
    tot = sum(w.values())
    fused[m] = len(R) >= n_min and count_flags(m) >= k and yes / tot >= rho
    for r in R:                              # agreement-learned trust
        agree = (flag[r, m] == fused[m])
        trust[r] = clip(trust[r] + eta * (1 if agree else -1), t_min, 1.0)
    if fused[m]:
        reports[pseudo(m)] += 1
        if reports[pseudo(m)] >= k_p:
            flagged_until[pseudo(m)] = now + tau
copy_flag(c) = local_flag(c) or fused[msg(c)] or now < flagged_until[pseudo(c)]
# k, k_p, rho, tau, eta chosen on 0709 only; reported on 1416 (and the reverse)
```

### 7. Threat model

**The attacker controls:**

- Content of its own BSMs under many pseudonyms (positions, speeds, replayed traces).
- Its own reports as a receiver: what it flags or vouches for, how often it reports.
- Which benign pseudonyms it accuses; it hears roughly what its neighbours hear.

**It does not control:** Honest receivers' local checks, the fusion rule, or which receivers hear a broadcast.

**Adaptive attacks to test:**

- Vouching: all attacker receivers report 'genuine' for their own ghosts.
- Framing: accuse benign senders near the attacker to burn their reputation.
- Trust farming: report honestly for a while to earn trust, then lie.
- Report flooding: many reports per period (one radio, many reporter pseudonyms) - links to MA2.
- Plausible ghosts: positions kept inside the range check so local flags stay low.

### 8. Experiments and metrics

**Baselines to beat** (measured numbers only, with source):

- Local `catch` detector (ours), per-copy F1

| subset | ours | paper |
|---|---|---|
| GridSybil_0709 | 0.727 | 0.742 |
| GridSybil_1416 | 0.692 | 0.737 |
| DataReplaySybil_0709 | 0.51 | 0.51 |
| DataReplaySybil_1416 | 0.537 | 0.524 |
| DoSRandomSybil_0709 | 0.887 | 0.989 |
| DoSRandomSybil_1416 | 0.885 | 0.988 |
| DoSDisruptiveSybil_0709 | 0.66 | 0.664 |
| DoSDisruptiveSybil_1416 | 0.697 | 0.691 |

Source: src/baselines/plausibility/README.md (summary.json), agent.md changelog 2026-10-29. per received copy F1 of the local `catch` plausibility detector (ours) vs the paper's Table III.

- Pilot sharing signal (not a result)

| run | extra_attack_copies_flagged_pp | extra_benign_copies_flagged_pp | mean_receivers |
|---|---|---|---|
| DoSRandomSybil 1416 | 6.8 | 0.18 | 3.8 |
| GridSybil 1416 | 3.9 | 0.22 | 5.4 |
| DataReplaySybil 1416 | 4.8 | 0.33 | 5.4 |

Source: comparison/novel-ideas.md, pilot table (read-only scratchpad run, one run per family, 1416, range check only, honest reporters; not a result).


**Splits:**

- Fit every free parameter on group 0709, report on 1416; then the reverse. Never tune on the reported group.
- Anything learned per vehicle (trust, clusters, thresholds) is split by vehicle: no physical vehicle in both the fit and the report set.
- Label = the sender's trace-file attack code != 0 (as in our Table III reproduction); the true `sender` field is ground truth only, never an input (F13).
- F14 caveat: GridSybil_0709 is a 2025 re-simulation (different vehicle ids, 62-65% attack messages vs about 46% implied by the paper); report GridSybil 0709 and 1416 separately and do not pool them.
- F1 caveat: the four families reuse the same benign traffic; never fit on one family and report on another family of the same group as if it were new traffic.

**Metrics:**

- Per-copy accuracy / precision / recall / F1 per subset (paper Table II definitions)
- Benign per-copy FPR
- Per-pseudonym recall and time to first flag
- Reports per second (communication cost)

**Ablations:**

- Fusion rule: OR vs k-of-n vs mean score vs trust-weighted
- Shared checks: range only vs all
- Liars: 0 / 10 / 30% of reporters
- Pseudonym propagation on / off, tau
- Report period length

**Figures we would plot:**

- Delta F1 vs `catch` per subset (bar, 0709 -> 1416 and reverse)
- F1 vs fraction of lying reporters, one line per fusion rule
- Attack recall vs benign FPR as k varies
- Reporter trust over time, honest vs attacker reporters
- Time-to-first-flag CDF per family

### 9. Solution

Every receiver runs the local checks and sends short reports; a fusion agent at the RSU / misbehaviour authority combines them per messageID with k-of-n and learned reporter trust, and sends back a flag list that receivers apply for tau seconds.

**Expected contribution.** First measured, leakage-safe evaluation of cooperative verdict fusion on the receiver-view VeReMi-Extension Sybil subsets, including Sybil attackers who poison the fusion.

**Success (hypotheses, fixed before running):**

- Hypothesis H-MA1a: with k, tau chosen on 0709, fusion raises per-copy F1 over `catch` by >= 0.05 on at least 2 of the 4 families on 1416 (and the reverse), with benign FPR up by <= 0.5 pp.
- Hypothesis H-MA1b: with 30% lying reporters, the robust variant stays at or above local `catch` F1 on every family.

**Failure (and what we would report):**

- Gain < 0.05 F1 on all families, or FPR cost > 0.5 pp: report the null result (RULE 1).
- Robust fusion falls below `catch` under 30% liars: cooperation is unsafe without identity linking (motivates MA2).

### 10. Risks, effort, scores

- Small gains where local checks already work (GridSybil precision 0.997; recall limited by plausible ghosts).
- DataReplay copies look plausible everywhere, so sharing may add little.
- 'Any receiver flagged' inflates FPR with many receivers; k-of-n must control it.
- Labels are the sender's file code: unmodified attacker messages count as attacks (as in Table III).
- F14: GridSybil_0709 class balance differs; do not pool groups.
- Lying-reporter behaviour is simulated on real logs; the attacker model is ours, not VeReMi's.

**Effort:** 6–7 days. No training, no new data; reuses the verified plausibility baseline. Plan: novel-ideas.md, 7-day plan.

**Scores:** N 3 · F 5 · T 3 · P 3 = 14.

---

## MA2 — Sybil-aware consensus over linked identities

*Consensus / voting multi-agent system made Sybil-resistant by identity linking. From I2, I3.*

### 1. Intro

Votes, reputations and consensus checks assume attackers are a minority of identities. A Sybil attacker breaks that: one radio speaks under many pseudonyms and can outvote honest vehicles. But all ghosts of one radio are physically sent from one place, so they are heard by the same receivers at almost the same instants, whatever position they claim. If agents can link pseudonyms into estimated physical senders, they can vote per sender instead of per pseudonym.

VeReMi-Extension logs who heard each message and when, so hearer sets per pseudonym are available from existing logs with no new protocol. GridSybil ghosts fake a traffic jam by count, which is a natural per-identity vote. The true `sender` field lets us score the linking, but it must stay ground truth only (F13: GridSybil ghosts share pseudonym 1).

**Pitch.** Passive co-reception linking turns many outvoting ghosts back into one attacker vote.

### 2. Why novel

| Paper | What it does | What it misses |
|---|---|---|
| A19 | Pairwise Fréchet distance on reported trajectories + DBSCAN to find Sybils. | Uses claimed trajectories, which the attacker controls; not on VeReMi. |
| A09 | Sybil trajectories found as cliques of overlapping RSU location proofs. | Needs RSU-signed proofs and a protocol change; not usable on passive logs. |
| A33 | Sybil-TraceGuard: semi-supervised GNN links fragmented pseudonyms to the attacker. | Needs labels; dataset unclear; no consensus study. |
| A34 | Time-space-communication features as a graph for Sybil traceability under pseudonym change. | Authors and dataset not verified; no voting or consensus study. |
| B35 | Ghost Riders: Sybil devices on Waze; defence by authenticated co-location graphs. | Needs device-to-device attestations; not raw BSM logs. |
| C01 | Sybil-resistant truth discovery by grouping similar accounts. | Groups by similarity of submitted data, not physical reception; non-vehicular. |
| B01 | ROBOSAC: consensus over random teammate subsets in collaborative perception. | States that attackers are a minority; Sybils break this. |
| B08 | CP-Guard: sample consensus against malicious collaborators. | Consensus fails if Sybils make attackers the majority. |
| B03 | Benign vehicles jointly check fabricated perception data. | Assumes non-colluding identities; Sybil voting not addressed. |

**Gap.** No paper links pseudonyms passively from co-reception (who heard it, when) in existing V2X logs, and none measures how much identity linking restores a vote or consensus against Sybil ghosts on a public dataset.

**Honest caveat.** The voting task on VeReMi is partly artificial: there is no perception data, so the reputation vote uses simulated reporter behaviour on real identities and reception sets. The congestion vote uses real claimed data only. Linking quality bounds everything.

### 3. Agents

| Agent | Runs on | Inputs | Outputs | Local state | Honest / adversarial | Note |
|---|---|---|---|---|---|---|
| **Attacker radio with ghost pseudonyms g1..gk** (`A`) | attacker vehicle OBU | own position | ghost BSMs under many pseudonyms, all sent from one place | ghost list | adversarial | Claimed positions differ; the physical send point does not. |
| **Benign senders** (`B`) | vehicle OBU | own GNSS state | BSMs under one pseudonym each (rotating over time) | pseudonym | honest | Platoons can share hearer sets too (false-link risk). |
| **Hearer agents (receivers)** (`H`) | vehicle OBU / RSU | received BSMs | reception records (pseudonym, messageID, sendTime, rcvTime, receiver id) | none beyond the log | honest | Attacker receivers also report (they may omit or fake records: threat model). |
| **Linker agent** (`L`) | RSU / edge | reception records over a sliding window | pseudonym -> estimated physical sender cluster, with confidence | hearer-set time series per pseudonym; similarity graph | honest | Never reads the true `sender` field (F13). |
| **Voting agents** (`V`) | vehicle OBU (one vote per pseudonym) | a question: 'jam in cell c?' (from claimed speed / presence) or 'is pseudonym x genuine?' | vote | — | either | Ghosts vote too, all in the attacker's favour. |
| **Tally agent** (`T`) | RSU / cloud | votes + linker clusters | decision counted per pseudonym and per linked sender; flagged clusters | — | honest | Weight of a cluster = 1, split equally among its pseudonyms. |

### 4. Interaction protocol

| # | From | To | Content | When |
|---|---|---|---|---|
| 1 | A | H | Ghost BSMs g1, g2, g3 sent from one radio at nearly the same instants | every beacon |
| 2 | B | H | Benign BSMs, each heard by its own neighbours | every beacon |
| 3 | H | L | Reception records: who heard which pseudonym, and when (hearer sets light up together for g1..g3) | each window |
| 4 | L | L | Similarity of hearer-set time series (Jaccard + send-time alignment) -> clusters = estimated radios | each window |
| 5 | L | T | Pseudonym -> cluster map with confidence | each window |
| 6 | V | T | Votes on the question (ghosts all vote for the attacker's answer) | per question |
| 7 | T | T | Tally per pseudonym (ghosts win) vs per linked sender (attacker counts once and loses) | per question |
| 8 | T | H | Decision + flag clusters with many pseudonyms whose claimed positions disagree with their shared hearer set | per question |

### 5. Working methodology

| Stage | Input | Operation | Output | Reuse |
|---|---|---|---|---|
| 1. Reception records | raw data/VeReMi-Dataset/ (read only) or MA1's per-copy index | Per received copy: run, receiver, messageID, senderPseudo, sendTime, rcvTime (true `sender` kept in a separate ground-truth column) | reception table per run | src/baselines/plausibility/parser.py |
| 2. Fingerprints | reception table | Per pseudonym: hearer set per time bin (1 s), send-time phase within the beacon period | hearer-set time series per pseudonym | — |
| 3. Pairwise similarity | fingerprints of pseudonyms alive in the same window | Mean Jaccard of hearer sets over shared bins + send-time alignment; only pairs that co-exist in time | similarity graph | — |
| 4. Linking | similarity graph | Threshold + connected components or agglomerative clustering; threshold chosen on 0709 | pseudonym -> estimated sender | — |
| 5. Linking evaluation | clusters + true `sender` (ground truth only) | ARI, pairwise precision / recall, per family; compare with claimed-trajectory Fréchet + DBSCAN (A19-style) and with no linking | linking table | — |
| 6. Votes | real claimed data + clusters | V1 congestion vote: per road cell and time, count identities claiming slow presence (GridSybil fakes this); V2 reputation vote: each identity in range votes on a target's genuineness, ghosts vouch for each other (simulated behaviour) | vote outcomes per pseudonym and per cluster | MA1 local verdicts as honest votes |
| 7. Detection and metric | cluster features (size, hearer-set agreement, claimed-position spread) | Flag clusters; per-copy F1 vs `catch`; vote flip rate vs number of ghosts | final tables and figures | src/baselines/plausibility/metrics.py |

### 6. Algorithm

```python
# linker agent, one window W
for p in pseudonyms_alive(W):
    F[p] = {b: hearers(p, b) for b in bins(W)}       # who heard p in each 1 s bin
for p, q in pairs_alive_together(W):
    shared = bins(F[p]) & bins(F[q])
    if len(shared) < b_min: continue
    s = mean(jaccard(F[p][b], F[q][b]) for b in shared)
    s *= phase_align(send_times(p), send_times(q))   # same radio, same beacon slot
    if s >= s_link: G.add_edge(p, q)
cluster = connected_components(G)                    # estimated radios
# tally agent, one question
for v in votes:
    per_pseudo[v.answer] += 1
    per_sender[v.answer] += 1 / size(cluster[v.pseudo])
decision_naive  = argmax(per_pseudo)                  # ghosts can win
decision_linked = argmax(per_sender)                  # one radio = one vote
# b_min, s_link chosen on 0709; true sender used only to score clusters
```

### 7. Threat model

**The attacker controls:**

- Number of ghosts and their claimed positions / speeds.
- Send timing of each ghost (it can jitter send times to break phase alignment).
- Its own reception records if it acts as a hearer (omit or fake).
- Transmit power, if modelled (changes who hears it; VeReMi does not vary it).

**It does not control:** Which honest receivers physically hear its radio.

**Adaptive attacks to test:**

- Send-time jitter between ghosts to break alignment.
- Ghost rotation: short-lived ghosts so fingerprints are tiny (as DoS families already do, F11).
- Hide in a platoon: drive with benign vehicles so benign pseudonyms share its hearer set (forced false links, framing benign cars).
- Two colluding radios splitting ghosts (cluster count no longer = attacker count).

### 8. Experiments and metrics

**Baselines to beat** (measured numbers only, with source):

- Local `catch` detector (ours), per-copy F1

| subset | ours | paper |
|---|---|---|
| GridSybil_0709 | 0.727 | 0.742 |
| GridSybil_1416 | 0.692 | 0.737 |
| DataReplaySybil_0709 | 0.51 | 0.51 |
| DataReplaySybil_1416 | 0.537 | 0.524 |
| DoSRandomSybil_0709 | 0.887 | 0.989 |
| DoSRandomSybil_1416 | 0.885 | 0.988 |
| DoSDisruptiveSybil_0709 | 0.66 | 0.664 |
| DoSDisruptiveSybil_1416 | 0.697 | 0.691 |

Source: src/baselines/plausibility/README.md (summary.json), agent.md changelog 2026-10-29. per received copy F1 of the local `catch` plausibility detector (ours) vs the paper's Table III.

- No linking (one pseudonym = one vote) (to be built; no number yet)
- Claimed-trajectory Fréchet + DBSCAN linking (A19-style, re-implemented) (to be built; no number yet)

**Splits:**

- Fit every free parameter on group 0709, report on 1416; then the reverse. Never tune on the reported group.
- Anything learned per vehicle (trust, clusters, thresholds) is split by vehicle: no physical vehicle in both the fit and the report set.
- Label = the sender's trace-file attack code != 0 (as in our Table III reproduction); the true `sender` field is ground truth only, never an input (F13).
- F14 caveat: GridSybil_0709 is a 2025 re-simulation (different vehicle ids, 62-65% attack messages vs about 46% implied by the paper); report GridSybil 0709 and 1416 separately and do not pool them.
- F1 caveat: the four families reuse the same benign traffic; never fit on one family and report on another family of the same group as if it were new traffic.
- Linking thresholds use no labels except on the fit group; the true `sender` scores clusters only.

**Metrics:**

- Linking: ARI, pairwise precision / recall per family
- Vote flip rate vs number of ghosts, per pseudonym vs per linked sender
- Congestion-vote error per cell vs benign ground truth (benign vehicles' own GNSS rows)
- Detection: per-copy F1 and benign FPR vs `catch`

**Ablations:**

- Hearer-set only vs + send-time alignment
- Window length and bin size
- Clustering rule
- Benign platoon density (false links)
- Adaptive jitter strength

**Figures we would plot:**

- Hearer-set heat strip: ghosts of one radio light up together
- ARI per family, co-reception vs trajectory linking
- Vote outcome vs number of ghosts, two tallies side by side
- False-link rate vs traffic density

### 9. Solution

Receivers stream reception records to an RSU linker; the linker keeps clusters of pseudonyms that are heard together; every vote, reputation or consensus at the RSU counts one vote per cluster, and clusters with many disagreeing ghosts are flagged.

**Expected contribution.** A passive, protocol-free Sybil linking signal from co-reception in existing V2X logs, and the first measurement of how identity linking restores voting under Sybil attack on VeReMi-Extension.

**Success (hypotheses, fixed before running):**

- Hypothesis H-MA2a: co-reception linking gives higher ARI than claimed-trajectory linking on GridSybil (both groups, thresholds from the other group).
- Hypothesis H-MA2b: on the congestion vote, the attacker flips the per-pseudonym outcome but not the per-linked-sender outcome in most attacked cells.
- Hypothesis H-MA2c: cluster features add per-copy F1 over `catch` at <= 0.5 pp extra benign FPR.

**Failure (and what we would report):**

- Benign platoons give as many false links as true ones: report it; linking then needs trajectory consistency too.
- DoS / DataReplay fingerprints too short to link: restrict the claim to GridSybil and say so.

### 10. Risks, effort, scores

- False links in dense benign traffic and platoons (they share hearer sets).
- DoS and DataReplay pseudonyms live 1-2 messages (F11): fingerprints are tiny.
- F13: GridSybil ghosts share pseudonym 1; grouping by true sender is privileged and must stay ground truth.
- The reputation vote is simulated behaviour on real logs; the congestion vote is the only fully data-driven one.
- F14 on GridSybil_0709 (re-simulation).
- Effort beyond a week: depends on MA1's index or an equivalent export.

**Effort:** 10–12 days. Linking (I2) 6-8 days + voting study (I3) about 4 days; reuses MA1's per-copy index if MA1 is done first.

**Scores:** N 4 · F 2 · T 4 · P 4 = 14.

---

## MA3 — Agentic detection pipeline: tool-using LLM orchestrator with specialist agents

*Hierarchical tool-using agent system (LLM planner + specialist tool agents + critic). From I7.*

### 1. Intro

Our detectors each see one side of a track: physics checks, reconstruction error, identity links, report history. An analyst or misbehaviour authority needs one verdict and a reason it can audit. Letting an LLM decide from text (as in A06) is costly, hard to trust and leakage-prone. Here the LLM only plans which tools to call and explains; a fixed, fitted rule makes the decision.

On VeReMi-Extension no single detector covers all families: `catch` is strong on DoSRandom but weak on DataReplay (F1 0.510 / 0.537), and the one-class AE is below chance on DataReplay (AUROC 0.391) while only GridSybil beats the length shortcut. Routing between complementary tools is a real need.

**Pitch.** An auditable agent pipeline whose every sentence is checked against a measured tool output.

### 2. Why novel

| Paper | What it does | What it misses |
|---|---|---|
| A06 | MistralBSM: fine-tuned Mistral-7B on BSM sequences as text; edge LLM + cloud LLM verification. | The LLM decides; selected VeReMi attacks; split by vehicle not discussed; 7B latency. |
| B30 | TrafficGPT: an LLM orchestrating traffic foundation models as tools. | No security or misbehaviour task. |
| B31 | AgentsCoDriver: LLM driving agents with memory, reflection and communication. | Agents trust each other's messages; no misbehaviour handling. |
| B32 | V2V-LLM: an MLLM fusing several CAVs' perception. | Fused inputs assumed honest. |
| B27 | LLMLight: LLM agents for signal control. | Reads reported traffic state verbatim; no adversarial reasoning. |

**Gap.** No paper builds a tool-using agent pipeline for V2X misbehaviour where the LLM does not decide, checks its explanations against tool outputs, and is evaluated leakage-safe on VeReMi-Extension.

**Honest caveat.** Tool-using LLM agents are a generic pattern; the novelty is the application and the faithfulness evaluation. Thesis fit is weak (the thesis is a self-supervised trajectory encoder), cost and latency are unmeasured, and detection quality cannot exceed the tools'. Lowest priority of the three.

### 3. Agents

| Agent | Runs on | Inputs | Outputs | Local state | Honest / adversarial | Note |
|---|---|---|---|---|---|---|
| **Trigger queue** (`Q`) | RSU / cloud | local `catch` flags, AE score above val p95, MA1 fused flags | flagged track (run, receiver, pseudonym, time span) | queue | honest | Only flagged tracks reach the LLM (bounds cost). |
| **Orchestrator (LLM planner)** (`O`) | cloud | track id + tool descriptions | tool calls (parallel), then hand-off to scorer, critic, reporter | scratchpad for this track only | honest | Never sees labels, the true sender, or raw test metrics. |
| **Physics plausibility agent** (`P`) | cloud tool | track messages | per-check scores and which checks fired | — | honest | = src/baselines/plausibility/ (F2MD catch checks). |
| **Reconstruction agent** (`AE`) | cloud tool | track windows (T = 32) | masked reconstruction error + percentile vs benign val | — | honest | = task/autoencoder_benign_t32 best.pt; scores only its test vehicles. |
| **Linking agent** (`LK`) | cloud tool | pseudonym + window | linked cluster id, size, hearer-set agreement | — | honest | = MA2's co-reception linker. |
| **History agent** (`HI`) | cloud tool | pseudonym / cluster | past reports, fused flags, report trust (MA1) | evidence store | honest | — |
| **Scorer (fixed rule, not an LLM)** (`SC`) | cloud tool | evidence vector | score + verdict | weights fitted on 0709 | honest | Logistic regression or MA1 fusion over tool outputs; the decision lives here. |
| **Critic agent** (`CR`) | cloud (LLM or rules) | draft verdict + evidence | accept / ask for one more tool / reject explanation | — | honest | Checks consistency and that no claim lacks a tool output. |
| **Reporter agent** (`RP`) | cloud (LLM) | final evidence + verdict | short auditable report with evidence citations | — | honest | Every sentence maps to a tool output (checked automatically). |
| **Adversarial content in the track** (`ADV`) | attacker vehicle | — | crafted BSM fields (e.g. odd values aimed at the LLM's text input) | — | adversarial | The attacker cannot talk to the agents but controls the data they read. |

### 4. Interaction protocol

| # | From | To | Content | When |
|---|---|---|---|---|
| 1 | Q | O | Flagged track: pseudonym p, receiver, time span, trigger reason | on flag |
| 2 | O | P, AE, LK, HI | Tool calls in parallel: physics checks, reconstruction score, linked cluster, report history | immediately |
| 3 | P | O | Evidence: which checks fired, scores | tool returns |
| 4 | AE | O | Evidence: reconstruction error percentile vs benign val | tool returns |
| 5 | LK | O | Evidence: cluster size, hearer-set agreement | tool returns |
| 6 | HI | O | Evidence: past reports and fused flags | tool returns |
| 7 | O | SC | Evidence vector -> fixed scoring rule -> verdict | all evidence in |
| 8 | O | CR | Draft verdict + evidence for a consistency check | after scoring |
| 9 | CR | O | Accept, or ask for one more tool call (at most once) | after check |
| 10 | O | RP | Final evidence + verdict | accepted |
| 11 | RP | Q | Auditable report card to the analyst / misbehaviour authority | end |

> **Illustrative example, not a result.** Verdict: likely Sybil ghost (from the fixed scorer). Evidence: the range check fired on most of this pseudonym's messages [physics agent]; reconstruction error is high relative to benign validation windows [reconstruction agent]; this pseudonym is linked with several others that were heard by the same receivers at the same instants [linking agent]; two other receivers reported it earlier [history agent]. Not used: true sender, labels.

### 5. Working methodology

| Stage | Input | Operation | Output | Reuse |
|---|---|---|---|---|
| 1. Tools as functions | existing code | Wrap each detector as a deterministic tool with a JSON schema: physics (src/baselines/plausibility/), reconstruction (task/autoencoder_benign_t32 best.pt, scoring.py), linking (MA2), history (MA1) | tool registry | src/baselines/plausibility/, task/autoencoder_benign_t32/ |
| 2. Track set | flags from `catch` / AE / MA1 | Build flagged tracks on the AE's test vehicles only (all/T32_benign split) so no tool scores data it was fitted on | evaluation track list | src/data/encoder_input/all/T32_benign/ (test vehicles) |
| 3. Fixed scorer | tool outputs on 0709 tracks | Fit a small logistic regression (or MA1 fusion) over tool outputs; freeze | scorer weights | — |
| 4. Orchestration | track + tool registry | LLM plans tool calls; variants: call all tools (no LLM), LLM routing, LLM-only verdict (A06-style ablation) | evidence + verdict per track | — |
| 5. Critic and report | evidence + verdict | Critic checks every claim has a tool output; reporter writes the card | report cards | — |
| 6. Faithfulness check | report cards + tool logs | Automatic: each sentence's cited tool and value must match the log; count unsupported claims | faithfulness rate | — |
| 7. Metrics and cost | verdicts + labels (after scoring) | Per-track F1 / AUROC vs best single tool and vs call-all; tool calls, tokens and latency per track | final table | — |

### 6. Algorithm

```python
# orchestrator, one flagged track
plan = llm.plan(track.summary, tools=[physics, recon, linker, history])
evidence = parallel_call(plan.tools, track)            # deterministic tools
score, verdict = scorer(evidence)                       # fixed rule fitted on 0709
review = critic(verdict, evidence)
if review.needs_more and not track.retried:
    evidence |= call(review.tool, track); track.retried = True
    score, verdict = scorer(evidence)
report = llm.explain(verdict, evidence)
for claim in report.claims:                             # faithfulness gate
    assert claim.tool in evidence and matches(claim.value, evidence[claim.tool])
return verdict, score, report        # the LLM never changes verdict or score
# forbidden inputs for every agent: labels, true sender, test metrics
```

### 7. Threat model

**The attacker controls:**

- Every BSM field of its own messages (the text the LLM reads).
- Timing and number of ghosts (which tools fire).
- Its own reports, if MA1 history is used.

**It does not control:** The tools, the fixed scorer, or the prompts.

**Adaptive attacks to test:**

- Prompt injection through crafted field values (the LLM must only see tool outputs, not raw text it can be steered by).
- Evade the trigger: stay below every tool's threshold so no track is flagged.
- Tool disagreement: make physics pass and AE fire (or the reverse) to test the scorer and critic.
- Cost attack: trigger many flags to exhaust the LLM budget.

### 8. Experiments and metrics

**Baselines to beat** (measured numbers only, with source):

- Local `catch` detector (ours), per-copy F1

| subset | ours | paper |
|---|---|---|
| GridSybil_0709 | 0.727 | 0.742 |
| GridSybil_1416 | 0.692 | 0.737 |
| DataReplaySybil_0709 | 0.51 | 0.51 |
| DataReplaySybil_1416 | 0.537 | 0.524 |
| DoSRandomSybil_0709 | 0.887 | 0.989 |
| DoSRandomSybil_1416 | 0.885 | 0.988 |
| DoSDisruptiveSybil_0709 | 0.66 | 0.664 |
| DoSDisruptiveSybil_1416 | 0.697 | 0.691 |

Source: src/baselines/plausibility/README.md (summary.json), agent.md changelog 2026-10-29. per received copy F1 of the local `catch` plausibility detector (ours) vs the paper's Table III.

- One-class AE (ours), window AUROC

| attack | reconstruction | length_alone |
|---|---|---|
| GridSybil | 0.736 | 0.559 |
| DataReplay | 0.391 | 0.952 |
| DoSRandom | 0.973 | 0.967 |
| DoSDisruptive | 0.611 | 0.967 |

Source: agent.md changelog 2026-10-29 (task/autoencoder_benign_t32, run seed0, one seed). window AUROC of each attack test split vs benign test, all/T32_benign: reconstruction error / length alone.

- Supervised AE T = 32 (ours), window AUROC, GridSybil only

| attack | auroc |
|---|---|
| GridSybil | 0.926 |

Source: agent.md changelog 2026-11-03 (task/autoencoder_st_t32, run seed0, one seed). test window AUROC GridSybil vs benign, all/T32_benign_grid (labels used in training; says nothing about unseen attacks).

- Call-all-tools + fixed scorer, no LLM (to be built; no number yet)
- LLM-only verdict (A06-style ablation) (to be built; no number yet)

**Splits:**

- Fit every free parameter on group 0709, report on 1416; then the reverse. Never tune on the reported group.
- Anything learned per vehicle (trust, clusters, thresholds) is split by vehicle: no physical vehicle in both the fit and the report set.
- Label = the sender's trace-file attack code != 0 (as in our Table III reproduction); the true `sender` field is ground truth only, never an input (F13).
- F14 caveat: GridSybil_0709 is a 2025 re-simulation (different vehicle ids, 62-65% attack messages vs about 46% implied by the paper); report GridSybil 0709 and 1416 separately and do not pool them.
- F1 caveat: the four families reuse the same benign traffic; never fit on one family and report on another family of the same group as if it were new traffic.
- Only the AE's test vehicles (all/T32_benign split by vehicle key) are evaluated, so the AE never scores its own training data.

**Metrics:**

- Per-track F1 / AUROC per family
- Faithfulness: share of report claims matching tool logs
- Tool calls, tokens and latency per track
- Agreement with call-all (does routing lose detections?)

**Ablations:**

- Routing by LLM vs call all tools
- Critic on / off
- LLM-only verdict vs fixed scorer
- Model size of the orchestrator
- Prompt injection on / off

**Figures we would plot:**

- Detection vs cost (tool calls per track) for the variants
- Faithfulness rate per variant
- Which tools the planner calls per family
- Latency distribution

### 9. Solution

A queue of flagged tracks feeds an LLM planner that calls deterministic detector tools in parallel; a fixed scorer decides; a critic checks; a reporter writes a short card in which every claim cites a tool output.

**Expected contribution.** A faithfulness-checked agentic pipeline for V2X misbehaviour on VeReMi-Extension, with an honest comparison against plain tool fusion and an LLM-only verdict.

**Success (hypotheses, fixed before running):**

- Hypothesis H-MA3a: LLM routing keeps per-track F1 within 0.01 of call-all while making fewer tool calls.
- Hypothesis H-MA3b: >= 95% of report claims match tool logs; the LLM-only verdict is worse than the fixed scorer.

**Failure (and what we would report):**

- Routing loses detections or saves no calls: the LLM adds only explanation; say so.
- Unsupported claims > 5%: explanations are not auditable; the pipeline should not be used.

### 10. Risks, effort, scores

- Weak thesis fit: the thesis is a self-supervised trajectory encoder.
- Cost and latency of LLM calls are unmeasured and may be impractical on vehicles.
- Detection cannot exceed the tools'; the AE is weak outside GridSybil and DoSRandom.
- Leakage: tools must only score held-out vehicles; LLMs may have seen VeReMi descriptions in pretraining.
- Prompt injection through BSM fields.
- Depends on MA1 and MA2 for two of its four tools.

**Effort:** 8–10 days. Needs the tool agents to exist (plausibility baseline and AE exist; linker and history from MA1 / MA2). LLM serving cost and latency are unmeasured.

**Scores:** N 3 · F 2 · T 2 · P 2 = 9.

---

Out of scope for the thesis plan (RULE 7): all three need the user's say-so before any implementation.
