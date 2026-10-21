# Which VeReMi-Extension Sybil attacks can be recognised from vehicle trajectories?

*RoadFM-Lite research note · 2026-10-18 · sources: the VeReMi Extension paper (`varemi.txt`) and measurements on the
local VeReMi-Extension data (group 1416)*

> **Question.** "Which attacks in the dataset are useful for recognizing a Sybil attack based on vehicle trajectory?"

> **Answer.** **GridSybil** is the one attack whose Sybil behaviour is written into the trajectories themselves.
> Each ghost identity lives for minutes, sends at the normal rate and moves in formation with the attacker's other
> ghosts, but its claimed motion is physically inconsistent. **DataReplaySybil** is a secondary case: each message
> replays real, plausible motion, so the evidence is a *copy* of another vehicle's recent trajectory under fresh
> identities. That is only usable when the unit spans several identities. **DoSRandomSybil** and
> **DoSDisruptiveSybil** are not trajectory attacks: their identities last one or two messages, so there is no track
> to learn from, and their signal is message rate, identity churn and (for Disruptive) replay.
>
> The paper defines 19 misbehaviours in all, but **only these four run in Sybil mode** (§1.1). The other 15 keep one
> identity, so none of them is a Sybil attack. Two groups of them are still useful: the **position / speed faults and
> EventualStop**, as real examples of falsified trajectories (hard negatives, and real counterparts of the violations
> we inject), and the **non-Sybil twins** (DataReplay, DoSRandom, Disruptive, DoSDisruptive), which isolate what the
> Sybil mode adds. They need the full 19-attack release; our local copy holds only the four Sybil variants.

Context: per the professor, the thesis does not build a detection tool. The question is which attacks give a
trajectory-representation model something to learn and to be evaluated on.

---

## 1. What the paper defines

The VeReMi Extension paper (Kamel et al., 2020) generates its data with F2MD (VEINS on OMNeT++ and SUMO) on the
LuST scenario ("1.61km² … peak density of 67.4 Veh/km²", L199–200). It uses two time windows, "rush hour time
(7h-9h)" and "low traffic time (14h16h)" (L386–387), and a "penetration rate … set at 30%" for attackers (L393–394).
Each message carries `pos`, `spd`, `acl`, `hed` (each with a noise field), `sender`, `senderPseudo`, `messageID`,
`rcvTime` and `sendTime` (L370–383). Ground truth is in a separate file (L387–388).

Of the attacks defined in §IV-C, four run in Sybil mode, and those four are the ones in our local copy:

| Attack (our code) | Paper's definition (verbatim) | What it manipulates |
|---|---|---|
| **GridSybil** (A16), "Traffic congestion Sybil" | "generates a grid of fake vehicles in a chosen position by maintaining a new identity and a correct message frequency for each fake vehicle" (L343–347) | fabricated positions laid out as a formation; **one lasting pseudonym per ghost**; normal rate |
| **DataReplaySybil** (A17) | "sending information previously received from a specific target neighbor … Sybil mode with the attacker changing its identity on every new chosen target" (L332–335) | real but **replayed** motion of one neighbour; new pseudonym for each target |
| **DoSRandomSybil** (A18) | "all the messages fields set to random values … sybil mode with the attacker changing its identity on every send message" (L326–331) | every field random; **new pseudonym on every message**; higher rate |
| **DoSDisruptiveSybil** (A19) | "information replay of previously received data from random neighbors … executed in Sybil and DoS modes" (L336–339) | replays of **random** neighbours; pseudonym churn; higher rate |

### 1.1 All 19 behaviours in the paper, not only the Sybil ones

The paper's Table III covers **19 misbehaviour types** (plus the mixed `MixAll` set). Only the **four Sybil variants**
above use many identities; the other 15 keep **one identity** per vehicle. They are not Sybil attacks, but several
matter for a trajectory-based Sybil study. Our local copy contains only the four Sybil variants; the other 15 need the
full VeReMi-Extension release (optional download, waiting on approval, `agent.md` §5).

| Behaviour (paper name) | Kind | Identities | What changes in the trajectory | Paper F1 (0709 / 1416) | Use for a trajectory-based Sybil study |
|---|---|---|---|---|---|
| ConstPos | malfunction | one | position frozen at one point | 0.993 / 0.992 | **Real implausible motion** (no movement while driving): non-Sybil hard negative; real example for the physics heads |
| RandomPos | malfunction | one | position random each step | 0.9983 / 0.9985 | Real impossible jumps; same kinematic error as DoSRandomSybil but one identity |
| ConstPosOffset | malfunction | one | true path shifted by a fixed offset | 0.940 / 0.931 | Plausible shape, wrong place: shows a track can look physical yet be false |
| RandomPosOffset | malfunction | one | true path + random jitter | 0.977 / 0.979 | Noisy track: control for GPS-like noise vs Sybil |
| ConstSpeed / RandomSpeed | malfunction | one | speed field constant / random | 0.884 / 0.899 · 0.963 / 0.968 | Position-vs-speed inconsistency, the GridSybil kinematic error without the Sybil part |
| ConstSpeedOffset / RandomSpeedOffset | malfunction | one | speed field offset | 0.581 / 0.568 · 0.783 / 0.787 | Subtle kinematic error; hard for rules |
| DelayedMessages | malfunction | one | correct data, sent late (Δt) | 0.910 / 0.891 | Timing only; tests Δτ features, no trajectory change |
| EventualStop | attack | one | position frozen, speed set to 0 (fake stop) | 0.680 / 0.692 | **Coherent but false motion**: a trajectory-only lie, hard for rules |
| DoS | attack | one | normal data, too many messages | 0.987 / 0.989 | Rate only; no trajectory change |
| DoSRandom | attack | one | random fields + high rate | 0.9994 / 0.9995 | Non-Sybil twin of DoSRandomSybil |
| DataReplay | attack | one | replays one neighbour's motion | 0.892 / 0.880 | **Non-Sybil twin of DataReplaySybil** (F1 0.89 vs 0.51: the Sybil mode is what makes it hard) |
| Disruptive | attack | one | replays random neighbours | 0.971 / 0.971 | Non-Sybil twin of DoSDisruptiveSybil |
| DoSDisruptive | attack | one | replays random neighbours + high rate | 0.988 / 0.988 | Non-Sybil twin of DoSDisruptiveSybil |

How these help even though they are not Sybil attacks:

1. **Matched pairs isolate the Sybil effect.** DataReplay vs DataReplaySybil, DoSRandom vs DoSRandomSybil, and
   Disruptive / DoSDisruptive vs DoSDisruptiveSybil differ only in the identity behaviour. The paper's own numbers
   show it matters: DataReplay drops from F1 0.89 to 0.51 in Sybil mode, DoSDisruptive from 0.99 to 0.66–0.69.
   Comparing a model on each pair answers *what the trajectory model gains or loses from many identities*.
2. **Real kinematic anomalies.** ConstPos, RandomPos, the offsets, the speed faults and EventualStop are *real*
   physically inconsistent tracks under one identity. They can validate, or replace, the violations RoadFM-Lite
   currently injects artificially (physics heads P1–P3), and they are the natural **hard negatives**: a Sybil-aware
   representation should tell "one vehicle lying about its motion" apart from "one vehicle posing as many".
3. **What they cannot do:** none of them shows the Sybil signature itself (many identities, coordinated). For that,
   GridSybil remains the only clean trajectory case.

The paper's own baseline is a set of plausibility checks on position, speed, heading and acceleration
(absolute plausibility, temporal consistency, relative consistency, Kalman filter, overlap, beacon frequency,
sudden appearance, range; L401–412). Its Table III gives:

| Attack | Window | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| DoSRandomSybil | 0709 / 1416 | 0.990 / 0.989 | 0.999 / 0.999 | 0.978 / 0.977 | **0.989 / 0.988** |
| Traffic congestion Sybil (GridSybil) | 0709 / 1416 | 0.820 / 0.800 | 0.997 / 0.997 | 0.590 / 0.584 | **0.742 / 0.737** |
| DoSDisruptiveSybil | 0709 / 1416 | 0.770 / 0.779 | 0.982 / 0.988 | 0.501 / 0.532 | **0.664 / 0.691** |
| DataReplaySybil | 0709 / 1416 | 0.801 / 0.795 | 0.920 / 0.892 | 0.353 / 0.371 | **0.510 / 0.524** |

The paper attributes weak results "mainly … [to] a lower Recall" (L490). It notes that replay attacks lower
precision by "tricking the detection system into flagging the target genuine vehicles" (L501–504).

## 2. What the data shows (group 1416)

**Method.**
- **Data:** all of group 1416 in the prepared receiver-view data: 4 families, 1,688 receivers, 1,202 benign senders,
  519 attacker vehicles, messages deduplicated by `messageID`. Group 0709 is left out because of the GridSybil_0709
  re-simulation issue (F14).
- **True position and speed:** the raw ground-truth files were not available on this machine, so they are taken from
  each sender's own GPS log, interpolated to the message time and used only when the gap is ≤ 2 s. Benign error is
  therefore ≈ 0 by construction.
- **Grouping:** tracks are grouped by (true sender, pseudonym). This uses information a receiver does not have,
  especially for GridSybil, where all ghosts broadcast pseudonym 1 (F13).
- **Subsamples:** seeded (seed 0) for the formation and replay checks.
- **Script and output:** `traj_signals.py` / `traj_signals.json` (analysis scratch, not in the repo).

| Measure | Benign | GridSybil | DataReplay | DoSRandom | DoSDisruptive |
|---|---|---|---|---|---|
| **Does a track exist?** | | | | | |
| Messages per pseudonym, p50 / p90 | 91 / 158 | **68 / 149** | 1 / 4 | 2 / 4 | 2 / 4 |
| Pseudonyms with only 1–3 messages | 0.9% | **1.9%** | 89.0% | 89.8% | 89.8% |
| Pseudonyms per physical sender, p50 | 1 | 6 | 49 | 100 | 100 |
| Attack messages in receiver links of ≥ 4 messages | 98.8% | **98.3%** | 25.5% | 3.3% | 3.3% |
| **Is the motion wrong?** | | | | | |
| Claimed vs true position, p50 / p95 (m) | 0 / 7 | 25 / 193 | 24 / 503 | 705 / 1,310 | 373 / 1,165 |
| \|implied − claimed speed\|, p50 (m/s) | 0.09 | **10.1** | 0.21 | 15.8 | 5.9 |
| Tracks with an implied speed > 50 m/s | 0% | **49.7%** | 3.1% | 0%* | 0%* |
| **Do the identities move together?** | | | | | |
| Same-second pairs: distance / speed difference, p50 | 37 m / 4.7 m/s (nearest benign) | **13 m / 0.19 m/s** (same attacker) | – | 763 m / 20.5 m/s | 309 m / 10.7 m/s |
| Messages per active second per sender | 1.0 | 5.4 | 1.0 | 2.0 | 2.0 |
| **Is it a replay?** | | | | | |
| Matches another sender's earlier message (≤ 0.5 m, ≤ 0.2 m/s) | 19.3% | 9.6% | **81.0%** | 0% | **98.3%** |
| Lag to the matched message, p50 (s) | 435 | 547 | **3.1** | – | **3.7** |

\* DoS tracks have ~50 s gaps between their few messages, so no consecutive pair qualifies as a jump.

## 3. Attack by attack

**GridSybil: useful (strong, carried by motion).**
- **Tracks are long:** about as long as benign ones (median 68 messages; 98% of messages fall in links of ≥ 4).
- **Coordinated formation:** an attacker's ghosts sit 13 m apart and differ by only 0.19 m/s.
- **Inconsistent kinematics:** claimed and implied speed disagree by ~10 m/s, and half of the tracks contain jumps
  faster than 50 m/s. This matches the paper's "grid of fake vehicles … correct message frequency".
- **Hard for the paper's rules:** the plausibility checks reach only 0.58–0.59 recall, which leaves room for a learned
  trajectory representation.
- **Caveat:** all ghosts broadcast pseudonym 1 (F13), so part of the "impossible jump" signal comes from several ghosts
  being merged into one stream. Any analysis must state how tracks are keyed.

**DataReplaySybil: partially useful (identity and timing, not physics).**
- **Inside a track the motion is real** (implied vs claimed speed 0.21 m/s), so a single-track model sees normal
  driving.
- **The Sybil evidence is relational:** 81% of its messages copy a neighbour's message from ~3 s earlier, and each
  attacker uses ~49 pseudonyms.
- **Usable only with a wider unit:** one that compares tracks across identities, e.g. everything one receiver hears in
  a time window. Only 26% of its messages reach a 4+-message link in the current per-pseudonym view.
- **Hardest for the paper's checks** (F1 0.51–0.52), which is why it is the natural zero-shot hold-out in the plan.

**DoSRandomSybil: not useful as a trajectory attack.**
- **Nothing like real motion:** positions are random (705 m from the truth), so single messages are trivially
  implausible (paper F1 0.99).
- **No track to learn from:** every message carries a new pseudonym and 97% of messages sit in 1–3-message links.
- **Signal is rate and identity churn:** 100 pseudonyms, 2 messages per second.

**DoSDisruptiveSybil: not useful as a trajectory attack.**
- **It is a replay:** 98% of its messages copy another vehicle's message from ~4 s earlier.
- **Same churn and rate as DoSRandom,** and only 3% of messages reach a 4+-message link.
- **Its signal is replay and rate, not motion.**

## 4. Summary

| Attack | Trajectory signal | Carried by | Paper F1 (rules) | Role in a trajectory-based thesis |
|---|---|---|---|---|
| GridSybil | **Strong** | motion: inconsistent kinematics + formation of long-lived ghosts | 0.74 | **Primary attack** for learning and evaluation |
| DataReplaySybil | Partial | copied recent trajectories + identity churn | 0.51 | Secondary; needs a multi-identity unit (receiver time window) |
| DoSRandomSybil | Weak | random single messages, rate, churn | 0.99 | Out of trajectory scope (control only) |
| DoSDisruptiveSybil | Weak | replay + rate + churn | 0.68 | Out of trajectory scope (control only) |

## 5. Implications for RoadFM-Lite

1. **Benign + GridSybil (the T = 64 input) is the right core dataset** for a trajectory model. It is the only pair where
   the Sybil behaviour lives in the tracks.
2. **DataReplaySybil should be added only through a unit that spans identities** (the receiver time window, D15
   proposal), never as per-pseudonym 1–3-message windows.
3. **The two DoS variants should not be used to argue for trajectory learning.** Window length alone separates them
   (AUROC 0.95–0.97 in our tests). Including them inflates or distorts results; keep them as controls, if at all.
4. **This explains the `model-all` results:** only GridSybil beat the length shortcut (0.61 vs 0.53). The other three
   failed exactly where this analysis predicts.

## 6. Limitations

- One time window (1416) measured; 0709 excluded (F14).
- "True" position from the sender's own GPS log instead of the ground-truth file (not available locally).
- Tracks keyed by the true sender for analysis; a receiver only sees pseudonyms, and GridSybil reuses pseudonym 1 (F13).
- Only the four Sybil attacks of our local copy were analysed. The paper's non-Sybil attacks and malfunctions were
  not, and the full 19-attack release is not downloaded.

## References

- J. Kamel, M. Wolf, R. W. van der Heijden, A. Kaiser, P. Urien, F. Kargl, "VeReMi Extension: A Dataset for
  Comparable Evaluation of Misbehavior Detection in VANETs," IEEE ICC 2020 (text: `varemi.txt`; line numbers above refer to it).
- Project notes: `docs/research-notes/data_understanding/attack-taxonomy.md`, `agent.md` §4 (F11, F13, F14),
  `model-all-report.html`.
