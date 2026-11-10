# VeReMi-Extension plausibility-check baseline — reproduction spec (tracker 4.5)

Goal: re-implement, offline on the local raw `data/VeReMi-Dataset/`, the "simple detection algorithm" of
Kamel et al., *VeReMi Extension* (ICC 2020), Section V / Table III (`varemi.txt` lines 398–484), and compare our
per-message Accuracy / Precision / Recall / F1 with the paper's for the 8 Sybil subsets. This note is the
specification for `src/baselines/plausibility/`. Written 2026-10-29 from the F2MD source code (read-only);
nothing was run.

Every value below is either quoted from the source (URL + file + line) or marked **UNKNOWN** with a proposed
default. A reproduction that misses the paper is a valid result (RULE 1).

## 1. Sources and which code version

| Item | Where |
|---|---|
| Paper | `varemi.txt` (repo root), §V lines 398–484, Table III lines 430–470; refs [19] F2MD, [20] CaTCh (WCNC 2019), [21] MA Sybil (UEMCON 2019) |
| F2MD top repo | https://github.com/josephkamel/F2MD — only servers / scripts; the detector lives in the submodule `veins-f2md` |
| Detector code, **HEAD** | https://github.com/josephkamel/veins-f2md/tree/340e3cb41cc7717160a381a52b384a4318d7083e (2022-09-29; the commit pinned by F2MD master), path `src/veins/modules/application/f2md/` — written below as `f2md/` |
| Detector code, **closest to the paper** | https://github.com/josephkamel/veins-f2md/tree/3b306aac398b82bc1db19493c9631ce5a3b29adf (2020-01-10, "added F2MD configuration to all omnetpp.ini"; the commit pinned by F2MD `e079154594`, 2020-01-10). The paper appeared at ICC June 2020; no earlier public commit of the detector exists, so this is the best available proxy. |
| Scenario config | `networks/LuSTScenario/omnetpp.ini` (Jan 2020) / `f2md-networks/LuSTScenario/omnetpp.ini` (HEAD) — the `*.node[*].appl.*` lines override the defaults in `f2md/F2MDParameters.h` |

Differences between the Jan-2020 and HEAD detector that matter here (all else identical up to formatting):

| Item | Jan 2020 (`3b306aa`) | HEAD (`340e3cb`) |
|---|---|---|
| `MAX_SA_RANGE` (ini + header) | **210** | 420 |
| Proximity plausibility check (`ProximityPlausibilityCheck`, `MAX_PROXIMITY_*`) | **absent** | added 2020-06-17 (commit `2812165bfa`, "add proximity plausibility check"), and added to `ThresholdApp` |
| `MAX_BEACON_FREQUENCY` | 0.99 (ini and header) | 0.99 in the ini, 0.9 in the header default; the ini wins → **0.99** |

**Decision proposed:** reproduce the Jan-2020 version (no proximity check). `MAX_SA_RANGE` does not affect the
decision anyway (sudden appearance is not part of the threshold rule, §4).

**Caveat on the data (not the code):** the local raw files are not the files behind Table III. Run folders are
dated `2022-9-13` (e.g. `DataReplaySybil_0709/VeReMi_25200_28800_2022-9-13_21:7:46/`) and `GridSybil_0709` is a
2025-11-15 re-simulation (agent.md F14). They were generated with the same simulator family but are not
byte-identical to the 2020 dataset, so exact agreement is not expected even with perfect code.

## 2. Which F2MD "app" and check version produced Table III

Paper text (§V): checks "(1) Absolute plausibility, (2) Temporal consistency, (3) Relative consistency of a
field with respect to another, (4) Consistency with respect to a Kalman Filter, (5) Overlap of two vehicles,
(6) Beacon frequency compliance, (7) Sudden appearance plausibility, (8) Transmission range plausibility …
calculated then passed through a threshold mechanism. If the threshold is reached for a certain check, the
vehicle message is considered misbehaving."

What the code says:

- F2MD runs two detectors in parallel, V1 and V2. The shipped LuST config (both versions, ini lines 156–161):
  `checksVersionV1 = 2` (ExperiChecks), `checksVersionV2 = 2`, `appTypeV1 = 0` (**ThresholdApp**),
  `appTypeV2 = 2` (BehavioralApp). Enum comment: `0:LegacyChecks, 1:CatchChecks, 2:ExperiChecks`;
  `0:ThresholdApp, 1:AggrigationApp, 2:BehavioralApp, …`.
- `ThresholdApp` is the only app that matches "if the threshold is reached for a certain check, the message is
  misbehaving". Its threshold is **0.28125**: `f2md/F2MDVeinsApp.h` line 173
  `ThresholdApp ThreV1 = ThresholdApp(1, 0.28125);` (same in Jan 2020). (`AggrigationApp(1, 0.28125, 0.5, 10.0, 3)`,
  line 176, aggregates over a sender's history — not the paper's description.)
- `ExperiChecks` = `CaTChChecks` plus negative "augmentation" values below 0 for ML training
  (`#define AUG_FACTOR 20`, `mdChecks/ExperiChecks.cc` line 24). Thresholded at 0.28125 it gives the same
  decision as CaTCh wherever CaTCh returns 0 (both ≤ 0.28125). So **ExperiChecks + ThresholdApp ≡ CaTChChecks +
  ThresholdApp(0.28125)** for every check we can compute (the only differences are inside the map check, §5 C10,
  which we cannot compute).
- `LegacyChecks` (version 0) are the binary 0 / 1 versions of the same checks (the CaTCh paper's baseline). With
  binary outputs any threshold in (0, 1) gives the same decision.

**Most likely configuration of Table III (UNKNOWN, stated honestly):** V1 = CaTCh/Experi checks + ThresholdApp
(θ = 0.28125), because that is the shipped default and the paper cites CaTCh [20] for the check definitions. It is
possible the authors used LegacyChecks. **Proposal:** implement **LegacyChecks first** (exact, binary,
self-contained formulas, §5) as configuration `legacy`, and **CaTCh + θ = 0.28125** second as configuration
`catch` (port the confidence-factor functions verbatim from `MDMLib.cc`, §6). Report both; name `catch` as the
closer-to-paper configuration.

## 3. Ground truth, labels and which messages count

Raw format (verified on `DataReplaySybil_0709/VeReMi_25200_28800_2022-9-13_21:7:46/`, read-only):

| File | Rows |
|---|---|
| `traceJSON-<vehicleId>-<moduleId>-A<attackCode>-<startTime>-<slice>.json` | the receiver's log: `type 2` = its own GNSS state (`rcvTime, pos, pos_noise, spd, spd_noise, acl, acl_noise, hed, hed_noise`), `type 3` = a received BSM (`rcvTime, sendTime, sender, senderPseudo, messageID, pos, …, hed_noise`) |
| `traceGroundTruthJSON-<slice>.json` | `type 4`, one row per BSM sent: `sendTime, sender, senderPseudo, messageID, pos, …` — **no `attackerType` field** in this release (checked 189,402 rows) |

Writer: `f2md/mdSupport/VeReMiPrintable.cc` — file name line 57 (`-A<attackerType>`), `serializeGroundTruth`
lines 76–163 (called in `populateWSM` *after* the attack overwrote the BSM, `F2MDVeinsApp.cc` line 1595, so the
ground truth holds the transmitted, possibly fake, values), `serializeBeacon` lines 165–256 (`rcvTime` =
`bsm->getArrivalTime()`). `*_noise` = the BSM's confidence fields (`getSenderPosConfidence()` etc.), i.e.
exactly the inputs CaTCh uses as confidence radii.

**Label (primary, as briefed):** a received BSM is *misbehaving* iff its `sender` (true vehicle id) is an
attacker in that run folder, i.e. the sender's own trace file has `A<code>` with code ≠ 0 (A16–A19). Join key
`(folder, run, sender)` (ground truth `sender` = file `vehicleId`, verified in
`docs/research-notes/data_understanding/veremi-dataset-structure.md` §3). `messageID` joins a received row to
its ground-truth row; it is not needed for the label because labels are per sender vehicle.

**How F2MD itself labels (for the record, may explain part of a gap):** the label is `bsm->getSenderMbType()`
(`ThresholdApp.cc` lines 36–37, 72–73). An attacker sets `LocalAttacker` on every BSM (`F2MDVeinsApp.cc`
line 1498) **except** when its attack produced nothing that beacon (`attackBsm.getSenderPseudonym() == 0`),
then it sends its true state labelled `Genuine` (`F2MDVeinsApp.cc` lines 1525–1572; e.g. `DataReplaySybil` with
no neighbour yet, `mdAttacks/MDAttack.cc` line 908 `setSenderPseudonym(0)`; and every vehicle before
`START_ATTACK = 10 s`). So Table III counts an attacker's unmodified BSMs as genuine. **Secondary diagnostic label
(proposed):** `f2md_label = attacker AND the received pos differs from the sender's own type-2 pos at that time`
— report how many attacker messages it would flip; do not use it as the headline.

**Which messages count:** every `type 3` row in every `traceJSON-*` file of the subset (both run folders of the
group), i.e. every received copy at every receiver, **including receivers that are attackers**
(`LocalMisbehaviorDetection` runs in `onBSM` for every node, `F2MDVeinsApp.cc` lines 564–579; `incAll` counts
every BSM, `ThresholdApp.cc` line 36). Metrics are per received message: TP = flagged & misbehaving, FP =
flagged & genuine, etc. (paper Table II).

Note for sanity checks: paper Table I gives per-subset averages for the 0709 group — attacker vehicles 1,220 /
924,251 messages, genuine 2,846 / 2,221,825 messages (1416: 505 / 249,612 and 1,179 / 569,723). Whether these are
sent or received counts is not stated (UNKNOWN); report both our sent (ground-truth rows) and received counts.

## 4. Per-receiver state and the decision rule

Per receiver (one `traceJSON` file), process rows in file order (`rcvTime` ascending):

- **Own state** `myPosition, myPositionConfidence, mySpeed, myHeading, myHeadingConfidence` = the latest
  `type 2` row with `rcvTime ≤` the BSM's `rcvTime` (F2MD uses the current mobility state at reception).
- **History** keyed by `senderPseudo` (the receiver never sees the true `sender`): the latest previous BSM of
  that pseudonym (`NodeHistory`, `bsmList[0]`, `mdBase/NodeHistory.cc` lines 30–32, 85–93) and five Kalman
  filters per pseudonym (`mdBase/NodeMDMHistory.cc` lines 82–97). The table holds at most
  `MAX_NODES_LENGTH = 200` pseudonyms, evicting the one with the oldest latest `sendTime`
  (`mdBase/NodeTable.cc` `put` / `getOldestNode`; `F2MDParameters.h` line 28). Entries are never expired by time.
- **Order per BSM** (`onBSM`, `F2MDVeinsApp.cc` lines 564–611): run all checks against the history *before*
  the BSM; then append the BSM to its pseudonym's history; a new pseudonym gets its Kalman filters initialised
  from this BSM (`initKalman`).
- **Δt** everywhere = `|rcvTime_new − rcvTime_old|` (`MDMLib::calculateDeltaTime`, `MDMLib.cc` lines 67–70).
- Checks needing a previous BSM (C3–C9, C11–C17) run only when the pseudonym has ≥ 1 previous BSM; otherwise they
  keep their default value **1** (`mdBase/BsmCheck.cc` lines 14–33).

**Decision (ThresholdApp, `mdApplications/ThresholdApp.cc` lines 30–82, Jan-2020 version = same without the
proximity term):** flag the BSM iff **any** of the following is ≤ θ = **0.28125**:
any intersection value (C6), range plausibility (C1), position consistency (C3), position–speed consistency (C5),
position–speed max consistency (C5b), speed consistency (C4), speed plausibility (C2), position plausibility
(C10, map), beacon frequency (C8), position–heading consistency (C7), and the seven Kalman outputs
KalmanPSCP, KalmanPSCS, KalmanPSCSP, KalmanPSCSS, KalmanPCC, KalmanPACS, KalmanSCC (C11–C15).
**Sudden appearance (C9) is computed but not part of the rule** (it is also commented out of `calculateMinFactor`,
`MDApplication.cc` lines 77–78). The decision is per message; there is no per-(receiver, pseudonym) or per-vehicle
aggregation in ThresholdApp.

## 5. LegacyChecks — every check (configuration `legacy`)

Source: `f2md/mdChecks/LegacyChecks.cc` (HEAD line numbers; Jan-2020 bodies identical except C0). All outputs are
0 (implausible) or 1. Notation: `d(a,b)` = 2-D Euclidean distance (z ignored, `MDMLib.cc` 16–19);
`|v|` = norm of the 3-vector (`calculateSpeedPtr`, 21–23); `cur` = this BSM, `old` = previous BSM of the same
pseudonym; `Δt` as above.

Receiver-vehicle limits (`F2MDVeinsApp.cc` lines 214–216): `MAX_PLAUSIBLE_SPEED = maxSpeed + 0.01`,
`MAX_PLAUSIBLE_ACCEL = accel + 0.01`, `MAX_PLAUSIBLE_DECEL = decel + 0.01` of **the receiver's own SUMO vType**
(not the `MIN_MAX_*` parameters, which no check reads). The vType is not in the data → **UNKNOWN per receiver.**
LuST `vtypes.add.xml` (https://github.com/josephkamel/veins-f2md/blob/340e3cb41cc7717160a381a52b384a4318d7083e/f2md-networks/LuSTScenario/vtypes.add.xml):
passenger1 (p = .40) accel 2.6 / decel 4.5 / maxSpeed 70 / length 5.0; passenger2a (.20) 3.0 / 4.5 / 50 / 4.5;
passenger2b (.20) 2.8 / 4.5 / 50 / 4.5; passenger3 (.10) 2.7 / 4.5 / 70 / 6.0; passenger4 (.05) 2.4 / 4.5 / 30 / 5.5;
passenger5 (.05) 2.3 / 4.5 / 30 / 7.0; bus 2.6 / 4.5 / 30 / 12.0. **Proposed default:** the most frequent type,
passenger1 → `MAX_PLAUSIBLE_SPEED = 70.01 m/s`, `ACCEL = 2.61`, `DECEL = 4.51` m/s²; sensitivity run with the
most permissive accel 3.01 (fewer false positives).

| # | Check (paper class) | Formula (output 0 = implausible) | Parameters | Source lines |
|---|---|---|---|---|
| C0 | Proximity plausibility | not in the paper's code (Jan 2020); uses the simulator's true positions of all vehicles (`realDynamicMap`), unavailable offline → **skip** | `MAX_PROXIMITY_RANGE_L 30, _W 3, _DISTANCE 2` (HEAD only) | 53–82 |
| C1 | Range plausibility (8) | `0 if d(myPos, senderPos) ≥ MAX_PLAUSIBLE_RANGE else 1` | `MAX_PLAUSIBLE_RANGE = 420` m | 84–96 |
| C2 | Speed plausibility (1) | `0 if |spd| ≥ MAX_PLAUSIBLE_SPEED else 1` | receiver vType | 204–213 |
| C3 | Position consistency (2) | `0 if d(curPos, oldPos) ≥ MAX_PLAUSIBLE_SPEED · Δt else 1` | — | 98–109 |
| C4 | Speed consistency (2) | `δ = |spd_cur| − |spd_old|`; if δ > 0: `0 if δ ≥ MAX_PLAUSIBLE_ACCEL·Δt`; else `0 if |δ| ≥ MAX_PLAUSIBLE_DECEL·Δt`; else 1 | — | 111–140 |
| C5 | Position–speed consistency (3) | only if `Δt < MAX_TIME_DELTA`: `(minD, maxD) = calculateMaxMinDist(|spd_cur|, |spd_old|, Δt, ACCEL, DECEL, SPEED)` (`MDMLib.cc` 594–644); `v_min = min(|spd_cur|, |spd_old|)`; `addon = max(0, MAX_MGT_RNG_DOWN + 0.3571·v_min − 0.01694·v_min²)`; `0 if (dist − minD + addon < 0) or (maxD − dist + MAX_MGT_RNG_UP < 0) else 1`; `Δt ≥ MAX_TIME_DELTA` → 1 | `MAX_TIME_DELTA = 3.1` s, `MAX_MGT_RNG_DOWN = 6.2`, `MAX_MGT_RNG_UP = 2.1` m | 172–202 |
| C5b | Position–speed max consistency (3) | only if `Δt < MAX_TIME_DELTA`: `v_th = dist/Δt`; `0 if max(|spd|) − v_th > (DECEL + MAX_MGT_RNG)·Δt`; `0 if v_th − min(|spd|) > (ACCEL + MAX_MGT_RNG)·Δt`; else 1 | `MAX_MGT_RNG = 4` | 142–170 |
| C6 | Overlap / intersection (5) | for the receiver itself (Δt fixed 0.11) and every other pseudonym in the table whose latest BSM has `Δt < MAX_DELTA_INTER`: `inter = RectRectFactor(...) · (MAX_DELTA_INTER − Δt)/MAX_DELTA_INTER`, `RectRectFactor = A∩ / (A1 + A2 − A∩)` of two rotated rectangles (`MDMLib.cc` 691–699, `rectIntLib/RectIntLib.cc`; rectangle `w` = width, `h` = length, rotated by the heading angle of `calculateHeadingAngle`, `MDMLib.cc` 25–39); value `0 if inter > 0.5 else 1`; every value < 1 is kept, the rule flags if any is ≤ θ | `MAX_DELTA_INTER = 2.0` s | 215–318 |
| C7 | Position–heading consistency (3) | only if `Δt < POS_HEADING_TIME`: 1 if `dist < 1` m or `|spd_cur| < 1` m/s; else `α = |angle(hed_cur) − angle(curPos − oldPos)|`, wrap `α > 180 → 360 − α`; `0 if α > MAX_HEADING_CHANGE else 1` (angles from `calculateHeadingAngle`, degrees) | `POS_HEADING_TIME = 1.1` s, `MAX_HEADING_CHANGE = 90`° | 363–397 |
| C8 | Beacon frequency (6) | `0 if rcvTime_cur − rcvTime_old < MAX_BEACON_FREQUENCY else 1` (same pseudonym) | `MAX_BEACON_FREQUENCY = 0.99` s (ini) | 351–361 |
| C9 | Sudden appearance (7) — **not in the decision** | first BSM of a pseudonym, or `Δt > MAX_SA_TIME`: `0 if d(senderPos, myPos) < MAX_SA_RANGE else 1` | `MAX_SA_RANGE = 210` (Jan 2020; 420 HEAD), `MAX_SA_TIME = 2.1` s | 320–332, 772–776, 863–866 |
| C10 | Position plausibility (map) (1) | `1 if |spd| ≤ MAX_NON_ROUTE_SPEED`; else `0 if distance(senderPos, nearest road lane) > MAX_DISTANCE_FROM_ROUTE else 1` (`LinkControl::calculateDistance(pos, 50, 50)` on `lust.net.xml`). `MAX_NON_ROUTE_SPEED = −1` → always evaluated. **Needs the LuST road map, not in the data → UNKNOWN; default: skip (value 1)** and report it as missing. The map is public (`f2md-networks/LuSTScenario/lust.net.xml`, 10.9 MB) but downloading it is a user decision (agent.md §5). | `MAX_DISTANCE_FROM_ROUTE = 2` m | 334–349 |
| C11 | Kalman position–speed (4) → KalmanPSCP, KalmanPSCS | Kalman_SVI (§5.1) with `U = (acl.x, acl.x, acl.y, acl.y)`, `R = diag(cx, cvx, cy, cvy)` where `cx, cy = max(pos_noise.x/.y, KALMAN_MIN_POS_RANGE)`, `cvx, cvy = max(spd_noise.x/.y, KALMAN_MIN_SPEED_RANGE)`; `Δ = |X_corrected − Z|`; `PSCP = 1 − √(Δx² + Δy²)/(KALMAN_POS_RANGE·cx·Δt)`, `PSCS = 1 − √(Δvx² + Δvy²)/(KALMAN_SPEED_RANGE·cvx·Δt)`; NaN → 0; then `0 if < 0.5 else 1`. If `Δt ≥ MAX_KALMAN_TIME`: both 1 and the filter is re-initialised to the current state | `MAX_KALMAN_TIME = 3.1`, `KALMAN_POS_RANGE = 1.0`, `KALMAN_SPEED_RANGE = 4.0`, `KALMAN_MIN_POS_RANGE = 4.0`, `KALMAN_MIN_SPEED_RANGE = 1.0` | 399–480 |
| C12 | Kalman scalar → KalmanPSCSP, KalmanPSCSS | Kalman_SC on `(Σ distance, |spd|)`: call `getDeltaPos(Δt, d(curPos, oldPos), |spd|, |acl|, |acl|, cx, cvx)` (`pos += distance`, `Z = (pos, |spd|)`, `U = (|acl|, |acl|)`); `PSCSP = 1 − Δ0/(KALMAN_POS_RANGE·cx·Δt)`, `PSCSS = 1 − Δ1/(KALMAN_SPEED_RANGE·cvx·Δt)`; binarise at 0.5; re-init `(0, |spd|)` when `Δt ≥ MAX_KALMAN_TIME` | as C11 | 482–549 |
| C13 | Kalman position → KalmanPCC | Kalman_SI on position, `predict()` without input, `Z = curPos`, `R = diag(cx, cy)`; `1 − √(Δx²+Δy²)/(4·KALMAN_POS_RANGE·cx·Δt)`; binarise at 0.5 | as C11 | 551–596 |
| C14 | Kalman position + speed input → KalmanPACS | **the same Kalman_SI instance as C13** (both read `getKalmanSiv1()`, lines 789–795, 839–853), updated a second time with `U = old BSM's spd` (`X_pred = X + Δt·U`); same formula as C13 | as C11 | 598–643 |
| C15 | Kalman speed → KalmanSCC | Kalman_SI on speed (separate `kalmanVI`), `Z = spd`, `U = acl`, `R = diag(cvx, cvy)`; `1 − √(Δvx²+Δvy²)/(KALMAN_SPEED_RANGE·cvx·Δt)`; binarise at 0.5 | as C11 | 644–685 |

Order inside `CheckBSM` (lines 687–871): C0, C1, C2, C6, C10 always; then, if history exists, C3, C4, C8, C7, C5,
C5b, C9 (if `Δt > MAX_SA_TIME`), C11, C12, C13, C14, C15; without history only C9.

### 5.1 Kalman filters (`f2md/mdSupport/kalmanLib/`)

Generic step (`KalmanFilterJ_*.cc`): `predict`: `X = A·X0 (+ B∘U)`, `P = A·P0·Aᵀ + Q`; `correct(Z)` with `H = I`:
`K = P(P + R)⁻¹`, `X = X + K(Z − X)`, `P = (I − K)P`, then `X0 = X`, `P0 = P`. **`B∘U` is element-wise**
(`MatrixOp_*::multiply1D`: `ret[i] = B[i][0]·U[i]`), not a matrix product. Arithmetic is C++ `float`.

| Filter | State | A | B | Q | P0 at init | R | Source |
|---|---|---|---|---|---|---|---|
| Kalman_SVI | (x, vx, y, vy) | [[1,Δt,0,0],[0,1,0,0],[0,0,1,Δt],[0,0,0,1]] | (Δt²/2, Δt, Δt²/2, Δt) | diag(50, 10, 50, 10) | I₄ | diag(cx, cvx, cy, cvy) per call | `Kalman_SVI.cc` 15–284 |
| Kalman_SC | (Σdist, speed) | [[1,Δt],[0,1]] | (Δt²/2, Δt) | diag(10, 10) | diag(10, 10) | diag(cx, cvx) | `Kalman_SC.cc` 15–145 |
| Kalman_SI | (x, y) | I₂ | (Δt, Δt) | diag(10, 10) | diag(10, 10) | diag(c1, c2) | `Kalman_SI.cc` 15–139 |

Initialisation on a pseudonym's first BSM (`NodeMDMHistory.cc` 82–97): SVI ← (x, vx, y, vy); SC ← (0, |spd|);
SI (position, used by C13 + C14) ← (x, y); SAI ← (x, y) (unused); VI (speed) ← (vx, vy).

## 6. CaTCh checks (configuration `catch`, θ = 0.28125)

Source: `f2md/mdChecks/CaTChChecks.cc` (HEAD lines; `ExperiChecks.cc` gives the same thresholded decision, §2).
Same structure, history, Kalman filters, Δt and decision rule as §4–§5; outputs are continuous in [0, 1] using
the BSM confidence radii (`pos_noise.x`, `|spd_noise|`, `hed_noise.x`) and the receiver's own `pos_noise.x`. Port
these helper functions **verbatim** from `f2md/mdSupport/MDMLib.cc`: `calculateCircleSegment` (81–113),
`calculateCircleCircleIntersection` (115–150), `SafeAcos` (152–158), `CircleCircleFactor` (160–208),
`SegmentSegmentFactor` (210–274), `OneSidedCircleSegmentFactor` (371–443), `calculateMaxMinDist` (594–644),
`EllipseEllipseIntersectionFactor` (701–774, uses `ellipseIntLib/`).

| # | CaTCh formula | Lines |
|---|---|---|
| C1 | `CircleCircleFactor(d, conf_sender, conf_receiver, MAX_PLAUSIBLE_RANGE)` | 78–91 |
| C2 | `1 if |spd| + c/2 < MAX_SPEED; 0 if |spd| − c/2 > MAX_SPEED; else (c/2 + (MAX_SPEED − |spd|))/c`, `c = |spd_noise|` | 131–145 |
| C3 | `CircleCircleFactor(dist, conf_cur, conf_old, MAX_PLAUSIBLE_SPEED·Δt)` | 93–103 |
| C4 | `SegmentSegmentFactor(|δ|, c_cur, c_old, ACCEL·Δt or DECEL·Δt)` | 105–129 |
| C5 | mean of three `calculateMaxMinDist` variants (speeds ± confidence) through `CircleCircleFactor` / `OneSidedCircleSegmentFactor`; `min(factorMin, factorMax)` | 212–276 |
| C5b | `min` of two `OneSidedCircleSegmentFactor` with radii `conf_pos/Δt + conf_spd` | 147–210 |
| C6 | `clip(1.01 − EllipseEllipseIntersectionFactor(...)·(MAX_DELTA_INTER − Δt)/MAX_DELTA_INTER, 0, 1)` | 278–365 |
| C7 | heading interval `α ± hed_noise.x` turned into circle-segment fractions of the two position-confidence circles, mean of four | 457–530 |
| C8 | identical to Legacy (binary) | 446–455 |
| C9 | `1 − (circle ∩ (MAX_SA_RANGE + conf_receiver)) / (π conf_sender²)` — not in the decision | 367–394 |
| C10 | fraction of sample points in the confidence circle that are on a road — needs the map → skip | 396–444 |
| C11–C15 | Legacy formulas without the 0.5 binarisation; values `< 0` clipped to 0 (continuous) | 607–890 |

## 7. Inputs that the dataset does not contain

| Needed by | Missing | Proposed default |
|---|---|---|
| C2–C5b, C11–C15 | receiver's SUMO vType (speed / accel / decel limits) | passenger1: 70.01 m/s, 2.61, 4.51 m/s² (**UNKNOWN**) |
| C6 | sender width / length (`bsm->getSenderWidth/Length`), receiver width / length | length 5.0 m (passenger1), width **1.8 m** (SUMO default passenger width; vtypes.add.xml sets no width) for every vehicle (**UNKNOWN**) |
| C10 | LuST road network | skip (value 1); report "map check missing" (**UNKNOWN impact**, likely lowers recall on DoSRandomSybil, whose random positions fall off-road) |
| C0 | true positions of all vehicles | skip (post-paper check) |
| label | per-message attacker flag | sender's file code ≠ 0 (§3) |

## 8. Parameter table (all values, Jan-2020 ini = HEAD ini unless noted)

From `networks/LuSTScenario/omnetpp.ini` (Jan 2020) lines 220–244 / `f2md-networks/LuSTScenario/omnetpp.ini`
(HEAD) lines 224–248, defaults in `f2md/F2MDParameters.h` lines 136–165.

| Parameter | Value | Used by |
|---|---|---|
| θ (ThresholdApp) | 0.28125 | decision (`F2MDVeinsApp.h` 173) |
| MAX_PLAUSIBLE_RANGE | 420 m | C1 |
| MAX_TIME_DELTA | 3.1 s | C5, C5b |
| MAX_DELTA_INTER | 2.0 s | C6 |
| MAX_SA_RANGE / MAX_SA_TIME | 210 m (HEAD 420) / 2.1 s | C9 (not in decision) |
| MAX_KALMAN_TIME | 3.1 s | C11–C15 |
| KALMAN_POS_RANGE / KALMAN_SPEED_RANGE | 1.0 / 4.0 | C11–C15 |
| KALMAN_MIN_POS_RANGE / KALMAN_MIN_SPEED_RANGE | 4.0 / 1.0 | C11–C15 |
| MAX_MGT_RNG / _DOWN / _UP | 4 / 6.2 / 2.1 m | C5b / C5 / C5 |
| MAX_BEACON_FREQUENCY | 0.99 s | C8 |
| MAX_DISTANCE_FROM_ROUTE / MAX_NON_ROUTE_SPEED | 2 m / −1 | C10 |
| MAX_HEADING_CHANGE / POS_HEADING_TIME | 90° / 1.1 s | C7 |
| MIN_MAX_SPEED / ACCEL / DECEL | 40 / 3 / 4.5 | **not read by any check** |
| MAX_CONFIDENCE_RANGE, DELTA_BSM_TIME | 10, 5 | not used by the checks above |
| MAX_NODES_LENGTH / MAX_BSM_LENGTH | 200 / 20 | history table (`F2MDParameters.h` 25–28) |
| Beacon interval | 1 s (ini line 114); attackers shorten it (DoS*Sybil ÷ `DosMultipleFreqSybil = 2`, GridSybil ÷ (`SybilVehNumber` 5 + 1)) | context for C8 |
| START_ATTACK | 10 s | attackers genuine before |

## 9. Known sources of a gap to Table III (to report, not to tune away)

1. Different data files (2022 / 2025 re-simulations, §1).
2. Check version (Legacy vs CaTCh) not stated in the paper (§2).
3. Map check C10 and the vehicle-size / vType inputs are unavailable (§7).
4. Label definition: F2MD labels an attacker's unmodified BSMs as genuine (§3).
5. Code drift: the earliest public detector is 2020-01-10; the paper's code may differ.

No parameter may be tuned to approach the paper's numbers (RULE 1); sensitivity runs (accel 3.01, `f2md_label`)
are reported as such.
