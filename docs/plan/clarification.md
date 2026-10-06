# Clarification — corrected Stage 1/2 plan and Q&A

**Date:** 2026-10-05 · **Companions:** `docs/plan/stage1-ssl-wbs.md` (evidence, decisions D1–D9),
`stage1-plan.html` (interactive review and tracker)

The professor's plan, in the same structure, with each recommendation from the plan review applied.
**Status: adopted 2026-10-05** (student decision; D1–D9 in `docs/plan/stage1-ssl-wbs.md`). It is shared with the
professor (WBS task S1.0.1); any objection is recorded as a change to the WBS.

---

## Part 1 — Corrected plan

### 1. Input Representation

Each sample is the sequence of messages heard from one sender pseudonym, with copies of the same message (same
`messageID`) heard by different receivers removed. Its length is **T = 64** messages. Beacons arrive at **1 Hz** in
VeReMi-Extension, so T = 64 is about 64 seconds of driving and keeps 79% of vehicles; T = 128 is run as a
sensitivity check. Every sample is cut to exactly T messages, with no padding, and an equal number of windows is
drawn per vehicle, so sequence length cannot reveal the class.

Every timestep carries a 13-dimensional feature vector built from BSM fields:

- **Position (4):** sender-claimed position (pos_x, pos_y), and the receiver's own GPS position at rcvTime (from
  its type-2 record).
- **Velocity (4):** sender-claimed velocity vector and the receiver's own velocity vector.
- **Acceleration (2):** sender-claimed acl_x, acl_y.
- **Relative geometry (2):** range and bearing from the receiver to the claimed position.
- **Timing (1):** log of the inter-message gap Δτ = rcvTime(t) − rcvTime(t−1).

Comparing what a vehicle claims with where the receiver actually is gives the key signal for position
falsification. The claimed position is a median 81 m from the receiver for benign senders and 743 m for DoSRandom
attackers.

Δτ catches timing anomalies. Benign beacons arrive about every 1 s with occasional missed messages, while
GridSybil sends bursts (15% of gaps under 0.5 s).

Time of day is not used: it alone separates the 07–09 h and 14–16 h scenario groups and would invalidate the
transfer test. The true sender ID, the `*_noise` fields and ground-truth positions are never inputs.

Train, validation and test splits are grouped by (time window, physical vehicle) across all four attack scenarios,
so no vehicle appears in more than one split.

### 2. Architecture

- **Encoder:** TimesNet with four stacked TimesBlocks (inner width d_ff = 64, 3 Inception kernel sizes, top-k = 3
  periods) and no time-of-day embedding.
- **Periods:** chosen per sample from FFT frequencies f ≥ 2.
- **Outputs:** per-timestep states H (T × d), and an embedding z (dimension d ∈ {128, 256, 512}) from global
  average pooling of H over time.
- **How it works:** TimesNet uses FFT to find dominant periods and reshapes the sequence into 2-D, so convolutions
  capture both short- and long-range temporal patterns.
- **Comparison:** a Transformer encoder with the same inputs, outputs and pretraining is trained as an ablation, to
  measure whether TimesNet's period modelling helps on mostly aperiodic vehicle motion.

### 3. Stage 1: Self-Supervised Pretraining (no attack labels)

Three objectives are trained jointly, using training-split vehicles only. The reconstruction decoder reads H; all
other heads attach to z. All extra heads are discarded after pretraining.

**Masked reconstruction.**

- Each sample uses one of two masking strategies with equal probability: random masking (exactly 25% of timesteps)
  or block masking (one contiguous block of round(r·T) steps, r ~ U(0.20, 0.30), i.e. 13–19 steps at T = 64).
- Masked steps are replaced by a learned mask token.
- A two-layer MLP decoder reconstructs the masked steps from H, trained with mean squared error over masked steps.
  The reconstruction targets are sender position (relative to the window), velocity, acceleration and log-Δτ.
- The model must beat a linear-interpolation baseline on block masks.

**Physical constraint heads (P1–P3).** Three binary classifiers (MLPs, BCE loss). With probability 0.5, one
physics violation is injected into the window, and each head predicts whether its violation was injected:

- **P1, kinematic violation:** a speed spike or position jump at a random step.
- **P2, speed consistency violation:** the claimed speed is scaled without moving the positions to match.
- **P3, heading feasibility violation:** an impossible sharp turn of heading or path at one step.

Because the labels come from our own random injection, they never encode the attack label and need no map. The
original rules computed on real data are reported only as diagnostics: on real windows, H1 fired only on
attackers, H2 found no violations above 70 km/h, and H3 flagged 33% of benign windows.

**Contrastive learning (SimCLR style, InfoNCE loss).** Two augmented views of each sample form a positive pair,
using physically consistent augmentations:

- a common shift of 3–5 m applied to both the sender and the receiver positions;
- a time stretch s ∈ [0.95, 1.05] (velocity × s, acceleration × s², Δτ ÷ s);
- a random 80–100% temporal crop taken by subsampling steps, without interpolation.

Augmentations must not imitate attack artifacts. Per-step GPS jitter and speed scaling on their own are not used.

Negatives are the other samples in the batch (one window per pseudonym per batch), plus hard negatives from the
same 50 m grid cell and same scenario group, from a different pseudonym at least 10 minutes apart, down-weighted by
β = 0.5. Settings: temperature τ = 0.1, batch size 256, and a projection head d → d → 128 that is discarded
afterward.

**Joint loss.**

- L = λ1·L_rec + λ2·L_InfoNCE + λ3·L_P1 + λ4·L_P2 + λ5·L_P3, with each loss normalised to a comparable scale.
- λ1 = 1, λ3 = λ4 = λ5 = 0.3, λ2 ∈ {0.1, 0.3, 1}, plus one run with automatic uncertainty weighting.
- Checkpoints are chosen without labels (held-out reconstruction error and embedding quality on a pretraining
  validation slice).

### 4. Stage 2: Supervised Contrastive Fine-Tuning (few-shot)

We initialize from the Stage 1 checkpoint and fine-tune with a 10× lower learning rate.

- **Training classes (local data):** A16 GridSybil, A18 DoSRandomSybil, A19 DoSDisruptiveSybil, plus benign.
- **Held out for zero-shot testing:** A17 DataReplaySybil.
- **Label budget:** n ∈ {10, 20, 30, 50} examples per class, with at least 5 resampled support sets per n to measure
  sensitivity.
- **If the full 19-attack VeReMi-Extension is downloaded:** train on A1 constant position, A2 constant position
  offset, A3 random position, A4 random position offset and A9 eventual stop. Exclude A5–A8 (speed attacks, no
  trajectory shape anomaly) and hold out A17 DataReplaySybil.

### 5. Memory Bank and Anomaly Scoring

After fine-tuning, we freeze the encoder.

- **Building the bank:** pass the training-split trajectories through it, without using their labels, and store the
  L2-normalised embeddings as the memory bank.
- **Scoring:** for a new trajectory, retrieve its K nearest embeddings with FAISS (cosine similarity),
  K ∈ {5, 10, 20}. The anomaly score is the mean cosine similarity to those K neighbours.
- **Threshold:** a score below θ means anomalous. K and θ are chosen on the validation split, and the test split is
  used once.
- **Deployment:** an anomalous vehicle's update would be excluded from a federated-learning round. That is the
  motivating deployment; FL itself is not simulated.

### 6. Hyperparameters to Sweep

- T ∈ {64, 128}
- d ∈ {128, 256, 512}
- K, θ
- λ2 and the loss-weighting method
- mask ratio r
- augmentation strengths: shift size and time-stretch range
- hard-negative weight β
- temperature τ
- few-shot n

---

## Part 2 — What changed and why (one line each)

1. **1 Hz, not 100 ms:** the measured median gap is 1.000 s.
2. **Receiver "observed" position = receiver's own GPS:** VeReMi has no receiver-side measurement of the sender.
3. **Time of day removed, range and bearing added:** time of day perfectly separates the two scenario groups
   (shortcut). The vector stays 13-dimensional.
4. **T = 64 with fixed-length, length-matched samples:** attackers send longer histories, so length would leak the
   class.
5. **New vehicle-grouped split:** 89.7% of benign test vehicles currently have a near-copy in train.
6. **TimesNet details fixed:** d_ff = 64 keeps it at 2.3–9.2M parameters instead of 37.5–600M; per-sample periods;
   no time embedding.
7. **Block length = ratio × T:** a 20–40-step block doesn't fit the 20–30% mask ratio at T = 64.
8. **The decoder reads H, not z:** reconstructing individual timesteps needs the per-step states.
9. **Physics heads detect injected violations:** the original rules leaked the label (H1), were empty (H2) or caught
   simulator quirks (H3). The heads are renamed P1–P3 to avoid clashing with the thesis hypotheses.
10. **Augmentations made physically consistent:** jitter and speed scaling resembled attack artifacts. The timestamp
    offset is dropped because time of day is no longer a feature.
11. **Road-segment negatives become grid-cell negatives within the same group:** there is no map, and "other time of
    day" means the other scenario group.
12. **Small λ sweep and label-free checkpoint selection instead of a full grid:** a full grid is 243 configs × 3
    seeds, and choosing by accuracy uses labels.
13. **Stage 2 uses the attack codes we have:** A1–A9 are not in our data, and data replay is A17, not A16.
14. **Memory bank built from unlabeled training data:** choosing only "legitimate" trajectories needs labels. FL is
    treated as the deployment motivation only.

---

## Part 3 — Q&A

### What are range and bearing?

Both describe where the sender *claims* to be, as seen from the receiving car.

- **Range** is the straight-line distance from the receiver to the claimed position, in metres: √(dx² + dy²),
  where dx and dy are how far the claimed position is from the receiver along x and y. Example: "the sender says
  it is 81 m away from me."
- **Bearing** is the direction of that claimed position from the receiver, as an angle: atan2(dy, dx). Example:
  "…and it's ahead-left of me."

**Why they help:** honest cars claim positions close to the receiver, at a median of 81 m and rarely beyond radio
range (about 384 m at p99). DoSRandom attackers claim positions a median of 743 m away. A sudden jump in range or
bearing between messages is also suspicious. Bearing is stored as **one angle**, atan2(dy, dx) wrapped to (−π, π],
so the input stays at 13 features (decided 2026-10-06). Its jump at ±π (directly behind-left vs behind-right) is a
known wrap-around that the model sees as a large change.

### Does the data contain latitude/longitude?

No. Raw records in all three file types (own GPS, received messages and ground truth) store every position as
`pos: [x, y, 0.0]` in metres, for example `[1346.63, 1060.61, 0.0]`. There are no latitude or longitude fields.
These are flat simulator coordinates (SUMO metres from the map's local origin), not GPS degrees.

### Can the coordinates be pinned on OpenStreetMap? (the dataset is Luxembourg street data)

Probably, but not directly.

- **The coordinates are not lat/lon.** VeReMi-Extension was simulated with SUMO on the LuST scenario (Luxembourg).
  Its `pos` values are flat SUMO metres, roughly −3.6 km to +2.8 km, measured from the simulator's own origin.
- **Converting needs the LuST network file.** `lust.net.xml` from the public LuSTScenario GitHub repository has a
  `<location>` tag with `netOffset` and `projParameter`. To convert:
  1. Subtract `netOffset` from (x, y) to get UTM coordinates (believed to be zone 32N — to be checked in the file).
  2. Convert UTM to lat/lon, e.g. with `pyproj`.
  3. Plot the points on OpenStreetMap.
- **It may not line up.** Negative coordinates suggest VeReMi may use a different offset from the stock
  `lust.net.xml`. The test is to plot a few trajectories and check that they follow real streets, within a few
  metres.
- **If it lines up, it unlocks:** the speed-limit head (OSM `maxspeed`), real road-segment negatives for
  contrastive learning, and map views of attacks.
- **Needed first:** an OK to download `lust.net.xml` (external data). A read-only alignment test in a notebook
  follows.

### After pretraining, how do we test the model — simulation, or SUMO-generated T = 64 data?

Test on the held-out VeReMi test split. No simulator or new SUMO runs are needed for the main results.

1. **Main test: the VeReMi test split, cut into T = 64 windows.**
   - VeReMi is already SUMO-generated, so the test split is the "SUMO data". It must use the new split grouped by
     vehicle, so no test car was seen during pretraining.
   - Freeze the pretrained encoder, then run each check:
     - **Linear / kNN probe:** fit a tiny classifier on z using training labels, then score test windows
       (macro-F1, AUROC, recall per attack type).
     - **Few-shot:** give the probe only n = 10/20/30/50 labelled examples per class.
     - **Transfer:** pretrain on 0709 and test on 1416, and the reverse.
     - **Label-free anomaly score:** kNN distance to unlabeled training embeddings; AUROC on test.
   - Compare against the same encoder trained from scratch and against the Transformer. That comparison is the
     thesis result.
2. **The simulator (`simulation/` dashboard):** for looking, not scoring. Replay test windows the model got wrong
   (map, kinematics, heatmap) to understand its errors. It produces no metrics.
3. **New SUMO runs:** optional and later, as an extra out-of-distribution test (new traffic or attacks). It is new
   data, so only if the student and professor want it.
