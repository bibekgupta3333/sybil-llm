# WBS — RoadFM-Lite TimesNet plan: Stage 1 (now) and Stage 2 (deferred)

**Created:** 2026-09-30 · **Updated:** 2026-10-06 · **Plan text:** `docs/plan/clarification.md` Part 1 + Part 2
(the adopted plan; source of truth for every task below).
**Scope — Stage 1 (now):** plan §1 (input representation), §2 (TimesNet encoder), §3 (Stage 1 SSL pretraining,
no attack labels) and the Stage 1 evaluation — §2 of this WBS.
**Scope — Stage 2 · deferred:** plan §4 (supervised contrastive few-shot fine-tuning) and §5 (memory bank and
anomaly scoring) — §4 of this WBS; starts after Stage 1. The §6 sweep plan covering both stages is §5.
**Companions:** `docs/plan/research-plan.md` (decision record, "Stage 1 plan" section) · `stage1-plan.html`
(interactive version of this WBS, progress saved in the browser) · `docs/plan/clarification.md` (the corrected
plan, written in the professor's format, plus Q&A).

**Status 2026-10-05:** the corrected plan in `docs/plan/clarification.md` is **adopted** (student decision);
D1–D9 below are decided. The professor still sees the corrected plan (S1.0.1); any objection is recorded as a
change to this document.

Every number below was measured on the local data this session (read-only on `data/`) unless marked
*recalled*. Nothing has been implemented yet.

---

## 0. Read this first — findings that change the plan

| # | Finding | Evidence | Consequence |
|---|---|---|---|
| F1 | **Cross-scenario train/test leak in the current split.** The 4 attack scenarios reuse the same SUMO traffic: every benign `(group, time window, senderPseudo)` appears in all 4 scenario folders, with positions within 0.21 m median (max 1.0 m) of each other. **89.7% of test benign identities (2,283 / 2,546) have a near-copy in train.** | `window_metadata.parquet` + `idx_{train,test}.npy`, 300 sampled pairs from `X_windows.npy` (GridSybil_0709 excluded) | RULE 3. The existing `idx_*` split is sender-disjoint *within* a scenario but not *across* scenarios. Every existing number (incl. `models/results/`) is optimistic until re-split. New group key: `(time window, physical vehicle)` pooled over the 4 scenarios (S1.1.1, S1.1.5). |
| F2 | **Existing TCP objective is trivial.** `models/transformer_model.ipynb` cell 7 sets `tcp_label = torch.zeros(...)` and never applies a corruption; mask positions are drawn once per batch (cell 5). | notebook source | Current results are effectively MTR-only pretraining. They stay as what that code produced, but are not an MTR+TCP result. No seed recorded. |
| F3 | **Local data has only A0 + A16–A19.** A16 = GridSybil, A17 = DataReplaySybil, A18 = DoSRandomSybil, A19 = DoSDisruptiveSybil. No A1–A15. | traceJSON filename codes, all 8 folders | Stage 2 as written (A1–A4, A9 train, "A16 data replay" held out) needs the full 19-type VeReMi-Extension — a new dataset (RULE 7). The held-out replay class is A17, not A16. |
| F4 | **Beacons are 1 Hz, not 10 Hz.** Δτ median 1.000 s per (receiver, pseudonym); `rcvTime == sendTime` in every type-3 record, so Δτ(rcvTime) ≡ Δτ(sendTime). | 4.4M type-3 records from 4 runs | Δτ is informative (GridSybil 15% of gaps < 0.5 s; benign p99 7–9 s from missed receptions) but it measures attack mechanics and reception loss, not late BSMs. |
| F5 | **Time of day identifies the scenario group.** sin(tod) ∈ [0.70, 0.97] for 0709, [−0.87, −0.50] for 1416; moves ~15° within a run. | receiver logs | Including sin/cos lets the model tell 0709 from 1416 outright and breaks 0709↔1416 transfer (H3). Drop it, or ablate it and never use it in transfer runs. |
| F6 | **"Receiver observed" position has no direct source.** The receiver only logs its own GPS (type-2, every 1.000 s). `pos_noise` is simulator noise (~3–5 m); ground-truth `pos` of an attacker is a label oracle (matches claimed only 21% of attacker messages). | receiver logs | Use the receiver's own type-2 fix at rcvTime (p50 0.49 s old) plus relative features. Claimed-to-receiver range: benign p50 81 m / p99 384 m; DoSRandom p50 743 m. |
| F7 | **Sequence length depends on the unit.** Sender-centric (current prep): median 110 msgs. Receiver-link `(receiver, pseudonym)`: median 11–14, only 3–8% of benign ≥ 64. DataReplay / DoS attacker pseudonyms rotate (median 42.5 / 100 per vehicle) → their receiver-link sequences are median **1 message**. Attackers are longer in the sender-centric view (DoS 200, Grid 345 vs Benign 100). | prep metadata + receiver logs | T and the sequence unit are one joint decision (D1). Length must not separate classes (length-matched sampling, S1.1.4). |
| F8 | **Physical heads as specified are label proxies or empty.** 60k train windows: H1 (combined kinematic rule) flags **0.00% benign / 0.00% DataReplay / 84–88% DoS & Grid → P(attack \| flag) = 1.000**. H2 speed > 70/90 km/h flags 0% (max reported ~18.8 m/s); > 50 km/h flags 36% with no attack link. H3 a_lat > 8 m/s² flags 33% of benign (SUMO turns in 1–2 steps at 1 Hz). Reported \|acl\| never exceeds 4.51 m/s² in any class. | `X_windows.npy` sample, seed 0 | H1 would be supervised training on an attack-label proxy — incompatible with "no attack labels" in spirit. Resolved by D4: P1–P3 detect *injected* violations; the original rules are diagnostics only. |
| F9 | **TimesNet size explodes if d_ff = d.** 4 blocks, 6 Inception kernels: d = 128/256/512 → 37.5M / 150M / 600M params (2.4B with d_ff = 4d). With a bottleneck d_ff = 64, 3 kernels: 2.30M / 4.60M / 9.20M. | numpy param count from the THUML reference (MIT) | ~18k identities. Use a fixed d_ff = 64 bottleneck. |
| F10 | **Several augmentations match Sybil signatures.** Per-step GPS jitter of 3–5 m at 1 Hz adds 3–5 m/s implied-speed mismatch, the scale of GridSybil ghost spacing and DataReplay offsets (down to 0.3 m). Speed scaling without positions invents a DoSRandom-like mismatch. Crop + interpolated resample smears Δτ. | analysis | Contrastive invariance to these would blind the encoder to the attacks. Use physically consistent versions (S1.3.C1) and measure sensitivity (S1.3.C5). |

## 1. Decisions (adopted 2026-10-05)

All nine are decided; the adopted option is the corrected plan in `docs/plan/clarification.md`.

| ID | Decision | Options considered | Adopted |
|---|---|---|---|
| D1 | Sequence unit and T | (a) receiver-link `(receiver, pseudonym)`, T = 20 + padding; (b) network-heard pseudonym sequence (messages de-duplicated by `messageID`), T = 64 | **(b), T = 64 primary, T = 128 sensitivity**; fixed length, no padding, equal windows per vehicle. Rotating-pseudonym DataReplay/DoS broadcasts that rarely reach T are a documented blind spot. |
| D2 | Meaning of "receiver observed" pos/speed | receiver's own GPS · `pos_noise` · ground truth | **Receiver's own type-2 fix at rcvTime** (position + velocity). Never `sender`, `*_noise`, or ground truth as inputs. |
| D3 | Time features | keep sin/cos + Δτ · drop sin/cos | **sin/cos dropped** (F5); the two slots become **range + bearing** (receiver → claimed position); **log-Δτ kept**. d_in = 13. Bearing is one value, atan2(dy, dx) wrapped to (−π, π] — not sin/cos (decided 2026-10-06). |
| D4 | Physical heads | as specified · rule-on-real-data redesign · injected violations | **P1–P3 detect violations we inject** (p = 0.5 per window; label = injected or not): P1 speed spike / position jump, P2 speed scaled without matching positions, P3 impossible sharp turn. No map needed; labels cannot encode the attack label. The original H1–H3 rules on real windows are reported as diagnostics only. Renamed P1–P3 (clash with hypotheses H1–H4). |
| D5 | TCP | dropped (advisor) · kept as ablation | **Dropped.** Corruption detection is covered by the injected-violation heads (D4). The old pilot's TCP was a no-op (F2); its results stay a pilot. |
| D6 | Contrastive hard negatives | "same road segment, different time of day" (needs OSM + crosses groups) · map-free | **50 m grid cell on SUMO x/y, same group, different pseudonym, \|Δt\| ≥ 10 min**, β = 0.5, ablated. |
| D7 | Loss weights λ1–λ5 | full grid (3⁵ = 243 configs × 3 seeds) · reduced | **Losses normalised to a comparable scale; λ1 = 1, λ3 = λ4 = λ5 = 0.3, λ2 ∈ {0.1, 0.3, 1}, + one automatic (uncertainty) weighting run** ≈ 4 configs × 3 seeds; checkpoints selected **label-free** (S1.3.J2). |
| D8 | New split (F1) | re-split · keep | **Re-split** on `(time window, physical vehicle)` pooled over the 4 scenarios; the physical id is used for grouping only, never as input. **RULE 3 split change — adopted, not yet executed** (S1.1.5). |
| D9 | Deviation from `proposal/main.tex` | — | **Decided 2026-10-05 (student): accepted.** The advisor's Stage 1 plan supersedes the proposal's Transformer + MTR/TCP method; deviating from `main.tex` is not treated as an issue and needs no revision before Stage 1 work. The Transformer stays as an encoder ablation. |

## 2. WBS

Effort in working days. `→` marks a dependency. Tasks that write under `data/` need the user's go-ahead
(RULE 2).

### S1.0 Scope, decisions, environment

| ID | Task | Deliverable | Acceptance criterion | Dep | d |
|---|---|---|---|---|---|
| S1.0.1 | Share the adopted corrected plan with the professor | `docs/plan/clarification.md` | professor has seen it; any objection recorded as a change here | — | 1 |
| S1.0.2 | Decision record + hypotheses remap | `research-plan.md` "Stage 1 plan" section | each changed task ID cross-referenced (superseded / modified / kept) | — | 0.5 |
| S1.0.4 | Pin the training environment (RULE 4) | `requirements-train.txt` (`pip freeze`) | torch version, device and seed list recorded in every run config | — | 0.5 |
| S1.0.5 | Rename physics heads P1–P3 | all docs | no clash with H1–H4 | — | 0.1 |

### S1.1 Input representation (step 1)

| ID | Task | Deliverable | Acceptance criterion | Dep | d |
|---|---|---|---|---|---|
| S1.1.1 | Cross-scenario leak audit (F1) | `scripts/audit_cross_scenario_leak.py`, `docs/research-notes/data_understanding/cross-scenario-leak.md` | reports, per class, the share of test identities with a same-(window, vehicle) copy in train; reproduces the 89.7% benign figure | — | 1 |
| S1.1.2 | Streaming receiver-log parser | `scripts/build_receiver_sequences.py` → `data/prepared_receiver/` (**needs OK**) | row counts match raw files (23,032 files, 12.4 GB); asserts `rcvTime == sendTime`; < 30 min | data-write OK | 2 |
| S1.1.3 | Receiver-own join + relative features | `models/roadfm/features.py` + tests | no privileged field (`sender`, `*_noise`, ground truth) in inputs; benign range p99 ≈ 384 m reproduced | S1.1.2 | 2 |
| S1.1.4 | Sequence unit, T, length control; GridSybil_0709 pseudonym-1 handling | `configs/s1_input.json`, length histograms per class | every sample cut to exactly T messages (no padding), equal windows per vehicle, so length alone gives AUC ≈ 0.5; retained-identity share logged per T | S1.1.2, D1 | 1.5 |
| S1.1.5 | Group-aware split (**RULE 3 — split change**) | `scripts/make_receiver_splits.py`, `idx_*_rx.npy` | zero `(window, vehicle)` overlap across train/val/test and across 0709/1416; pretrain-val = 10% of train identities | S1.1.1, S1.1.4 | 1 |
| S1.1.6 | Shortcut probes | `notebooks/s1_leakage_probes.ipynb` | logistic probes on length, time-of-day (probe target only — never an input), run id: chance-level class separation | S1.1.5 | 1 |
| S1.1.7 | Manifest | `data/prepared_receiver/config.json` | feature list, T, seeds, normalisation (train-only) recorded | S1.1.3–6 | 0.5 |

### S1.2 Encoder — TimesNet (step 2)

| ID | Task | Deliverable | Acceptance criterion | Dep | d |
|---|---|---|---|---|---|
| S1.2.1 | Embedding: token conv + positional, no temporal (x_mark) path | `models/roadfm/embedding.py` | (B, T, 13) → (B, T, d); no time-of-day injection | S1.0.4 | 0.5 |
| S1.2.2 | Inception + TimesBlock, **per-sample** top-k periods, bins f ≥ 2 | `models/roadfm/timesnet.py` (vendored, MIT credit) | shapes for T ∈ {50, 64, 100, 128}; output invariant to batch composition (tol 1e-5) | S1.2.1 | 2 |
| S1.2.3 | `TimesNetEncoder` (4 blocks, shared LayerNorm, global average pooling of H over time) returning `(H, z)` | same file | param counts match F9 (±0.1%); z: (B, d); H feeds the decoder | S1.2.2 | 1 |
| S1.2.4 | Unit tests | `tests/test_timesnet.py` | shape, param count, seed determinism, CPU vs MPS agreement, GAP over all T steps (inputs are fixed-length, no padding) | S1.2.3 | 1 |
| S1.2.5 | Transformer encoder with the same `(H, z)` API (ablation) | `models/roadfm/transformer.py` | ≈ 1.19M params at d = 128; same tests pass | S1.2.4 | 1 |
| S1.2.6 | Period diagnostic on train only | notebook cell + figure | histogram of chosen periods per block | S1.2.3 | 0.5 |

Recommended config: T = 64 (T = 128 sensitivity), d_in = 13, d ∈ {128, 256, 512}, d_ff = 64, num_kernels = 3,
top_k = 3 (per-sample periods, f ≥ 2), no time-of-day embedding, dropout 0.1.

### S1.3 Stage 1 pretraining (step 3) — no attack labels

**S1.3.A Masked reconstruction**

| ID | Task | Deliverable | Acceptance criterion | Dep | d |
|---|---|---|---|---|---|
| S1.3.A1 | Random (exactly 25% of T) or block (one block, L = round(r·T), r ~ U(0.20, 0.30)) masking, chosen per sample with equal probability | `models/roadfm/masking.py`, `tests/test_masking.py` | ratio exact ±1 step for all T; seeded determinism; mask returned | S1.1.7 | 1 |
| S1.3.A2 | Learned mask token replaces masked steps (no zero-fill; d_in stays 13) | encoder patch + test | gradient reaches the token; FFT sees no zero blocks | S1.3.A1, S1.2.3 | 1 |
| S1.3.A3 | 2-layer MLP decoder on H[:, t∈M]; masked MSE on sender pos (window-relative), velocity, acceleration, log-Δτ | `models/roadfm/losses.py::MaskedRecLoss` | loss only on masked steps; per-feature MSE logged | S1.3.A2 | 1 |
| S1.3.A4 | Linear-interpolation baseline | `scripts/baseline_interp.py` | model beats interpolation on block masks (pretrain-val) | S1.3.A3 | 0.5 |

**S1.3.P Physical-constraint heads (P1–P3)** — injected violations (D4)

| ID | Task | Deliverable | Acceptance criterion | Dep | d |
|---|---|---|---|---|---|
| S1.3.P1 | Violation injectors: P1 speed spike / position jump, P2 speed scaled without matching positions, P3 impossible sharp turn; applied with p = 0.5, label = injected | `models/roadfm/injections.py` + tests | each injector changes only its target fields; labels 50% ± 2 pp; seeded determinism; an injected window breaks its rule, an untouched one is unchanged | S1.1.7 | 1.5 |
| S1.3.P2 | Diagnostic audit of the original H1–H3 rules on real windows (reporting only, never training labels) | `docs/research-notes/physics-heads-audit.md` | reproduces F8 rates ±1 pp; per-family flag rate and AUROC reported on train | S1.1.7 | 1 |
| S1.3.P3 | *Optional:* LuST map alignment test (**download needs OK**) — not needed by P1–P3 | note: source, `netOffset` check | lanes overlay VeReMi positions within ~5 m, or recorded as not aligned | download OK | 1.5 |
| S1.3.P4 | MLP heads on z, BCE on injected labels | loss module + config fields | heads train; per-head pretrain-val AUROC logged | S1.3.P1, S1.2.3 | 1 |

**S1.3.C Contrastive (SimCLR / InfoNCE)**

| ID | Task | Deliverable | Acceptance criterion | Dep | d |
|---|---|---|---|---|---|
| S1.3.C1 | Physically consistent augmentations: common-mode 3–5 m shift (sender + receiver), time stretch s ∈ [0.95, 1.05] (v·s, a·s², Δτ/s), crop 80–100% by subsampling (no interpolation) | `models/roadfm/augment.py`, `tests/test_augment.py` | tests: shift leaves Δpos and range unchanged; stretch preserves ‖Δpos − vΔτ‖; crop keeps true Δτ | S1.1.7 | 2 |
| S1.3.C3 | Identity-level sampler, pseudonym de-dup | `models/roadfm/sampler.py` + test | no pseudonym twice in a batch; physical `sender` never read | S1.1.5 | 1 |
| S1.3.C4 | NT-Xent (τ = 0.1, batch 256), projection head d→d→128, grid-cell hard negatives (D6) | `losses.py::InfoNCE`, `negatives.py` + tests | hard negatives: same 50 m cell, same group, different pseudonym, \|Δt\| ≥ 10 min; β = 0.5 configurable | S1.3.C1, C3 | 1.5 |
| S1.3.C5 | Attack-artifact sensitivity probe | `scripts/sensitivity_probe.py` | cosine shift under synthetic offsets / frozen blocks vs under augmentations; ratio ≫ 1 reported per checkpoint | S1.3.C4 | 1 |

**S1.3.J Joint training**

| ID | Task | Deliverable | Acceptance criterion | Dep | d |
|---|---|---|---|---|---|
| S1.3.J1 | L = λ1·L_rec + λ2·L_nce + λ3·L_P1 + λ4·L_P2 + λ5·L_P3; EMA normalisation + Kendall option | `losses.py::JointLoss` + test | all heads dropped at checkpoint export | A3, C4, P4 | 1 |
| S1.3.J2 | Label-free checkpoint selector | `models/roadfm/select.py` | pretrain-val masked MSE, alignment/uniformity, effective rank, head losses; asserts no label and no `idx_val` read | S1.3.J1 | 1 |
| S1.3.J3 | Weight sweep (D7), ≥ 3 seeds | `models/results/s1_sweep/*.json` (config, seed, pip freeze) | every run reproducible from its JSON | S1.3.J2, S1.0.4 | 3 |

### S1.4 Stage 1 evaluation (frozen encoder, labels for evaluation only)

| ID | Task | Deliverable | Acceptance criterion | Dep | d |
|---|---|---|---|---|---|
| S1.4.1 | Linear + kNN probes on frozen z | `models/results/s1_probe.json` | train labels fit the probe only; binary + 4-family macro-F1, AUROC; ≥ 3 seeds | S1.3.J3 | 2 |
| S1.4.2 | Per-family recall, incl. nulls | table | all 4 families + benign reported | S1.4.1 | 0.5 |
| S1.4.3 | 0709 ↔ 1416 transfer | 2×2 matrix + Δ | pretraining reads the source group's train identities only; no time-of-day feature | S1.4.1 | 1.5 |
| S1.4.4 | Label-free anomaly score (unlabeled train embeddings) | AUROC | θ chosen on val; labels used only for scoring | S1.4.1 | 1 |
| S1.4.5 | Controls: TimesNet from scratch; Transformer encoder under the same SSL | paired comparison | same config, splits, seeds | S1.4.1, S1.2.5 | 1.5 |
| S1.4.6 | Objective ablations: −P heads, −InfoNCE, −hard negatives | results JSON | same splits and seeds as S1.4.1 | S1.4.1 | 2 |

**Stage 1 total ≈ 45 working days** (44.6 d = S1.0 2.1 · S1.1 9.0 · S1.2 6.0 · S1.3 19.0 incl. 1.5 optional ·
S1.4 8.5; add ~15% for integration and re-runs) · **Stage 2 ≈ 14.5 d (deferred**, §4).

**Critical path:** S1.1.2 (data write OK) → S1.1.3 → S1.1.4 → S1.1.5 → S1.3.A1/C1/P1 →
S1.3.J1 → S1.3.J2 → S1.3.J3 → S1.4.1. S1.1.1 (leak audit) and S1.2.x (encoder) have no data-write
dependency and can start immediately.

## 3. Mapping to the existing research plan

| Status | `research-plan.md` tasks |
|---|---|
| Superseded | 2.2 (features), 3.1 (Transformer spec → TimesNet), 3.3 (TCP dropped, D5 — corruption detection moves to injected-violation heads P1–P3) |
| Modified | 2.1 (T, receiver logs), 2.3 (→ diagnostic physics rules, S1.3.P2), 2.4 (→ contrastive augmentations + violation injectors), 2.5 (random exactly 25% / block masking), 3.2 (MLP decoder), 3.4 (λ1–λ5, label-free selection), 3.6 (label-free anomaly score), 4.4 (from-scratch TimesNet), 5.5 (objective ablations), hypothesis H4 |
| Kept | 2.6 (splice fix, extended to receiver logs), 2.7 (re-verified under the new split), 3.7, 4.1–4.3, 5.4, 5.6, 5.7, Phases 6–8 |
| Deferred to Stage 2 (§4) | 3.5, 5.2, 5.3, hypothesis H2 |
| Invalidated until re-split | v1 pilot numbers in `models/results/` (F1, F2) |

## 4. Stage 2 WBS — deferred

Stage 2 · deferred — planned, starts after Stage 1 (plan §4–§5). FL ("excluded from the FL round") is the
deployment motivation only; FL is not simulated (no task).

| ID | Task | Deliverable | Acceptance criterion | Dep | d |
|---|---|---|---|---|---|
| S2.0.1 | Confirm Stage 2 class set with the professor: train A16 GridSybil, A18 DoSRandom, A19 DoSDisruptive (+ benign); hold out A17 DataReplay for zero-shot. Full 19-attack download stays optional (needs OK) | note in `research-plan.md` | class set confirmed in writing | S1.4.1 | 0.5 |
| S2.0.2 | Few-shot support-set sampler: n ∈ {10, 20, 30, 50} per class, ≥ 5 resampled sets per n, train split only | `models/roadfm/fewshot.py` + tests | seeded; sets disjoint from val/test; A17 never sampled | S2.0.1 | 1 |
| S2.1.1 | Supervised contrastive fine-tuning from the Stage 1 checkpoint, LR = Stage 1 LR / 10 | `models/roadfm/supcon.py`, `scripts/train_stage2.py` | loads the chosen S1 checkpoint (heads already dropped); config + seed + pip freeze saved | S2.0.2, S1.3.J3 | 2 |
| S2.1.2 | Fine-tuning runs: every n × 5 support sets × 3 seeds, for the pretrained encoder and the from-scratch control | `models/results/s2_finetune/*.json` | every run reproducible from its JSON | S2.1.1 | 3 |
| S2.2.1 | Memory bank: L2-normalised embeddings of training-split trajectories, built **without labels**; FAISS cosine index | `models/roadfm/memory_bank.py` + tests | asserts no label read; index size = number of training windows | S2.1.1 | 1 |
| S2.2.2 | kNN anomaly score = mean cosine similarity to the K nearest neighbours; choose K ∈ {5, 10, 20} and θ on **val** | `models/results/s2_threshold.json` | K, θ chosen on val only; test untouched | S2.2.1 | 1 |
| S2.2.3 | Evaluation, test used once: per-class recall, AUROC, FPR at θ; zero-shot result on held-out A17 | `models/results/s2_eval.json` | test read exactly once; nulls reported | S2.2.2, S2.1.2 | 1.5 |
| S2.3.1 | Hyperparameter sweep (§6), staged: Stage 1 settings chosen label-free first; K, θ, n on val | `configs/s2_sweep.json` + results | every config logged with seed | S2.2.3 | 3 |
| S2.3.2 | Label-efficiency curves (score vs n) vs from-scratch and Transformer; report nulls | figure + table in `models/results/` | ≥ 3 seeds; error bars | S2.3.1 | 1.5 |

**Stage 2 total = 14.5 working days** (deferred).

## 5. Hyperparameter sweep plan (§6 of the plan)

| Hyperparameter | Values | Chosen when | How (no test data) |
|---|---|---|---|
| T | 64 (primary), 128 (sensitivity) | Stage 1 | report both; 64 is default |
| d | 128, 256, 512 | Stage 1 | label-free checkpoint score, then probe on val |
| Mask ratio r | block 20–30% of T; random 25% | Stage 1 | label-free (held-out reconstruction error) |
| λ2 + weighting | λ2 ∈ {0.1, 0.3, 1} + one automatic-weighting run (λ1 = 1, λ3–λ5 = 0.3) | Stage 1 | label-free checkpoint selection |
| Augmentation strengths | shift 3–5 m; time stretch ±5%; crop 80–100% | Stage 1 | label-free + sensitivity probe (S1.3.C5) |
| Hard-negative weight β | 0.5 default (ablate 0) | Stage 1 | label-free |
| Temperature τ | 0.1 default | Stage 1 | label-free |
| K | 5, 10, 20 | Stage 2 | val |
| θ | threshold on the anomaly score | Stage 2 | val |
| Few-shot n | 10, 20, 30, 50 per class | Stage 2 | all reported (label-efficiency curve) |

## 6. Out of scope unless the advisor signs off

OSM map matching (beyond the optional S1.3.P3 alignment test) · federated learning simulation · receiver-side
plausibility checks beyond the relative features in S1.1.3 · the full 19-type VeReMi-Extension download (optional
per S2.0.1; needs OK).
