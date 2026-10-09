# WBS — RoadFM-Lite TimesNet plan: self-supervised pretraining (now) and few-shot fine-tuning (deferred)

**Created:** 2026-09-30 · **Updated:** 2026-10-08 · **Plan text:** `docs/plan/clarification.md` Part 1 + Part 2
(the adopted plan; source of truth for every task below).
**Scope — self-supervised pretraining (now, tasks S1.x):** plan §1 (input representation), §2 (TimesNet encoder), §3 (self-supervised pretraining,
no attack labels) and the pretraining evaluation — §2 of this WBS.
**Scope — few-shot fine-tuning (deferred, tasks S2.x):** plan §4 (supervised contrastive few-shot fine-tuning) and §5 (memory bank and
anomaly scoring) — §4 of this WBS; starts after pretraining. The §6 sweep plan covering both phases is §5.
**Companions:** `docs/plan/research-plan.md` (decision record, "Pretraining plan" section) · `stage1-plan.html`
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
| F3 | **Local data has only A0 + A16–A19.** A16 = GridSybil, A17 = DataReplaySybil, A18 = DoSRandomSybil, A19 = DoSDisruptiveSybil. No A1–A15. | traceJSON filename codes, all 8 folders | Fine-tuning as written (A1–A4, A9 train, "A16 data replay" held out) needs the full 19-type VeReMi-Extension — a new dataset (RULE 7). The held-out replay class is A17, not A16. |
| F4 | **Beacons are 1 Hz, not 10 Hz.** Δτ median 1.000 s per (receiver, pseudonym); `rcvTime == sendTime` in every type-3 record, so Δτ(rcvTime) ≡ Δτ(sendTime). | 4.4M type-3 records from 4 runs | Δτ is informative (GridSybil 15% of gaps < 0.5 s; benign p99 7–9 s from missed receptions) but it measures attack mechanics and reception loss, not late BSMs. |
| F5 | **Time of day identifies the scenario group.** sin(tod) ∈ [0.70, 0.97] for 0709, [−0.87, −0.50] for 1416; moves ~15° within a run. | receiver logs | Including sin/cos lets the model tell 0709 from 1416 outright and breaks 0709↔1416 transfer (H3). Drop it, or ablate it and never use it in transfer runs. |
| F6 | **"Receiver observed" position has no direct source.** The receiver only logs its own GPS (type-2, every 1.000 s). `pos_noise` is simulator noise (~3–5 m); ground-truth `pos` of an attacker is a label oracle (matches claimed only 21% of attacker messages). | receiver logs | Use the receiver's own type-2 fix at rcvTime (p50 0.49 s old) plus relative features. Claimed-to-receiver range: benign p50 81 m / p99 384 m; DoSRandom p50 743 m. |
| F7 | **Sequence length depends on the unit.** Sender-centric (current prep): median 110 msgs. Receiver-link `(receiver, pseudonym)`: median 11–14, only 3–8% of benign ≥ 64. DataReplay / DoS attacker pseudonyms rotate (median 42.5 / 100 per vehicle) → their receiver-link sequences are median **1 message**. Attackers are longer in the sender-centric view (DoS 200, Grid 345 vs Benign 100). | prep metadata + receiver logs | T and the sequence unit are one joint decision (D1). Length must not separate classes (length-matched sampling, S1.1.4). |
| F8 | **Physical heads as specified are label proxies or empty.** 60k train windows: H1 (combined kinematic rule) flags **0.00% benign / 0.00% DataReplay / 84–88% DoS & Grid → P(attack \| flag) = 1.000**. H2 speed > 70/90 km/h flags 0% (max reported ~18.8 m/s); > 50 km/h flags 36% with no attack link. H3 a_lat > 8 m/s² flags 33% of benign (SUMO turns in 1–2 steps at 1 Hz). Reported \|acl\| never exceeds 4.51 m/s² in any class. | `X_windows.npy` sample, seed 0 | H1 would be supervised training on an attack-label proxy — incompatible with "no attack labels" in spirit. Resolved by D4: P1–P3 detect *injected* violations; the original rules are diagnostics only. |
| F9 | **TimesNet size explodes if d_ff = d.** 4 blocks, 6 Inception kernels: d = 128/256/512 → 37.5M / 150M / 600M params (2.4B with d_ff = 4d). With a bottleneck d_ff = 64, 3 kernels: 2.30M / 4.60M / 9.20M. D10 (2026-10-08) briefly used d_ff = d = 128 (4.60M); **D12 (2026-10-08) returned to d_ff = 64, d = 128, 3 kernels → 2,301,312.** | numpy param count from the THUML reference (MIT) | ~18k identities. Use a fixed d_ff = 64 bottleneck. |
| F10 | **Several augmentations match Sybil signatures.** Per-step GPS jitter of 3–5 m at 1 Hz adds 3–5 m/s implied-speed mismatch, the scale of GridSybil ghost spacing and DataReplay offsets (down to 0.3 m). Speed scaling without positions invents a DoSRandom-like mismatch. Crop + interpolated resample smears Δτ. | analysis | Contrastive invariance to these would blind the encoder to the attacks. Use physically consistent versions (S1.3.C1) and measure sensitivity (S1.3.C5). |
| F11 | **Pseudonym-level sequences lose 3 of 4 attacks at T = 64** (found 2026-10-06 by the S1.1.2 dry run). DataReplay / DoS attackers rotate pseudonyms every 1–2 messages, so almost no attacker pseudonym reaches 64 messages: in the 1416/54000 runs, 1 of 9,569 A17, 0 of 18,877 A18 and 0 of 18,873 A19 sequences survive; GridSybil keeps 48%, benign 63%. | `src/raw_logs.py` + `src/sequences.py` dry run (read-only; output in scratchpad) | D1 as written leaves pretraining probes and fine-tuning with no DataReplay / DoS samples. **Decided 2026-10-06 (user): keep pseudonym-level sequences.** DataReplay / DoSRandom / DoSDisruptive are a documented blind spot of pretraining; fine-tuning's training classes need redesign (only A16 + benign have samples). |
| F12 | **Primary-listener copies are closer than all copies.** Benign claimed-to-receiver range is p50 81 / p99 384 m over every received copy, but 36 / 348 m after keeping one copy per message (the most-hearing receiver tends to be near). Duplicate copies' rcvTime differ by up to 1.4 µs; 17 senders' files sit in the neighbouring time-window folder. | same dry run | S1.1.3 acceptance compares against the all-copies figure; the de-dup ignores µs rcvTime jitter; sender labels are looked up across a group's time windows. |
| F13 | **GridSybil ghosts share pseudonym 1 in every GridSybil run** (0709: 566 / 667 senders; 1416: 292 / 203), not only in the v1 0709 ground truth. With `sentinel_policy = split_by_sender`, 840 of 4,231 GridSybil samples (20%) are built by grouping pseudonym 1 by the true sender — a privileged grouping, the same kind the user declined for DoS. | `build_report_T64.json`, `meta_T64.parquet` (`seq_id` with `#`) | Decide: keep (document), or rebuild with `sentinel_policy = drop` (realistic receiver view, −840 GridSybil samples). |

## 1. Decisions (adopted 2026-10-05)

All nine are decided; the adopted option is the corrected plan in `docs/plan/clarification.md`.

| ID | Decision | Options considered | Adopted |
|---|---|---|---|
| D1 | Sequence unit and T | (a) receiver-link `(receiver, pseudonym)`, T = 20 + padding; (b) network-heard pseudonym sequence (messages de-duplicated by `messageID`), T = 64 | **(b), T = 64 primary, T = 128 sensitivity**; fixed length, no padding, equal windows per vehicle. Rotating-pseudonym DataReplay/DoS broadcasts that rarely reach T are a documented blind spot. |
| D1′ | **Revised 2026-10-06 (user): keep the receiver's view as VeReMi recorded it.** Every received copy is kept (no de-duplication); one prepared file per raw trace file in the same folder tree; a *link* = what one receiver heard from one (sender pseudonym, sender) pair; the first message of a link keeps `log_dtau = null`. Fixed-T windows are cut later from links. | — | Supersedes the de-duplicated network-heard sequences of D1 for the prepared data. |
| D2 | Meaning of "receiver observed" pos/speed | receiver's own GPS · `pos_noise` · ground truth | **Receiver's own type-2 fix at rcvTime** (position + velocity). Never `sender`, `*_noise`, or ground truth as inputs. |
| D3 | Time features | keep sin/cos + Δτ · drop sin/cos | **sin/cos dropped** (F5); the two slots become **range + bearing** (receiver → claimed position); **log-Δτ kept**. d_in = 13. Bearing is one value, atan2(dy, dx) wrapped to (−π, π] — not sin/cos (decided 2026-10-06). |
| D4 | Physical heads | as specified · rule-on-real-data redesign · injected violations | **P1–P3 detect violations we inject** (p = 0.5 per window; label = injected or not): P1 speed spike / position jump, P2 speed scaled without matching positions, P3 impossible sharp turn. No map needed; labels cannot encode the attack label. The original H1–H3 rules on real windows are reported as diagnostics only. Renamed P1–P3 (clash with hypotheses H1–H4). |
| D5 | TCP | dropped (advisor) · kept as ablation | **Dropped.** Corruption detection is covered by the injected-violation heads (D4). The old pilot's TCP was a no-op (F2); its results stay a pilot. |
| D6 | Contrastive hard negatives | "same road segment, different time of day" (needs OSM + crosses groups) · map-free | **50 m grid cell on SUMO x/y, same group, different pseudonym, \|Δt\| ≥ 10 min**, β = 0.5, ablated. |
| D7 | Loss weights λ1–λ5 | full grid (3⁵ = 243 configs × 3 seeds) · reduced | **Losses normalised to a comparable scale; λ1 = 1, λ3 = λ4 = λ5 = 0.3, λ2 ∈ {0.1, 0.3, 1}, + one automatic (uncertainty) weighting run** ≈ 4 configs × 3 seeds; checkpoints selected **label-free** (S1.3.J2). |
| D8 | New split (F1) | re-split · keep | **Re-split** on `(scenario group, physical vehicle)` pooled over the 4 scenarios and all time windows of the group (a vehicle can cross a window boundary — refined 2026-10-06); the physical id is used for grouping only, never as input. **RULE 3 split change — executed 2026-10-06** in `src/pipeline/input_representation.ipynb` (per-split JSON files in `src/data/prepared_data/`). **Update 2026-10-08 (RULE 3):** for the benign + GridSybil encoder input the split is now 90 / 10 train / test by sender vehicle (seed 0, stratified by group × class; 338,001 / 38,426 windows); pretraining's check set = 10% of train vehicles. |
| D9 | Deviation from `proposal/main.tex` | — | **Decided 2026-10-05 (student): accepted.** The advisor's pretraining plan supersedes the proposal's Transformer + MTR/TCP method; deviating from `main.tex` is not treated as an issue and needs no revision before pretraining work. The Transformer stays as an encoder ablation. |
| D10 | TimesNet inner width d_ff | d_ff = 64 bottleneck (F9) · d_ff = d | **Width part superseded by D12 (2026-10-08).** Was: **decided 2026-10-08 (student): d_ff = d = 128, no bottleneck**, to give TimesNet full capacity in every block (the encoder of the plan must be the strongest). 4 blocks, 3 kernels: **4,595,840 params at d = 128** (was 2,301,312 with d_ff = 64); d = 256 / 512 would be 18.4M / 73.4M. Overfitting risk is higher (~5.7k sender vehicles): watch the pretrain-val loss and the collapse / length monitors; d_ff = 64 stays available as an ablation. Same day, the last bottleneck was removed too: the P1–P3 heads use hidden width d (Linear 128→128→1) instead of d/2 (heads only, discarded after pretraining) — this head width stays in force. |
| D11 | Model width d and encoder scope | d ∈ {128, 256, 512} · Transformer ablation kept | **d = 512 superseded by D12 (2026-10-08); TimesNet-only stays in force.** Was: **decided 2026-10-08 (student): d = 512** with d_ff = d (D10): encoder 73,433,600 params, 74,820,765 with the pretraining heads; **TimesNet only** in the code (the Transformer ablation S1.2.5 was removed from the code; can return later). Code: `src/model/benign_gridsybil/timesnet/`. ~16× the compute per step of d = 128; overfitting risk is higher — watch the check-set loss and the collapse / length monitors. |
| D12 | Main model width d and inner width d_ff | d = 128 / d_ff = 64 (professor's plan) · d_ff = d (D10) · d = 512 (D11) | **Decided 2026-10-08 (student): d = 128, d_ff = 64** — the professor's plan. Encoder **2,301,312** params (Inception 128→64→128 bottleneck; one TimesBlock 574,016), heads 101,136 (P1–P3 keep hidden width d, Linear 128→128→1), encoder + heads 2,402,448 (2,402,461 with the mask token). Reasons: the plan's setting; half the convolution weights of d_ff = 128 (2.30M vs 4.60M); less overfitting risk with ~5.7k sender vehicles; faster. d_ff = 128 is the first ablation if the model underfits; d = 256 / 512 are later ablations, chosen only by evidence. TimesNet only in the code (D11). Code: `src/model/benign_gridsybil/timesnet/config.py` (d_model = 128, d_ff = 64); `--check` passes (encoder 2,301,312, padding Δ = 0). |
| D13 | Scaling of heavy-tailed features and contrastive false negatives (EDA fixes) | keep z-scores · clip · robust | **Decided 2026-10-08 (student):** (a) `claimed_pos_x`, `claimed_pos_y`, `range` are re-scaled at load time by the pretraining code: (raw − median) / IQR of train real rows, then a soft tail beyond |z| = 5 (sign·(5 + log1p(|z| − 5)); order kept, invertible); other features stay train z-scores; stored files unchanged. Effect (one-shard check): range 388 m → 1.00 (was 0.65), 12.8 km ghost → 8.8 (was 36.4). (b) NT-Xent never uses windows of the same broadcasting pseudonym in the same run as negatives (copies / crops of the same broadcasts; ghost pseudonym 1 = one key per run); key uses only observable run + pseudonym, not the true sender. Code: `src/model/benign_gridsybil/timesnet/` (`config.py` robust_features / soft_clip / mask_same_broadcast, `data.py` FeatureSpace.fit_robust / broadcast_groups, `losses.py` nt_xent). `--check` passes (padding Δ = 0, scaling round-trip 1e-7). |
| D14 | Training speed | random batches · length-bucketed batches | **Decided 2026-10-08:** length-bucketed batches (shuffled within each window length; batch order shuffled per epoch; every train window once per epoch), batch 256. Reason: the per-window FFT / period grouping runs once per distinct length / period; random batches of 256 hold ~60 lengths, bucketed ones 2–3. Side effect: NT-Xent negatives share the anchor's length (no length cue). `config.length_bucketed = False` restores random batches. |

## 2. Pretraining tasks (S1.x)

Effort in working days. `→` marks a dependency. Tasks that write under `data/` need the user's go-ahead
(RULE 2).

### S1.0 Scope, decisions, environment

| ID | Task | Deliverable | Acceptance criterion | Dep | d |
|---|---|---|---|---|---|
| S1.0.1 | Share the adopted corrected plan with the professor | `docs/plan/clarification.md` | professor has seen it; any objection recorded as a change here | — | 1 |
| S1.0.2 | Decision record + hypotheses remap | `research-plan.md` "Pretraining plan" section | each changed task ID cross-referenced (superseded / modified / kept) | — | 0.5 |
| S1.0.4 | Pin the training environment (RULE 4) — ✅ **Done 2026-10-07**: `.venv-train` (Python 3.14.5, torch 2.14.1, MPS), `requirements-train.txt`, Jupyter kernel `roadfm-train` | `requirements-train.txt` (`pip freeze`) | torch version, device and seed list recorded in every run config | — | 0.5 |
| S1.0.5 | Rename physics heads P1–P3 | all docs | no clash with H1–H4 | — | 0.1 |

### S1.1 Input representation

| ID | Task | Deliverable | Acceptance criterion | Dep | d |
|---|---|---|---|---|---|
| S1.1.1 | Cross-scenario leak audit (F1) — ⬜ **Not done** (code + prepared data removed 2026-10-06 at the user's request; findings F11–F13 and the leak-audit note are kept) | `scripts/audit_cross_scenario_leak.py`, `docs/research-notes/data_understanding/cross-scenario-leak.md` | reports, per class, the share of test identities with a same-(window, vehicle) copy in train; reproduces the 89.7% benign figure | — | 1 |
| S1.1.2 | Streaming receiver-log parser — ✅ **Done 2026-10-06 (D1′) in `src/pipeline/input_representation.ipynb`** — mirrored tree `src/data/prepared_data/<Family>_<group>/<run>/traceJSON-*.json` (23,032 files, 2.9 GB JSON) + `index.json`; all 20,364,658 received messages kept, 66 s. | `scripts/build_receiver_sequences.py` → `data/prepared_receiver/` (**needs OK**) | row counts match raw files (23,032 files, 12.4 GB); asserts `rcvTime == sendTime`; < 30 min | data-write OK | 2 |
| S1.1.3 | Receiver-own join + relative features — ✅ **Done 2026-10-06 (D1′) in `src/pipeline/input_representation.ipynb`** — mirrored tree `src/data/prepared_data/<Family>_<group>/<run>/traceJSON-*.json` (23,032 files, 2.9 GB JSON) + `index.json`; 13 features per message, finite except the first log_dtau of each link (null). | `models/roadfm/features.py` + tests | no privileged field (`sender`, `*_noise`, ground truth) in inputs; benign range p99 ≈ 384 m reproduced | S1.1.2 | 2 |
| S1.1.4 | Sequence unit, T, length control; GridSybil_0709 pseudonym-1 handling — ✅ **Done 2026-10-06 (D1′) in `src/pipeline/input_representation.ipynb`** — mirrored tree `src/data/prepared_data/<Family>_<group>/<run>/traceJSON-*.json` (23,032 files, 2.9 GB JSON) + `index.json`; 5,321,657 links, all messages kept; encoder windows cut 2026-10-06 in `src/pipeline/benign_gridsybil/encoder_input_T64.ipynb` for benign + GridSybil: crop at 64, every window padded to 64 + mask, 376,427 windows, 71.0% padding (DataReplay/DoS links ≥ 64: 0–1 → not covered yet). | `configs/s1_input.json`, length histograms per class | every sample cut to exactly T messages (no padding), equal windows per vehicle, so length alone gives AUC ≈ 0.5; retained-identity share logged per T | S1.1.2, D1 | 1.5 |
| S1.1.5 | Group-aware split (**RULE 3 — split change**) — ✅ **Done 2026-10-06 (D1′) in `src/pipeline/input_representation.ipynb`** — mirrored tree `src/data/prepared_data/<Family>_<group>/<run>/traceJSON-*.json` (23,032 files, 2.9 GB JSON) + `index.json`; **RULE 3** vehicle split over sender vehicles in `index.json`: 4,730 / 526 / 1,126 / 1,126 vehicles. | `scripts/make_receiver_splits.py`, `idx_*_rx.npy` | zero `(window, vehicle)` overlap across train/val/test and across 0709/1416; pretrain-val = 10% of train identities | S1.1.1, S1.1.4 | 1 |
| S1.1.6 | Shortcut probes — 🟨 **Partly done 2026-10-06** in `src/eda/benign_gridsybil/eda_encoder_input_T64.ipynb`: length-only AUC 0.531 (val); unmasked mean leaks length; time-of-day / run-id probes not redone | `notebooks/s1_leakage_probes.ipynb` | logistic probes on length, time-of-day (probe target only — never an input), run id: chance-level class separation | S1.1.5 | 1 |
| S1.1.7 | Manifest — ✅ **Done 2026-10-06 (D1′) in `src/pipeline/input_representation.ipynb`** — mirrored tree `src/data/prepared_data/<Family>_<group>/<run>/traceJSON-*.json` (23,032 files, 2.9 GB JSON) + `index.json`; `index.json`: labels, split, train-only normalisation (per-feature counts), counts, provenance. | `data/prepared_receiver/config.json` | feature list, T, seeds, normalisation (train-only) recorded | S1.1.3–6 | 0.5 |

### S1.2 Encoder — TimesNet

| ID | Task | Deliverable | Acceptance criterion | Dep | d |
|---|---|---|---|---|---|
| S1.2.1 | Embedding: token conv + positional, no temporal (x_mark) path — ✅ **Built 2026-10-07** in `src/model/benign_gridsybil/encoder_T64.ipynb` (zero padding, not circular; × mask) | `models/roadfm/embedding.py` | (B, T, 13) → (B, T, d); no time-of-day injection | S1.0.4 | 0.5 |
| S1.2.2 | Inception + TimesBlock, **per-sample** top-k periods, bins f ≥ 2 — ✅ **Built 2026-10-07** (same notebook): periods from an FFT over each window's real rows only; batch-composition |Δz| 8.3e-7 (re-run 2026-10-08 at d_ff = 128; D12 back to d_ff = 64) | `models/roadfm/timesnet.py` (vendored, MIT credit) | shapes for T ∈ {50, 64, 100, 128}; output invariant to batch composition (tol 1e-5) | S1.2.1 | 2 |
| S1.2.3 | `TimesNetEncoder` (4 blocks, shared LayerNorm, global average pooling of H over time) returning `(H, z)` — ✅ **Built 2026-10-07**: masked mean pooling over real rows; **2,301,312 params at d = 128, d_ff = 64 (D12, 2026-10-08)**; 4,595,840 with d_ff = 128 (D10, superseded) | same file | param counts match D12 (2,301,312 at d = 128, d_ff = 64, exact); z: (B, d); H feeds the decoder | S1.2.2 | 1 |
| S1.2.4 | Unit tests — ✅ **as notebook checks 2026-10-07** for both encoders: T ∈ {50, 64, 100, 128}, params exact, padding Δ = 0, batch |Δz| ≤ 1e-6, no NaN, seed, CPU vs MPS ~1.4e-6 at d_ff = 64 (1.8e-6 in the d_ff = 128 re-run; D12 back to d_ff = 64; no `tests/` file)| `tests/test_timesnet.py` | shape, param count, seed determinism, CPU vs MPS agreement, GAP over all T steps (inputs are fixed-length, no padding) | S1.2.3 | 1 |
| S1.2.5 | Transformer encoder with the same `(H, z)` API (ablation) — ⬜ built 2026-10-07, **removed from the code 2026-10-08 (D11: TimesNet only)**| `models/roadfm/transformer.py` | ≈ 1.19M params at d = 128; same tests pass | S1.2.4 | 1 |
| S1.2.6 | Period diagnostic on train only — 🟨 one train shard (10,000 windows), untrained; 26% period-1 choices | notebook cell + figure | histogram of chosen periods per block | S1.2.3 | 0.5 |

Recommended config: T = 64 (T = 128 sensitivity), d_in = 13, d = 128, d_ff = 64 (D12; d_ff = 128 and d = 256 / 512 are ablations chosen only by evidence), num_kernels = 3,
top_k = 3 (per-sample periods, f ≥ 2), no time-of-day embedding, dropout 0.1.

### S1.3 Self-supervised pretraining objectives — no attack labels

**S1.3.A Masked reconstruction**

| ID | Task | Deliverable | Acceptance criterion | Dep | d |
|---|---|---|---|---|---|
| S1.3.A1 | Random (exactly 25% of T) or block (one block, L = round(r·T), r ~ U(0.20, 0.30)) masking, chosen per sample with equal probability — ⬜ not started (a draft script was written and removed 2026-10-07)| `models/roadfm/masking.py`, `tests/test_masking.py` | ratio exact ±1 step for all T; seeded determinism; mask returned | S1.1.7 | 1 |
| S1.3.A2 | Learned mask token replaces masked steps (no zero-fill; d_in stays 13) — ⬜ not started (a draft script was written and removed 2026-10-07)| encoder patch + test | gradient reaches the token; FFT sees no zero blocks | S1.3.A1, S1.2.3 | 1 |
| S1.3.A3 | 2-layer MLP decoder on H[:, t∈M]; masked MSE on sender pos (window-relative), velocity, acceleration, log-Δτ — ⬜ not started (a draft script was written and removed 2026-10-07)| `models/roadfm/losses.py::MaskedRecLoss` | loss only on masked steps; per-feature MSE logged | S1.3.A2 | 1 |
| S1.3.A4 | Linear-interpolation baseline | `scripts/baseline_interp.py` | model beats interpolation on block masks (pretrain-val) | S1.3.A3 | 0.5 |

**S1.3.P Physical-constraint heads (P1–P3)** — injected violations (D4)

| ID | Task | Deliverable | Acceptance criterion | Dep | d |
|---|---|---|---|---|---|
| S1.3.P1 | Violation injectors: P1 speed spike / position jump, P2 speed scaled without matching positions, P3 impossible sharp turn; applied with p = 0.5, label = injected — ⬜ not started (a draft script was written and removed 2026-10-07)| `models/roadfm/injections.py` + tests | each injector changes only its target fields; labels 50% ± 2 pp; seeded determinism; an injected window breaks its rule, an untouched one is unchanged | S1.1.7 | 1.5 |
| S1.3.P2 | Diagnostic audit of the original H1–H3 rules on real windows (reporting only, never training labels) | `docs/research-notes/physics-heads-audit.md` | reproduces F8 rates ±1 pp; per-family flag rate and AUROC reported on train | S1.1.7 | 1 |
| S1.3.P3 | *Optional:* LuST map alignment test (**download needs OK**) — not needed by P1–P3 | note: source, `netOffset` check | lanes overlay VeReMi positions within ~5 m, or recorded as not aligned | download OK | 1.5 |
| S1.3.P4 | MLP heads on z, BCE on injected labels — ⬜ not started (a draft script was written and removed 2026-10-07)| loss module + config fields | heads train; per-head pretrain-val AUROC logged | S1.3.P1, S1.2.3 | 1 |

**S1.3.C Contrastive (SimCLR / InfoNCE)**

| ID | Task | Deliverable | Acceptance criterion | Dep | d |
|---|---|---|---|---|---|
| S1.3.C1 | Physically consistent augmentations: common-mode 3–5 m shift (sender + receiver), time stretch s ∈ [0.95, 1.05] (v·s, a·s², Δτ/s), crop 80–100% by subsampling (no interpolation) — ⬜ not started (a draft script was written and removed 2026-10-07)| `models/roadfm/augment.py`, `tests/test_augment.py` | tests: shift leaves Δpos and range unchanged; stretch preserves ‖Δpos − vΔτ‖; crop keeps true Δτ | S1.1.7 | 2 |
| S1.3.C3 | Identity-level sampler, pseudonym de-dup | `models/roadfm/sampler.py` + test | no pseudonym twice in a batch; physical `sender` never read | S1.1.5 | 1 |
| S1.3.C4 | NT-Xent (τ = 0.1, batch 256), projection head d→d→128, grid-cell hard negatives (D6) — ⬜ not started (a draft script was written and removed 2026-10-07)| `losses.py::InfoNCE`, `negatives.py` + tests — **2026-10-08 (D13b):** same-broadcast / same-link pairs masked as false negatives | hard negatives: same 50 m cell, same group, different pseudonym, \|Δt\| ≥ 10 min; β = 0.5 configurable | S1.3.C1, C3 | 1.5 |
| S1.3.C5 | Attack-artifact sensitivity probe | `scripts/sensitivity_probe.py` | cosine shift under synthetic offsets / frozen blocks vs under augmentations; ratio ≫ 1 reported per checkpoint | S1.3.C4 | 1 |

**S1.3.J Joint training**

| ID | Task | Deliverable | Acceptance criterion | Dep | d |
|---|---|---|---|---|---|
| S1.3.J1 | L = λ1·L_rec + λ2·L_nce + λ3·L_P1 + λ4·L_P2 + λ5·L_P3; EMA normalisation + Kendall option — ⬜ not started (a draft script was written and removed 2026-10-07)| `losses.py::JointLoss` + test | all heads dropped at checkpoint export | A3, C4, P4 | 1 |
| S1.3.J2 | Label-free checkpoint selector — ⬜ not started (a draft script was written and removed 2026-10-07)| `models/roadfm/select.py` | pretrain-val masked MSE, alignment/uniformity, effective rank, head losses; asserts no label and no `idx_val` read | S1.3.J1 | 1 |
| S1.3.J3 | Weight sweep (D7), ≥ 3 seeds | `models/results/s1_sweep/*.json` (config, seed, pip freeze) | every run reproducible from its JSON | S1.3.J2, S1.0.4 | 3 |

### S1.4 Pretraining evaluation (frozen encoder, labels for evaluation only)

| ID | Task | Deliverable | Acceptance criterion | Dep | d |
|---|---|---|---|---|---|
| S1.4.1 | Linear + kNN probes on frozen z | `models/results/s1_probe.json` | train labels fit the probe only; binary + 4-family macro-F1, AUROC; ≥ 3 seeds | S1.3.J3 | 2 |
| S1.4.2 | Per-family recall, incl. nulls | table | all 4 families + benign reported | S1.4.1 | 0.5 |
| S1.4.3 | 0709 ↔ 1416 transfer | 2×2 matrix + Δ | pretraining reads the source group's train identities only; no time-of-day feature | S1.4.1 | 1.5 |
| S1.4.4 | Label-free anomaly score (unlabeled train embeddings) | AUROC | θ chosen on val; labels used only for scoring | S1.4.1 | 1 |
| S1.4.5 | Controls: TimesNet from scratch; Transformer encoder under the same SSL | paired comparison | same config, splits, seeds | S1.4.1, S1.2.5 | 1.5 |
| S1.4.6 | Objective ablations: −P heads, −InfoNCE, −hard negatives | results JSON | same splits and seeds as S1.4.1 | S1.4.1 | 2 |

**Pretraining tasks (S1.x) total ≈ 45 working days** (44.6 d = S1.0 2.1 · S1.1 9.0 · S1.2 6.0 · S1.3 19.0 incl. 1.5 optional ·
S1.4 8.5; add ~15% for integration and re-runs) · **few-shot fine-tuning ≈ 14.5 d (deferred**, §4).

**Critical path:** S1.1.2 (data write OK) → S1.1.3 → S1.1.4 → S1.1.5 → S1.3.A1/C1/P1 →
S1.3.J1 → S1.3.J2 → S1.3.J3 → S1.4.1. S1.1.1 (leak audit) and S1.2.x (encoder) have no data-write
dependency and can start immediately.

## 3. Mapping to the existing research plan

| Status | `research-plan.md` tasks |
|---|---|
| Superseded | 2.2 (features), 3.1 (Transformer spec → TimesNet), 3.3 (TCP dropped, D5 — corruption detection moves to injected-violation heads P1–P3) |
| Modified | 2.1 (T, receiver logs), 2.3 (→ diagnostic physics rules, S1.3.P2), 2.4 (→ contrastive augmentations + violation injectors), 2.5 (random exactly 25% / block masking), 3.2 (MLP decoder), 3.4 (λ1–λ5, label-free selection), 3.6 (label-free anomaly score), 4.4 (from-scratch TimesNet), 5.5 (objective ablations), hypothesis H4 |
| Kept | 2.6 (splice fix, extended to receiver logs), 2.7 (re-verified under the new split), 3.7, 4.1–4.3, 5.4, 5.6, 5.7, Phases 6–8 |
| Deferred to fine-tuning (§4) | 3.5, 5.2, 5.3, hypothesis H2 |
| Invalidated until re-split | v1 pilot numbers in `models/results/` (F1, F2) |

## 4. Few-shot fine-tuning tasks (S2.x) — deferred

Few-shot fine-tuning · deferred — planned, starts after pretraining (plan §4–§5). FL ("excluded from the FL round") is the
deployment motivation only; FL is not simulated (no task).

| ID | Task | Deliverable | Acceptance criterion | Dep | d |
|---|---|---|---|---|---|
| S2.0.1 | Confirm fine-tuning class set with the professor: train A16 GridSybil, A18 DoSRandom, A19 DoSDisruptive (+ benign); hold out A17 DataReplay for zero-shot. Full 19-attack download stays optional (needs OK) | note in `research-plan.md` | class set confirmed in writing | S1.4.1 | 0.5 |
| S2.0.2 | Few-shot support-set sampler: n ∈ {10, 20, 30, 50} per class, ≥ 5 resampled sets per n, train split only | `models/roadfm/fewshot.py` + tests | seeded; sets disjoint from val/test; A17 never sampled | S2.0.1 | 1 |
| S2.1.1 | Supervised contrastive fine-tuning from the pretrained checkpoint, LR = pretraining LR / 10 | `models/roadfm/supcon.py`, `scripts/train_stage2.py` | loads the chosen S1 checkpoint (heads already dropped); config + seed + pip freeze saved | S2.0.2, S1.3.J3 | 2 |
| S2.1.2 | Fine-tuning runs: every n × 5 support sets × 3 seeds, for the pretrained encoder and the from-scratch control | `models/results/s2_finetune/*.json` | every run reproducible from its JSON | S2.1.1 | 3 |
| S2.2.1 | Memory bank: L2-normalised embeddings of training-split trajectories, built **without labels**; FAISS cosine index | `models/roadfm/memory_bank.py` + tests | asserts no label read; index size = number of training windows | S2.1.1 | 1 |
| S2.2.2 | kNN anomaly score = mean cosine similarity to the K nearest neighbours; choose K ∈ {5, 10, 20} and θ on **val** | `models/results/s2_threshold.json` | K, θ chosen on val only; test untouched | S2.2.1 | 1 |
| S2.2.3 | Evaluation, test used once: per-class recall, AUROC, FPR at θ; zero-shot result on held-out A17 | `models/results/s2_eval.json` | test read exactly once; nulls reported | S2.2.2, S2.1.2 | 1.5 |
| S2.3.1 | Hyperparameter sweep (§6), staged: pretraining settings chosen label-free first; K, θ, n on val | `configs/s2_sweep.json` + results | every config logged with seed | S2.2.3 | 3 |
| S2.3.2 | Label-efficiency curves (score vs n) vs from-scratch and Transformer; report nulls | figure + table in `models/results/` | ≥ 3 seeds; error bars | S2.3.1 | 1.5 |

**Few-shot fine-tuning tasks (S2.x) total = 14.5 working days** (deferred).

## 5. Hyperparameter sweep plan (§6 of the plan)

| Hyperparameter | Values | Chosen when | How (no test data) |
|---|---|---|---|
| T | 64 (primary), 128 (sensitivity) | Pretraining | report both; 64 is default |
| d | 128, 256, 512 | Pretraining | label-free checkpoint score, then probe on val |
| Mask ratio r | block 20–30% of T; random 25% | Pretraining | label-free (held-out reconstruction error) |
| λ2 + weighting | λ2 ∈ {0.1, 0.3, 1} + one automatic-weighting run (λ1 = 1, λ3–λ5 = 0.3) | Pretraining | label-free checkpoint selection |
| Augmentation strengths | shift 3–5 m; time stretch ±5%; crop 80–100% | Pretraining | label-free + sensitivity probe (S1.3.C5) |
| Hard-negative weight β | 0.5 default (ablate 0) | Pretraining | label-free |
| Temperature τ | 0.1 default | Pretraining | label-free |
| K | 5, 10, 20 | Fine-tuning | val |
| θ | threshold on the anomaly score | Fine-tuning | val |
| Few-shot n | 10, 20, 30, 50 per class | Fine-tuning | all reported (label-efficiency curve) |

## 6. Out of scope unless the advisor signs off

OSM map matching (beyond the optional S1.3.P3 alignment test) · federated learning simulation · receiver-side
plausibility checks beyond the relative features in S1.1.3 · the full 19-type VeReMi-Extension download (optional
per S2.0.1; needs OK).
