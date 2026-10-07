# agent.md — start here (current state of the project)

> **Every agent reads this first, then root `CLAUDE.md` (rules 1–7).** It says what the project is *now*,
> which files are live vs. legacy, what is decided, and what waits on the user. **Keep it current** (see the
> last section). Last updated: **2026-10-07** (status: input done for benign + GridSybil; next = torch env → encoder with mask).

## 1. The project in five lines

- Master's thesis (Florida Polytechnic): **RoadFM-Lite**, self-supervised trajectory model for **Sybil attack
  detection** on the local **VeReMi-Extension** data (4 Sybil attacks × 2 scenario groups, 0709 / 1416).
- **Adopted method (2026-10-05):** a **TimesNet** encoder pretrained **without attack labels** (Stage 1:
  masked reconstruction + injected-violation physics heads P1–P3 + SimCLR contrastive), then few-shot
  fine-tuning + memory-bank anomaly scoring (Stage 2, **deferred**).
- The method **replaces** the proposal's Transformer + MTR/TCP design. Deviating from `proposal/main.tex` is
  **accepted** (decision D9) — do not flag it as an issue.
- **Prepared data = the receiver's view, mirroring VeReMi** (D1′): `src/pipeline/input_representation.ipynb` writes one
  JSON per raw trace file under `src/data/prepared_data/<Family>_<group>/<run>/` (every received copy kept,
  13 features per message, grouped into links) + `index.json` (labels, vehicle split, normalisation). Windows for
  benign + GridSybil are cut (below); DataReplay / DoS (1–2-message links) still need a receiver time window.
- **Encoder input (benign + GridSybil, T = 64: crop at 64, every window padded to 64 + mask; user choice)** is
  built by `src/pipeline/benign_gridsybil/encoder_input_T64.ipynb` → `src/data/encoder_input/benign_gridsybil/T64/<split>/b64/`: 376,427 windows of 64 × 13,
  every message used once, 71.0% padding rows (buckets 8/16/32/64 would give 26.3%; one setting, `buckets`). Full description: `docs/plan/stage1-preprocessing-feature-engineering.md`.
  The encoder must use the mask (masked pooling + losses on real rows; FFT periods from real rows). Masking code waits on the torch env.
- **Scope now (user, 2026-10-07):** benign + GridSybil is a first test run of the whole pipeline; all attack types
  come after it works (DataReplay / DoS need a different window, their links are 1–2 messages).
- All previous model numbers (v1 pilot) are **invalid as evidence** (F1 leak, F2 TCP bug) — history only.

## 2. What is live, what is legacy

| Path | Status | What it is |
|---|---|---|
| `agent.md` (this) | **live** | orientation + current state |
| `CLAUDE.md` | **live** | project rules 1–7 (binding) |
| `docs/plan/clarification.md` | **live — the plan** | adopted plan in the professor's format + why it changed + Q&A |
| `docs/plan/stage1-ssl-wbs.md` | **live — the task list** | findings F1–F10, decisions D1–D9, tasks S1.0–S1.4 (38, ≈ 45 d) and S2.0–S2.3 (9, 14.5 d, deferred), sweep plan |
| `docs/plan/research-plan.md` | **live** | thesis-wide phases 0–8 + "Stage 1 plan" decision record; keep in sync with the WBS |
| `src/pipeline/input_representation.ipynb` | **live** | raw trace file → prepared trace file (receiver view, all copies, 13 features, links) + `index.json` |
| `src/eda/eda_window_size.ipynb` | **live** | EDA for choosing T: link-length distributions per class, window-size sweep (no padding / padding / overlap), class + split balance, "Choosing T" summary (read-only) |
| `src/eda/benign_gridsybil/eda_encoder_input_T64.ipynb` | **live** | EDA of the encoder input (`src/data/encoder_input/benign_gridsybil/T64/`): counts, padding, length shortcut, per-class feature ranges + outliers, normalisation drift, correlations, single-feature AUC, repeated broadcasts (read-only) |
| `src/pipeline/benign_gridsybil/encoder_input_T64.ipynb` | **live** | benign + GridSybil links → crop at 64 → every window padded to 64 + mask (`buckets = (64,)`) → normalised sharded JSON |
| `src/data/encoder_input/benign_gridsybil/T64/` | live (gitignored, 1.8 GB JSON) | `<split>/b64/part-*.json` (id, x 64×13, mask) + `part-*_info.json` (labels, n_messages, bucket) + `metadata.json` |
| `docs/plan/stage1-encoder-notes.md` | **live** | study notes on the encoder: TimesNet from the ground up, masking + checks, heads, plan vs built, professor Q&A, glossary |
| `docs/plan/stage1-preprocessing-feature-engineering.md` | **live** | the preprocessing + feature-engineering description (features, links, crop/buckets/mask, outputs, limitations) |
| `src/data/prepared_data/` | live (gitignored, 2.9 GB JSON) | mirrors `data/VeReMi-Dataset` (8 scenario folders → runs → `traceJSON-*.json`) + `index.json` (labels, split, normalisation, counts) |
| `src/model/benign_gridsybil/timesnet_encoder_T64.ipynb` | **live** | masked TimesNet encoder + untrained heads + loss definitions + checks, **no training** (kernel `roadfm-train`) |
| `.venv-train/`, `requirements-train.txt` | **live** | pinned PyTorch env (torch 2.14.1, MPS); Jupyter kernel `roadfm-train` |
| `src/README.md` | **live** | what is in `src/`, layout, run order, data format, rules |
| `src/agent.md` | **live** | rules for code in `src/` |
| `stage1-input-slides.html` | **live** | 40-slide deck explaining Stage 1 input representation (raw data, design decisions, leakage findings, pipeline + outputs, caveats) for the professor |
| `encoder-shapes.html` | **live** | one-page matrix math of the encoder (input → embedding → TimesBlock → H, z → heads) with every size and parameter count |
| `stage1-plan.html` | **live** | per-task tracker for Stage 1/2 + plan review, diagrams, slides |
| `tracker.html` | **live** | whole-thesis tracker (phases, alignment table, critical path) |
| `simulation/` | live tool | TypeScript window simulator (data understanding); see `simulation/README.md` |
| `notebooks/eda_veremi.ipynb`, `benign_vs_attack_maps.ipynb` | live (data understanding) | EDA, raw-data maps |
| `notebooks/refresher_*.ipynb` | study notes | numpy / pandas / pytorch / DL refreshers |
| `docs/research-notes/` | reference | literature + Phase 1 data notes (`data_understanding/`) |
| `models/transformer_model.ipynb`, `models/*.pt`, `models/results/` | **legacy v1** | Transformer pilot; TCP was a no-op; numbers invalid. Do not extend. |
| `scripts/prepare_data.py`, `scripts/export_simulation_sample.py` | legacy v1 / tooling | v1 20 × 13 window prep (+ splice fix), simulator export. House-style reference for new code. |
| `data/` | **protected** (gitignored, 13 GB) | raw `VeReMi-Dataset/` + v1 `prepared_data/` — read-only (RULE 2) |
| `proposal/` | background | original proposal; method superseded (D9) |
| `README.md` | **stale overview** | still describes the v1 method; trust this file + the plan docs |

## 3. Decisions in force (details: WBS §1)

| ID | Decision |
|---|---|
| D1 | Sample = network-heard pseudonym sequence (dedup by `messageID`), **T = 64** (128 sensitivity), fixed length, equal windows per vehicle |
| D2 | "Receiver observed" = receiver's own GPS position/velocity at rcvTime |
| D3 | **13 features**: sender-claimed pos/vel, receiver pos/vel, acl, range, bearing (one wrapped angle, not sin/cos), log-Δτ. **No time of day.** |
| D4 | Physics heads **P1–P3 detect violations we inject** (p = 0.5); real-data rules are diagnostics only |
| D5 | TCP dropped |
| D6 | Contrastive hard negatives: same 50 m grid cell, same group, other pseudonym, ≥ 10 min apart |
| D7 | Normalised losses, λ1 = 1, λ3–λ5 = 0.3, λ2 ∈ {0.1, 0.3, 1}; label-free checkpoint selection |
| D8 | **New split** by (scenario group, physical vehicle) across the 4 scenarios — RULE 3 change, **in force** (`src/data/prepared_data/`, from the notebook) |
| D9 | Deviation from `proposal/main.tex` accepted |

Stage 2 defaults: train A16 GridSybil, A18 DoSRandom, A19 DoSDisruptive + benign; hold out **A17 DataReplay**;
n ∈ {10, 20, 30, 50}; unlabeled memory bank; K, θ on val; FL = motivation only.

## 4. Facts that trip up new agents

- Beacons are **1 Hz** (not 100 ms). Positions are SUMO **metres**, not lat/lon; there is **no map** in the data.
- Local data has only attack codes **A0, A16–A19** (A16 = GridSybil, A17 = DataReplay). No A1–A15.
- **F1:** the 4 scenarios reuse the same traffic → the v1 split leaks (89.7% of benign test identities have a
  near-copy in train). `scripts/audit_splits.py` passes only because it checks within one scenario.
- **F2:** `models/transformer_model.ipynb` TCP labels are all zero → v1 is MTR-only.
- `GridSybil_0709` pseudonym `1` is a shared sentinel (653 vehicles) — the v1 splice defect.
- Time of day alone separates 0709 from 1416 → never an input.
- Original physics rules: H1 fires only on attackers (label proxy), H2 has no violations > 70 km/h, H3 flags
  33% of benign windows — hence D4.
- Physics heads are **P1–P3**; **H1–H4** are the thesis hypotheses.
- **F11:** DataReplay / DoS rotate pseudonyms every 1–2 messages → at T = 64 pseudonym-level sequences kept
  1 DataReplay and 0 DoS samples (benign 70.5%, GridSybil 48.5%). Decide the sequence unit again before
  re-implementing S1.1.
- **F13:** GridSybil ghosts share pseudonym 1 in every GridSybil run (not only 0709); splitting it by the true
  sender is a privileged grouping.
- The removed S1.1 run measured: shortcut probe length AUC 0.551, time of day 0.544, run id 0.886.
- **F12:** one copy per message (primary listener) gives benign range p50 36 / p99 348 m (all copies: 81 / 384 m).
- A repo hook (`scripts/organize_repo.sh`, after every subagent) moves stray root files; keep-list includes
  `agent.md` and `stage1-plan.html` — add new root files there or they get moved.

## 5. Waiting on the user (do not proceed without an explicit OK)

| Item | Blocks |
|---|---|
| DataReplay / DoS windows (1–2-message links → receiver time window?) — benign + GridSybil are done | Stage 2 classes, not S1.2 |
| Optional: download LuST map (`lust.net.xml`) | S1.3.P3 only |
| Optional: download the full 19-attack VeReMi-Extension | original Stage 2 classes only |
| Professor's confirmation of the corrected plan (S1.0.1) | nothing blocks; objections become WBS changes |

## 6. Next steps

1. User reads the encoder notebook (no training until they say so).
2. S1.3: masking strategies, augmentations, violation injectors, joint loss, training loop.
3. S1.3 objectives → joint training → S1.4 frozen-encoder evaluation. Then Stage 2.

Follow `src/agent.md` §5 (per-task workflow) for every task.

## 7. Keeping this file current (every agent, every substantive turn)

- When a task completes, a decision is made, or an approval arrives: update §3–§6 here **and** the WBS /
  `research-plan.md` in the same turn, add a line to the changelog below, and remind the user to tick
  `stage1-plan.html` and `tracker.html`.
- If a file's status changes (live → legacy, new doc), update §2.
- Keep it short: this file is orientation, not a duplicate of the plan. Link instead of copying.

## Changelog

- **2026-10-07** — `encoder-shapes.html`: simple one-page matrix-size walkthrough of the encoder (one 28-row window); added to the root keep-list.
- **2026-10-07** — `docs/plan/stage1-encoder-notes.md`: written study notes on the encoder (≈ 5.3k words, numbers from the notebook).
- **2026-10-07** — torch env pinned (`.venv-train`, torch 2.14.1, MPS; S1.0.4). Encoder notebook `src/model/benign_gridsybil/timesnet_encoder_T64.ipynb` (no training): masked TimesNet, 2,301,312 params; periods from an FFT over real rows only; heads + losses defined; all checks pass.
- **2026-10-07** — folders reflect the trial scope: `src/pipeline/benign_gridsybil/`, `src/eda/benign_gridsybil/`, data moved to `src/data/encoder_input/benign_gridsybil/T64/` (content unchanged; `metadata.json` out_dir updated); all-scenario notebooks stay at `src/pipeline/` and `src/eda/`; links updated.
- **2026-10-07** — `src/` organized into folders (code stays in the notebooks, user choice): `src/pipeline/` (input_representation, encoder_input_T64), `src/eda/` (eda_window_size, eda_encoder_input_T64), `src/data/`. Notebooks find the repo root by walking up to `CLAUDE.md`; links in docs / HTML updated.
- **2026-10-07** — status update: input representation done for benign + GridSybil (S1.1.2–S1.1.5), S1.1.6 partly (length AUC 0.531); masking is in the data, model-side masking waits on the torch env (S1.0.4).
- **2026-10-06** — encoder input switched to one fixed length (user: compute cost accepted): every window padded to 64 (`buckets = (64,)`), 376,427 windows, 71.0% padding, 39 shards, 1.8 GB; checks pass; EDA re-run (same takeaways); `stage1-input-slides.html` + `stage1-plan.html` updated (buckets shown as the alternative). EDA gained a padding/mask section: half the windows are padding from position 13–16; an unmasked mean leaks length (log_dtau AUC 0.492 → 0.442).
- **2026-10-06** — `src/eda/benign_gridsybil/eda_encoder_input_T64.ipynb`: EDA of the bucketed input; length-only AUC 0.531; no single feature has |AUC − 0.5| > 0.1; |z| > 10 only in GridSybil claimed_pos / range; max split drift 0.10 std; repeated broadcasts never cross splits (894 cross-split matches are all standing cars).
- **2026-10-06** — encoder input switched to crop + length buckets + mask (max T = 64): 376,427 windows, all
  messages used, 26.3% padding; doc `docs/plan/stage1-preprocessing-feature-engineering.md`; slides + stage1-plan.html updated.
- **2026-10-06** — `src/eda/eda_window_size.ipynb`: link lengths + T sweep + balance; largest T keeping ≥ 50% of messages without padding = 24; benign share most stable at T = 16.
- **2026-10-06** — padding removed from the encoder input (user): 22,976 full 64-message windows (was 376,427
  padded); uses 1.47 M of 6.98 M kept messages (benign 18%, GridSybil 23%).
- **2026-10-06** — `stage1-input-slides.html`: 26-slide presentation of the input representation.
- **2026-10-06** — `src/pipeline/benign_gridsybil/encoder_input_T64.ipynb`: benign + GridSybil encoder input, T = 64, padded windows +
  mask (376,427 windows; padding ≈ 70%; padding-only AUC 0.531 on val).
- **2026-10-06** — prepared data rebuilt in the receiver's view (D1′): all copies kept, VeReMi folder mirror,
  `index.json`; de-duplicated T64/T128 output replaced.
- **2026-10-06** — input representation rebuilt as `src/pipeline/input_representation.ipynb` (JSON only, no npy/parquet)
  → `src/data/prepared_data/T64|T128`; vehicle-grouped split in force again (RULE 3).
- **2026-10-06** — S1.1 code (`src/*.py`, tests, notebook, config) and prepared data (`src/data/`,
  `data/prepared_receiver/`) **removed at the user's request**; findings F1/F11–F13 kept; S1.1 tasks reset to
  not done; D8 not executed. (A copy of the removed code exists only in the agent's session scratchpad.)
- **2026-10-06** — `src/` simplified to a flat layout (one file per step; old `model/`, `helpers/`,
  `scripts/`, `configs/` removed); outputs verified byte-identical; command is now `python -m src.prepare`.
- **2026-10-06** — S1.1 run on full data → `data/prepared_receiver/` (15,904 samples); new split executed
  (RULE 3, 0% copies); F11 decided (pseudonym-level), F13 found; probe: length AUC 0.551.
- **2026-10-06** — S1.1 coded in `src/` (helpers, `model/data.py`, `model/splits.py`, `model/evaluation.py`,
  `scripts/prepare.py`, notebook, 85 tests); S1.1.1 leak audit done; F11 + F12 found; D8 key refined;
  `agent.md` restored to the root after the tidy hook moved it.
- **2026-10-06** — `src/agent.md` added (flat layout, OOP 90/10, guardrails); bearing fixed as one angle
  (13 features); WBS gained Stage 2 tasks + sweep plan; `tracker.html` rewritten for the adopted plan; this file
  created.
- **2026-10-05** — corrected plan adopted (D1–D9); `clarification.md` written; deviation from `main.tex`
  accepted.
- **2026-10-04** — plan review (19 → 18 issues) added to `stage1-plan.html`.
- **2026-09-30** — advisor's TimesNet Stage 1 plan received; WBS + findings F1–F10; F1 leak confirmed.
