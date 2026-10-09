# agent.md — start here (current state of the project)

> **Every agent reads this first, then root `CLAUDE.md` (rules 1–7).** It says what the project is *now*,
> which files are live vs. legacy, what is decided, and what waits on the user. **Keep it current** (see the
> last section). Last updated: **2026-10-08** (docs reorganised; D12: d = 128, d_ff = 64; D14 length-bucketed batches; pretraining package ready, no real run yet).

## 1. The project in five lines

- Master's thesis (Florida Polytechnic): **RoadFM-Lite**, self-supervised trajectory model for **Sybil attack
  detection** on the local **VeReMi-Extension** data (4 Sybil attacks × 2 scenario groups, 0709 / 1416).
- **Adopted method (2026-10-05):** a **TimesNet** encoder pretrained **without attack labels** (self-supervised pretraining:
  masked reconstruction + injected-violation physics heads P1–P3 + SimCLR contrastive), then few-shot
  fine-tuning + memory-bank anomaly scoring (fine-tuning and detection, **deferred**).
- The method **replaces** the proposal's Transformer + MTR/TCP design. Deviating from `proposal/main.tex` is
  **accepted** (decision D9) — do not flag it as an issue.
- **Prepared data = the receiver's view, mirroring VeReMi** (D1′): `src/pipeline/input_representation.ipynb` writes one
  JSON per raw trace file under `src/data/prepared_data/<Family>_<group>/<run>/` (every received copy kept,
  13 features per message, grouped into links) + `index.json` (labels, vehicle split, normalisation). Windows for
  benign + GridSybil are cut (below); DataReplay / DoS (1–2-message links) still need a receiver time window.
- **Encoder input (benign + GridSybil, T = 64: crop at 64, every window padded to 64 + mask; user choice)** is
  built by `src/pipeline/benign_gridsybil/encoder_input_T64.ipynb` → `src/data/encoder_input/benign_gridsybil/T64/<split>/b64/`: 376,427 windows of 64 × 13,
  every message used once, 71.0% padding rows (buckets 8/16/32/64 would give 26.3%; one setting, `buckets`). Full description: `docs/plan/stage1-preprocessing-feature-engineering.md`.
  The encoder uses the mask (masked pooling + losses on real rows; FFT periods from real rows).
- **Scope now (user, 2026-10-07):** benign + GridSybil is a first test run of the whole pipeline; all attack types
  come after it works (DataReplay / DoS need a different window, their links are 1–2 messages).
- All previous model numbers (v1 pilot) are **invalid as evidence** (F1 leak, F2 TCP bug) — history only.

## 2. What is live, what is legacy

| Path | Status | What it is |
|---|---|---|
| `agent.md` (this) | **live** | orientation + current state |
| `CLAUDE.md` | **live** | project rules 1–7 (binding) |
| `docs/plan/clarification.md` | **live — the plan** | adopted plan in the professor's format + why it changed + Q&A |
| `docs/plan/stage1-ssl-wbs.md` | **live — the task list** | findings F1–F13 (F14 in §4 below), decisions D1–D14 (full text, the source of truth), tasks S1.0–S1.4 (38, ≈ 45 d) and S2.0–S2.3 (9, 14.5 d, deferred), sweep plan |
| `docs/plan/research-plan.md` | **live** | thesis-wide phases 0–8 + "Pretraining plan" decision record (D1–D14 digest, RULE 3 split record); keep in sync with the WBS |
| `docs/README.md` · `docs/changelog.md` | **live** | index of every doc and page (live / reference / proposal-era / legacy) · changelog entries older than the newest 10 |
| `index.html` | **live** | landing page: links to the five explainer / tracker pages, docs, notebooks, npm commands |
| `src/pipeline/input_representation.ipynb` | **live** | raw trace file → prepared trace file (receiver view, all copies, 13 features, links) + `index.json` |
| `src/eda/eda_window_size.ipynb` | **live** | EDA for choosing T: link-length distributions per class, window-size sweep (no padding / padding / overlap), class + split balance, "Choosing T" summary (read-only) |
| `src/eda/benign_gridsybil/eda_encoder_input_T64.ipynb` | **live** | EDA of the encoder input (`src/data/encoder_input/benign_gridsybil/T64/`): counts, padding, length shortcut, per-class feature ranges + outliers, normalisation drift, correlations, single-feature AUC, repeated broadcasts (read-only) |
| `src/pipeline/benign_gridsybil/encoder_input_T64.ipynb` | **live** | benign + GridSybil links → crop at 64 → every window padded to 64 + mask (`buckets = (64,)`) → normalised sharded JSON |
| `src/data/encoder_input/benign_gridsybil/T64/` | live (gitignored, 1.8 GB JSON) | `<split>/b64/part-*.json` (id, x 64×13, mask) + `part-*_info.json` (labels, n_messages, bucket) + `metadata.json` |
| `docs/plan/stage1-encoder-notes.md` | **live** | study notes on the encoder: TimesNet from the ground up, masking + checks, heads, plan vs built, professor Q&A, glossary |
| `docs/plan/stage1-preprocessing-feature-engineering.md` | **live** | the preprocessing + feature-engineering description (features, links, crop/buckets/mask, outputs, limitations) |
| `src/data/prepared_data/` | live (gitignored, 2.9 GB JSON) | mirrors `data/VeReMi-Dataset` (8 scenario folders → runs → `traceJSON-*.json`) + `index.json` (labels, split, normalisation, counts) |
| `src/model/benign_gridsybil/encoder_T64.ipynb` | **live — the encoder explained** | TimesNet only; imports `timesnet/encoder.py`, shows the code, checks (2,301,312 params at d = 128, d_ff = 64 (D12), padding Δ = 0, batch ≤ 1e-6, CPU vs MPS ~1.4e-6) and the period diagnostic |
| `src/model/benign_gridsybil/timesnet/` | **live — self-supervised pretraining (TimesNet only)** | `config.py` (PretrainConfig, d = 128, d_ff = 64; D12), `data.py`, `encoder.py`, `heads.py` (reconstruction, P1–P3, projection), `views.py`, `losses.py`, `monitors.py`, `train.py` (`.venv-train/bin/python -m src.model.benign_gridsybil.timesnet.train --check|--smoke|--resume`), `pretrain_monitor.ipynb` (read-only plots); not trained |
| `scripts/hf_hub.py` · `docs/ec2-training.md` | **live** | Hugging Face transfers (user `bibekgupta3333`, **private** repos): dataset `veremi-extension-raw` (raw `data/VeReMi-Dataset`, download restores the same tree, sha256-checked) and model `roadfm-lite-timesnet` (`runs/<run_id>/`, SHA256SUMS); npm `hf:whoami`, `data:upload`, `data:download`, `model:upload`, `model:download`, `model:list`, `train:one-batch`; EC2 guide |
| `docker/Dockerfile`, `docker-compose.yml`, `requirements/linux.txt`, `scripts/run_notebook.py` | **live** | Ubuntu 24.04 image (Python 3.14.5 via uv, `/opt/venv`, torch 2.14.1 `cu126` / `cpu`, build args `TORCH_VARIANT` / `TORCH_VERSION`; no cu128 wheel of 2.14.1 for py3.14) for EC2; compose services `gpu` / `cpu`, repo bind-mounted at `/workspace`; npm `docker:build`, `docker:build:cpu`, `docker:shell`, `docker:shell:cpu`, `docker:check:cpu`, `docker:train`, `pipeline:prepare`, `pipeline:encoder-input`; npm python commands use `${PY:-.venv-train/bin/python}` (Mac unchanged); guide `docs/ec2-training.md` |
| `package.json` (root) | **live** | npm task shortcuts only (Python code; npm is the task runner) — groups `setup`, `pipeline:*`, `train:*`, `hf:*` / `data:*` / `model:*`, `docker:*`, `format` / `test` / `check`, `sim*`; see the file |
| `scripts/tests/`, `src/tests/` | **live** | pytest suites (CPU, synthetic data): legacy tools + exporters + HF in `scripts/tests/`, the pretraining package in `src/tests/`; run all with `npm test` |
| `src/runs/pretraining/benign_gridsybil/T64/<run_id>/` | live (gitignored) | one folder per training run (config.json, env.json, metrics.jsonl, checkpoints); so far only test runs (1–30 steps), no result |
| `.venv-train/`, `requirements-train.txt` · `.venv/`, `requirements.txt` | **live** | pinned PyTorch env (torch 2.14.1, MPS; Jupyter kernel `roadfm-train`) · EDA / notebook env (no torch); Linux / Docker pins in `requirements/linux.txt` |
| `src/README.md` | **live** | what is in `src/`, layout, run order, data format, rules |
| `src/agent.md` | **live** | rules for code in `src/` |
| `stage1-input-slides.html` | **live** | 44-slide deck explaining the pretraining input representation (raw data, design decisions, leakage findings, pipeline + outputs, caveats) for the professor |
| `encoder-shapes.html` | **live** | one-page matrix math of the encoder (input → embedding → TimesBlock → H, z → heads) with every size and parameter count |
| `src/pipeline/all/encoder_input_T24.ipynb` → `src/data/encoder_input/all/T24/` | **live — not for training until F14 is decided** | all 8 folders, all classes, T = 24, compact gzipped JSON (real rows only; reader pads + masks), 5,633,856 windows, 0.62 GB; 90/10 vehicle split reusing T64's; 1–3-row windows: DataReplay 94.7%, DoS 99.7% (F11) |
| `pretraining-explained.html` | **live** | the pretraining run explained in detail (data, check set, scaling, length-bucketed batches, one step, losses, joint loss, update + LR, epoch evaluation, checkpoints, health checklist, commands, Q&A) with 6 step-by-step animations; no results |
| `stage1-plan.html` | **live** | per-task tracker for pretraining / fine-tuning + plan review, diagrams, slides |
| `tracker.html` | **live** | whole-thesis tracker (phases, alignment table, critical path) |
| `simulation/` | live tool | TypeScript simulator: legacy v1 window replay + **"Encoder input (T = 64)" tab** (`simulation/src/encoder/`, sample from `scripts/export_encoder_sample.py` = `npm run sim:data`, 7.7 MB, gitignored `public/`); see `simulation/README.md` |
| `notebooks/eda_veremi.ipynb`, `benign_vs_attack_maps.ipynb` | live (data understanding) | EDA, raw-data maps |
| `notebooks/refresher_*.ipynb` | study notes | numpy / pandas / pytorch / DL refreshers |
| `docs/research-notes/` | reference | literature + Phase 1 data notes (`data_understanding/`) |
| `models/transformer_model.ipynb`, `models/*.pt`, `models/results/` | **legacy v1** | Transformer pilot; TCP was a no-op; numbers invalid. Do not extend. |
| `scripts/prepare_data.py`, `scripts/export_simulation_sample.py` | legacy v1 / tooling | v1 20 × 13 window prep (+ splice fix), simulator export. House-style reference for new code. |
| `data/` | **protected** (gitignored, 13 GB) | raw `VeReMi-Dataset/` + v1 `prepared_data/` — read-only (RULE 2) |
| `proposal/` | background | original proposal; method superseded (D9) |
| `README.md` | **live** | front door: what the project is, method, repo map, quick start |

## 3. Decisions in force (details: WBS §1)

| ID | Decision |
|---|---|
| D1 | Sample = network-heard pseudonym sequence (dedup by `messageID`), **T = 64** (128 sensitivity), fixed length, equal windows per vehicle |
| D1′ | Prepared data = the receiver's view as VeReMi recorded it: every received copy kept, one prepared file per raw trace file, links = (receiver, sender pseudonym, sender) (2026-10-06, user) |
| D2 | "Receiver observed" = receiver's own GPS position/velocity at rcvTime |
| D3 | **13 features**: sender-claimed pos/vel, receiver pos/vel, acl, range, bearing (one wrapped angle, not sin/cos), log-Δτ. **No time of day.** |
| D4 | Physics heads **P1–P3 detect violations we inject** (p = 0.5); real-data rules are diagnostics only |
| D5 | TCP dropped |
| D6 | Contrastive hard negatives: same 50 m grid cell, same group, other pseudonym, ≥ 10 min apart |
| D7 | Normalised losses, λ1 = 1, λ3–λ5 = 0.3, λ2 ∈ {0.1, 0.3, 1}; label-free checkpoint selection |
| D8 | **New split** by (scenario group, physical vehicle) across the 4 scenarios — RULE 3 change, **in force** (`src/data/prepared_data/`, from the notebook) |
| D8′ | **⚠ 2026-10-08: benign + GridSybil encoder input split 90 / 10 train / test by sender vehicle** (seed 0, stratified; 338,001 / 38,426 windows); pretraining check set = 10% of train vehicles; no val split (RULE 3, recorded in research-plan) |
| D9 | Deviation from `proposal/main.tex` accepted |
| D10 | ~~d_ff = d = 128~~ (width part superseded by D12); **still in force:** P1–P3 heads use hidden width d (Linear 128→128→1) |
| D11 | ~~d = 512~~ (superseded by D12); **still in force:** Transformer ablation removed from the code (TimesNet only) (2026-10-08, student) |
| D12 | **d = 128, d_ff = 64** — the professor's plan (2026-10-08, student): encoder 2,301,312 params, 2,402,448 with heads (2,402,461 incl. the 13-number mask token, as runs report); d_ff = 128 and d = 256 / 512 are ablations, chosen only by evidence |
| D13 | Pretraining re-scales `claimed_pos_x/y` and `range` (median / IQR of train + soft tail at |z| = 5, at load time) and masks same-broadcast pairs (run + pseudonym) as NT-Xent false negatives (2026-10-08, from the EDA verdicts) |
| D14 | Pretraining speed (2026-10-08, user: "full speed"): **length-bucketed batches** (each batch = windows of ~the same length, shuffled within length, batch order shuffled per epoch; ≈ 2–3 distinct lengths per batch instead of ~60) and batch 256 (the plan); `npm run train:full` lifts the 8 h cap. Every window still seen once per epoch |

Few-shot fine-tuning defaults: train A16 GridSybil, A18 DoSRandom, A19 DoSDisruptive + benign; hold out **A17 DataReplay**;
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
- **F14 (2026-10-08, open):** GridSybil_0709 is a separate re-simulation of the 0709 traffic (run folders dated 2025-11-15, different vehicle ids); a shared id is never the same car. In the all-data T24 input the vehicle-key split therefore lets 2.5–5.9% of GridSybil_0709 test benign identities have a near-copy in train within 1 m (≈ 8–10% within 3 m; near-full copies 0.4–1.0%). 1416 and the other 0709 families: 0% (no leak). Fix option: link cars across families by trajectory (same start ±1 s / 10 m, median gap ≤ 3 m over ≥ half the track), union-find, split by component (RULE 3 change). The benign + GridSybil T64 input is not affected (one family only).
- **F12:** one copy per message (primary listener) gives benign range p50 36 / p99 348 m (all copies: 81 / 384 m).
- A repo hook (`scripts/organize_repo.sh`, after every subagent) moves stray root files; keep-list includes
  `agent.md`, `index.html` and the root HTML pages — add new root files there or they get moved.

## 5. Waiting on the user (do not proceed without an explicit OK)

| Item | Blocks |
|---|---|
| **F14:** fix the GridSybil_0709 cross-family near-copies in the T24 split (trajectory linking + union-find) or document as a limitation | using `encoder_input/all/T24` |
| DataReplay / DoS windows (1–2-message links → receiver time window?) — benign + GridSybil are done | Fine-tuning classes, not S1.2 |
| Optional: download LuST map (`lust.net.xml`) | S1.3.P3 only |
| Optional: download the full 19-attack VeReMi-Extension | original fine-tuning classes only |
| Professor's confirmation of the corrected plan (S1.0.1) | nothing blocks; objections become WBS changes |

## 6. Next steps

1. When the user says so: `npm run train:smoke` (measures the speed with D14 batches), then `npm run train:full` (Mac) or the Docker / EC2 route (`docs/ec2-training.md`); read the run with `src/model/benign_gridsybil/timesnet/pretrain_monitor.ipynb`. Main model d = 128, d_ff = 64 (D12).
2. After a run: S1.4 frozen-encoder evaluation (probes, transfer). Then few-shot fine-tuning (S2.x, deferred).

Follow `src/agent.md` §5 (per-task workflow) for every task.

## 7. Keeping this file current (every agent, every substantive turn)

- When a task completes, a decision is made, or an approval arrives: update §3–§6 here **and** the WBS /
  `research-plan.md` in the same turn, add a line at the top of the changelog below (keep the newest 10; move the
  oldest to the top of `docs/changelog.md`), and remind the user to tick `stage1-plan.html` and `tracker.html`.
- If a file's status changes (live → legacy, new doc), update §2 and the one-line entry in `docs/README.md`.
- Keep it short: this file is orientation, not a duplicate of the plan. Link instead of copying.

## Changelog

Newest 10 entries; older ones are in [`docs/changelog.md`](docs/changelog.md).

- **2026-10-08** — docs reorganised (multi-agent): `README.md` rewritten for the adopted plan; `CLAUDE.md` project map refreshed (rules unchanged); new `docs/README.md` (index of every doc) and root `index.html` (landing page for the explainer pages, added to the hook keep-list); `src/agent.md` matched to the real layout (no `src/tools.py` / `SafeWriter`, black 120, tests in `scripts/tests/` + `src/tests/`); decision tables put in ID order with D14; changelog older than the newest 10 entries moved to `docs/changelog.md`.
- **2026-10-08** — test model `twenty-windows` (one real step on a batch of 20 windows, `--smoke --batch-size 20 --max-steps 1`) uploaded to the private HF model repo (`runs/twenty-windows`, commit 9619f80) and downloaded back: sha256 match, checkpoint loads. A pipeline test only — not a result.
- **2026-10-08** — one-command setup `scripts/setup.sh` (`npm run setup` / `setup:cpu`; never downloads the dataset, checks it). Tested in Docker (cpu) on a fresh copy with the dataset mounted read-only: prepared data 84 s, T64 input 167 s (byte-identical shards, counts match), `train:check` 12 s, re-run idempotent. HF inside the container: whoami, one-batch training, model:list and model:download (sha256 match) pass; **model:upload from the container blocked by the session's permission check — the user runs it**. GPU mode untested (no NVIDIA here).
- **2026-10-08** — Docker image verified on the Mac (cpu service, arm64): build ≈ 1.5 min, 2.8 GB; Python 3.14.5, torch 2.14.1+cpu, kernels `python3` + `roadfm-train`; `train:check` passes in 12 s; `scripts/tests` 66 passed after adding `pyarrow==25.0.1` to `requirements/linux.txt`; `train:smoke -- --max-steps 3` OK (≈ 10 s/step CPU). GPU image (`cu126`, amd64) resolves but was not built here (no GPU, little disk); first real test is on EC2.
- **2026-10-08** — Docker route for EC2 (Ubuntu 24.04, compose services `gpu` / `cpu`, no data or token in the image); `docs/ec2-training.md` rewritten for it (instance choice, plain-Ubuntu driver + Docker + NVIDIA Container Toolkit install, ≥ 100 GB disk, build → `hf auth login` → `data:download` → `pipeline:prepare` → `pipeline:encoder-input` → `train:check` / `train:smoke` → `train:full` in tmux → `model:upload`, costs); venv route kept as a Mac appendix; `src/README.md` run table gained a Docker row.
- **2026-10-08** — `--no-resource-guard` CLI flag (config `resource_guard`): no MPS memory cap, no memory-budget stop (memory still logged; recorded in config.json; also accepted with `--resume`); npm `train:full:no-guard`. `--check --no-resource-guard` passes.
- **2026-10-08** — Hugging Face set up (`huggingface_hub` 2.2.0 + hf-xet pinned in `requirements-train.txt`): `scripts/hf_hub.py` (+25 tests), npm commands, `docs/ec2-training.md`. One-batch training run (`--max-steps` now counts real updates; first real update logged) → uploaded to the private model repo `bibekgupta3333/roadfm-lite-timesnet/runs/one-batch` and downloaded back: sha256 identical, checkpoint loads. Dataset upload command ready, **not run yet** (13 GB, waits on the user).
- **2026-10-08** — ⚠ RULE 3: all-data encoder input at T = 24 (`src/pipeline/all/encoder_input_T24.ipynb`): 5,633,856 windows (train 5,071,584 / test 562,272), 0.62 GB gzipped compact JSON, benign kept ×4, split 90/10 by vehicle reusing the T64 assignment (5,755 vehicles) + seeded for 1,756 new. Reviewer agent found F14 (GridSybil_0709 re-simulation → 2.5–5.9% near-copy leak for its test benign); waiting on the user.
- **2026-10-08** — single-batch test found a bug (length-bucketed batch of 1-row windows → all losses 0 → loss scale 1e-8 → val_joint 2.6e8). Fixed: losses with no applicable windows are skipped (not 0), each loss scale = mean of its first 200 present values, empty batches skip the update (logged `skipped_steps`, `losses_present`), eval averages each loss over batches where present. Re-checked: one-step smoke val_joint 2.61; a typical batch sends gradients to all 72 weight tensors, one AdamW step changes all 72, 25 repeats on the same batch lower every loss (joint 1.85 → 0.61; recon 0.78 → 0.18) at ~2.2 s/step.
- **2026-10-08** — `stage1-input-slides.html` +4 slides after "Encoder input" (multi-agent): a real unlabelled train window as the encoder sees it (64 × 13 heat map, normalised rows, mask), the same window in real units with a map, where the labels live (`_info` + LabelFirewall), and a two-window quiz with an evaluation-only label reveal. Deck now 44 slides.
