# agent.md — start here (current state of the project)

> **Every agent reads this first, then root `CLAUDE.md` (rules 1–7).** It says what the project is *now*,
> which files are live vs. legacy, what is decided, and what waits on the user. **Keep it current** (see the
> last section). Last updated: **2026-10-10** (full-GPU (g4dn T4) Ubuntu driver fix; `--dataset grid|all` → `model-grid` / `model-all` + HF upload commands; docs reorganised; D12: d = 128, d_ff = 64; D14 length-bucketed batches; pretraining package ready, no real run yet).

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
| `scripts/hf_hub.py` · `docs/ec2-training.md` | **live** | Hugging Face transfers (user `bibekgupta3333`, **private** repos): dataset `veremi-extension-raw` (raw `data/VeReMi-Dataset`, download restores the same tree, sha256-checked) encoder input `roadfm-lite-encoder-input` (every `src/data/encoder_input/<name>/<T>/` tree at `<name>/<T>/` + `manifests/<name>/<T>.json`; npm `data:upload-input[:all]`, `data:download-input[:all]`, `-- --input all/T24`) and model `roadfm-lite-timesnet` (`runs/<run_id>/`, SHA256SUMS; the two named runs `runs/model-grid`, `runs/model-all`; re-upload of a name replaces its folder in one commit, stale files deleted); npm `hf:whoami`, `data:upload`, `data:download`, `model:upload`, `model:download`, `model:list`, `model:upload:grid` / `:all` (with `last.pt` + `*.log`), `model:download:grid` / `:all`, `hf:upload:grid` / `:all` + `hf:download:grid` / `:all` (official `hf` CLI, same layout; `hf_hub.py sums <run_dir>` before, `verify <dir> --install-to <run_dir>` after), `train:one-batch`; a run's `SHA256SUMS` covers every uploaded file (logs included); EC2 guide §6 |
| `docker/Dockerfile`, `docker-compose.yml`, `requirements/linux.txt`, `scripts/run_notebook.py` | **live** | Ubuntu 24.04 image (Python 3.14.5 via uv, `/opt/venv`, torch 2.14.1 `cu126` / `cpu`, build args `TORCH_VARIANT` / `TORCH_VERSION`; no cu128 wheel of 2.14.1 for py3.14) for EC2; compose services `gpu` / `cpu`, repo bind-mounted at `/workspace`; npm `docker:build`, `docker:build:cpu`, `docker:shell`, `docker:shell:cpu`, `docker:check:cpu`, `docker:train`, `pipeline:prepare`, `pipeline:encoder-input`; npm python commands use `${PY:-.venv-train/bin/python}` (Mac unchanged); guide `docs/ec2-training.md` |
| `scripts/ec2_bootstrap.sh` · `scripts/setup_venv.sh` · `scripts/gpu_check.py` · `docker/ec2-sim/` | **live** | EC2 = the Mac setup, **no Docker on EC2** (user, 2026-10-09). Bootstrap: base tools, Node 22 + npm, uv, `hf`, `gh`, both venvs via `setup_venv.sh`, GPU report; `--allow-root`, `--check`, `--dry-run`, `--install-driver` (on g6f / gr6f vGPU hosts: AWS's GRID driver; `--driver auto|grid|ubuntu`), `--docker`. `setup_venv.sh` (`npm run setup:venv`, `setup:venv:check`): `.venv-train` + `.venv` with uv, pins (macOS `requirements-train.txt`; Linux `requirements/linux.txt` + torch cu126 / cpu), kernels, verify. `gpu_check.py` (`npm run gpu:check`, `gpu:require`). Then `npm run setup:native`. `docker/ec2-sim/` = test-only simulation of a fresh instance (`npm run ec2:sim`, `ec2:sim:ubuntu`) |
| `package.json` (root) | **live** | npm task shortcuts only (Python code; npm is the task runner) — groups `setup`, `pipeline:*`, `train:*` (`train:grid`, `train:grid:check` / `:smoke` / `:no-guard`, same for `train:all`, = `--dataset grid` / `all`; plain `train:*` = grid), `hf:*` / `data:*` / `model:*`, `docker:*`, `format` / `test` / `check`, `sim*`; see the file |
| `scripts/tests/`, `src/tests/` | **live** | pytest suites (CPU, synthetic data): legacy tools + exporters + HF in `scripts/tests/`, the pretraining package in `src/tests/`; run all with `npm test` |
| `src/runs/pretraining/benign_gridsybil/T64/<run_id>/` · `src/runs/pretraining/all/T24/<run_id>/` | live (gitignored) | one folder per training run (config.json, env.json, metrics.jsonl, checkpoints); the full runs are `model-grid` (`--dataset grid`) and `model-all` (`--dataset all`); **`model-all` finished** 2026-10-10 (10 epochs, best.pt + last.pt + train.log; pretraining health only, no detection result), `model-grid` training |
| `.venv-train/`, `requirements-train.txt` · `.venv/`, `requirements.txt` | **live** | pinned PyTorch env (torch 2.14.1, MPS; Jupyter kernel `roadfm-train`) · EDA / notebook env (no torch); Linux / Docker pins in `requirements/linux.txt` |
| `src/README.md` | **live** | what is in `src/`, layout, run order, data format, rules |
| `src/agent.md` | **live** | rules for code in `src/` |
| `stage1-input-slides.html` | **live** | 44-slide deck explaining the pretraining input representation (raw data, design decisions, leakage findings, pipeline + outputs, caveats) for the professor |
| `encoder-shapes.html` | **live** | one-page matrix math of the encoder (input → embedding → TimesBlock → H, z → heads) with every size and parameter count |
| `src/pipeline/all/encoder_input_T24.ipynb` → `src/data/encoder_input/all/T24/` | **live — pretraining input of `--dataset all` (train split only); evaluation on its test split waits on F14** | all 8 folders, all classes, T = 24, compact gzipped JSON (real rows only; reader pads + masks), 5,633,856 windows, 0.62 GB; 90/10 vehicle split reusing T64's; 1–3-row windows: DataReplay 94.7%, DoS 99.7% (F11) |
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
- **g6f / gr6f** EC2 instances have a fractional L4 (NVIDIA vGPU, `g6f.xlarge` = 3 GB GPU memory) → they need AWS's **GRID** driver, not Ubuntu's `nvidia-driver-*` (`bash scripts/ec2_bootstrap.sh --install-driver` handles it).
- **g4dn** (Tesla T4, a full Turing GPU) takes Ubuntu's server driver with **open** prebuilt modules (`linux-modules-nvidia-<ver>-server-open-aws`; on Ubuntu 26.04 the 615 branch has no proprietary `aws` modules), then a **reboot** (the modules package pulls a newer kernel) — `--install-driver` does both picks.
- **F12:** one copy per message (primary listener) gives benign range p50 36 / p99 348 m (all copies: 81 / 384 m).
- A repo hook (`scripts/organize_repo.sh`, after every subagent) moves stray root files; keep-list includes
  `agent.md`, `index.html` and the root HTML pages — add new root files there or they get moved.

## 5. Waiting on the user (do not proceed without an explicit OK)

| Item | Blocks |
|---|---|
| **F14:** fix the GridSybil_0709 cross-family near-copies in the T24 split (trajectory linking + union-find) or document as a limitation | evaluation on the `encoder_input/all/T24` test split (pretraining `--dataset all` reads train only) |
| DataReplay / DoS windows (1–2-message links → receiver time window?) — benign + GridSybil are done | Fine-tuning classes, not S1.2 |
| Optional: download LuST map (`lust.net.xml`) | S1.3.P3 only |
| Optional: download the full 19-attack VeReMi-Extension | original fine-tuning classes only |
| Professor's confirmation of the corrected plan (S1.0.1) | nothing blocks; objections become WBS changes |

## 6. Next steps

1. `model-all` (all/T24) finished: S1.4 frozen-encoder evaluation (probes, transfer) of it can start — on the train split / 1416 only until F14 is decided (`all/T24` test split, §5); read the run with `src/model/benign_gridsybil/timesnet/pretrain_monitor.ipynb`.
2. `model-grid` (benign_gridsybil/T64) is training on the g4dn (tmux); when it finishes: save its `train.log`, `npm run model:upload:grid`, then S1.4 on it. Then few-shot fine-tuning (S2.x, deferred).

Follow `src/agent.md` §5 (per-task workflow) for every task.

## 7. Keeping this file current (every agent, every substantive turn)

- When a task completes, a decision is made, or an approval arrives: update §3–§6 here **and** the WBS /
  `research-plan.md` in the same turn, add a line at the top of the changelog below (keep the newest 10; move the
  oldest to the top of `docs/changelog.md`), and remind the user to tick `stage1-plan.html` and `tracker.html`.
- If a file's status changes (live → legacy, new doc), update §2 and the one-line entry in `docs/README.md`.
- Keep it short: this file is orientation, not a duplicate of the plan. Link instead of copying.

## Changelog

Newest 10 entries; older ones are in [`docs/changelog.md`](docs/changelog.md).

- **2026-10-10** — `model-all` (`--dataset all`, T = 24) finished on the g4dn.xlarge (T4): 10 epochs, 179,420 steps, 13.345 h, stop_reason max_epochs, best_val_joint 0.3773 (epoch 10); `best.pt` + `last.pt` (28.9 MB each), config.json, env.json, metrics.jsonl, `train.log` (console log captured from the tmux scrollback, 1,096 lines) in `src/runs/pretraining/all/T24/model-all/`. Pretraining health only — not a detection result (RULE 1). HF commands updated (multi-agent): `SHA256SUMS` covers every uploaded file incl. logs, verified on download; `hf_hub.py sums` / `verify`; npm `hf:upload:grid|all`, `hf:download:grid|all` (official `hf` CLI) beside `model:upload|download:grid|all`; docs in `docs/ec2-training.md` §5–§6, README. Uploaded: uploaded with `npm run hf:upload:all` (commit 0ec8097), round-trip download verified (sha256 ok for all 6 files). `model-grid` still training (upload after it finishes).
- **2026-10-10** — Ubuntu-driver path for full GPUs fixed in `scripts/ec2_bootstrap.sh` (bug on a g4dn.xlarge T4, Ubuntu 26.04: `--install-driver` picked `nvidia-driver-615-server`, a transitional package, with no prebuilt proprietary `aws` modules → DKMS). Now: newest ≥ 560 non-transitional `nvidia-headless-no-dkms-<ver>-server[-open]` + `nvidia-utils` + prebuilt `linux-modules-nvidia-<ver>-server[-open]-<flavour>` whose `nvidia-kernel-common` requirement accepts the driver version; open modules on Turing+, proprietary for GK/GM/GP/GV (`ROADFM_NVIDIA_MODULES` override); DKMS only without prebuilt modules; refuses to mix with another installed driver version; `--check` shows the pick or `fix: sudo reboot` when the modules are installed for a newer kernel. GRID path unchanged. `gpu_check.py` hint split: full GPU → Ubuntu driver + reboot, vGPU → GRID (test adjusted; `scripts/tests` 104 passed). Verified on the g4dn: `--dry-run --install-driver` → 615 open, modules for 7.0.0-1014-aws, already installed (by hand in the same session) → reboot; `--check` → `fix: sudo reboot`. shellcheck not installed here (not run). Guide: `docs/ec2-training.md` (g4dn / g5 / g6 / p3).
- **2026-10-09** — two named training sets (multi-agent; nothing trained, nothing uploaded): `--dataset grid|all` → fixed run folders `src/runs/pretraining/benign_gridsybil/T64/model-grid` and `src/runs/pretraining/all/T24/model-all`; npm `train:grid[:check|:smoke|:no-guard]`, `train:all[…]`, `model:upload:grid` / `:all` (`--include-last`), `model:download:grid` / `:all`. `hf_hub.py`: re-upload of a run id replaces `runs/<id>/` in one commit (stale files deleted, so it always matches SHA256SUMS), `download --force` drops local `.pt` files the repo run no longer has, a run `README.md` is uploaded when present, model card lists both runs; +4 fake-hub tests (round trip for `model-all` / `model-grid` via the CLI, clean re-upload). README "Two training sets" block (F14 note, all/T24 RAM ≈ 6.3 GB, ≈ 3.5–9 h per epoch → set `--max-epochs`).
- **2026-10-09** — vGPU fix in `scripts/ec2_bootstrap.sh` (bug: `--install-driver` put Ubuntu's `nvidia-driver-595-open` on a g6f.xlarge; its fractional L4 is an NVIDIA vGPU that only NVIDIA's GRID guest driver supports, so `nvidia-smi` failed and torch saw no CUDA). Detection: instance type `g6f.` / `gr6f.` from IMDSv2, or NVIDIA PCI subsystem `0x1733`, or dmesg `NVIDIA vGPU`; override `--driver auto|grid|ubuntu`. On a vGPU `--install-driver` purges Ubuntu's versioned nvidia driver packages, downloads the newest `latest/*grid-aws.run` from AWS's public bucket (size checked against the listing, `sh --check`), installs it `--silent --dkms --no-questions --ui=none`, blacklists nouveau, rebuilds the initramfs (update-initramfs or dracut), loads `nvidia` + `nvidia-uvm` and checks `nvidia-smi` (steps match the sequence that worked by hand on this host); `--check` reports the driver kind + license and fails on "vGPU with non-GRID driver". `gpu_check.py` prints a GRID hint for a vGPU without a driver (+4 tests; `scripts/tests` 100 passed); shellcheck clean; `--check` and `--dry-run` verified on the g6f.xlarge (GRID 595.91.07, L4-3Q, 3 GB, licensed). The install path itself was not run by the script (the driver was installed by hand in the same session). Guide: `docs/ec2-training.md` (g6f / gr6f).
- **2026-10-09** — Colab setup removed (user): .mcp.json, colab/train_colab.ipynb, docs/colab-training.md and their doc mentions; kept: `--runs-dir` train flag and the `data:upload-input` / `data:download-input` HF commands (generic, tested).
- **2026-10-09** — Colab Pro route (multi-agent; Mac MPS measured ~1.6–5 s/step, 40–100 min/epoch in run `20261009-154416-joint`): project `.mcp.json` with `colab-mcp` (Claude Code runs cells in the open Colab tab), `colab/train_colab.ipynb`, `hf_hub.py` `upload-encoder-input` / `download-encoder-input` (private dataset `bibekgupta3333/roadfm-lite-encoder-input`, npm `data:upload-input` / `data:download-input`), train flag `--runs-dir` (runs on Google Drive survive disconnects), guide `docs/colab-training.md`, README quick-start paragraph. No change to data, split, model or losses. Waits on the user: OK for the 1.8 GB upload, commit + push, restart Claude Code.
- **2026-10-09** — EC2 setup mirrors the Mac (multi-agent): `scripts/setup_venv.sh` (both venvs, same pins + kernels, `--check` fails on pin drift; Mac `--check` passes, 92 + 14 pins), `scripts/gpu_check.py` (+18 tests; nvidia-smi, torch CUDA / MPS, matmul smoke; `gpu:require` exits 1 without CUDA), `gh` from GitHub's apt repo, pango libs for weasyprint; bootstrap calls both; `setup.sh --native` re-creates missing venvs. Simulated fresh EC2 (`ec2:sim` root 26.04, `ec2:sim:ubuntu` 24.04): all 14 checks pass in both (hf, gh, venvs, kernels, gpu checks, `npm test` 102 passed, setup:native up to the missing dataset). Mac: `npm run check` 102 passed. GPU path untested (no NVIDIA here).
- **2026-10-09** — EC2 route is **native (no Docker)**, user: "I am not going to use docker compose in ec2". `ec2_bootstrap.sh` now installs `hf` + `.venv-train` by default (Docker only with `--docker`), links `hf` / `uv` into `/usr/local/bin` (before: `hf` missing in new shells), `--allow-root` is a real flag, `sudo -n true` before `sudo -v` (the `ubuntu` user also matches the password `%sudo` rule). `setup.sh --native` (= `npm run setup:native`) runs pipeline + `train:check` with `.venv-train`. Test-only simulation `docker/ec2-sim/` (`npm run ec2:sim` root 26.04, `ec2:sim:ubuntu` 24.04): bootstrap + idempotent re-run, then npm, hf on PATH, torch, `hf:whoami`, `npm test` 84 passed, `setup:native` stops at the missing dataset, `ec2:check` — all pass. GPU parts untested (no NVIDIA here).
- **2026-10-09** — `ec2_bootstrap.sh` fixed after the first EC2 run (user: Ubuntu 26.04, root shell → refused): root / `sudo` runs are now allowed (the sudo caller gets the docker group), Ubuntu 22.04 / 24.04 / 26.04 accepted, Docker repo falls back to `noble` if a release has no repo yet, driver fallback picks the newest packaged server driver ≥ 560. Tested as root in `ubuntu:26.04`: dry-run, real install (Node 22.23, npm 10.9, Docker 29.9, compose 5.6, uv), re-run all exit 0; shellcheck clean.
- **2026-10-09** — `scripts/ec2_bootstrap.sh` (+ npm `ec2:bootstrap`, `ec2:bootstrap:native`, `ec2:check`, `ec2:dry-run`): one script that installs every host dependency on a fresh Ubuntu 24.04 EC2 instance (base tools, Node.js 22 + npm, Docker + buildx + compose, NVIDIA Container Toolkit + runtime on GPU hosts, optional driver / native `.venv-train` / host `hf`), idempotent, `--check` / `--dry-run`. Tested in a clean ubuntu:24.04 container (arm64, non-root + sudo): dry-run, check, real install (Node 22.23, npm 10.9, Docker 29.9 + compose 5.6 + buildx, uv), idempotent re-run, `--native --hf-cli` (.venv-train with torch 2.14.1+cpu, kernels; `npm test` 84 passed); CPU host exits 0; two small fixes (CPU next-commands, hf version row). GPU driver / toolkit install not testable here (repo lines and driver parsing verified). `docs/ec2-training.md` starts with it; README quick start mentions it.
