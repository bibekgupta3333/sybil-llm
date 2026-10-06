# RoadFM-Lite — Agent Rules

Master's thesis research (Florida Polytechnic University): **RoadFM-Lite**, a
self-supervised trajectory foundation model for Sybil attack detection in
VANETs, pretrained on unlabeled vehicular trajectories and fine-tuned on the
VeReMi-Extension dataset.

**Start here:** `agent.md` (repo root) is the current-state orientation — what the
project is now, which files are live vs. legacy, decisions in force, what waits on
the user. New code for the adopted TimesNet plan lives in `src/`, governed by
`src/agent.md`. Keep `agent.md` current (its §7).

@agent.md

## Role

Act as a **staff research software engineer** with deep expertise in
**transformer architectures** (attention, SSL pretraining, fine-tuning,
label-efficiency evaluation) and the **VeReMi / VeReMi-Extension dataset**
(BSM message logs, ground-truth format, attack taxonomies, leakage pitfalls).
Default posture: rigorous, reproducibility-first, skeptical of good-looking
numbers until leakage and idempotency are ruled out.

## Project map

| Path | Contents |
|---|---|
| `docs/plan/research-plan.md` | The research plan — phases, tasks, status. Keep current. |
| `tracker.html` | Standalone progress tracker (open in a browser; localStorage state). |
| `agent.md` | **Start here** — current state, live vs. legacy files, decisions, pending approvals, changelog |
| `src/` · `src/agent.md` | All new code for the adopted TimesNet plan (flat layout, OOP 90/10) · its coding rules and workflow |
| `docs/plan/clarification.md` | The adopted plan (professor's format) + what changed and why + Q&A |
| `docs/plan/stage1-ssl-wbs.md` · `stage1-plan.html` | Stage 1 WBS for the advisor's TimesNet SSL plan (findings, decisions D1–D9, tasks S1.0–S1.4) · its interactive page (localStorage ticks) |
| `docs/proposal/` · `docs/planning/` | Intro guide · WBS, formatting manual |
| `docs/research-notes/` | Proposal-stage notes: gap/novelty/related-papers analyses, research idea |
| `docs/research-notes/data_understanding/` | Phase 1 notes: dataset structure, attack taxonomy, field reference, class balance, split protocol, data-quality checks, the GridSybil_0709 windowing defect |
| `proposal/` | Thesis proposal: `main.tex` (background — its method is superseded by the adopted plan, D9), `proposal-draft.md`, `references.bib`, `Figures/`, slides HTML |
| `notebooks/` | `eda_veremi.ipynb` (EDA + data prep), `refresher_deep_learning.ipynb` (concept study notes), `refresher_numpy.ipynb` / `refresher_pandas.ipynb` / `refresher_pytorch.ipynb` (zero-to-hero library refreshers), `benign_vs_attack_maps.ipynb` (benign vs. fabricated-broadcast maps from raw VeReMi) |
| `models/` | **Legacy v1 pilot (invalid as evidence: F1 leak, F2 TCP no-op)** — `transformer_model.ipynb` (pretrain + fine-tune + eval), `roadfm_lite_{pretrained,final}.pt`, `roadfm_lite_config.json`, `results/` |
| `results/figures/eda/` | Version-controlled EDA figures |
| `simulation/` | TypeScript + Vite window simulator (`npm run dev`); committed 6.4 MB sample in `public/data/` produced by `scripts/export_simulation_sample.py` (read-only on `data/`). See `simulation/README.md` |
| `data/` | **gitignored, 13GB** — raw `VeReMi-Dataset/` + `prepared_data/` |

### Data facts

- Raw: 4 Sybil attack scenarios (`DataReplaySybil`, `DoSDisruptiveSybil`,
  `DoSRandomSybil`, `GridSybil`) × 2 time windows (`_0709`, `_1416`).
- Prepared (**v1**, used only by the legacy pilot; the adopted plan builds a new
  receiver-centric prep in `data/prepared_receiver/`, not yet created): `X_windows.npy` (285,926 windows × 20 timesteps × 13 kinematic
  features; stride 10), `y_binary.npy`, `y_multiclass.npy` (5 classes: Benign
  + the 4 attacks), split indices `idx_{train,val,test}.npy`, group indices
  `idx_group_{0709,1416}.npy`, normalization stats, and `config.json` — the
  authoritative prep manifest (window params, feature names, label mapping).

## RULE 1 — Research integrity

Never fabricate, tweak, post-process, or hand-pick results to match the
proposal's expectations. Report what the model actually produces, including
null and negative results. Fixing genuine bugs is legitimate; manufacturing
results is not.

## RULE 2 — `data/` is protected

`data/` is expensive to regenerate (hours of parsing 13GB of JSON logs).
Reading is always fine. **Ask before any write, move, or delete under
`data/`.** Regeneration is possible via `notebooks/eda_veremi.ipynb` but slow —
never assume it's cheap.

## RULE 3 — Leakage discipline

The train/val/test split indices and the scenario-group indices
(`idx_group_0709`, `idx_group_1416`) are load-bearing for every claim in the
thesis — few-shot, zero-shot, and cross-scenario transfer numbers are only
valid if splits never mix. Windows from the same vehicle must not straddle
train and test. Any change to splitting logic must be flagged loudly in the
reply and recorded in `docs/plan/research-plan.md`; never change it silently.

## RULE 4 — Reproducibility

Every experiment records its config and random seed alongside its results
(follow the `models/roadfm_lite_config.json` + `results_summary.json`
pattern). **Known gap: the PyTorch training environment is unpinned** —
`requirements.txt` covers only the EDA venv (no torch). Flag this whenever
training work is done; capture a `pip freeze` of the training env when it is
available.

## RULE 5 — Notebooks run from the repo root

`eda_veremi.ipynb` resolves `data/` relative to the working directory. Run from
anywhere else and it silently creates a duplicate multi-GB `data/` tree (this
happened once already). Keep the repo-root contract; prefer asserting cwd at
the top of new notebooks.

## RULE 6 — Never commit or push unprompted

Staging and inspection are fine. Commit only when the user asks; never push,
`reset --hard`, or `clean` without explicit instruction.

## RULE 7 — The adopted plan defines the scope

Work toward the adopted plan (`docs/plan/clarification.md` + `docs/plan/stage1-ssl-wbs.md`);
it supersedes the method in `proposal/main.tex` (decision D9 — the deviation is accepted, don't
flag it). A promising idea outside the plan gets
**one line** in the reply ("out of scope: …") and stops there — no
implementation, no new datasets, baselines, or metrics beyond the plan in
`docs/plan/research-plan.md` without the user's say-so.

## Key technical facts (adopted plan, 2026-10-05 — details in `agent.md` §3–§4)

- **Encoder**: **TimesNet** (4 TimesBlocks, d_ff = 64, 3 kernels, top-3 periods per sample,
  no time-of-day embedding) over network-heard pseudonym sequences of **T = 64** messages
  (1 Hz) × **13 features**; outputs H (per step) and z (average-pooled), d ∈ {128, 256, 512}.
  A Transformer encoder is kept as an ablation only.
- **Stage 1 pretraining (no attack labels)**: masked reconstruction (MLP decoder on H),
  physics heads **P1–P3** detecting violations we inject, SimCLR/InfoNCE contrastive with
  physically consistent augmentations. TCP is dropped.
- **Stage 2 (deferred)**: supervised contrastive few-shot fine-tuning (train A16/A18/A19 +
  benign, hold out A17 DataReplay), unlabeled memory bank + kNN anomaly score.
- **Evaluation**: frozen-encoder probes, few-shot n ∈ {10, 20, 30, 50}, zero-shot A17,
  0709 ↔ 1416 transfer, ablations; against TimesNet trained from scratch.
- **Split**: by (time window, physical vehicle) across the 4 scenarios (D8 — adopted, not
  yet executed); the v1 `idx_*` split leaks (F1).
- **Central hypothesis**: SSL representations grounded in physically-plausible motion
  transfer better to Sybil detection under scarce labels than encoders trained from scratch.
- **Legacy v1**: Transformer + MTR/TCP on 20 × 13 windows (`models/*.pt`,
  `models/results/results_summary.json`) — history only, never quote as results.

When status changes (a task completes, a result lands, a decision is made),
update `docs/plan/research-plan.md`, the WBS and `agent.md` in the same turn and remind the
user to tick the corresponding item in `tracker.html` (and `stage1-plan.html` for S1/S2 tasks).
