# Documentation index

Every document and page in the repo, one line each. Status: **live** (current, kept up to date) · **reference**
(still valid background) · **proposal-era** (written before the adopted plan) · **legacy** (v1 pilot, history only).
The current project state is in the root [`agent.md`](../agent.md); file paths below are relative to the repo root.

## Start here

| File | Status | What it is |
|---|---|---|
| [`agent.md`](../agent.md) | live | current state: live vs. legacy files, decisions, open questions, next steps, newest changelog |
| [`CLAUDE.md`](../CLAUDE.md) | live | project rules 1–7 and the stable folder map |
| [`README.md`](../README.md) | live | front door: method, repo map, quick start |
| [`docs/changelog.md`](changelog.md) | live | changelog entries older than the newest 10 in `agent.md` |
| [`src/README.md`](../src/README.md) | live | code layout, run order, data format |
| [`src/agent.md`](../src/agent.md) | live | coding rules and per-task workflow for `src/` |

## Plan — `docs/plan/` (live)

| File | What it is |
|---|---|
| [`clarification.md`](plan/clarification.md) | the adopted plan in the professor's format, what changed and why, Q&A |
| [`stage1-ssl-wbs.md`](plan/stage1-ssl-wbs.md) | task list (pretraining S1.x, fine-tuning S2.x), findings F1–F13, decisions D1–D14 in full (source of truth) |
| [`research-plan.md`](plan/research-plan.md) | thesis-wide phases 0–8, decision digest, RULE 3 split record (its intro still describes the v1 method; D9) |
| [`stage1-encoder-notes.md`](plan/stage1-encoder-notes.md) | study notes on the TimesNet encoder: masking, checks, heads, professor Q&A, glossary |
| [`stage1-preprocessing-feature-engineering.md`](plan/stage1-preprocessing-feature-engineering.md) | preprocessing + feature engineering: features, links, crop / pad / mask, outputs, limitations |

## Operations

| File | Status | What it is |
|---|---|---|
| [`docs/ec2-training.md`](ec2-training.md) | live | runbook: Docker image, EC2 GPU setup, private Hugging Face data / model transfers, training commands, costs |
| [`simulation/README.md`](../simulation/README.md) | live | the TypeScript simulator: tabs, sample data (`npm run sim:data`), architecture |

## Pages (open in a browser; links collected in [`index.html`](../index.html))

| File | Status | What it is |
|---|---|---|
| [`tracker.html`](../tracker.html) | live | whole-thesis tracker: phases, alignment table, critical path (localStorage ticks) |
| [`stage1-plan.html`](../stage1-plan.html) | live | per-task tracker for pretraining / fine-tuning, plan review, diagrams (localStorage ticks) |
| [`stage1-input-slides.html`](../stage1-input-slides.html) | live | 44-slide deck on the pretraining input representation, for the professor |
| [`encoder-shapes.html`](../encoder-shapes.html) | live | matrix shapes and parameter counts through the encoder, with animations |
| [`pretraining-explained.html`](../pretraining-explained.html) | live | the pretraining run step by step: data, batches, losses, update, evaluation, checkpoints |
| [`model-all-report.html`](../model-all-report.html) | live (generated) | findings on `model-all` for the professor: pipeline, training health, label-free test scores, frozen probes (when run), diagnosis, fixes; built by `npm run report:model-all` (`scripts/build_model_report.py`) from the run summary, detection manifest, `docs/reports/model-all-findings.json` |
| [`proposal/thesis-proposal-presentation.html`](../proposal/thesis-proposal-presentation.html) | proposal-era | proposal defence slides |

## Data understanding — `docs/research-notes/data_understanding/` (reference)

Raw-data facts are still valid; window and split sections describe the legacy v1 prep (F1 leak). The current input
is described in `docs/plan/stage1-preprocessing-feature-engineering.md`.

| File | What it is |
|---|---|
| [`README.md`](research-notes/data_understanding/README.md) | index of the Phase 1 notes |
| [`veremi-dataset-structure.md`](research-notes/data_understanding/veremi-dataset-structure.md) | folder layout and log record types of VeReMi-Extension |
| [`field-reference.md`](research-notes/data_understanding/field-reference.md) | every field in the BSM / GPS records |
| [`attack-taxonomy.md`](research-notes/data_understanding/attack-taxonomy.md) | the attack codes and what each Sybil attack does |
| [`class-balance.md`](research-notes/data_understanding/class-balance.md) | class counts per scenario |
| [`split-protocol.md`](research-notes/data_understanding/split-protocol.md) | the v1 split protocol (superseded: D8 / D8′) |
| [`cross-scenario-leak.md`](research-notes/data_understanding/cross-scenario-leak.md) | finding F1: the v1 split leaks across scenarios |
| [`data-quality-checks.md`](research-notes/data_understanding/data-quality-checks.md) | data-quality checks on the raw logs |
| [`gridsybil-windowing-defect.md`](research-notes/data_understanding/gridsybil-windowing-defect.md) | GridSybil_0709 pseudonym-1 sentinel and the v1 splice defect |

## Literature and proposal — reference / proposal-era

| File | Status | What it is |
|---|---|---|
| [`research-notes/related-papers.md`](research-notes/related-papers.md) | reference | related-work survey |
| [`research-notes/gap-analysis.md`](research-notes/gap-analysis.md) | reference | research-gap analysis |
| [`research-notes/novelty-analysis.md`](research-notes/novelty-analysis.md) | reference | novelty analysis |
| [`research-notes/introduction-simplified.md`](research-notes/introduction-simplified.md) | proposal-era | plain-language introduction to the project |
| [`research-notes/research-idea-seed.txt`](research-notes/research-idea-seed.txt) | proposal-era | the original research idea |
| [`proposal/proposal-introduction-guide.md`](proposal/proposal-introduction-guide.md) | proposal-era | long-form introduction and dataset guide for the proposal |
| [`planning/wbs.md`](planning/wbs.md) | proposal-era | proposal WBS (not the current task list — that is `plan/stage1-ssl-wbs.md`) |
| [`planning/thesis-format-manual.txt`](planning/thesis-format-manual.txt) · [`planning/example-proposal-template.txt`](planning/example-proposal-template.txt) | reference | university thesis format manual · proposal template |
| [`proposal/main.tex`](../proposal/main.tex) · [`proposal/proposal-draft.md`](../proposal/proposal-draft.md) · [`proposal/references.bib`](../proposal/references.bib) | proposal-era | thesis proposal (background; method superseded, D9) |

## Legacy v1 pilot

| File | What it is |
|---|---|
| [`models/transformer_model.ipynb`](../models/transformer_model.ipynb), `models/results/` | Transformer + MTR/TCP pilot; numbers invalid (F1 leak, F2 TCP no-op) — never quote as results |
| `results/figures/eda/` | v1 EDA figures |
