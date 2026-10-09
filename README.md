# RoadFM-Lite

**Self-supervised trajectory representations for Sybil attack detection in VANETs** — master's thesis,
Florida Polytechnic University.

## What it is

A Sybil attacker in a vehicular network broadcasts beacons for several fake identities. RoadFM-Lite learns what
physically plausible, network-heard motion looks like **without attack labels**, then detects Sybil identities with
very few labels. Data: the local [VeReMi-Extension](https://github.com/josephkamel/VeReMi-Dataset) logs
(4 Sybil attacks — GridSybil, DataReplay, DoSRandom, DoSDisruptive — × 2 scenario groups, 0709 / 1416).

**Method (adopted plan, 2026-10-05):**

- **Input:** what a receiving car heard from one sender pseudonym (1 Hz beacons), 13 features per message, windows of
  T = 64 messages padded + masked. Split by sender vehicle, never by window.
- **Self-supervised pretraining:** a TimesNet encoder (4 TimesBlocks, d = 128, d_ff = 64, 2.3 M parameters) trained
  with masked reconstruction, physics heads P1–P3 that detect violations we inject, and SimCLR / InfoNCE contrastive
  learning. No attack labels are read.
- **Few-shot fine-tuning (deferred):** supervised contrastive fine-tuning on n ∈ {10, 20, 30, 50} labels per class,
  DataReplay held out for zero-shot, memory-bank kNN anomaly score; compared with TimesNet trained from scratch.

**Status:** the input pipeline is built for benign + GridSybil (first trial); the pretraining package passes its
checks; no real training run yet. The current state, decisions and open questions are in [`agent.md`](agent.md).
The earlier Transformer + MTR/TCP pilot in `models/` is legacy and its numbers are invalid (split leak, TCP no-op).

## Where to start

| Read | For |
|---|---|
| [`agent.md`](agent.md) | current state, live vs. legacy files, decisions D1–D14, what waits on the user |
| [`docs/README.md`](docs/README.md) | index of every document and page |
| [`index.html`](index.html) | landing page for the explainer pages (open in a browser) |
| [`docs/plan/clarification.md`](docs/plan/clarification.md) · [`docs/plan/stage1-ssl-wbs.md`](docs/plan/stage1-ssl-wbs.md) | the adopted plan · the task list |
| [`src/README.md`](src/README.md) | code layout, run order, data format |
| [`CLAUDE.md`](CLAUDE.md) | project rules (data protection, leakage, reproducibility) |

## Repository map

```
agent.md · CLAUDE.md · index.html     current state · rules · landing page
*.html                                explainer + tracker pages (open via file://)
docs/                                 plan, notes, EC2 runbook, changelog (index: docs/README.md)
src/                                  adopted-plan code: pipeline/ and eda/ notebooks, model/ (TimesNet package), tests/
scripts/                              setup, Hugging Face, notebook runner, simulator exporters, legacy v1 tools, tests/
simulation/                           TypeScript simulator of the windows the encoder reads
notebooks/                            data understanding (EDA, raw maps) + study refreshers
docker/ · docker-compose.yml          Ubuntu 24.04 training image (services gpu / cpu)
models/ · results/                    legacy v1 pilot · v1 EDA figures
proposal/                             thesis proposal (its method is superseded)
data/                                 raw VeReMi-Dataset, 13 GB, gitignored, read-only
```

## Quick start

All commands run from the repo root; `npm` is only a task runner for the Python code (`npm run x -- <arg>` passes
arguments through).

**Mac (two venvs, Python 3.14.5):**

```bash
python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt              # EDA / notebooks (no torch)
python3.14 -m venv .venv-train && .venv-train/bin/pip install -r requirements-train.txt   # training + tests (torch 2.14.1, MPS)
npm run pipeline:prepare && npm run pipeline:encoder-input   # raw data -> src/data/ (needs data/VeReMi-Dataset/)
npm run train:check        # fast correctness checks
npm run train:smoke        # a short run; then npm run train:full
npm test                   # pytest (scripts/tests + src/tests); npm run check = format check + tests
```

**Fresh EC2 instance (Ubuntu 24.04):** `bash scripts/ec2_bootstrap.sh` first (installs Node.js 22 + npm, Docker +
compose, the NVIDIA Container Toolkit when there is a GPU, uv; `--check`, `--dry-run`, `--install-driver`, `--native`),
then `newgrp docker` and the Docker command below.

**Docker / EC2 (one command):** `npm run setup` (NVIDIA GPU) or `npm run setup:cpu` builds the image, runs both
pipeline notebooks if their outputs are missing and runs `train:check`. Full guide:
[`docs/ec2-training.md`](docs/ec2-training.md).

**Data and models:** the raw dataset and training runs move through two **private** Hugging Face repos
(`npm run data:download` / `data:upload`, `model:upload` / `model:download` / `model:list`; log in first with
`hf auth login`). The raw data is never committed.

**Simulator:** `npm run sim:data` (sample from `src/data/`, gitignored) then `npm run sim`; `sim:test`,
`sim:typecheck`, `sim:build`.

## Where outputs go

| Output | Path (gitignored) |
|---|---|
| prepared data · encoder input | `src/data/prepared_data/` · `src/data/encoder_input/<subset>/T<n>/` |
| training runs (config.json, env.json, metrics, checkpoints) | `src/runs/pretraining/benign_gridsybil/T64/<run_id>/` (`npm run train:runs`) |
| simulator sample | `simulation/public/data/` |

`data/` is protected: never write there (rules in `CLAUDE.md`).
