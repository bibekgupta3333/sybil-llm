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
npm run setup:venv         # .venv (EDA, requirements.txt) + .venv-train (training + tests, torch 2.14.1, MPS) + kernels; --check to verify
npm run gpu:check          # which GPU / MPS torch sees
npm run pipeline:prepare && npm run pipeline:encoder-input   # raw data -> src/data/ (needs data/VeReMi-Dataset/)
npm run train:check        # fast correctness checks
npm run train:smoke        # a short run; then npm run train:full
npm test                   # pytest (scripts/tests + src/tests); npm run check = format check + tests
```

**Training commands (examples).** `npm run train -- <options>`; every option is recorded in the run's `config.json`.
Runs go to `src/runs/pretraining/benign_gridsybil/T64/<run_id>/`; watch them with
`src/model/benign_gridsybil/timesnet/pretrain_monitor.ipynb`.

```bash
# Pilot on the Mac: MPS, 16 GB cap, 2 hours, 2 epochs (the LR schedule spans max-epochs, so cap epochs with time)
npm run train -- --device mps --mem-gb 16 --max-hours 2 --max-epochs 2

# Same pilot with a smaller batch (less memory per step, ~2x the steps; 256 is the plan, other sizes are ablations)
npm run train -- --device mps --mem-gb 16 --max-hours 2 --max-epochs 2 --batch-size 128

# Overnight on the Mac: 10 epochs, early stopping still on, named run folder
npm run train -- --device mps --mem-gb 16 --max-epochs 10 --max-hours 18 --run-id mac-10ep

# Full run (plan: batch 256, max 20 epochs, patience 5, 30 h cap); on EC2 the GPU is picked automatically
npm run train:full
npm run train -- --device cuda --max-hours 30         # same, forcing the GPU (EC2: run npm run gpu:require first)

# No memory cap / memory-budget stop (memory is still logged)
npm run train -- --device mps --max-epochs 10 --no-resource-guard

# Reconstruction loss only (sanity-baseline preset; = npm run train:recon-only), or the contrastive weight λ2 from the sweep {0.1, 0.3, 1}
npm run train -- --preset recon_only --max-epochs 10
npm run train -- --lambda-nce 0.1 --max-epochs 10

# Quick checks and a fixed number of updates
npm run train:check                                   # pre-flight checks, no training
npm run train:smoke                                   # one shard, 30 steps: real speed + memory
npm run train -- --smoke --max-steps 5 --batch-size 20 --run-id tiny

# Resume a stopped run (uses its config.json and last.pt), list runs, all options
npm run train:resume -- src/runs/pretraining/benign_gridsybil/T64/<run_id>
npm run train:runs
npm run train:help
```

| Option | Default | Meaning |
|---|---|---|
| `--device` | `auto` (mps → cuda → cpu) | `mps`, `cuda` or `cpu` |
| `--batch-size` | 256 (plan) | windows per step; changing it changes the run (no LR re-scaling), so treat it as an ablation |
| `--max-epochs` · `--max-hours` · `--max-steps` | 20 · 8 (`train:full`: 30) · none | stop at whichever comes first; early stopping (patience 5) can stop sooner |
| `--mem-gb` · `--no-resource-guard` | 22 | memory budget (MPS cap + stop); the guard off = no cap, no stop |
| `--preset` · `--lambda-nce` | `joint` · 0.3 | loss mix: all losses or reconstruction only; contrastive weight λ2 |
| `--run-id` · `--seed` | timestamp + preset · 0 | run folder name; random seed |
| `--check` · `--smoke` · `--resume` | — | pre-flight checks · short real run · continue a run |

**Fresh EC2 instance (Ubuntu 22.04 / 24.04 / 26.04, no Docker):** `bash scripts/ec2_bootstrap.sh` installs
everything the Mac has (Node 22 + npm, uv, `hf`, `gh`, the same two venvs + kernels) and prints a GPU report; then
`gh auth login`, `hf auth login`, `npm run data:download`, `npm run setup:native`, `npm run train:full` in tmux.
Tested locally in Ubuntu containers (`npm run ec2:sim`, `ec2:sim:ubuntu`). Guide: [`docs/ec2-training.md`](docs/ec2-training.md).

**Docker (optional, Mac / CPU tests):** `npm run setup` (NVIDIA GPU) or `npm run setup:cpu` builds the image, runs both
pipeline notebooks if their outputs are missing and runs `train:check`. Full guide:
[`docs/ec2-training.md`](docs/ec2-training.md).

**Data and models:** the raw dataset, the ready encoder input and training runs move through **private** Hugging Face
repos (log in first with `hf auth login`); none of it is committed.

```bash
npm run data:upload / data:download                 # raw data/VeReMi-Dataset/ (13 GB)
npm run data:upload-input:all                       # every tree in src/data/encoder_input/ (benign_gridsybil/T64, all/T24)
npm run data:download-input:all                     # restore them all, sha256-checked (skips trees that already match)
npm run data:download-input -- --input all/T24      # one tree (default benign_gridsybil/T64)
npm run model:upload -- <run_dir> / model:download -- <run_id> / model:list
```

**Simulator:** `npm run sim:data` (sample from `src/data/`, gitignored) then `npm run sim`; `sim:test`,
`sim:typecheck`, `sim:build`.

## Where outputs go

| Output | Path (gitignored) |
|---|---|
| prepared data · encoder input | `src/data/prepared_data/` · `src/data/encoder_input/<subset>/T<n>/` |
| training runs (config.json, env.json, metrics, checkpoints) | `src/runs/pretraining/benign_gridsybil/T64/<run_id>/` (`npm run train:runs`) |
| simulator sample | `simulation/public/data/` |

`data/` is protected: never write there (rules in `CLAUDE.md`).
