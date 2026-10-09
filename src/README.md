# src — RoadFM-Lite input pipeline

This folder holds the code for the adopted plan: turning raw VeReMi-Extension logs into the input a TimesNet
encoder reads. The code lives inside notebooks. Rules for writing code here are in [`agent.md`](agent.md); the
project's current state is in the root [`agent.md`](../agent.md).

**Current scope:** benign + GridSybil is the **first trial** of the whole pipeline. All attack types come after it
works. Trial-only files sit in `benign_gridsybil/` folders.

## Layout

```
src/
  README.md                            this file
  agent.md                             coding rules for src/
  pipeline/                            notebooks that write data
    input_representation.ipynb         all scenarios: raw logs -> prepared_data/
    benign_gridsybil/
      encoder_input_T64.ipynb          trial: links -> 64 x 13 windows + mask
  model/
    benign_gridsybil/
      encoder_T64.ipynb                trial: the TimesNet encoder explained and checked (imports timesnet/encoder.py)
      timesnet/                        trial: self-supervised pretraining package (config, data, encoder, heads, views, losses,
                                       monitors, train) + pretrain_monitor.ipynb
                                       model: TimesNet only, d = 128, d_ff = 64 (D12; encoder 2,301,312 params)
  eda/                                 read-only analysis notebooks
    eda_window_size.ipynb              all scenarios: link lengths, choice of T
    benign_gridsybil/
      eda_encoder_input_T64.ipynb      trial: EDA of the encoder input
  data/                                outputs (gitignored)
    prepared_data/                     2.9 GB, mirrors data/VeReMi-Dataset
    encoder_input/benign_gridsybil/T64/   1.8 GB, the encoder input
```

## How to run

Data and EDA notebooks use the repo's `.venv`. The model notebook needs PyTorch: use `.venv-train`
(`python -m venv .venv-train && .venv-train/bin/pip install -r requirements-train.txt`) or the Jupyter kernel
`roadfm-train`. The notebooks find the repo root
by themselves, so they can be opened from any folder. Run them in this order:

| # | Notebook | Reads | Writes | Time |
|---|---|---|---|---|
| 1 | `pipeline/input_representation.ipynb` | `data/VeReMi-Dataset/` (read only) | `data/prepared_data/` | ~1 min |
| 2 | `pipeline/benign_gridsybil/encoder_input_T64.ipynb` | `data/prepared_data/` | `data/encoder_input/benign_gridsybil/T64/` | ~2.5 min |
| 3 | `model/benign_gridsybil/encoder_T64.ipynb` (kernel `roadfm-train`) | one train shard | nothing | ~1 min |
| 4 | `.venv-train/bin/python -m src.model.benign_gridsybil.timesnet.train --check` → `--smoke` → no flag | train shards only (10% of train vehicles = check set) | `src/runs/pretraining/benign_gridsybil/T64/<run_id>/` (older runs: `src/runs/stage1/…`) | hours |
| — | `eda/*.ipynb` | the outputs above | nothing | seconds |
| Docker | Linux / EC2: `docker compose build gpu` (or `cpu`), `docker compose run --rm gpu bash`, then inside `npm run pipeline:prepare` → `npm run pipeline:encoder-input` → `npm run train:check` → `train:smoke` → `train:full` (image venv via `PY=/opt/venv/bin/python`; torch 2.14.1 `cu126` for `gpu`, `cpu` for `cpu`; guide `docs/ec2-training.md`) | as rows 1–4 (repo bind-mounted at `/workspace`) | as rows 1–4 | first build ≈ 1.5 min (`cpu`, 2.8 GB image; `gpu` ≈ 10 GB, longer); then as rows 1–4 |

One command (Docker, Ubuntu / EC2 or Mac): `./scripts/setup.sh [--cpu] [--rebuild-data] [--skip-build]` runs the checks, the image build, both pipeline notebooks (if their outputs are missing) and `train:check` (raw data must already be in `data/VeReMi-Dataset/`; see `docs/ec2-training.md`).

Shortcuts from the repo root: `npm run train:check`, `npm run train:smoke`, `npm run train`, `npm run train:resume -- src/runs/pretraining/benign_gridsybil/T64/<run_id>`, `npm run train:runs` (see `package.json`).

Paths in the table are under `src/`, except `data/VeReMi-Dataset/`, which is the protected raw data at the repo
root (never written to).

## What the data is

- **Message:** one beacon a car heard (1 per second), described by **13 features**: claimed position and velocity
  of the sender, the receiver's own position and velocity, claimed acceleration, range, bearing, and log time gap.
  Never inputs: time of day, true sender id, ground truth.
- **Link:** every message one receiving car heard from one sender pseudonym, in time order. Median 12 messages.
- **Window (encoder input):** a link cut into pieces of at most 64 messages, each **padded to 64 rows** with a
  **mask** (1 = real message, 0 = padding). 376,427 windows, 71% padding rows, normalised with train-only statistics.
- **Split:** 90 / 10 train / test by sender vehicle (D8′, seed 0, stratified; 338,001 / 38,426 windows); no vehicle
  appears in both. Pretraining carves a check set of 10% of train vehicles; there is no val split.

Output files in `data/encoder_input/benign_gridsybil/T64/`:

- `<split>/b64/part-XXXXX.json`: `id`, `x` (64 × 13) and `mask` (64). There are no labels here.
- `part-XXXXX_info.json`: labels, `n_messages` and the sender/receiver of each window. Use it for evaluation only.
- `metadata.json`: features, settings, normalisation, counts and provenance.

## Rules that matter here

- Never write under the root `data/` folder (raw data, slow to rebuild).
- No attack labels in pretraining; labels are read only for evaluation.
- The encoder **must use the mask**: average and compute losses over real rows only, and take FFT periods from real
  rows. Never feed the mask or `n_messages` as a feature (window length is a weak shortcut, AUC 0.531).
- Don't change the split silently (root `CLAUDE.md`, RULE 3).

## Next

1. Read the encoder notebook (`model/benign_gridsybil/encoder_T64.ipynb`) and its study notes (`docs/plan/stage1-encoder-notes.md`); nothing is trained yet.
2. Self-supervised pretraining: the package `model/benign_gridsybil/timesnet/` is built and `--check` passes; `--smoke`
   and the full run wait for the user. Read runs with `timesnet/pretrain_monitor.ipynb`.
3. The all-attack version (DataReplay / DoS need a per-receiver time window; their links are 1–2 messages).
