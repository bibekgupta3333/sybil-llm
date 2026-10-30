# src — RoadFM-Lite code for the adopted plan

This folder holds the code for the adopted plan: turning raw VeReMi-Extension logs into the input a TimesNet
encoder reads (notebooks), and the self-supervised pretraining package (`model/benign_gridsybil/timesnet/`). Rules for writing code here are in [`agent.md`](agent.md); the
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
      encoder_input_T48.ipynb          same, T = 48 (crop stride 48, bucket 48)
      encoder_input_T46.ipynb          same, T = 46 (crop stride 46, bucket 46)
      encoder_input_T48_tvt.ipynb      T = 48 with train / val / test = 80 / 10 / 10 (test = T48's test)
      encoder_input_T48_tvt_s24.ipynb  T48_tvt with 50% overlap (crop stride 24 + end-aligned last window), same split
      encoder_input_T48_benign.ipynb   T48_tvt benign only (one-class training; test_gridsybil = test-vehicle attacks)
    all/
      encoder_input_T24.ipynb          all attacks, T = 24 (training set `--dataset all`; F14 open for its test split)
      encoder_input_T32_benign.ipynb   all 8 folders, benign only, T = 32, train / val / test 80 / 10 / 10 by vehicle (RULE 3)
                                       + test_<attack> splits of the test vehicles (one-class training)
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
  tests/                               pytest for the timesnet package (synthetic tensors, CPU)
  data/                                outputs (gitignored)
    prepared_data/                     2.9 GB, mirrors data/VeReMi-Dataset
    encoder_input/benign_gridsybil/T64/   1.8 GB, the encoder input
    encoder_input/benign_gridsybil/T48/   1.5 GB, same messages + split at T = 48 (395,851 windows, 63.3% padding)
    encoder_input/benign_gridsybil/T46/   1.5 GB, same messages + split at T = 46 (399,362 windows, 62.0% padding)
    encoder_input/benign_gridsybil/T48_tvt/  1.5 GB, T = 48, train / val / test 80 / 10 / 10 by vehicle (316,050 / 39,332 / 40,469 windows; RULE 3)
    encoder_input/benign_gridsybil/T48_tvt_s24/  1.7 GB, T48_tvt with 50% overlap (331,215 / 41,440 / 42,653 windows, 55.6% padding; same messages, split, normalisation)
    encoder_input/benign_gridsybil/T48_benign/   0.66 GB, T48_tvt benign only (112,890 / 13,584 / 13,405 windows + test_gridsybil 27,064; normalisation from benign train; no RULE 3 change, subset of T48_tvt's split)
    encoder_input/all/T24/             0.6 GB, all-attack input (F14 open)
    encoder_input/all/T32_benign/      3.3 GB JSON, benign only from all 8 folders (502,773 / 65,313 / 60,142 windows) + test_gridsybil /
                                       test_datareplay / test_dosrandom / test_dosdisruptive (28,087 / 76,283 / 199,834 / 199,826; RULE 3, F14 limitation)
  runs/                                training runs (gitignored)
    pretraining/benign_gridsybil/T64/model-grid/ `--dataset grid`: config.json, env.json, metrics.jsonl, checkpoints
    pretraining/all/T24/model-all/               `--dataset all`, same files (other run ids sit next to them)
```

## How to run

Data and EDA notebooks use the repo's `.venv`. The model notebook needs PyTorch: use `.venv-train`
(`python -m venv .venv-train && .venv-train/bin/pip install -r requirements-train.txt`) or the Jupyter kernel
`roadfm-train`. The notebooks find the repo root
by themselves, so they can be opened from any folder. Run them in this order:

| # | Notebook | Reads | Writes | Time |
|---|---|---|---|---|
| 1 | `pipeline/input_representation.ipynb` (`npm run pipeline:prepare`) | `data/VeReMi-Dataset/` (repo root, read only) | `src/data/prepared_data/` | ~1 min |
| 2 | `pipeline/benign_gridsybil/encoder_input_T64.ipynb` (`npm run pipeline:encoder-input`) | `src/data/prepared_data/` | `src/data/encoder_input/benign_gridsybil/T64/` | ~2.5 min |
| 2t | `pipeline/benign_gridsybil/encoder_input_T48.ipynb` (`npm run pipeline:encoder-input:t48`) | `src/data/prepared_data/` | `src/data/encoder_input/benign_gridsybil/T48/` | ~2 min |
| 2u | `pipeline/benign_gridsybil/encoder_input_T46.ipynb` (`npm run pipeline:encoder-input:t46`) | `src/data/prepared_data/` | `src/data/encoder_input/benign_gridsybil/T46/` | ~2 min |
| 2v | `pipeline/benign_gridsybil/encoder_input_T48_tvt.ipynb` (`npm run pipeline:encoder-input:t48-tvt`) | `src/data/prepared_data/` | `src/data/encoder_input/benign_gridsybil/T48_tvt/` (train / val / test) | ~2 min |
| 2w | `pipeline/benign_gridsybil/encoder_input_T48_tvt_s24.ipynb` (`npm run pipeline:encoder-input:t48-tvt-s24`) | `src/data/prepared_data/` | `src/data/encoder_input/benign_gridsybil/T48_tvt_s24/` (train / val / test, 50% overlap) | ~2 min |
| 2x | `pipeline/benign_gridsybil/encoder_input_T48_benign.ipynb` (`npm run pipeline:encoder-input:t48-benign`) | `src/data/prepared_data/` (+ `T48_tvt/metadata.json`, normalisation comparison only) | `src/data/encoder_input/benign_gridsybil/T48_benign/` (train / val / test benign, test_gridsybil) | ~2 min |
| 2b | `pipeline/all/encoder_input_T24.ipynb` (input of `--dataset all`) | `src/data/prepared_data/` | `src/data/encoder_input/all/T24/` | minutes |
| 2c | `pipeline/all/encoder_input_T32_benign.ipynb` (`npm run pipeline:encoder-input:all-t32-benign`) | `src/data/prepared_data/` | `src/data/encoder_input/all/T32_benign/` (train / val / test benign + 4 test_<attack> splits) | ~1 min |
| 3 | `model/benign_gridsybil/encoder_T64.ipynb` (kernel `roadfm-train`) | one train shard | nothing | ~1 min |
| 4 | `npm run train:check` → `train:smoke` → `train:full` (= `.venv-train/bin/python -m src.model.benign_gridsybil.timesnet.train …`) | train shards only (10% of train vehicles = check set) | `src/runs/pretraining/benign_gridsybil/T64/<run_id>/` | hours |
| 4g | `npm run train:grid:check` → `train:grid:smoke` → `train:grid` (`--dataset grid`, 30 h cap) → `model:upload:grid` | `encoder_input/benign_gridsybil/T64/` train shards | `src/runs/pretraining/benign_gridsybil/T64/model-grid/` → HF `runs/model-grid` | hours |
| 4a | `npm run train:all:check` → `train:all:smoke` → `train:all -- --max-epochs 3` (`--dataset all`) → `model:upload:all` | `encoder_input/all/T24/` train shards (≈ 6.3 GB in RAM; minutes to load) | `src/runs/pretraining/all/T24/model-all/` → HF `runs/model-all` | ≈ 3.5–9 h per epoch (CUDA) |
| — | `eda/*.ipynb` | the outputs above | nothing | seconds |
| Docker | Linux / EC2 / Mac with Docker: `npm run setup` (GPU) or `npm run setup:cpu` — builds the image and runs rows 1, 2 and the check; full runbook in [`docs/ec2-training.md`](../docs/ec2-training.md) | as rows 1–4 (repo bind-mounted at `/workspace`) | as rows 1–4 | image build ≈ 1.5 min (`cpu`) |

Rows 4g / 4a write one fixed folder each (`model-grid`, `model-all`); if it exists the trainer stops — continue with `npm run train:resume -- <folder>` or pass `--run-id model-all-2`. Download a run back with `npm run model:download:grid` / `model:download:all`. For `all`, F14 is open: evaluation on its test split waits on the F14 decision (pretraining reads train only). Details: root [`README.md`](../README.md) "Two training sets".

More shortcuts from the repo root: `npm run train:resume -- src/runs/pretraining/benign_gridsybil/T64/<run_id>`, `npm run train:runs`, `npm test` (pytest for `scripts/tests` + `src/tests`), `npm run check` (format check + tests); all in `package.json`.

Notebook paths in the table are under `src/`; data paths are written in full. `data/VeReMi-Dataset/` is the
protected raw data at the repo root (never written to).

## What the data is

- **Message:** one beacon a car heard (1 per second), described by **13 features**: claimed position and velocity
  of the sender, the receiver's own position and velocity, claimed acceleration, range, bearing, and log time gap.
  Never inputs: time of day, true sender id, ground truth.
- **Link:** every message one receiving car heard from one sender pseudonym, in time order. Median 12 messages.
- **Window (encoder input):** a link cut into pieces of at most 64 messages, each **padded to 64 rows** with a
  **mask** (1 = real message, 0 = padding). 376,427 windows, 71% padding rows, normalised with train-only statistics.
- **Split:** 90 / 10 train / test by sender vehicle (D8′, seed 0, stratified; 338,001 / 38,426 windows); no vehicle
  appears in both. Pretraining carves a check set of 10% of train vehicles; there is no val split.

Output files in `src/data/encoder_input/benign_gridsybil/T64/`:

- `<split>/b64/part-XXXXX.json`: `id`, `x` (64 × 13) and `mask` (64). There are no labels here.
- `part-XXXXX_info.json`: labels, `n_messages` and the sender/receiver of each window. Use it for evaluation only.
- `metadata.json`: features, settings, normalisation, counts and provenance.

## Rules that matter here

- Never write under the root `data/` folder (raw data, slow to rebuild); outputs go to `src/data/` and `src/runs/`.
- No attack labels in pretraining; labels are read only for evaluation.
- The encoder **must use the mask**: average and compute losses over real rows only, and take FFT periods from real
  rows. Never feed the mask or `n_messages` as a feature (window length is a weak shortcut, AUC 0.531).
- Don't change the split silently (root `CLAUDE.md`, RULE 3).

## Next

1. Read the encoder notebook (`model/benign_gridsybil/encoder_T64.ipynb`) and its study notes (`docs/plan/stage1-encoder-notes.md`); nothing is trained yet.
2. Self-supervised pretraining: the package `model/benign_gridsybil/timesnet/` is built and `--check` passes; `--smoke`
   and the full run wait for the user. Read runs with `timesnet/pretrain_monitor.ipynb`.
3. The all-attack version (DataReplay / DoS need a per-receiver time window; their links are 1–2 messages).
