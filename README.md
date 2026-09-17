# RoadFM-Lite

**A Self-Supervised Foundation Model for Road-Network-Grounded Trajectory Representation, applied to Sybil Attack Detection in VANETs**

Master's thesis project — Florida Polytechnic University.

## Overview

Vehicular crowdsensing (VCS) systems reward drivers for reporting traffic and environmental data, which makes them a target for **Sybil attacks**: a single adversary forges multiple fake vehicle identities to claim unearned rewards and inject corrupted data. This project develops **RoadFM-Lite**, a self-supervised trajectory foundation model that pretrains on unlabeled vehicular trajectory data — using masked trajectory reconstruction and trajectory-consistency prediction (replay, shuffle, speed-scale, and positional-offset corruptions) — to learn road-network-grounded representations, then fine-tunes on the [VeReMi](https://github.com/josephkamel/VeReMi-Dataset) dataset for Sybil detection under few-shot, zero-shot, and cross-scenario evaluation settings.

Central hypothesis: representations grounded in road-network structure and physically-plausible motion transfer more effectively to Sybil detection — especially with scarce labels — than trajectory-only encoders trained from scratch.

## Repository structure

```
sybil-llm/
├── README.md
├── CLAUDE.md               # Agent rules: staff research SWE persona, research-integrity + leakage rules
├── tracker.html            # Standalone research-progress tracker (open via file://, localStorage-persisted)
├── requirements.txt        # EDA/notebook environment (not the model-training env — see note in file)
├── docs/
│   ├── plan/                # research-plan.md — the 6-phase executable research plan
│   ├── proposal/            # Long-form proposal introduction/dataset guide
│   ├── research-notes/      # Gap analysis, novelty analysis, related-papers survey, simplified intro
│   │   └── data_understanding/  # Phase 1 notes: dataset structure, attack taxonomy, field reference,
│   │                             #   class balance, split protocol, data-quality checks, windowing defect
│   └── planning/            # WBS, thesis formatting manual, example proposal template
├── notebooks/                # EDA notebook, DL/transformer refresher notebook, utility script
├── proposal/                 # Thesis proposal LaTeX source, Markdown draft, references.bib, slides HTML, Figures/
├── models/
│   ├── transformer_model.ipynb   # Pretraining + fine-tuning + evaluation notebook
│   ├── roadfm_lite_pretrained.pt
│   ├── roadfm_lite_final.pt
│   ├── roadfm_lite_config.json
│   └── results/                  # training_loss.png, classification_metrics.png, embeddings_tsne.png, results_summary.json
├── results/
│   └── figures/eda/           # VeReMi EDA plots (moved out of data/, which is gitignored, so they're version-controlled)
└── data/                      # gitignored (13GB) — raw VeReMi-Dataset scenarios + prepared_data/ (windows, splits, norm stats)
```

## Data

`data/` is gitignored and not checked in. It contains:
- `data/VeReMi-Dataset/` — raw simulation traces for 4 attack types (DataReplaySybil, DoSDisruptiveSybil, DoSRandomSybil, GridSybil) across two time windows (`_0709`, `_1416`).
- `data/prepared_data/` — windowed/normalized tensors (`X_windows.npy`, `y_binary.npy`, `y_multiclass.npy`), train/val/test/group split indices, normalization stats, and `config.json` documenting the windowing pipeline (window size, stride, feature columns, label mapping).

`notebooks/eda_veremi.ipynb` regenerates `data/prepared_data/` when run from the **repo root** (it resolves the dataset path relative to cwd) — running it from another working directory will silently create a duplicate `data/` under that directory.

## Environment

`requirements.txt` covers the EDA/notebook environment only. The environment used to train `transformer_model.ipynb` (PyTorch) is currently undocumented — export its `pip freeze` separately if you need to reproduce training.

## Status

Active thesis research. See `docs/planning/wbs.md` for task status and `proposal/main.tex` / `proposal/proposal-draft.md` for the current proposal draft.
