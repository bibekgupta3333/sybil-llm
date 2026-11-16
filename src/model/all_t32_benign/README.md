# TimesNet on `all/T32_benign` (one-class pretraining + evaluation, EC2)

TimesNet (d = 128, d_ff = 64, D12; objectives D7; scaling D13; length-bucketed batches D14) pretrained on the
**benign** windows of all 8 scenario folders, T = 32 (`src/data/encoder_input/all/T32_benign`, on-disk 80 / 10 / 10
vehicle split), then scored one-class against each attack split of the test vehicles. Same split and test windows as
`task/autoencoder_benign_t32` (one-class ConvAE), so the two are directly comparable. Plan: `PLAN.md`.

**Setting.** The code never opens a label file during training, but the training data is benign-only, selected with
attack labels: this is one-class (semi-supervised) training, not the label-free pretraining of `model-grid` /
`model-all` (user's choice, 2026-11-15).

**Self-contained.** Every piece of the TimesNet model lives in this package: `timesnet/encoder.py`, `heads.py`,
`views.py`, `losses.py`, `monitors.py`, `base_config.py`, `base_data.py`, `base_train.py` are this package's own copies
of `src/model/benign_gridsybil/timesnet/` (copied 2026-11-15; same math, parameter names, state_dict keys and
random-number order, so existing checkpoints load and seeds give the same numbers). Nothing is imported from
`src.model.benign_gridsybil` (checked by `timesnet/tests/test_isolation.py`); that package stays as it was for the
other runs.

## Layout

| Path | What |
|---|---|
| `src/pipeline/all_t32_benign/cache.py` | JSON feature shards → npy cache (`src/data/cache/all/T32_benign/<split>/x.npy, mask.npy, ids.json, manifest.json`), idempotent, `--verify`; no labels |
| `timesnet/encoder.py` · `heads.py` · `views.py` · `losses.py` · `monitors.py` | TimesNet encoder (embedding, TimesBlock, Inception, FFT periods, masked pooling), `PretrainingModel` + reconstruction / P1–P3 / projection heads, masking / augmentations / injected violations, NT-Xent + `LossScales` + `Objective`, label-free `Evaluator` (package copies, 2026-11-15) |
| `timesnet/base_config.py` · `base_data.py` · `base_train.py` | `PretrainConfig`; feature order, shard readers, `FeatureShards`, `FeatureSpace`, `LabelFirewall`; base `Trainer`, `preflight`, `BatchPrefetcher`, `ResourceGuard` (package copies; the old grid / all dataset registry and command line dropped) |
| `timesnet/config.py` | `T32BenignConfig` (extends `PretrainConfig`: T = 32, cache, on-disk val as check set, `min_train_rows` X3 flag default 1) |
| `timesnet/data.py` | `PretrainFirewall` (train / val features only), `EvalFirewall`, `load_split`, `space_state` / `restore_space` |
| `timesnet/train.py` | `T32Trainer` (subclass of `base_train.Trainer`): LR horizon = real updates (X1), GPU / provenance logging (X9), checkpoints with the feature scaling |
| `timesnet/evaluate.py` · `scoring.py` | frozen encoder → recon error, Mahalanobis (train z), kNN cosine K 5 / 10 / 20 (train bank); thresholds benign val p95 / p99; scores saved before labels are read; length-only baseline, n ≤ 3 / n ≥ 4, per track; `--random-init` control |
| `timesnet/history.py` | per-epoch logging: `history.jsonl` (train vs val for every loss, LR, grad norm, P1–P3 AUROC, representation health), `feature_history.jsonl` (per-feature recon train / val / interpolation baseline), `steps.jsonl` (every update), `summary.json` |
| `timesnet_T32_benign.ipynb` · `timesnet/plots.py` | analysis notebook in the `task/autoencoder_benign_t32` style: input, model, training record (all loss curves), reconstruction per feature, test (scores first, labels after), context, analysis template; `npm run t32b:notebook` (newest run + evaluation; `RUN` / `EVAL` parameters) |
| `timesnet/tests/` | 79 CPU tests (synthetic data; `test_isolation.py`: no import of `src.model.benign_gridsybil`, run with it blocked; `test_model.py`: sizes, shapes, padding invariance) |

Runs: `src/runs/pretraining/all/T32_benign/model-t32b-seed<k>/` · evaluation: `src/runs/evaluation/all/T32_benign/<run>/`
(both gitignored).

## The six commands (SEED=0|1|2, default 0)

| Command | What it does |
|---|---|
| `npm run benign:download` | HF input `all/T32_benign` (private, sha256-checked) → npy cache → verify |
| `npm run benign:train` | full train split, no memory cap (`--no-resource-guard`), up to 72 h; logs everything into the run folder |
| `npm run benign:pause` | writes `PAUSE` in the run folder: the run stops after the current update, evaluates, saves `last.pt` with the partial epoch (Ctrl+C / SIGTERM do the same; a second Ctrl+C interrupts at once) |
| `npm run benign:resume` | continues the paused / stopped run at the next batch (no epoch or step logged twice) |
| `npm run benign:eval` | one-class evaluation + random-init control + executes the notebook |
| `npm run benign:fetch` | download a run + its evaluation folders from HF back to their local places, sha256-checked, never overwrites (`SEED=k`) |
| `npm run benign:upload` | every file of the run folder + its evaluation folders → HF model repo `runs/<run_id>/`, `evaluations/<name>/`, each with SHA256SUMS (`src/pipeline/all_t32_benign/hub.py`) |

Run folder `src/runs/pretraining/all/T32_benign/model-t32b-seed<k>/`: config.json, env.json (versions, git, GPU),
train.log (full console), metrics.jsonl, history.jsonl, feature_history.jsonl, steps.jsonl, summary.json, best.pt,
last.pt. Run `benign:train` inside `tmux` on EC2.

## EC2 runbook (g4dn, after `scripts/ec2_bootstrap.sh` + `npm run setup:native`)

```bash
npm run gpu:require                 # CUDA visible
npm run t32b:data:download          # HF dataset all/T32_benign (private, sha256-checked)
npm run t32b:cache && npm run t32b:cache:verify   # ~10 s on the Mac, ~1 min on 4 vCPUs; 1.9 GB
npm run t32b:test                   # 70 tests
npm run t32b:check                  # preflight, no training
npm run t32b:smoke                  # 30 updates, plumbing only
tmux new -s t32b                    # then, inside tmux:
npm run t32b:train                  # seed 0 (SEED=1 / SEED=2 for others; t32b:train:seeds = 0, 1, 2)
npm run t32b:eval && npm run t32b:eval:random-init   # one-class scores + untrained control
npm run t32b:upload                 # HF model repo runs/model-t32b-seed0 (incl. last.pt for resume)
```

`npm run t32b:ec2` chains gpu check → download → cache → verify → check → train (seed 0) → eval → random-init eval.
Quick complete run on the Mac (2 train shards, 2–3 epochs, ~13 min, plumbing only): `npm run t32b:short && npm run t32b:eval:short && npm run t32b:notebook` (`--short` caps the LR / loss-scale warm-up at 25 updates). Resume after a stop: `SEED=0 npm run t32b:resume`. Download a finished run elsewhere: `SEED=0 npm run t32b:download`.
Commit before the real run so `env.json` records `git_dirty: false` (X9). `t32b:upload` uploads the run folder
(incl. `train.log`) only; copy `src/runs/evaluation/all/T32_benign/<run>/` back from EC2 yourself (e.g. `scp -r`).

## Rules kept

- Model choice is label-free (`val_joint` on benign val); thresholds from benign val; Mahalanobis / kNN fitted on train
  z; the length baseline has a fixed direction (−n rows). Labels are read only after every score file is written.
- Report each AUROC next to the length-only baseline and the random-init control (an untrained TimesNet already scores
  above chance on kNN / Mahalanobis). One seed is not a result: run seeds 0, 1, 2.
- Windows with < 2 real rows have no reconstruction error (nothing to hide); they get the benign val median, so the
  "all" reconstruction AUROC of DataReplay / DoS (70–89% 1-row windows) is fill-dependent. Compare on the `n ≥ 2` /
  `n ≥ 4` subsets; the like-for-like comparison with `task/autoencoder_benign_t32` is GridSybil on n ≥ 4.
- Caveats: F14 (GridSybil_0709 near-copies may straddle the split), F1 (benign traffic repeated per family), F11
  (DataReplay / DoS windows are 95–100% 1–3 rows, so length separates them).
