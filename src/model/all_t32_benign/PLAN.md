# TimesNet on `all/T32_benign` — EC2 package plan (2026-11-15)

User request (2026-11-15): pretrain TimesNet on EC2 on `src/data/encoder_input/all/T32_benign` (benign windows of all
8 folders, T = 32, on-disk 80 / 10 / 10 vehicle split), new folders under `src/model/` and `src/pipeline/`, npm scripts
for EC2, **existing setup untouched** (`src/model/benign_gridsybil/`, existing npm scripts, `scripts/`, data).

## Facts this plan rests on

- HF: `bibekgupta3333/roadfm-lite-encoder-input` (dataset, private) holds `all/T32_benign/` (233 files) and
  `manifests/all/T32_benign.json`; verified 2026-11-15: same 233 files as local, sizes equal, 114 LFS files sha256-equal.
  Download on EC2: `npm run data:download-input -- --input all/T32_benign` (existing script).
- Layout: `<split>/b32/part-XXXXX.json` (id, x 32 × 13 normalised, mask) + `part-XXXXX_info.json` (labels);
  splits `train` / `val` / `test` (benign) + `test_gridsybil` / `test_datareplay` / `test_dosrandom` /
  `test_dosdisruptive` (attack windows of the test vehicles). Normalisation = benign train real rows.
- Same split and test vehicles as `task/autoencoder_benign_t32` (one-class ConvAE, GridSybil 0.736) → the TimesNet
  one-class result is directly comparable to it (same test windows).

## Decision to record (flag)

Training data is benign-only, selected with attack labels (one-class / semi-supervised setting). The pretraining code
itself never opens a label file, but the *data choice* uses labels; this differs from the label-free pretraining of
`model-grid` / `model-all`. User's choice (2026-11-15); record in `docs/plan/research-plan.md` and `agent.md`.
No split change (on-disk split reused) → no RULE 3 change.

## Layout (new, nothing existing edited except `package.json` additions and docs)

```
src/pipeline/all_t32_benign/          data pipeline for EC2
  cache.py                            JSON shards -> npy cache (features only) + manifest, verify
src/model/all_t32_benign/timesnet/    pretraining + one-class evaluation
  config.py   data.py   train.py      (encoder / heads / losses / views / monitors / base_* are the package's
  evaluate.py scoring.py              own copies of src/model/benign_gridsybil/timesnet, 2026-11-15; no import)
  tests/                              (README.md one level up)
src/runs/pretraining/all/T32_benign/<run_id>/   gitignored run folders (model-t32b-seed<k>)
src/runs/evaluation/all/T32_benign/<run_id>/    gitignored evaluation output
```

## Fixes over the model-all run (`model-all-report.html`)

- **X1** LR schedule: horizon = real optimizer updates; test that the LR reaches its floor at the last update.
- **X5** model choice / thresholds on the on-disk benign `val` split (no hold-out carved from train).
- **X6** every AUROC next to the length-only baseline and split into n ≤ 3 / n ≥ 4 rows.
- **X7** seeds 0, 1, 2 (npm script loops them).
- **X9** GPU name + peak memory in env.json / metrics.jsonl; git commit + dirty flag recorded.
- **X3** (drop windows with < 2 real rows from the SSL train set) = flag, **default off** (needs the user's OK).

## Evaluation (one-class, labels read only after every score is computed)

Scores per window from the frozen best checkpoint: masked-reconstruction error (deterministic masks, several passes),
Mahalanobis on z (fit on train z), kNN distance to a train memory bank (K ∈ {5, 10, 20}, fixed before results).
Thresholds at benign val p95 / p99. Each `test_<attack>` vs benign `test`: AUROC, PR-AUC, recall at val p95,
benign FPR; length-only baseline; n ≤ 3 / n ≥ 4; per-track max / mean. Saved per-window score files.

## npm scripts (prefix `t32b:`)

`t32b:data:download`, `t32b:cache`, `t32b:check`, `t32b:smoke`, `t32b:train` (seed 0), `t32b:train:seeds` (0, 1, 2),
`t32b:resume`, `t32b:eval`, `t32b:test`, `t32b:upload` / `t32b:download` (HF model repo, `runs/model-t32b-seed<k>`),
`t32b:ec2` (download → cache → check → train seed 0 → eval).

## Work split (agents)

- Agent A: `src/pipeline/all_t32_benign/` + `config.py`, `data.py`, `train.py`, their tests; run `--check` and a
  30-step `--smoke` on the Mac (verification only).
- Agent B: `evaluate.py`, `scoring.py`, their tests (against a tiny random checkpoint).
- Lead: npm scripts, README, docs; then Agent R (independent review) on the whole package.
