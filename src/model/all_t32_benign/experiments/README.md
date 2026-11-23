# Contamination experiments mix10 (one-class TimesNet on all/T32 with 10% of one attack in train)

Four runs of the one-class TimesNet of `src/model/all_t32_benign/` (see its `README.md`), each trained on the benign
windows of `all/T32_benign` **mixed with 10% of one attack**, then evaluated one-class exactly like the clean run
`model-t32b-seed0`. Question: how much does a contaminated "benign" training set hurt one-class detection, of the
contaminating attack itself and of the others? Side experiments of the one-class setting; one seed each (not a result
until seeds 1, 2 exist; RULE 1). No model code is changed: the scripts call `timesnet.train` / `timesnet.evaluate`
with other data, cache, run and output paths.

| Experiment | Contaminating attack (code) | Input / cache |
|---|---|---|
| `mix10_gridsybil` | GridSybil (16) | `src/data/encoder_input/all/T32_mix10_gridsybil/` · `src/data/cache/all/T32_mix10_gridsybil/` |
| `mix10_datareplay` | DataReplay (17) | `.../T32_mix10_datareplay/` |
| `mix10_dosrandom` | DoSRandom (18) | `.../T32_mix10_dosrandom/` |
| `mix10_dosdisruptive` | DoSDisruptive (19) | `.../T32_mix10_dosdisruptive/` |

## Spec

- **Train:** the `all/T32_benign` benign train windows + attack windows making up 10% of train. **Val:** mixed the same
  way (it is the label-free check set for checkpoint selection, and the threshold source of the evaluation).
  **Test:** the same splits as `all/T32_benign` (benign `test` + `test_gridsybil` / `test_datareplay` /
  `test_dosrandom` / `test_dosdisruptive` of the test vehicles), same windows. Normalisation refitted on the mixed
  train. Inputs are built by a separate pipeline step (not in this folder); the split is that of `all/T32_benign`.
- **Training:** seed 0, the clean run's config (`T32BenignConfig` preset `joint`, D7 / D12 / D13 / D14, 20 epochs,
  `--no-resource-guard --max-hours 72`); only `--data-dir`, `--cache-dir`, `--runs-dir`, `--run-id` differ (checked:
  a `--check` with these flags gives a config.json equal to the clean run's except paths and the check-mode settings).
- **Evaluation:** `timesnet.evaluate` with the experiment's data + cache (recon / Mahalanobis / kNN k 5 / 10 / 20,
  length-only reference, n ≥ 2 / n ≥ 4 subsets, thresholds from val p95 / p99), plus the `--random-init` control.

## Layout

```
src/runs/pretraining/all/T32_mix10_<attack>/model-t32m10-<attack>-seed0/      run (config, env, train.log, *.jsonl, summary, best.pt, last.pt, launcher.log)
src/runs/pretraining/all/T32_mix10_<attack>/model-t32m10-<attack>-seed0.launcher.log   console of mix10_train.sh (appending)
src/runs/evaluation/all/T32_mix10_<attack>/model-t32m10-<attack>-seed0/       evaluation + eval.log + executed notebook
src/runs/evaluation/all/T32_mix10_<attack>/model-t32m10-<attack>-seed0-random-init/
src/model/all_t32_benign/experiments/mix10_<attack>/timesnet_T32_mix10_<attack>.ipynb   per-experiment notebook (outputs stripped)
src/model/all_t32_benign/experiments/mix10_compare.ipynb                    clean vs the 4 mixes
HF (private bibekgupta3333/roadfm-lite-timesnet): runs/model-t32m10-<attack>-seed0/, evaluations/model-t32m10-<attack>-seed0[-random-init]/
```

## Commands (repo root; `<attack>` = gridsybil | datareplay | dosrandom | dosdisruptive)

| Command | What it does |
|---|---|
| `bash src/model/all_t32_benign/experiments/mix10_train.sh <attack>` | trains into the run folder under `nice -n 10`; resumes from `last.pt` if the run folder exists; a finished run exits with "nothing to do". Run inside tmux. Pause: `touch <run>/PAUSE` (or Ctrl+C once), resume = the same command |
| `bash .../mix10_sync.sh <attack>` | mid-training: snapshot of config / env / train.log / launcher.log / *.jsonl / summary / last.pt → HF `runs/model-t32m10-<attack>-seed0/`, SHA256SUMS + round-trip check; read-only on the run; repeatable (e.g. hourly); refuses a finished run (`FORCE_SYNC=1`) |
| `bash .../mix10_eval.sh <attack>` | evaluation + random-init control into the eval folders (each with `eval.log`), then executes the notebook into `<eval>/timesnet_T32_mix10_<attack>.ipynb`; skips evaluations that already have summary.json; `NOTEBOOK_ONLY=1` re-executes only the notebook; `ALLOW_UNFINISHED=1` evaluates the current best.pt of an unfinished run |
| `bash .../mix10_upload.sh <attack>` | final upload of the run folder (every file, incl. launcher.log) and both eval folders (incl. the executed notebook), each with a fresh SHA256SUMS; each repo folder mirrors the local one; round-trip sha256 check after each |
| `bash scripts/hf_fetch.sh runs/model-t32m10-<attack>-seed0 <run folder> ...` | download (sha256-checked), see `hf:download:mix10` below |

Every script takes `--dry-run` (or `DRY_RUN=1`: prints the commands, runs nothing) and `--help`. Environment: `SEED`
(default 0), `PY` (default `.venv-train/bin/python`), `HF` (default `hf`), `HF_MODEL_REPO`, `NICE` (default 10),
`EVAL_DEVICE` (default auto), `NB_TIMEOUT`. Shared paths and helpers: `_mix10_common.sh`.

The upload uses the `hf` CLI (like `scripts/hf_fetch.sh` and the npm `hf:upload:*` commands) instead of
`src/pipeline/all_t32_benign/hub.py`, whose run / evaluation roots are fixed to `T32_benign`; the SHA256SUMS format is
the same (checked: `write_sums` reproduces hub.py's file of `model-t32b-seed0-random-init` byte for byte), so
`scripts/hf_hub.py verify` and `scripts/hf_fetch.sh` work on the uploaded folders.

## Proposed npm scripts (for `package.json`, not added)

```json
"mix10:train": "bash src/model/all_t32_benign/experiments/mix10_train.sh",
"mix10:sync": "bash src/model/all_t32_benign/experiments/mix10_sync.sh",
"mix10:eval": "bash src/model/all_t32_benign/experiments/mix10_eval.sh",
"mix10:upload": "bash src/model/all_t32_benign/experiments/mix10_upload.sh",
"mix10:notebooks": "for a in gridsybil datareplay dosrandom dosdisruptive; do NOTEBOOK_ONLY=1 bash src/model/all_t32_benign/experiments/mix10_eval.sh $a || exit 1; done && ${PY:-.venv-train/bin/python} scripts/run_notebook.py src/model/all_t32_benign/experiments/mix10_compare.ipynb --timeout 3600",
"hf:download:mix10": "a=${ATTACK:?set ATTACK=gridsybil|datareplay|dosrandom|dosdisruptive}; s=${SEED:-0}; bash scripts/hf_fetch.sh runs/model-t32m10-$a-seed$s src/runs/pretraining/all/T32_mix10_$a/model-t32m10-$a-seed$s evaluations/model-t32m10-$a-seed$s src/runs/evaluation/all/T32_mix10_$a/model-t32m10-$a-seed$s evaluations/model-t32m10-$a-seed$s-random-init src/runs/evaluation/all/T32_mix10_$a/model-t32m10-$a-seed$s-random-init",
"hf:download:mix10:all": "for a in gridsybil datareplay dosrandom dosdisruptive; do ATTACK=$a npm run hf:download:mix10 || true; done"
```

Usage: `npm run mix10:train -- gridsybil`, `npm run mix10:eval -- dosrandom`, `ATTACK=datareplay npm run hf:download:mix10`.
`mix10:notebooks` executes the compare notebook in place (outputs written into the repo file, as `t32b:notebook` does).

## Known issues (read before running the evaluation)

- **Mixed val vs `evaluate.py` (resolved 2026-11-23):** `evaluate.py` by default asserts that every `val` window has
  label 0. It now has an opt-in `--allow-val-attack CODE` (repeatable, default off, recorded in `summary.json`
  `eval_settings.val_attack_codes`; tests in `timesnet/tests/test_evaluate.py`). `mix10_eval.sh` counts the val label
  codes (val only, never a test split) and passes each non-zero code, so the alarm thresholds (p95 / p99) come from the
  mixed val, label-free as in deployment. The clean `all/T32_benign` evaluation is unchanged.
- **Text in env.json:** `train.py` writes the fixed note `data_selection` ("train / val hold benign windows only ...")
  into `env.json`; for these runs it is wrong (train / val are 10% attack). The run's `config.json` `data_dir` names the
  mixed input; the notebooks warn if it does not.
- Val-based numbers (val joint, val recon, thresholds) of a mix run and the clean run come from different val splits;
  only the test AUROCs are on the same windows. The random-init control also differs between runs (each uses its run's
  normalisation and scaling).
- Caveats of the clean run apply: F11 (DataReplay / DoS windows are 95–100% 1–3 rows, length separates them), F14, F1.
