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
HF (private bibekgupta3333/roadfm-lite-timesnet): runs/model-t32m10-<attack>-seed0/, evaluations/model-t32m10-<attack>-seed0[-random-init]/, judges/model-t32m10-<attack>-seed0/
src/runs/evaluation/all/T32_mix10_<attack>/judge-model-t32m10-<attack>-seed0/      judge report (verdict.md, recompute.json, decompose.json), from npm run hf:download:mix10:judges
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

## npm scripts (in `package.json`)

```json
"mix10:train": "bash src/model/all_t32_benign/experiments/mix10_train.sh",
"mix10:sync": "bash src/model/all_t32_benign/experiments/mix10_sync.sh",
"mix10:eval": "bash src/model/all_t32_benign/experiments/mix10_eval.sh",
"mix10:upload": "bash src/model/all_t32_benign/experiments/mix10_upload.sh",
"mix10:notebooks": "for a in gridsybil datareplay dosrandom dosdisruptive; do NOTEBOOK_ONLY=1 bash src/model/all_t32_benign/experiments/mix10_eval.sh $a || exit 1; done && ${PY:-.venv-train/bin/python} scripts/run_notebook.py src/model/all_t32_benign/experiments/mix10_compare.ipynb --timeout 3600",
"hf:download:mix10": "a=${ATTACK:?set ATTACK=gridsybil|datareplay|dosrandom|dosdisruptive}; s=${SEED:-0}; bash scripts/hf_fetch.sh runs/model-t32m10-$a-seed$s src/runs/pretraining/all/T32_mix10_$a/model-t32m10-$a-seed$s evaluations/model-t32m10-$a-seed$s src/runs/evaluation/all/T32_mix10_$a/model-t32m10-$a-seed$s evaluations/model-t32m10-$a-seed$s-random-init src/runs/evaluation/all/T32_mix10_$a/model-t32m10-$a-seed$s-random-init",
"hf:download:mix10:all": "for a in gridsybil datareplay dosrandom dosdisruptive; do ATTACK=$a npm run hf:download:mix10 || true; done",
"hf:download:mix10:judges": "for a in gridsybil datareplay dosrandom dosdisruptive; do bash scripts/hf_fetch.sh judges/model-t32m10-$a-seed0 src/runs/evaluation/all/T32_mix10_$a/judge-model-t32m10-$a-seed0 || true; done"
```

Usage: `npm run mix10:train -- gridsybil`, `npm run mix10:eval -- dosrandom`, `ATTACK=datareplay npm run hf:download:mix10`.
`mix10:notebooks` executes the compare notebook in place (outputs written into the repo file, as `t32b:notebook` does).

## Results (seed 0, 2026-11-24)

EC2 g4dn T4, 2026-11-23 → 2026-11-24. All four runs: 20 epochs, `stop_reason` max_epochs, LR → 0 at the last update,
0 skipped steps, best epoch 19–20 (chosen on the mixed val, label-free); config.json = the clean run's except
`data_dir` / `cache_dir` / `runs_dir`. Three ran in parallel on the one GPU (~2.2 s / step, ~25 h), DoSDisruptive after
them (10.6 h). Data checked by the judges: train 502,773 benign + 55,864 attack (share 0.1000), val 65,313 + 7,257;
benign train / val ids = `all/T32_benign`'s; no vehicle in two splits; test window ids identical in order to
`all/T32_benign` (`_info` sha256-equal; x differs only by the refitted normalisation). Every evaluation (pretrained,
random-init, and the clean pair) was recomputed by an independent judge on CPU from the saved scores with its own
labels: max |Δ| = 0 for all four, verdict PASS-WITH-NOTES each. Judge reports: HF `judges/model-t32m10-<attack>-seed0`
(`npm run hf:download:mix10:judges`). Side experiment, one seed, not a result (RULE 1).

**The contaminant itself** (test window AUROC, all windows; source: `summary.json` of each evaluation; length only =
−n_messages, the same for every run):

| Contaminant | Score | mix10 pretrained | mix10 random-init | clean pretrained | clean random-init | length only |
|---|---|---|---|---|---|---|
| GridSybil | kNN k5 | **0.778** | 0.774 | 0.886 | 0.839 | 0.559 |
| GridSybil | Mahalanobis | **0.732** | 0.647 | 0.880 | 0.673 | |
| GridSybil | recon | 0.677 | 0.533 | 0.741 | 0.553 | |
| DataReplay | kNN k5 | **0.525** | 0.619 | 0.840 | 0.867 | 0.952 |
| DataReplay | Mahalanobis | **0.852** | 0.882 | 0.946 | 0.944 | |
| DataReplay | recon | 0.449 | 0.463 | 0.486 | 0.466 | |
| DoSRandom | kNN k5 | **0.208** | 0.698 | 0.971 | 0.993 | 0.967 |
| DoSRandom | Mahalanobis | **0.244** | 0.597 | 0.960 | 0.995 | |
| DoSRandom | recon | 0.551 | 0.543 | 0.550 | 0.537 | |
| DoSDisruptive | kNN k5 | **0.444** | 0.563 | 0.859 | 0.903 | 0.967 |
| DoSDisruptive | Mahalanobis | **0.736** | 0.849 | 0.954 | 0.982 | |
| DoSDisruptive | recon | 0.552 | 0.540 | 0.549 | 0.535 | |

Every contaminant loses detectability; DoSRandom falls **below chance** (median kNN-5 distance of test DoSRandom 0.0056
vs benign 0.018: near-duplicates in the bank), DoSDisruptive kNN too. The drop is not a length artefact: the judges
recomputed AUROC inside length bins (n = 1 / 2 / 3 / ≥ 4 real rows) and it appears in every bin (e.g. GridSybil n ≥ 4
kNN 0.897 → 0.791; DoSRandom kNN 0.962 / 0.996 / 0.993 / 1.000 → 0.258 / 0.279 / 0.196 / 0.499). Recall at the val p95
threshold of the contaminant: GridSybil kNN 0.717 → 0.142, DataReplay 0.326 → 0.060, DoSRandom 0.844 → 0.007,
DoSDisruptive 0.41 → 0.04.

**Spill-over to the other attacks** (window AUROC kNN k5 / Mahalanobis; clean pretrained in the last row):

| Contaminant ↓ / test split → | GridSybil | DataReplay | DoSRandom | DoSDisruptive |
|---|---|---|---|---|
| GridSybil | *0.778 / 0.732* | 0.821 / 0.946 | 0.972 / 0.986 | 0.835 / 0.971 |
| DataReplay | 0.874 / 0.875 | *0.525 / 0.852* | 0.809 / 0.845 | **0.434** / 0.794 |
| DoSRandom | 0.886 / 0.875 | 0.807 / 0.910 | *0.208 / 0.244* | 0.720 / 0.824 |
| DoSDisruptive | 0.883 / 0.882 | 0.655 / 0.879 | 0.855 / 0.823 | *0.444 / 0.736* |
| clean (`model-t32b-seed0`) | 0.886 / 0.880 | 0.840 / 0.946 | 0.971 / 0.960 | 0.859 / 0.954 |

GridSybil contamination spills little (≤ 0.03 outside GridSybil, either direction); the short-window attacks spill onto
each other (DataReplay → DoSDisruptive kNN 0.859 → 0.434; DoSDisruptive → DoSRandom 0.971 / 0.960 → 0.855 / 0.823),
consistent with 1-row contaminant windows filling the bank's 1-row region (DoS test windows: 99.7% are 1–3 rows, F11).
GridSybil detection is unchanged by DataReplay / DoS contamination (kNN 0.874–0.886). Executed comparison incl. the
n ≥ 2 subsets: `mix10_compare.ipynb`.

**Decomposition: fit set vs encoder training.** The one-class scorers are fitted on the train split (Mahalanobis mean /
covariance, kNN bank of 200k train windows), so with contamination both the encoder *and* the reference set see the
attack. The GridSybil and DoSDisruptive judges refitted Mahalanobis and kNN on a seeded subsample (seed 12345; 40,000
mixed train windows, or only their benign rows; 15,000 benign test + 10,000 GridSybil / 12,000 per-attack test
windows), for both the mix10 and the clean encoder:

| Contaminant (score) | mix10 encoder, mixed fit | mix10 encoder, benign-only fit | clean encoder, mixed fit | clean encoder, benign-only fit |
|---|---|---|---|---|
| GridSybil (Maha / kNN-5) | 0.730 / 0.763 | **0.887 / 0.878** | 0.724 / 0.739 | 0.879 / 0.881 |
| DoSDisruptive (Maha / kNN-5) | 0.739 / 0.486 | **0.989 / 0.928** | 0.626 / 0.420 | 0.952 / 0.861 |

For GridSybil and DoSDisruptive nearly the whole drop comes from fitting the scorers on the contaminated train; with a
benign-only fit the mix10 encoder is as good as the clean one (GridSybil) or better on this seed (DoSDisruptive, +0.04
/ +0.07; needs more seeds before it is quoted). The DataReplay and DoSRandom judges did not refit; they split the drop
with the random-init control instead (same weights, only the fit set changes): DataReplay kNN-5 total −0.315 = fit set
−0.248 + encoder training −0.067 (Mahalanobis −0.094 = −0.062 − 0.032); DoSRandom: random-init with the mixed fit
already drops 0.993 → 0.698 (kNN-5), the trained encoder drops further to 0.208 — here encoder training adds a large
second drop.

**Caveats.**
- One seed (seeds 1, 2 not run); single-seed differences of ~0.01 are not evidence; the decomposition is a subsample.
- Length alone (0.952 DataReplay, 0.967 DoS) still beats every learned score on DataReplay / DoS, clean and
  contaminated (F11); only GridSybil is detected beyond length.
- **Stale strings:** `summary.md` ("thresholds from benign val") and `summary.json` `protocol.thresholds_from`
  ("benign val") are wrong for these runs: the p95 / p99 thresholds are computed on all (mixed) val windows; likewise
  the recon fill rule text ("median of the defined benign val errors") is the median over the mixed val. Confirmed by
  all four judges. Benign-only val thresholds would move the contaminant's recall at p95 by ≤ 0.05 (GridSybil,
  DataReplay, DoSRandom) and DoSDisruptive Mahalanobis 0.19 → 0.27, so the recall collapse comes from the scores,
  not the threshold source; with the mixed-val threshold the benign test FPR at p95 is 0.036–0.044 (GridSybil /
  DataReplay runs) instead of ~0.05. The `summary.md` header "Like-for-like with task/autoencoder_benign_t32" holds only for the
  clean run.
- Val-based numbers come from different (mixed) val splits; F14 applies (79% of the GridSybil contaminants are
  GridSybil_0709).

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
