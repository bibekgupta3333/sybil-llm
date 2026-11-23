#!/usr/bin/env bash
# mix10_upload.sh <attack> [--dry-run]
#
# Final upload of one contamination experiment to the private HF model repo $HF_MODEL_REPO
# (default bibekgupta3333/roadfm-lite-timesnet), every file of each folder, each with a fresh SHA256SUMS:
#   src/runs/pretraining/all/T32_mix10_<attack>/model-t32m10-<attack>-seed0/     -> runs/model-t32m10-<attack>-seed0/
#   src/runs/evaluation/all/T32_mix10_<attack>/model-t32m10-<attack>-seed0/      -> evaluations/model-t32m10-<attack>-seed0/
#     (incl. the executed notebook timesnet_T32_mix10_<attack>.ipynb and eval.log)
#   .../model-t32m10-<attack>-seed0-random-init/                                 -> evaluations/model-t32m10-<attack>-seed0-random-init/
# The launcher log (mix10_train.sh console) is copied into the run folder as launcher.log first. Each repo folder
# mirrors its local folder (stale remote files, e.g. from mix10_sync.sh, deleted in the same commit); after each upload
# the folder is downloaded into a temp dir and every file checked against SHA256SUMS (round trip). Missing evaluation
# folders are skipped with a note. Refuses an unfinished run unless ALLOW_UNFINISHED=1 (use mix10_sync.sh mid-run).
# Not hub.py: src/pipeline/all_t32_benign/hub.py has the T32_benign run / evaluation roots built in; this script uses
# the `hf` CLI (as scripts/hf_fetch.sh and the npm hf:upload:* commands) with the same SHA256SUMS format.
source "$(dirname "${BASH_SOURCE[0]}")/_mix10_common.sh"
mix10_args "$@"

if [ ! -f "$RUN_DIR/config.json" ]; then
  say "error: $RUN_DIR is not a run folder (no config.json)"
  [ "$DRY_RUN" = 1 ] || exit 3
elif ! is_finished "$RUN_DIR"; then
  say "run not finished (stop_reason '$(stop_reason "$RUN_DIR")'); mid-run uploads: mix10_sync.sh $ATTACK"
  [ "${ALLOW_UNFINISHED:-0}" = 1 ] || [ "$DRY_RUN" = 1 ] || exit 3
fi
[ "$DRY_RUN" = 1 ] || "$HF" auth whoami >/dev/null || { say "error: not logged in to Hugging Face (hf auth login)"; exit 3; }

if [ -f "$LAUNCH_LOG" ]; then run cp "$LAUNCH_LOG" "$RUN_DIR/launcher.log"; fi
for pair in "$RUN_DIR|$HF_RUN" "$EVAL_DIR|$HF_EVAL" "$EVAL_RI_DIR|$HF_EVAL_RI"; do
  dir=${pair%%|*} prefix=${pair##*|}
  if [ "$dir" != "$RUN_DIR" ] && [ ! -f "$dir/summary.json" ]; then
    say "skip $dir (no evaluation yet)"
    continue
  fi
  if [ "$DRY_RUN" = 1 ]; then echo "+ write_sums $dir"; else write_sums "$dir"; fi
  hf_put_verified "$dir" "$prefix"
done
say "done: https://huggingface.co/$HF_MODEL_REPO/tree/main/$HF_RUN"
