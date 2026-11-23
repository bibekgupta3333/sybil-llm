#!/usr/bin/env bash
# mix10_train.sh <attack> [--dry-run]
#
# Trains the one-class TimesNet (same config as the clean run model-t32b-seed0, seed $SEED = 0) on
# src/data/encoder_input/all/T32_mix10_<attack> (cache src/data/cache/all/T32_mix10_<attack>) into
# src/runs/pretraining/all/T32_mix10_<attack>/model-t32m10-<attack>-seed0/ (full train split, --no-resource-guard,
# --max-hours 72), under `nice -n $NICE` (default 10).
# Idempotent: if the run folder has a config.json the run is resumed (`train --resume`: continues from last.pt; a
# finished run exits at once with "nothing to do"). A run folder without config.json is refused (move it away).
# train.py writes the run's console to <run>/train.log itself (appending on resume); this script tees its own console
# (incl. anything before train.py's logger starts) to src/runs/pretraining/all/T32_mix10_<attack>/<run_id>.launcher.log
# (appending), which mix10_sync.sh / mix10_upload.sh copy into the run folder as launcher.log.
# Pause: touch <run>/PAUSE (the run checkpoints after the current update); resume: run this script again.
# Run inside tmux on EC2. DRY_RUN=1 or --dry-run prints the command only.
source "$(dirname "${BASH_SOURCE[0]}")/_mix10_common.sh"
mix10_args "$@"

if [ ! -f "$DATA_DIR/metadata.json" ]; then
  say "error: $DATA_DIR/metadata.json missing (input not built yet)"
  [ "$DRY_RUN" = 1 ] || exit 3
fi
if [ ! -f "$CACHE_DIR/train/manifest.json" ] || [ ! -f "$CACHE_DIR/val/manifest.json" ]; then
  say "warning: no npy cache in $CACHE_DIR (train.py falls back to parsing the JSON shards: slower)"
fi

ARGS=(--no-resource-guard --max-hours 72)
if [ -f "$RUN_DIR/config.json" ]; then
  if is_finished "$RUN_DIR"; then
    say "run already finished ($(stop_reason "$RUN_DIR")): $RUN_DIR; nothing to do"
    exit 0
  fi
  say "resuming $RUN_DIR (stop_reason so far: '$(stop_reason "$RUN_DIR")')"
  CMD=("$PY" -m src.model.all_t32_benign.timesnet.train --resume "$RUN_DIR" "${ARGS[@]}")
elif [ -e "$RUN_DIR" ]; then
  say "error: $RUN_DIR exists but has no config.json (not a run folder); move it away first"
  exit 3
else
  say "new run $RUN_DIR"
  CMD=("$PY" -m src.model.all_t32_benign.timesnet.train --seed "$SEED" --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR"
    --runs-dir "$RUNS_PARENT" --run-id "$RUN_ID" "${ARGS[@]}")
fi
[ -f "$RUN_DIR/PAUSE" ] && { say "removing the old PAUSE flag"; run rm -f "$RUN_DIR/PAUSE"; }

if [ "$DRY_RUN" = 1 ]; then
  run nice -n "$NICE" "${CMD[@]}"
  echo "+ (tee -a $LAUNCH_LOG)"
  exit 0
fi
mkdir -p "$RUNS_PARENT"
{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) mix10_train.sh $ATTACK (seed $SEED, host $(hostname), git $(git rev-parse --short HEAD 2>/dev/null || echo ?))"
  printf '+'; printf ' %q' nice -n "$NICE" "${CMD[@]}"; printf '\n'
} | tee -a "$LAUNCH_LOG"
set +e
nice -n "$NICE" "${CMD[@]}" 2>&1 | tee -ai "$LAUNCH_LOG"  # -i: Ctrl+C reaches train.py (graceful pause), not tee
rc=${PIPESTATUS[0]}
set -e
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) exit $rc; stop_reason '$(stop_reason "$RUN_DIR")'" | tee -a "$LAUNCH_LOG"
exit "$rc"
