#!/usr/bin/env bash
# mix10_sync.sh <attack> [--dry-run]
#
# Mid-training sync of one contamination run to HF runs/model-t32m10-<attack>-seed0/ (private $HF_MODEL_REPO): a
# snapshot of the logs, metrics and last.pt (config.json, env.json, train.log, launcher.log, metrics.jsonl,
# history.jsonl, feature_history.jsonl, steps.jsonl, summary.json if present, last.pt; never best.pt or a *.tmp being
# written) is copied into a temp dir first, so the files uploaded together belong together (train.py replaces last.pt
# atomically), then uploaded with its own SHA256SUMS and round-trip verified. Read-only on the run folder; safe to run
# repeatedly and while training runs (e.g. `watch -n 3600` or a cron line). The repo folder mirrors the snapshot, so
# after the run mix10_upload.sh replaces it with the full folder (best.pt, evaluations).
# Download a synced run to resume elsewhere: bash scripts/hf_fetch.sh runs/model-t32m10-<attack>-seed0 <run folder>.
source "$(dirname "${BASH_SOURCE[0]}")/_mix10_common.sh"
mix10_args "$@"

if [ ! -f "$RUN_DIR/config.json" ]; then
  say "error: $RUN_DIR has no config.json yet (training not started)"
  [ "$DRY_RUN" = 1 ] || exit 3
fi
if is_finished "$RUN_DIR" && [ "${FORCE_SYNC:-0}" != 1 ]; then
  say "run finished: use mix10_upload.sh $ATTACK (a sync would replace the full upload with a snapshot; FORCE_SYNC=1 to sync anyway)"
  exit 0
fi
[ "$DRY_RUN" = 1 ] || "$HF" auth whoami >/dev/null || { say "error: not logged in to Hugging Face (hf auth login)"; exit 3; }

SNAP=$(mktemp -d)
trap 'rm -rf "$SNAP"' EXIT
FILES="config.json env.json train.log metrics.jsonl history.jsonl feature_history.jsonl steps.jsonl summary.json last.pt"
for f in $FILES; do
  if [ -f "$RUN_DIR/$f" ]; then
    if [ "$DRY_RUN" = 1 ]; then echo "+ cp $RUN_DIR/$f -> snapshot"; else cp -p "$RUN_DIR/$f" "$SNAP/$f"; fi
  fi
done
if [ -f "$LAUNCH_LOG" ]; then
  if [ "$DRY_RUN" = 1 ]; then echo "+ cp $LAUNCH_LOG -> snapshot/launcher.log"; else cp -p "$LAUNCH_LOG" "$SNAP/launcher.log"; fi
fi
if [ "$DRY_RUN" = 1 ]; then echo "+ write_sums snapshot"; else write_sums "$SNAP"; fi
say "sync of $RUN_DIR (stop_reason so far: '$(stop_reason "$RUN_DIR")')"
hf_put_verified "$SNAP" "$HF_RUN"
