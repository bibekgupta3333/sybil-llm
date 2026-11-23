#!/usr/bin/env bash
# mix10_eval.sh <attack> [--dry-run]
#
# One-class evaluation of the finished run src/runs/pretraining/all/T32_mix10_<attack>/model-t32m10-<attack>-seed0
# (best.pt) on the experiment's own data + cache (same test windows as all/T32_benign, normalised with the mixed-train
# statistics), then the random-init control, then executes the experiment notebook into an executed copy in the
# evaluation folder:
#   src/runs/evaluation/all/T32_mix10_<attack>/model-t32m10-<attack>-seed0/               (evaluate.py)
#   src/runs/evaluation/all/T32_mix10_<attack>/model-t32m10-<attack>-seed0-random-init/   (evaluate.py --random-init)
#   .../model-t32m10-<attack>-seed0/timesnet_T32_mix10_<attack>.ipynb                     (executed copy)
# Each evaluation's console goes to eval.log in its folder. Idempotent: an evaluation folder with summary.json is kept
# (skipped); one without summary.json (an interrupted evaluation) is refused: move it away first. The notebook copy
# is re-executed every time. Refuses an unfinished run unless ALLOW_UNFINISHED=1.
# Mixed val: the val label codes are counted first (val only, never a test split); every non-zero code is passed to
# evaluate.py as --allow-val-attack CODE (opt-in option added 2026-11-23), so the alarm thresholds come from the mixed
# val (label-free, as in deployment). The clean all/T32_benign evaluation is unchanged (val must be benign).
# NOTEBOOK_ONLY=1: skip the run checks and both evaluations, only (re-)execute the notebook copy (needs $EVAL_DIR).
# Environment: NICE (default 10), EVAL_DEVICE (default auto), NB_TIMEOUT (per cell, default 3600), DRY_RUN=1.
source "$(dirname "${BASH_SOURCE[0]}")/_mix10_common.sh"
mix10_args "$@"
NOTEBOOK_ONLY=${NOTEBOOK_ONLY:-0}

if [ "$NOTEBOOK_ONLY" = 1 ]; then
  [ -f "$EVAL_DIR/summary.json" ] || { say "skip notebook: no evaluation yet ($EVAL_DIR)"; exit 0; }
elif [ ! -f "$RUN_DIR/best.pt" ]; then
  say "error: $RUN_DIR/best.pt missing (train first: mix10_train.sh $ATTACK)"
  [ "$DRY_RUN" = 1 ] || exit 3
elif ! is_finished "$RUN_DIR"; then
  say "run not finished (stop_reason '$(stop_reason "$RUN_DIR")')"
  [ "${ALLOW_UNFINISHED:-0}" = 1 ] || [ "$DRY_RUN" = 1 ] || { say "set ALLOW_UNFINISHED=1 to evaluate the current best.pt"; exit 3; }
fi

VAL_ARGS=()
if [ "$NOTEBOOK_ONLY" != 1 ] && [ -d "$DATA_DIR/val" ]; then
  say "pre-check: label codes of the val split ($DATA_DIR/val)"
  codes=$("$PY" - "$DATA_DIR/val" <<'EOF'
import collections, json, sys
from pathlib import Path
counts = collections.Counter()
for p in sorted(Path(sys.argv[1]).glob("b*/part-?????_info.json")):
    counts.update(int(w["label"]) for w in json.loads(p.read_text())["windows"])
print("val label counts:", dict(sorted(counts.items())), file=sys.stderr)
print(" ".join(str(c) for c in sorted(counts) if c != 0))
EOF
)
  for c in $codes; do VAL_ARGS+=(--allow-val-attack "$c"); done
  [ ${#VAL_ARGS[@]} = 0 ] || say "mixed val: evaluate.py ${VAL_ARGS[*]} (alarm thresholds from the mixed val)"
fi

# evaluate <out dir> [--random-init]
evaluate() {
  local out=$1; shift
  if [ -f "$out/summary.json" ]; then
    say "kept: $out (summary.json exists)"
    return 0
  fi
  if [ -e "$out" ]; then
    say "error: $out exists without summary.json (interrupted evaluation?); move it away and re-run"
    [ "$DRY_RUN" = 1 ] && return 0 || exit 3
  fi
  local cmd=(nice -n "$NICE" "$PY" -m src.model.all_t32_benign.timesnet.evaluate --run "$RUN_DIR" --data-dir "$DATA_DIR"
    --cache-dir "$CACHE_DIR" --out "$out" --device "${EVAL_DEVICE:-auto}" "${VAL_ARGS[@]}" "$@")
  if [ "$DRY_RUN" = 1 ]; then run "${cmd[@]}"; return 0; fi
  mkdir -p "$EVAL_PARENT"
  local log
  log=$(mktemp "$EVAL_PARENT/.eval-XXXXXX")
  printf '+'; printf ' %q' "${cmd[@]}"; printf '\n'
  set +e
  "${cmd[@]}" 2>&1 | tee -ai "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ -d "$out" ]; then mv "$log" "$out/eval.log"; else mv "$log" "$EVAL_PARENT/$(basename "$out").failed-eval.log"; fi
  [ $rc = 0 ] || { say "error: evaluation failed (exit $rc), log next to $out"; exit $rc; }
}

if [ "$NOTEBOOK_ONLY" != 1 ]; then
  evaluate "$EVAL_DIR"
  evaluate "$EVAL_RI_DIR" --random-init
fi

NB_COPY=$EVAL_DIR/$(basename "$NOTEBOOK")
say "executing $NOTEBOOK -> $NB_COPY"
if [ "$DRY_RUN" = 1 ]; then
  run cp "$NOTEBOOK" "$NB_COPY"
  run env MIX10_SEED="$SEED" MIX10_RUN="$RUN_DIR" MIX10_EVAL="$EVAL_DIR" MIX10_RANDOM_EVAL="$EVAL_RI_DIR" \
    nice -n "$NICE" "$PY" scripts/run_notebook.py "$NB_COPY" --timeout "${NB_TIMEOUT:-3600}"
  exit 0
fi
cp "$NOTEBOOK" "$NB_COPY"
MIX10_SEED="$SEED" MIX10_RUN="$RUN_DIR" MIX10_EVAL="$EVAL_DIR" MIX10_RANDOM_EVAL="$EVAL_RI_DIR" \
  nice -n "$NICE" "$PY" scripts/run_notebook.py "$NB_COPY" --timeout "${NB_TIMEOUT:-3600}"
say "done: $EVAL_DIR/summary.md, $EVAL_RI_DIR/summary.md, $NB_COPY"
