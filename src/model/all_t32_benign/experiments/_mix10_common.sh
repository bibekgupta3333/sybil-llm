# Sourced by the mix10_*.sh scripts (not run directly): paths of one contamination experiment + small helpers.
#
# mix10_setup <attack> sets, for attack in gridsybil | datareplay | dosrandom | dosdisruptive and SEED (default 0):
#   DATA_DIR    src/data/encoder_input/all/T32_mix10_<attack>
#   CACHE_DIR   src/data/cache/all/T32_mix10_<attack>
#   RUNS_PARENT src/runs/pretraining/all/T32_mix10_<attack>        RUN_ID model-t32m10-<attack>-seed<SEED>
#   RUN_DIR     $RUNS_PARENT/$RUN_ID
#   EVAL_DIR    src/runs/evaluation/all/T32_mix10_<attack>/$RUN_ID   EVAL_RI_DIR $EVAL_DIR-random-init
#   NOTEBOOK    src/model/all_t32_benign/experiments/mix10_<attack>/timesnet_T32_mix10_<attack>.ipynb
#   LAUNCH_LOG  $RUNS_PARENT/$RUN_ID.launcher.log (full console of mix10_train.sh; copied into the run folder as
#               launcher.log by mix10_sync.sh / mix10_upload.sh; train.py itself writes $RUN_DIR/train.log)
# Environment: PY (default .venv-train/bin/python), HF (default hf), HF_MODEL_REPO (default
# bibekgupta3333/roadfm-lite-timesnet), SEED (default 0), NICE (default 10), DRY_RUN=1 (print commands, run nothing).

set -euo pipefail

MIX10_ATTACKS="gridsybil datareplay dosrandom dosdisruptive"
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "$REPO_ROOT"
[ -f CLAUDE.md ] && [ -d src/model/all_t32_benign ] || { echo "error: repo root not found ($REPO_ROOT)" >&2; exit 2; }

PY=${PY:-.venv-train/bin/python}
HF=${HF:-hf}
HF_MODEL_REPO=${HF_MODEL_REPO:-bibekgupta3333/roadfm-lite-timesnet}
SEED=${SEED:-0}
NICE=${NICE:-10}
DRY_RUN=${DRY_RUN:-0}
FINISHED_REASONS="max_epochs early_stopping"

# Parses "<attack> [--dry-run]" from the script arguments; sets ATTACK and DRY_RUN.
mix10_args() {
  ATTACK=""
  for arg in "$@"; do
    case "$arg" in
      --dry-run) DRY_RUN=1 ;;
      -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
      -*) echo "error: unknown option $arg" >&2; exit 2 ;;
      *) [ -z "$ATTACK" ] || { echo "error: one attack only" >&2; exit 2; }; ATTACK=$arg ;;
    esac
  done
  [ -n "$ATTACK" ] || { echo "usage: $(basename "$0") <$(echo $MIX10_ATTACKS | tr ' ' '|')> [--dry-run]" >&2; exit 2; }
  mix10_setup "$ATTACK"
}

mix10_setup() {
  local a=$1 ok=0 x
  for x in $MIX10_ATTACKS; do [ "$x" = "$a" ] && ok=1; done
  [ $ok = 1 ] || { echo "error: attack must be one of: $MIX10_ATTACKS (got '$a')" >&2; exit 2; }
  [[ "$SEED" =~ ^[0-9]+$ ]] || { echo "error: SEED must be an integer (got '$SEED')" >&2; exit 2; }
  DATA_DIR=src/data/encoder_input/all/T32_mix10_$a
  CACHE_DIR=src/data/cache/all/T32_mix10_$a
  RUNS_PARENT=src/runs/pretraining/all/T32_mix10_$a
  RUN_ID=model-t32m10-$a-seed$SEED
  RUN_DIR=$RUNS_PARENT/$RUN_ID
  EVAL_PARENT=src/runs/evaluation/all/T32_mix10_$a
  EVAL_DIR=$EVAL_PARENT/$RUN_ID
  EVAL_RI_DIR=$EVAL_DIR-random-init
  NOTEBOOK=src/model/all_t32_benign/experiments/mix10_$a/timesnet_T32_mix10_$a.ipynb
  LAUNCH_LOG=$RUNS_PARENT/$RUN_ID.launcher.log
  HF_RUN=runs/$RUN_ID
  HF_EVAL=evaluations/$RUN_ID
  HF_EVAL_RI=evaluations/$RUN_ID-random-init
}

say() { echo "[mix10 ${ATTACK:-?}] $*"; }

# Prints the command; runs it unless DRY_RUN=1.
run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  [ "$DRY_RUN" = 1 ] || "$@"
}

# stop_reason of a run's summary.json ("" if none).
stop_reason() {
  [ -f "$1/summary.json" ] || { echo ""; return 0; }
  "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("stop_reason") or "")' "$1/summary.json"
}

is_finished() {
  local r; r=$(stop_reason "$1")
  [ -n "$r" ] && [[ " $FINISHED_REASONS " == *" $r "* ]]
}

# Writes <dir>/SHA256SUMS: every regular file below <dir> (relative paths), except SHA256SUMS, PAUSE, *.tmp.
write_sums() {
  "$PY" - "$1" <<'EOF'
import hashlib, sys
from pathlib import Path
root = Path(sys.argv[1])
lines = []
for p in sorted(root.rglob("*")):
    if not p.is_file() or p.name in ("SHA256SUMS", "PAUSE") or p.suffix == ".tmp" or ".cache" in p.parts:
        continue
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    lines.append(f"{h.hexdigest()}  {p.relative_to(root).as_posix()}\n")
(root / "SHA256SUMS").write_text("".join(lines))
print(f"SHA256SUMS: {len(lines)} files in {root}")
EOF
}

# Uploads <local dir> to <repo path> (the repo folder mirrors the local one: stale remote files deleted in the same
# commit), then downloads it again into a temp dir and checks every file against SHA256SUMS + the local SHA256SUMS.
hf_put_verified() {
  local dir=$1 prefix=$2 tmp
  run "$HF" upload "$HF_MODEL_REPO" "$dir" "$prefix" --repo-type model --private \
    --exclude PAUSE --exclude '*.tmp' --delete '*' --commit-message "Upload $prefix (mix10)"
  [ "$DRY_RUN" = 1 ] && { echo "+ (dry run) round-trip sha256 check of $prefix"; return 0; }
  tmp=$(mktemp -d)
  "$HF" download "$HF_MODEL_REPO" --repo-type model --include "$prefix/*" --local-dir "$tmp" >/dev/null
  "$PY" scripts/hf_hub.py verify "$tmp/$prefix"
  if ! cmp -s "$tmp/$prefix/SHA256SUMS" "$dir/SHA256SUMS"; then
    rm -rf "$tmp"; echo "error: remote SHA256SUMS of $prefix differs from $dir/SHA256SUMS" >&2; return 1
  fi
  local n_remote n_local
  n_remote=$(cd "$tmp/$prefix" && find . -type f ! -path './.cache/*' ! -name SHA256SUMS | wc -l)
  n_local=$(wc -l < "$dir/SHA256SUMS")
  rm -rf "$tmp"
  [ "$n_remote" -eq "$n_local" ] || { echo "error: $prefix has $n_remote files on HF, $n_local in SHA256SUMS" >&2; return 1; }
  say "round trip OK: $prefix ($n_local files, sha256 equal)"
}
