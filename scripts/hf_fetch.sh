#!/usr/bin/env bash
# hf_fetch.sh <repo_path> <local_dir> [<repo_path> <local_dir> ...]
#
# Downloads folders of the private HF model repo (default bibekgupta3333/roadfm-lite-timesnet, override with
# HF_MODEL_REPO) into a temp dir, verifies every file against the folder's SHA256SUMS (scripts/hf_hub.py verify, works
# with sub-folders such as console/ judge/ scores/), then copies the tree into <local_dir>. Refuses a non-empty
# <local_dir> unless FORCE=1. Portable (macOS + Linux). Used by the npm `hf:download:baselines*` / `hf:download:t32b`
# commands; read-only on everything except <local_dir>.
set -euo pipefail
[ $# -ge 2 ] && [ $(($# % 2)) -eq 0 ] || { echo "usage: $0 <repo_path> <local_dir> [...]" >&2; exit 2; }
REPO=${HF_MODEL_REPO:-bibekgupta3333/roadfm-lite-timesnet}
HF=${HF:-hf}
PY=${PY:-.venv-train/bin/python}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
while [ $# -gt 0 ]; do
  src=${1%/} dest=${2%/}; shift 2
  if [ -d "$dest" ] && [ -n "$(ls -A "$dest")" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "skip: $dest exists and is not empty (FORCE=1 to overwrite)" >&2
    continue
  fi
  echo "downloading $REPO/$src ..."
  "$HF" download "$REPO" --repo-type model --include "$src/*" --local-dir "$TMP" >/dev/null
  [ -f "$TMP/$src/SHA256SUMS" ] || { echo "error: $src not found in $REPO (or no SHA256SUMS)" >&2; exit 1; }
  "$PY" scripts/hf_hub.py verify "$TMP/$src"
  mkdir -p "$dest"
  cp -R "$TMP/$src/." "$dest/"
  echo "installed: $dest"
done
