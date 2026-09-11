#!/usr/bin/env bash
# Repo tidiness pass, run automatically after every subagent call (SubagentStop hook).
# Conservative on purpose: only acts on loose files sitting directly at repo root, since
# that's the one place agents have actually dropped stray output in this project. Never
# touches data/ (per CLAUDE.md rule 2) and never overwrites an existing destination file.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MOVED=0
NOTES=()

# 1. Remove stray .DS_Store outside data/ and .venv/ (already gitignored, just keep the
#    working tree clean so `git status` stays readable).
while IFS= read -r -d '' f; do
  rm -f "$f" && MOVED=$((MOVED + 1))
done < <(find . -maxdepth 4 -name '.DS_Store' \
  -not -path './.git/*' -not -path './data/*' -not -path './.venv/*' -print0 2>/dev/null)

# 2. Known root files that belong at the top level — never move these.
KEEP_AT_ROOT=(README.md CLAUDE.md tracker.html requirements.txt .gitignore)
is_kept() {
  local name="$1"
  for k in "${KEEP_AT_ROOT[@]}"; do [[ "$name" == "$k" ]] && return 0; done
  return 1
}

move_one() {
  local src="$1" destdir="$2"
  local base dest
  base="$(basename "$src")"
  mkdir -p "$destdir"
  dest="$destdir/$base"
  if [[ -e "$dest" ]]; then
    NOTES+=("SKIP (destination exists): $src -> $dest")
    return
  fi
  mv "$src" "$dest"
  NOTES+=("moved: $src -> $dest")
  MOVED=$((MOVED + 1))
}

# 3. Sort loose files sitting directly at repo root by extension/name, matching the
#    conventions in CLAUDE.md's project map. Only regular files directly in "." — never
#    recurses, never touches anything already inside a project subfolder.
for f in *; do
  [[ -f "$f" ]] || continue
  is_kept "$f" && continue
  case "$f" in
    *.ipynb) move_one "$f" "notebooks" ;;
    *.png|*.jpg|*.jpeg) move_one "$f" "results/figures/misc" ;;
    *.pt|*.pth) move_one "$f" "models" ;;
    *.md) move_one "$f" "docs/research-notes" ;;
    *.py) move_one "$f" "scripts" ;;
  esac
done

if [[ ${#NOTES[@]} -gt 0 ]]; then
  printf '%s\n' "${NOTES[@]}" >&2
fi

if [[ $MOVED -gt 0 ]]; then
  echo "{\"systemMessage\": \"organize_repo.sh: tidied $MOVED item(s) at repo root\"}"
else
  echo '{}'
fi
