#!/usr/bin/env bash
# Runs inside docker/ec2-sim/compose.yml: makes the container look like a fresh EC2 instance (sudo, an `ubuntu`
# user), copies the repo without data or venvs, runs scripts/ec2_bootstrap.sh, then checks what a user needs next:
# npm, hf, gh (also in a new login shell), both venvs + kernels, the GPU check (CPU path; gpu:require must fail), the npm
# hf / test commands and `setup --native` up to the dataset. Strings passed to as_user contain no `$` (the `sudo -i`
# layer would expand them); intermediate output goes to files instead.
set -euo pipefail
SIM_USER="${SIM_USER:-root}"
SIM_ARGS="${SIM_ARGS:-}"

# What a fresh EC2 Ubuntu AMI already has: sudo, an `ubuntu` user with passwordless sudo.
apt-get update -qq >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq sudo ca-certificates >/dev/null
id ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash ubuntu
echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' >/etc/sudoers.d/90-ubuntu

home="$(getent passwd "$SIM_USER" | cut -d: -f6)"
repo="$home/sybil-llm"
mkdir -p "$repo"
tar -C /src --exclude=./data --exclude=./src/data --exclude=./src/runs --exclude='./.venv*' \
  --exclude='*/node_modules' --exclude=./.git -cf - . | tar -C "$repo" -xf -
chown -R "$SIM_USER:" "$repo"

as_user() { if [[ "$SIM_USER" == root ]]; then bash -lc "$1"; else sudo -iu "$SIM_USER" bash -lc "$1"; fi; }

# shellcheck disable=SC1091
echo "== bootstrap as $SIM_USER ($(. /etc/os-release && echo "$PRETTY_NAME")): bash scripts/ec2_bootstrap.sh $SIM_ARGS =="
as_user "cd $repo && bash scripts/ec2_bootstrap.sh $SIM_ARGS"
echo "== re-run (idempotent) =="
as_user "cd $repo && bash scripts/ec2_bootstrap.sh $SIM_ARGS >/dev/null && echo 're-run exit 0'"

echo "== checks in a NEW login shell (what the user gets after logging in again) =="
fails=0
check() {
  if as_user "cd $repo && $2" >/tmp/out 2>&1; then
    echo "✓ $1: $(tail -1 /tmp/out)"
  else
    echo "✗ $1"
    sed 's/^/    /' /tmp/out | tail -15
    fails=$((fails + 1))
  fi
}
check "npm" "npm -v"
check "hf on PATH" "command -v hf && hf version"
check "hf auth (no token: expected to say not logged in)" "hf auth whoami 2>&1 | head -2; true"
check ".venv-train torch" ".venv-train/bin/python -c 'import torch, huggingface_hub; print(\"torch\", torch.__version__, \"hub\", huggingface_hub.__version__)'"
check "npm run hf:whoami runs (no token)" "npm run -s hf:whoami 2>&1 | tail -2; true"
check "gh on PATH" "command -v gh && gh --version >~/gh.out && head -1 ~/gh.out"
check "setup:venv:check (both venvs ready)" "npm run -s setup:venv:check >~/venv-check.out 2>&1 && tail -2 ~/venv-check.out"
check ".venv imports (pandas, seaborn, weasyprint)" \
  ".venv/bin/python -c 'import pandas, seaborn, weasyprint; print(\"pandas\", pandas.__version__, \"weasyprint\", weasyprint.__version__)'"
check "kernels python3 + roadfm-train" \
  ".venv-train/bin/jupyter kernelspec list >~/kernels.out 2>&1 && grep -qw python3 ~/kernels.out && grep -qw roadfm-train ~/kernels.out && echo 'kernels: python3, roadfm-train'"
check "gpu:check (CPU path, exit 0)" "npm run -s gpu:check >~/gpu-check.out 2>&1 && tail -3 ~/gpu-check.out"
check "gpu:require exits 1 (no GPU in the sim)" \
  "if npm run -s gpu:require >~/gpu-require.out 2>&1; then echo 'gpu:require exited 0 without a GPU'; false; else echo 'gpu:require exit 1 (expected)'; fi"
check "npm test" "npm test >~/npm-test.out 2>&1 && tail -1 ~/npm-test.out"
check "setup:native stops at the missing dataset" \
  "{ npm run -s setup:native >~/setup-native.out 2>&1 || true; } && grep -q 'raw dataset not found' ~/setup-native.out && echo 'stops: raw dataset not found (expected; no data in the sim)'"
check "ec2:check" "npm run -s ec2:check -- $SIM_ARGS >~/ec2-check.out 2>&1 && tail -2 ~/ec2-check.out"
echo "== $fails failed check(s) =="
exit "$fails"
