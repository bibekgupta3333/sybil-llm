#!/usr/bin/env bash
# One-command setup for RoadFM-Lite on Ubuntu (EC2, NVIDIA) or a Mac with Docker. Run from the repo root:
#
#   ./scripts/setup.sh [--native] [--cpu] [--rebuild-data] [--skip-build]
#
#   --native        no Docker: run everything with .venv-train/bin/python on this machine (EC2 after
#                   `bash scripts/ec2_bootstrap.sh`); completes the venvs with scripts/setup_venv.sh when its
#                   --check fails, prints the GPU check; the GPU is used when torch sees CUDA
#   --cpu           use the `cpu` compose service (default: `gpu`, needs an NVIDIA GPU + container toolkit)
#   --rebuild-data  re-run both pipeline notebooks even if their outputs exist
#   --skip-build    do not run `docker compose build` (the image must already exist)
#
# Idempotent: existing prepared data and encoder input are kept unless --rebuild-data is given. The raw dataset is
# never downloaded here (and never written to): it must already be in data/VeReMi-Dataset/ (`npm run data:download`).
# Guide: docs/ec2-training.md.
set -euo pipefail

SERVICE="gpu"
NATIVE=0
REBUILD_DATA=0
SKIP_BUILD=0
for arg in "$@"; do
  case "$arg" in
    --native) NATIVE=1 ;;
    --cpu) SERVICE="cpu" ;;
    --rebuild-data) REBUILD_DATA=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    -h | --help)
      sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown option: $arg (see --help)" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RAW_DIR="data/VeReMi-Dataset"
EXPECTED_RAW_FILES=23048
PREPARED_INDEX="src/data/prepared_data/index.json"
ENCODER_META="src/data/encoder_input/benign_gridsybil/T64/metadata.json"
MIN_FREE_GB=20

if [[ -t 1 ]]; then
  GREEN=$'\033[32m' RED=$'\033[31m' YELLOW=$'\033[33m' BOLD=$'\033[1m' RESET=$'\033[0m'
else
  GREEN="" RED="" YELLOW="" BOLD="" RESET=""
fi
banner() { printf '\n%s== %s ==%s\n' "$BOLD" "$1" "$RESET"; }
ok() { printf '%s✓%s %s\n' "$GREEN" "$RESET" "$1"; }
warn() { printf '%s!%s %s\n' "$YELLOW" "$RESET" "$1"; }
fail() {
  printf '%s✗%s %s\n' "$RED" "$RESET" "$1" >&2
  exit 1
}
if ((NATIVE)); then
  PY="${PY:-$REPO_ROOT/.venv-train/bin/python}"
  export PY
  SERVICE="native"
  # Same interface as the container route: `python` means the project interpreter, anything else runs as is.
  in_container() {
    if [[ "$1" == python ]]; then
      shift
      "$PY" "$@"
    else
      "$@"
    fi
  }
else
  in_container() { docker compose run --rm -T "$SERVICE" "$@"; }
fi
timed() {
  # Runs a command and prints how long it took.
  local label="$1"
  shift
  local start=$SECONDS
  "$@" || fail "$label failed (exit $?)"
  ok "$label ($((SECONDS - start)) s)"
}

banner "Host checks (service: $SERVICE)"
if ((NATIVE)); then
  command -v npm >/dev/null 2>&1 || fail "npm not found — run: bash scripts/ec2_bootstrap.sh"
  ok "npm $(npm -v)"
  if [[ "$PY" == "$REPO_ROOT/.venv-train/bin/python" ]]; then
    # Same venvs as the bootstrap / the Mac: complete them when the check fails (idempotent, no-op when ready).
    if bash scripts/setup_venv.sh --check >/dev/null 2>&1; then
      ok "venvs ready (scripts/setup_venv.sh --check)"
    else
      warn "venvs not ready — running scripts/setup_venv.sh"
      bash scripts/setup_venv.sh || fail "scripts/setup_venv.sh failed — see its output above"
    fi
  fi
  [[ -x "$PY" ]] || fail "$PY not found — run: bash scripts/setup_venv.sh (or bash scripts/ec2_bootstrap.sh)"
  ok "python: $PY"
  if [[ -f scripts/gpu_check.py ]]; then
    "$PY" scripts/gpu_check.py | sed 's/^/  /' || warn "gpu_check.py failed — see above (not fatal)"
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    ok "host GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  else
    warn "no working nvidia-smi — training will run on the CPU"
  fi
else
command -v docker >/dev/null 2>&1 || fail "docker not found — install Docker (docs/ec2-training.md §1a)"
ok "docker $(docker --version | sed 's/^Docker version //')"
docker compose version >/dev/null 2>&1 || fail "'docker compose' (v2 plugin) not found — install docker-compose-plugin"
ok "docker compose $(docker compose version --short)"
docker info >/dev/null 2>&1 || fail "the Docker daemon is not reachable — start Docker (or add your user to the docker group)"
ok "Docker daemon reachable"
if [[ "$SERVICE" == "gpu" ]]; then
  command -v nvidia-smi >/dev/null 2>&1 \
    || fail "nvidia-smi not found — install the NVIDIA driver (docs/ec2-training.md §1a), or use --cpu"
  ok "host GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  docker info 2>/dev/null | grep -qi nvidia \
    || fail "NVIDIA container runtime not registered with Docker — install nvidia-container-toolkit, then
    sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker   (docs/ec2-training.md §1a)"
  ok "NVIDIA container runtime registered"
fi
fi
free_gb=$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')
if ((free_gb < MIN_FREE_GB)); then
  warn "only ${free_gb} GB free on this disk (want ≥ ${MIN_FREE_GB} GB: image + prepared data + runs)"
else
  ok "${free_gb} GB free on this disk"
fi

banner "Docker image"
if ((NATIVE)); then
  ok "native route: no image"
elif ((SKIP_BUILD)); then
  docker image inspect "roadfm-lite:$SERVICE" >/dev/null 2>&1 \
    || fail "--skip-build given but image roadfm-lite:$SERVICE does not exist — drop --skip-build"
  ok "build skipped; image roadfm-lite:$SERVICE exists"
else
  timed "docker compose build $SERVICE" docker compose build "$SERVICE"
fi

banner "Container environment"
expect="$SERVICE"
((NATIVE)) && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 && expect="native-gpu"
in_container python - "$expect" <<'PY' || fail "container environment check failed (see above)"
import sys
import torch

service = sys.argv[1]
print(f"  python {sys.version.split()[0]}, torch {torch.__version__}")
cuda = torch.cuda.is_available()
print(f"  torch.cuda.is_available() = {cuda}")
if cuda:
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
if service in ("gpu", "native-gpu") and not cuda:
    sys.exit("CUDA not available inside the gpu container")
from jupyter_client.kernelspec import KernelSpecManager

specs = KernelSpecManager().find_kernel_specs()
missing = {"python3", "roadfm-train"} - set(specs)
if missing:
    sys.exit(f"Jupyter kernels missing: {sorted(missing)}")
print("  kernels: python3, roadfm-train")
PY
ok "python, torch and kernels OK"

banner "Raw dataset ($RAW_DIR)"
count_raw_files() { find "$1" -type f ! -name '.DS_Store' 2>/dev/null | wc -l | tr -d ' '; }
raw_files=0
[[ -d "$RAW_DIR" ]] && raw_files=$(count_raw_files "$RAW_DIR")
where="host"
if ((raw_files == 0)); then
  # The dataset may be bind-mounted into the container only (e.g. a compose override); count it there.
  raw_files=$(in_container bash -c "find /workspace/$RAW_DIR -type f ! -name '.DS_Store' 2>/dev/null | wc -l" \
    | tr -d ' \r' || echo 0)
  where="container"
fi
if ((raw_files == 0)); then
  printf '%s✗%s raw dataset not found in %s/\n' "$RED" "$RESET" "$RAW_DIR" >&2
  echo "  download it first (private HF dataset, needs 'hf auth login'):  npm run data:download" >&2
  echo "  then re-run:  ./scripts/setup.sh $*" >&2
  exit 1
fi
if ((raw_files != EXPECTED_RAW_FILES)); then
  warn "$raw_files files in $RAW_DIR ($where), expected $EXPECTED_RAW_FILES — incomplete download? (npm run data:download resumes)"
else
  ok "$raw_files files in $RAW_DIR ($where)"
fi

banner "Data pipeline ($SERVICE)"
if ((REBUILD_DATA)) || [[ ! -f "$PREPARED_INDEX" ]]; then
  timed "prepared data -> src/data/prepared_data/" in_container npm run pipeline:prepare
else
  ok "prepared data exists ($PREPARED_INDEX); use --rebuild-data to rebuild"
fi
if ((REBUILD_DATA)) || [[ ! -f "$ENCODER_META" ]]; then
  timed "encoder input -> src/data/encoder_input/benign_gridsybil/T64/" in_container npm run pipeline:encoder-input
else
  ok "encoder input exists ($ENCODER_META); use --rebuild-data to rebuild"
fi

banner "Pretraining check (train:check)"
timed "train:check passed" in_container npm run train:check

banner "Done"
ok "setup complete (service: $SERVICE)"
if ((NATIVE)); then
  cat <<EOF
Next commands (from the repo root):
  npm run train:smoke                         # ~30 steps, real speed + memory
  tmux new -s train   # then:
  npm run train:full                          # full run (30 h guard)
  npm run model:upload -- <run_dir>           # e.g. src/runs/pretraining/benign_gridsybil/T64/<run_id>
EOF
  exit 0
fi
cat <<EOF
Next commands (from the repo root):
  docker compose run --rm $SERVICE npm run train:smoke                    # ~30 steps, real speed + memory
  tmux new -s train   # then:
  docker compose run --rm $SERVICE npm run train:full                     # full run (30 h guard)
  docker compose run --rm $SERVICE npm run model:upload -- <run_dir>      # e.g. src/runs/pretraining/benign_gridsybil/T64/<run_id>
EOF
