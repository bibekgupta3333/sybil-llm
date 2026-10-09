#!/usr/bin/env bash
# Creates or completes the two Python environments at the repo root, the same way on the Mac and on EC2 (Ubuntu):
#
#   bash scripts/setup_venv.sh [--check] [--dry-run] [--recreate] [--no-eda] [--cpu | --cuda] [-h]
#
#   --check      report only (python, torch, pins, imports, kernels); exit 1 if an environment is not ready
#   --dry-run    print every command, change nothing (then report the current state)
#   --recreate   delete and rebuild the environments (needed when one exists with another Python version)
#   --no-eda     only .venv-train (skip the EDA venv .venv)
#   --cpu        Linux: torch 2.14.1+cpu even if an NVIDIA GPU is present
#   --cuda       Linux: torch 2.14.1+cu126 even if no NVIDIA GPU is detected
#   -h, --help   this help
#
# .venv-train  Python 3.14.5 (uv venv --seed). macOS: requirements-train.txt (the pinned freeze, torch with MPS).
#              Linux: requirements/linux.txt + torch 2.14.1 from download.pytorch.org (cu126 on a GPU, cpu otherwise).
#              Jupyter kernels python3 + roadfm-train (--sys-prefix, unless the name already resolves).
# .venv        Python 3.14.5, requirements.txt (EDA, no torch; weasyprint needs libpango on Linux), kernel python3.
# Idempotent: `uv pip install` of pins already satisfied is a fast no-op; nothing is ever uninstalled. Installs uv
# (pinned installer) into ~/.local/bin when it is missing. Used by scripts/ec2_bootstrap.sh and `npm run setup:venv`.
set -euo pipefail

CHECK_ONLY=0
DRY_RUN=0
RECREATE=0
WITH_EDA=1
FORCE_VARIANT=""
for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --recreate) RECREATE=1 ;;
    --no-eda) WITH_EDA=0 ;;
    --cpu) FORCE_VARIANT=cpu ;;
    --cuda) FORCE_VARIANT=cu126 ;;
    -h | --help)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown option: $arg (see --help)" >&2
      exit 2
      ;;
  esac
done

# Versions kept equal to docker/Dockerfile, scripts/ec2_bootstrap.sh and requirements-train.txt.
UV_VERSION=0.12.24
PYTHON_VERSION=3.14.5
TORCH_VERSION=2.14.1
TRAIN_IMPORTS="numpy,pandas,torch,huggingface_hub,nbclient,pytest,black"
EDA_IMPORTS="numpy,pandas,scipy,sklearn,matplotlib,seaborn,ipykernel,weasyprint"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_VENV="$REPO_ROOT/.venv-train"
EDA_VENV="$REPO_ROOT/.venv"
EDA_REQ="$REPO_ROOT/requirements.txt"
OS_NAME="$(uname -s)"
if [[ "$OS_NAME" == "Darwin" ]]; then
  TRAIN_REQ="$REPO_ROOT/requirements-train.txt"
else
  TRAIN_REQ="$REPO_ROOT/requirements/linux.txt"
fi
export PATH="$HOME/.local/bin:$PATH"

if [[ -t 1 ]]; then
  GREEN=$'\033[32m' RED=$'\033[31m' YELLOW=$'\033[33m' CYAN=$'\033[36m' BOLD=$'\033[1m' RESET=$'\033[0m'
else
  GREEN="" RED="" YELLOW="" CYAN="" BOLD="" RESET=""
fi
banner() { printf '\n%s== %s ==%s\n' "$BOLD" "$1" "$RESET"; }
ok() { printf '%s✓%s %s\n' "$GREEN" "$RESET" "$1"; }
bad() { printf '%s✗%s %s\n' "$RED" "$RESET" "$1"; }
step() { printf '%s→%s %s\n' "$CYAN" "$RESET" "$1"; }
warn() { printf '%s!%s %s\n' "$YELLOW" "$RESET" "$1"; }
die() {
  bad "$1"
  exit 1
}

# Runs a command (argv form), or only prints it with --dry-run.
run() {
  if ((DRY_RUN)); then
    printf '%s→ [dry-run]%s %s\n' "$CYAN" "$RESET" "$(printf '%q ' "$@")"
  else
    printf '%s→%s %s\n' "$CYAN" "$RESET" "$(printf '%q ' "$@")"
    "$@"
  fi
}
# Runs a shell string (for pipes), or only prints it with --dry-run.
run_sh() {
  if ((DRY_RUN)); then
    printf '%s→ [dry-run]%s %s\n' "$CYAN" "$RESET" "$1"
  else
    printf '%s→%s %s\n' "$CYAN" "$RESET" "$1"
    bash -o pipefail -c "$1"
  fi
}

has_nvidia_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    return 0
  fi
  if command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -qi nvidia; then
    return 0
  fi
  # 0x10de is NVIDIA's PCI vendor id; works before pciutils is installed.
  grep -qsi '0x10de' /sys/bus/pci/devices/*/vendor
}
torch_variant() {
  if [[ -n "$FORCE_VARIANT" ]]; then
    echo "$FORCE_VARIANT"
  elif has_nvidia_gpu; then
    echo cu126
  else
    echo cpu
  fi
}
venv_python_version() { "$1/bin/python" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || true; }

# Checks one environment with its own interpreter. Prints one ✓ / ✗ / ! line per item; exit 1 if any ✗.
# Arguments: venv dir, requirements file, expected torch version ("" = none), imports (csv), kernels (csv).
verify_venv() {
  local venv="$1" req="$2" torch_version="$3" imports="$4" kernels="$5"
  if [[ ! -x "$venv/bin/python" ]]; then
    printf '%s✗%s %s\n' "$RED" "$RESET" "no interpreter at $venv/bin/python"
    return 1
  fi
  "$venv/bin/python" - "$PYTHON_VERSION" "$req" "$torch_version" "$imports" "$kernels" "$GREEN" "$RED" "$YELLOW" \
    "$RESET" <<'PY'
import importlib
import os
import re
import sys
from importlib import metadata

want_python, req_file, want_torch, imports, kernels, green, red, yellow, reset = sys.argv[1:10]
failures = 0


def line(good, text, mark=None):
    global failures
    if mark is None:
        mark = f"{green}✓{reset}" if good else f"{red}✗{reset}"
        failures += 0 if good else 1
    print(f"  {mark} {text}")


have_python = sys.version.split()[0]
line(have_python == want_python, f"python {have_python} (want {want_python})")

drift = []
pins = 0
with open(req_file, encoding="utf-8") as handle:
    for raw in handle:
        match = re.match(r"^\s*([A-Za-z0-9_.\-]+)(\[[^\]]*\])?==([^\s;#]+)", raw)
        if not match:
            continue
        pins += 1
        name, version = match.group(1), match.group(3)
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = None
        if have is None or have.split("+")[0] != version.split("+")[0]:
            drift.append(f"{name} {have or 'missing'} (pin {version})")
shown = ", ".join(drift[:6]) + (" ..." if len(drift) > 6 else "")
line(not drift, f"{pins} pins of {os.path.basename(req_file)} satisfied" if not drift else f"pins: {shown}")

if want_torch:
    try:
        import torch

        have_torch = torch.__version__
        cuda = torch.version.cuda or "none"
        line(have_torch.split("+")[0] == want_torch, f"torch {have_torch} (built CUDA {cuda}; want {want_torch})")
    except Exception as exc:  # noqa: BLE001 - any import failure means "not ready"
        line(False, f"torch import failed: {exc}")

failed = []
for module in filter(None, imports.split(",")):
    try:
        importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - report every failing import (weasyprint raises OSError without pango)
        failed.append(f"{module} ({type(exc).__name__}: {str(exc).splitlines()[0][:80] if str(exc) else ''})")
line(not failed, f"imports {imports.replace(',', ' ')}" if not failed else "imports failed: " + "; ".join(failed))

try:
    from jupyter_client.kernelspec import KernelSpecManager

    found = KernelSpecManager().find_kernel_specs()
    missing = [k for k in filter(None, kernels.split(",")) if k not in found]
    where = ", ".join(f"{k} -> {found[k]}" for k in kernels.split(",") if k in found)
    line(not missing, f"kernels {where}" if not missing else f"kernels missing: {' '.join(missing)}")
except Exception as exc:  # noqa: BLE001
    line(False, f"kernel lookup failed: {exc}")

sys.exit(1 if failures else 0)
PY
}

# Name, venv dir, requirements file, torch version, imports, kernels. Returns 0 when ready.
check_env() {
  local name="$1" venv="$2" req="$3" torch_version="$4" imports="$5" kernels="$6"
  step "$name ($venv)"
  if verify_venv "$venv" "$req" "$torch_version" "$imports" "$kernels"; then
    ok "$name ready"
    return 0
  fi
  bad "$name not ready"
  return 1
}

report() {
  local failed=0
  banner "Check"
  check_env ".venv-train" "$TRAIN_VENV" "$TRAIN_REQ" "$TORCH_VERSION" "$TRAIN_IMPORTS" "python3,roadfm-train" || failed=1
  if ((WITH_EDA)); then
    check_env ".venv (EDA)" "$EDA_VENV" "$EDA_REQ" "" "$EDA_IMPORTS" "python3" || failed=1
  fi
  return "$failed"
}

# ---------------------------------------------------------------------------------------------------------------------
banner "Python environments"
step "repo: $REPO_ROOT ($OS_NAME $(uname -m))"
if [[ "$OS_NAME" != "Darwin" && "$OS_NAME" != "Linux" ]]; then
  die "unsupported OS: $OS_NAME (macOS or Linux)"
fi
if [[ "$OS_NAME" == "Darwin" && -n "$FORCE_VARIANT" ]]; then
  warn "--cpu / --cuda only apply on Linux; macOS uses requirements-train.txt (torch with MPS)"
fi

if ((CHECK_ONLY)); then
  if report; then
    ok "all environments ready"
    exit 0
  fi
  bad "not ready — run: bash scripts/setup_venv.sh"
  exit 1
fi
((DRY_RUN)) && step "--dry-run: printing commands only, nothing changes"

[[ -f "$TRAIN_REQ" ]] || die "$TRAIN_REQ not found"
((WITH_EDA)) && { [[ -f "$EDA_REQ" ]] || die "$EDA_REQ not found"; }

# ---------------------------------------------------------------------------------------------------------------------
banner "uv"
if command -v uv >/dev/null 2>&1; then
  ok "uv $(uv --version | awk '{print $2}') ($(command -v uv)) — kept"
else
  command -v curl >/dev/null 2>&1 || die "curl not found (needed to install uv)"
  run_sh "curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh"
  if ((!DRY_RUN)); then
    hash -r
    command -v uv >/dev/null 2>&1 || die "uv not found after install (expected ~/.local/bin/uv)"
    ok "uv $(uv --version | awk '{print $2}') in ~/.local/bin"
  fi
fi

banner "Python $PYTHON_VERSION"
if command -v uv >/dev/null 2>&1 && uv python find --system "$PYTHON_VERSION" >/dev/null 2>&1; then
  ok "Python $PYTHON_VERSION found: $(uv python find --system "$PYTHON_VERSION")"
else
  run uv python install "$PYTHON_VERSION"
fi

# Creates the venv when missing; stops on a Python version mismatch unless --recreate.
ensure_venv() {
  local name="$1" venv="$2" have
  if [[ -e "$venv" ]] && ((RECREATE)); then
    warn "--recreate: removing $venv"
    run rm -rf "$venv"
  fi
  if [[ -x "$venv/bin/python" ]]; then
    have="$(venv_python_version "$venv")"
    if [[ "$have" != "$PYTHON_VERSION" ]]; then
      die "$name exists with Python '${have:-?}', not $PYTHON_VERSION — re-run with --recreate (deletes $venv)"
    fi
    ok "$name exists (Python $have); completing it"
  elif [[ -e "$venv" ]] && ((!DRY_RUN)); then
    die "$venv exists but has no bin/python — re-run with --recreate (deletes $venv)"
  else
    run uv venv --seed --python "$PYTHON_VERSION" "$venv"
  fi
}

# Registers a kernel in the venv (--sys-prefix) unless that name already resolves from it.
ensure_kernel() {
  local venv="$1" kernel="$2" display="$3"
  if [[ -x "$venv/bin/python" ]] && "$venv/bin/python" -c \
    "import sys; from jupyter_client.kernelspec import KernelSpecManager as K; sys.exit(sys.argv[1] not in K().find_kernel_specs())" \
    "$kernel" >/dev/null 2>&1; then
    ok "kernel $kernel already registered"
  else
    run "$venv/bin/python" -m ipykernel install --sys-prefix --name "$kernel" --display-name "$display"
  fi
}

# ---------------------------------------------------------------------------------------------------------------------
banner ".venv-train (Python $PYTHON_VERSION + torch $TORCH_VERSION)"
ensure_venv ".venv-train" "$TRAIN_VENV"
run uv pip install --python "$TRAIN_VENV/bin/python" -r "$TRAIN_REQ"
if [[ "$OS_NAME" == "Linux" ]]; then
  variant="$(torch_variant)"
  step "torch variant: $variant$([[ -n "$FORCE_VARIANT" ]] && echo " (forced)" || true)"
  run uv pip install --python "$TRAIN_VENV/bin/python" --index-url https://pypi.org/simple \
    --extra-index-url "https://download.pytorch.org/whl/${variant}" "torch==${TORCH_VERSION}+${variant}"
fi
ensure_kernel "$TRAIN_VENV" python3 "Python 3"
ensure_kernel "$TRAIN_VENV" roadfm-train "Python (roadfm-train)"

if ((WITH_EDA)); then
  banner ".venv (EDA, Python $PYTHON_VERSION)"
  ensure_venv ".venv" "$EDA_VENV"
  run uv pip install --python "$EDA_VENV/bin/python" -r "$EDA_REQ"
  ensure_kernel "$EDA_VENV" python3 "Python 3"
fi

# ---------------------------------------------------------------------------------------------------------------------
if ((DRY_RUN)); then
  banner "Dry run done — nothing was changed; current state:"
  report || true
  exit 0
fi
report || die "environment check failed — see the lines above"
ok "environments ready (.venv-train$( ((WITH_EDA)) && echo ", .venv" || true))"
exit 0
