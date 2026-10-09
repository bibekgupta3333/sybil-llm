#!/usr/bin/env bash
# Prepares a fresh Ubuntu 24.04 EC2 instance (plain Ubuntu AMI or the Deep Learning Base AMI) with every HOST
# dependency RoadFM-Lite needs. Run it once as a normal user with sudo, from anywhere (npm is not installed yet):
#
#   bash scripts/ec2_bootstrap.sh [options]
#
#   --dry-run          print every command, change nothing
#   --check            only report what is installed (exit 1 if a required component is missing)
#   --install-driver   install the NVIDIA server driver (>= 560) when a GPU is present but nvidia-smi fails (reboot after)
#   --native           also set up the non-Docker route: Python 3.14.5 (uv), .venv-train (requirements/linux.txt + torch
#                      2.14.1 cu126 / cpu), Jupyter kernels roadfm-train + python3
#   --hf-cli           also install the Hugging Face CLI on the host (`hf`; login can also happen inside the container)
#   --no-start         do not start / restart services or run containers (also automatic when systemd is not PID 1)
#   --allow-root       allow running as root (default: refuse)
#   -h, --help         this help
#
# Installs: base tools (git, git-lfs, tmux, htop, jq, build-essential, ...), Node.js 22 LTS + npm (NodeSource),
# Docker Engine + buildx + compose plugin (Docker's apt repo, user added to the docker group), NVIDIA Container
# Toolkit when an NVIDIA GPU is present, uv. Idempotent: everything already installed is skipped. Log:
# ~/roadfm-bootstrap.log. Next step afterwards: `npm run setup` (GPU) or `npm run setup:cpu`. Guide: docs/ec2-training.md.
set -euo pipefail

DRY_RUN=0
CHECK_ONLY=0
INSTALL_DRIVER=0
NATIVE=0
HF_CLI=0
NO_START=0
ALLOW_ROOT=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --check) CHECK_ONLY=1 ;;
    --install-driver) INSTALL_DRIVER=1 ;;
    --native) NATIVE=1 ;;
    --hf-cli) HF_CLI=1 ;;
    --no-start) NO_START=1 ;;
    --allow-root) ALLOW_ROOT=1 ;;
    -h | --help)
      sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown option: $arg (see --help)" >&2
      exit 2
      ;;
  esac
done

# Versions kept equal to docker/Dockerfile and requirements/linux.txt.
NODE_MAJOR=22
UV_VERSION=0.12.24
PYTHON_VERSION=3.14.5
TORCH_VERSION=2.14.1
HF_HUB_VERSION=2.2.0
MIN_DRIVER=560
FALLBACK_DRIVER=570
BASE_PACKAGES=(ca-certificates curl wget gnupg git git-lfs tmux htop jq unzip build-essential pciutils lsb-release)
DOCKER_PACKAGES=(docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin)
DOCKER_CONFLICTS=(docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_USER="$(id -un)"
LOG_FILE="${HOME:-/tmp}/roadfm-bootstrap.log"
export PATH="$HOME/.local/bin:$PATH"

if [[ -t 1 ]]; then
  GREEN=$'\033[32m' RED=$'\033[31m' YELLOW=$'\033[33m' CYAN=$'\033[36m' BOLD=$'\033[1m' RESET=$'\033[0m'
else
  GREEN="" RED="" YELLOW="" CYAN="" BOLD="" RESET=""
fi
# Everything below is also appended to the log file.
exec > >(tee -a "$LOG_FILE") 2>&1
printf '\n# %s  ec2_bootstrap.sh %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"

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

has_systemd() { [[ -d /run/systemd/system ]]; }
can_start() { ((!NO_START)) && has_systemd; }
pkg_installed() { dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q 'install ok installed'; }
has_nvidia_gpu() {
  if command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -qi nvidia; then
    return 0
  fi
  # 0x10de is NVIDIA's PCI vendor id; works before pciutils is installed.
  grep -qsi '0x10de' /sys/bus/pci/devices/*/vendor
}
nvidia_smi_ok() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; }
driver_version() { nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true; }
driver_major() {
  local v
  v="$(driver_version)"
  v="${v%%.*}"
  [[ "$v" =~ ^[0-9]+$ ]] && echo "$v" || echo 0
}
node_major() {
  local v
  v="$(node -v 2>/dev/null || true)"
  v="${v#v}"
  echo "${v%%.*}"
}
in_docker_group() { id -nG "$TARGET_USER" 2>/dev/null | tr ' ' '\n' | grep -qx docker; }
docker_runtime_configured() { grep -qs '"nvidia"' /etc/docker/daemon.json; }
missing_base_packages() {
  local p
  for p in "${BASE_PACKAGES[@]}"; do pkg_installed "$p" || printf '%s ' "$p"; done
}
torch_variant() { if has_nvidia_gpu; then echo cu126; else echo cpu; fi; }
VENV_PY="$REPO_ROOT/.venv-train/bin/python"
native_ok() {
  [[ -x "$VENV_PY" ]] || return 1
  "$VENV_PY" - "$PYTHON_VERSION" "$TORCH_VERSION" <<'PY' >/dev/null 2>&1
import sys
from jupyter_client.kernelspec import KernelSpecManager
import torch

assert sys.version.split()[0] == sys.argv[1], sys.version
assert torch.__version__.split("+")[0] == sys.argv[2], torch.__version__
assert {"python3", "roadfm-train"} <= set(KernelSpecManager().find_kernel_specs())
PY
}

APT_UPDATED=0
apt_update() {
  if ((!APT_UPDATED)); then
    run "${SUDO[@]}" apt-get update
    APT_UPDATED=1
  fi
}
# Installs only the packages that are missing.
apt_install() {
  local missing=() p
  for p in "$@"; do pkg_installed "$p" || missing+=("$p"); done
  if ((${#missing[@]} == 0)); then
    ok "already installed: $*"
    return 0
  fi
  apt_update
  run "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
}
# Adds an apt repo (key + list file) once; a new repo forces a fresh `apt-get update`.
add_apt_repo() {
  local name="$1" key_url="$2" key_file="$3" list_file="$4" list_line="$5" dearmor="$6"
  if [[ -f "$list_file" && -f "$key_file" ]]; then
    ok "$name apt repo already configured ($list_file)"
    return 0
  fi
  step "adding the $name apt repo"
  run "${SUDO[@]}" install -m 0755 -d "$(dirname "$key_file")"
  if ((dearmor)); then
    run_sh "curl -fsSL '$key_url' | ${SUDO[*]} gpg --batch --yes --dearmor -o '$key_file'"
  else
    run "${SUDO[@]}" curl -fsSL "$key_url" -o "$key_file"
  fi
  run "${SUDO[@]}" chmod a+r "$key_file"
  run_sh "echo '$list_line' | ${SUDO[*]} tee '$list_file' >/dev/null"
  APT_UPDATED=0
}

# ---------------------------------------------------------------------------------------------------------------------
# Status report (used by --check and as the final summary).
REQUIRED_MISSING=0
row() { printf '  %-26s %-36s %s\n' "$1" "$2" "$3"; }
status_row() {
  # name, version, state (ok | missing | skip | note), required (1/0)
  local name="$1" version="$2" state="$3" required="${4:-1}" mark
  case "$state" in
    ok) mark="${GREEN}✓ installed${RESET}" ;;
    missing)
      if ((required)); then
        mark="${RED}✗ missing${RESET}"
        REQUIRED_MISSING=1
      else
        mark="${YELLOW}- not installed (optional)${RESET}"
      fi
      ;;
    *) mark="${YELLOW}${state}${RESET}" ;;
  esac
  row "$name" "${version:--}" "$mark"
}
report() {
  REQUIRED_MISSING=0
  banner "Summary"
  row "component" "version" "status"
  row "---------" "-------" "------"
  local miss
  miss="$(missing_base_packages)"
  if [[ -z "$miss" ]]; then
    status_row "base packages" "${#BASE_PACKAGES[@]} packages" ok
  else
    status_row "base packages" "missing: ${miss% }" missing
  fi
  local nm
  nm="$(node_major)"
  if [[ -n "$nm" ]] && ((nm >= NODE_MAJOR)); then
    status_row "Node.js" "$(node -v)" ok
  else
    status_row "Node.js (>= $NODE_MAJOR)" "$(node -v 2>/dev/null || true)" missing
  fi
  if command -v npm >/dev/null 2>&1; then status_row "npm" "$(npm -v 2>/dev/null)" ok; else status_row "npm" "" missing; fi
  if command -v docker >/dev/null 2>&1; then
    status_row "Docker Engine" "$(docker --version 2>/dev/null | sed 's/^Docker version //')" ok
  else
    status_row "Docker Engine" "" missing
  fi
  if docker compose version >/dev/null 2>&1; then
    status_row "docker compose plugin" "$(docker compose version --short 2>/dev/null)" ok
  else
    status_row "docker compose plugin" "" missing
  fi
  if docker buildx version >/dev/null 2>&1; then
    status_row "docker buildx plugin" "$(docker buildx version 2>/dev/null | awk '{print $2}')" ok
  else
    status_row "docker buildx plugin" "" missing
  fi
  if in_docker_group; then
    if id -nG 2>/dev/null | tr ' ' '\n' | grep -qx docker; then
      status_row "docker group" "$TARGET_USER" ok
    else
      status_row "docker group" "$TARGET_USER" "added; re-login or 'newgrp docker'"
    fi
  else
    status_row "docker group" "$TARGET_USER" missing
  fi
  if ! has_systemd; then
    status_row "docker service" "" "skipped (systemd is not PID 1)"
  elif systemctl is-active --quiet docker 2>/dev/null; then
    status_row "docker service" "active" ok
  else
    status_row "docker service" "inactive" "not running (start: sudo systemctl enable --now docker)"
  fi
  if has_nvidia_gpu; then
    local gpu
    gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
    if nvidia_smi_ok; then
      local dv
      dv="$(driver_version)"
      if (($(driver_major) >= MIN_DRIVER)); then
        status_row "NVIDIA driver" "$dv ($gpu)" ok
      else
        status_row "NVIDIA driver" "$dv (< $MIN_DRIVER, cu126 needs >= $MIN_DRIVER)" "too old"
      fi
    else
      status_row "NVIDIA driver" "GPU present, nvidia-smi fails" missing
    fi
    if command -v nvidia-ctk >/dev/null 2>&1; then
      status_row "NVIDIA Container Toolkit" "$(nvidia-ctk --version 2>/dev/null | head -1 | awk '{print $NF}')" ok
    else
      status_row "NVIDIA Container Toolkit" "" missing
    fi
    if docker_runtime_configured; then
      status_row "docker nvidia runtime" "/etc/docker/daemon.json" ok
    else
      status_row "docker nvidia runtime" "" missing
    fi
  else
    status_row "NVIDIA GPU" "none found" "skipped (CPU service works)"
  fi
  if command -v uv >/dev/null 2>&1; then status_row "uv" "$(uv --version | awk '{print $2}')" ok; else status_row "uv" "" missing; fi
  if command -v hf >/dev/null 2>&1; then
    status_row "Hugging Face CLI (host)" "$(hf version 2>/dev/null | head -1 || echo present)" ok 0
  else
    status_row "Hugging Face CLI (host)" "" missing "$HF_CLI"
  fi
  if native_ok; then
    status_row "native .venv-train" "$("$VENV_PY" -c 'import sys, torch; print(sys.version.split()[0], "torch", torch.__version__)')" ok 0
  else
    status_row "native .venv-train" "" missing "$NATIVE"
  fi
}

# ---------------------------------------------------------------------------------------------------------------------
banner "Host"
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  if [[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "24.04" ]]; then
    ok "Ubuntu ${VERSION_ID} (${VERSION_CODENAME:-}), $(uname -m)"
  else
    warn "this is ${PRETTY_NAME:-an unknown OS}; the script is written for Ubuntu 24.04 — continuing, check the result"
  fi
else
  warn "/etc/os-release not found; the script is written for Ubuntu 24.04"
fi
command -v apt-get >/dev/null 2>&1 || die "apt-get not found — this script needs Ubuntu / Debian"
if ((EUID == 0)); then
  ((ALLOW_ROOT)) || die "do not run as root: run as a normal user with sudo (or pass --allow-root)"
  SUDO=()
  warn "running as root (--allow-root): the docker group step targets root"
else
  SUDO=(sudo)
fi
has_systemd || {
  NO_START=1
  warn "systemd is not PID 1 (container?): services will not be started"
}
((NO_START)) && step "--no-start: no service start / restart, no container runs"
((DRY_RUN)) && step "--dry-run: printing commands only, nothing changes"
step "log: $LOG_FILE"

if ((CHECK_ONLY)); then
  report
  ((REQUIRED_MISSING)) && {
    bad "required components are missing — run: bash scripts/ec2_bootstrap.sh"
    exit 1
  }
  ok "all required components are installed"
  exit 0
fi

if ((!DRY_RUN)) && ((EUID != 0)); then
  command -v sudo >/dev/null 2>&1 || die "sudo not found"
  sudo -v || die "sudo failed — this script needs a user with sudo rights"
fi
ARCH="$(dpkg --print-architecture)"
CODENAME="${VERSION_CODENAME:-noble}"  # from /etc/os-release, sourced above
NEED_RELOGIN=0
NEED_REBOOT=0

# ---------------------------------------------------------------------------------------------------------------------
banner "Base packages"
apt_install "${BASE_PACKAGES[@]}"
if command -v git-lfs >/dev/null 2>&1 && ! git config --global --get filter.lfs.clean >/dev/null 2>&1; then
  run git lfs install --skip-repo
fi

# ---------------------------------------------------------------------------------------------------------------------
banner "Node.js $NODE_MAJOR LTS + npm"
nm="$(node_major)"
if [[ -n "$nm" ]] && ((nm >= NODE_MAJOR)) && command -v npm >/dev/null 2>&1; then
  ok "node $(node -v), npm $(npm -v) — kept"
else
  [[ -n "$nm" ]] && step "node v$nm found; replacing it with Node.js $NODE_MAJOR from NodeSource"
  # Ubuntu's separate npm package conflicts with NodeSource's nodejs (which bundles npm).
  if pkg_installed npm; then run "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get remove -y npm; fi
  add_apt_repo "NodeSource" "https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key" \
    /etc/apt/keyrings/nodesource.gpg /etc/apt/sources.list.d/nodesource.list \
    "deb [arch=$ARCH signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" 1
  apt_update
  run "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs
  ((DRY_RUN)) || ok "node $(node -v), npm $(npm -v)"
fi

# ---------------------------------------------------------------------------------------------------------------------
banner "Docker Engine + buildx + compose plugin"
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1 && docker buildx version >/dev/null 2>&1; then
  ok "docker $(docker --version | sed 's/^Docker version //'), compose $(docker compose version --short) — kept"
else
  conflicts=()
  for p in "${DOCKER_CONFLICTS[@]}"; do pkg_installed "$p" && conflicts+=("$p"); done
  if ((${#conflicts[@]})); then
    step "removing packages that conflict with Docker's own: ${conflicts[*]}"
    run "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get remove -y "${conflicts[@]}"
  fi
  add_apt_repo "Docker" "https://download.docker.com/linux/ubuntu/gpg" \
    /etc/apt/keyrings/docker.asc /etc/apt/sources.list.d/docker.list \
    "deb [arch=$ARCH signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $CODENAME stable" 0
  apt_install "${DOCKER_PACKAGES[@]}"
  ((DRY_RUN)) || ok "docker $(docker --version | sed 's/^Docker version //'), compose $(docker compose version --short)"
fi
if in_docker_group; then
  ok "$TARGET_USER is in the docker group"
else
  if ! getent group docker >/dev/null 2>&1; then run "${SUDO[@]}" groupadd docker; fi
  run "${SUDO[@]}" usermod -aG docker "$TARGET_USER"
  NEED_RELOGIN=1
  warn "$TARGET_USER added to the docker group — log out and back in (or run 'newgrp docker') before using docker"
fi
if can_start; then
  if systemctl is-active --quiet docker && systemctl is-enabled --quiet docker; then
    ok "docker service enabled and running"
  else
    run "${SUDO[@]}" systemctl enable --now docker
  fi
else
  step "docker service start skipped (--no-start or no systemd)"
fi

# ---------------------------------------------------------------------------------------------------------------------
banner "NVIDIA (driver + Container Toolkit)"
if ! has_nvidia_gpu; then
  ok "no NVIDIA GPU found — skipped (use the cpu service: npm run setup:cpu)"
else
  if nvidia_smi_ok; then
    dv="$(driver_version)"
    if (($(driver_major) >= MIN_DRIVER)); then
      ok "NVIDIA driver $dv works (nvidia-smi) — kept"
    else
      warn "NVIDIA driver $dv is older than $MIN_DRIVER; the cu126 torch wheels need >= $MIN_DRIVER — upgrade it"
    fi
  elif ((INSTALL_DRIVER)); then
    apt_install ubuntu-drivers-common
    best=""
    if command -v ubuntu-drivers >/dev/null 2>&1; then
      best="$(ubuntu-drivers list --gpgpu 2>/dev/null | grep -oE 'nvidia[-:a-z]*[0-9]+-server' | grep -oE '[0-9]+' \
        | sort -n | tail -1 || true)"
    fi
    if [[ -n "$best" ]] && ((best >= MIN_DRIVER)); then
      run "${SUDO[@]}" ubuntu-drivers install --gpgpu "nvidia:${best}-server"
      run "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y "nvidia-utils-${best}-server"
    else
      step "ubuntu-drivers offers no server driver >= $MIN_DRIVER${best:+ (best: $best)}; installing nvidia-driver-${FALLBACK_DRIVER}-server"
      apt_install "nvidia-driver-${FALLBACK_DRIVER}-server" "nvidia-utils-${FALLBACK_DRIVER}-server"
    fi
    NEED_REBOOT=1
    warn "driver installed — reboot (sudo reboot), reconnect, check nvidia-smi, then re-run this script"
  else
    bad "NVIDIA GPU present but nvidia-smi fails — re-run with --install-driver (or use the Deep Learning Base AMI)"
  fi

  if command -v nvidia-ctk >/dev/null 2>&1 && pkg_installed nvidia-container-toolkit; then
    ok "NVIDIA Container Toolkit $(nvidia-ctk --version 2>/dev/null | head -1 | awk '{print $NF}') — kept"
  else
    add_apt_repo "NVIDIA Container Toolkit" "https://nvidia.github.io/libnvidia-container/gpgkey" \
      /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg /etc/apt/sources.list.d/nvidia-container-toolkit.list \
      "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/\$(ARCH) /" 1
    apt_install nvidia-container-toolkit
  fi
  if docker_runtime_configured; then
    ok "docker already has the nvidia runtime (/etc/docker/daemon.json)"
  else
    run "${SUDO[@]}" nvidia-ctk runtime configure --runtime=docker
    if can_start; then run "${SUDO[@]}" systemctl restart docker; else step "docker restart skipped (--no-start)"; fi
  fi
  if can_start && ((!DRY_RUN)) && ((!NEED_REBOOT)) && nvidia_smi_ok; then
    step "checking that containers see the GPU"
    if "${SUDO[@]}" docker run --rm --gpus all ubuntu:24.04 nvidia-smi; then
      ok "GPU visible inside a container"
    else
      bad "docker run --gpus all failed — see docs/ec2-training.md §1a"
    fi
  elif ((DRY_RUN)) && can_start; then
    run "${SUDO[@]}" docker run --rm --gpus all ubuntu:24.04 nvidia-smi
  else
    step "container GPU check skipped (--no-start, reboot pending, or no working driver)"
  fi
fi

# ---------------------------------------------------------------------------------------------------------------------
banner "uv"
if command -v uv >/dev/null 2>&1; then
  ok "uv $(uv --version | awk '{print $2}') — kept"
else
  run_sh "curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh"
  ((DRY_RUN)) || ok "uv $("$HOME/.local/bin/uv" --version | awk '{print $2}') in ~/.local/bin"
fi

if ((HF_CLI)); then
  banner "Hugging Face CLI (host)"
  if command -v hf >/dev/null 2>&1; then
    ok "hf already installed ($(command -v hf))"
  else
    run uv tool install "huggingface_hub[cli,hf_xet]==${HF_HUB_VERSION}"
  fi
fi

# ---------------------------------------------------------------------------------------------------------------------
if ((NATIVE)); then
  banner "Native route (.venv-train without Docker)"
  variant="$(torch_variant)"
  if native_ok; then
    ok ".venv-train already has Python $PYTHON_VERSION, torch $TORCH_VERSION and both kernels"
  else
    [[ -f "$REPO_ROOT/requirements/linux.txt" ]] || die "requirements/linux.txt not found under $REPO_ROOT"
    run uv python install "$PYTHON_VERSION"
    if [[ -x "$VENV_PY" ]]; then
      have="$("$VENV_PY" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || true)"
      [[ "$have" == "$PYTHON_VERSION" ]] \
        || die ".venv-train exists with Python '${have:-?}', not $PYTHON_VERSION — remove it (rm -rf .venv-train) and re-run"
      ok ".venv-train exists (Python $have); completing it"
    else
      run uv venv "$REPO_ROOT/.venv-train" --python "$PYTHON_VERSION"
    fi
    run uv pip install --python "$VENV_PY" -r "$REPO_ROOT/requirements/linux.txt"
    run uv pip install --python "$VENV_PY" --index-url https://pypi.org/simple \
      --extra-index-url "https://download.pytorch.org/whl/${variant}" "torch==${TORCH_VERSION}+${variant}"
    run "$VENV_PY" -m ipykernel install --sys-prefix --name python3 --display-name "Python 3"
    run "$VENV_PY" -m ipykernel install --sys-prefix --name roadfm-train --display-name "Python (roadfm-train)"
    if ((!DRY_RUN)); then
      native_ok || die ".venv-train check failed (python / torch / kernels) — see the log"
      ok ".venv-train: $("$VENV_PY" -c 'import sys, torch; print(sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available())')"
    fi
  fi
fi

# ---------------------------------------------------------------------------------------------------------------------
if ((DRY_RUN)); then
  banner "Dry run done — nothing was changed; current state:"
fi
report
((NEED_REBOOT)) && warn "reboot needed for the NVIDIA driver: sudo reboot, then re-run: bash scripts/ec2_bootstrap.sh"
((NEED_RELOGIN)) && warn "docker group changed: run 'newgrp docker' (or log out and back in) before using docker"

banner "Next commands"
gpu_hint="npm run setup             # GPU (service gpu)"
has_nvidia_gpu || gpu_hint="npm run setup:cpu         # no NVIDIA GPU here (service cpu)"
cat <<EOF
  newgrp docker                         # or log out and back in, so docker works without sudo
  cd $REPO_ROOT
  # put the raw dataset in data/VeReMi-Dataset/ yourself, or once it is on the Hub:
  npm run docker:shell                  # then inside: hf auth login && npm run data:download && exit
  $gpu_hint
  tmux new -s train                     # then: npm run docker:train  (= train:full in the gpu container)
EOF
if ((NATIVE)); then
  echo "  # native route (no Docker): npm run train:check / train:smoke / train:full use .venv-train/bin/python"
fi
((REQUIRED_MISSING)) && ((!DRY_RUN)) && {
  bad "some required components are still missing (see the summary)"
  exit 1
}
exit 0
