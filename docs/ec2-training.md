# Training on an EC2 GPU instance (Docker)

Copy-paste guide for running self-supervised pretraining on a rented NVIDIA machine inside one reproducible
**Ubuntu 24.04** Docker image (`docker/Dockerfile`, `docker-compose.yml`). The raw data and the finished runs move
through two **private** Hugging Face repos (`scripts/hf_hub.py`):

| What | Repo (default; override with env `HF_DATASET_REPO` / `HF_MODEL_REPO` or `--dataset-repo` / `--model-repo`) |
|---|---|
| raw VeReMi-Extension tree (`data/VeReMi-Dataset/`, 23k files, 13 GB) | dataset `bibekgupta3333/veremi-extension-raw` |
| pretraining runs (`runs/<run_id>/` = config.json, env.json, metrics.jsonl, best.pt, [last.pt], SHA256SUMS) | model `bibekgupta3333/roadfm-lite-timesnet` |

Every command runs from the **repo root** (RULE 5). `npm` is only a task runner; `npm run x -- <arg>` passes `<arg>`.

## Fresh instance: host bootstrap (no Docker — the EC2 route)

On EC2 everything runs **directly on the host** and mirrors the Mac: the same two venvs (`.venv-train` for training,
pipeline and tests; `.venv` for EDA), Python 3.14.5, the same pins (Linux: `requirements/linux.txt` + torch 2.14.1
`cu126` on a GPU, `cpu` otherwise; EDA: `requirements.txt`), the same Jupyter kernels (`python3`, `roadfm-train`).
npm is not installed yet, so the first command is the bash script (as `ubuntu`, or in a root shell with `--allow-root`):

```bash
git clone <this repo> sybil-llm && cd sybil-llm
bash scripts/ec2_bootstrap.sh          # base tools, Node 22 + npm, uv, hf, gh, both venvs + kernels, then a GPU report
gh auth login && hf auth login         # GitHub (git pull / push) and Hugging Face (private repos)
npm run gpu:check                      # which GPU, driver, CUDA, what torch sees, a matmul smoke test
npm run data:download                  # raw dataset -> data/VeReMi-Dataset/ (or copy it there yourself)
npm run setup:native                   # prepared data, encoder input, train:check (re-creates the venvs if needed)
tmux new -s train                      # then: npm run train:full ; afterwards npm run model:upload -- <run_dir>
```

| Command | What it does |
|---|---|
| `bash scripts/ec2_bootstrap.sh` | everything above; idempotent; log `~/roadfm-bootstrap.log`; `--check`, `--dry-run`, `--allow-root`, `--install-driver` (GPU without a working `nvidia-smi`: Ubuntu's server driver ≥ 560, then reboot + re-run; on a vGPU host AWS's GRID driver, see below), `--driver auto\|grid\|ubuntu` (which driver; default `auto`), `--docker` (optional Docker route) |
| `npm run setup:venv` (`bash scripts/setup_venv.sh`) | only the venvs: creates / completes `.venv-train` + `.venv` with uv, installs the pins (macOS: `requirements-train.txt`; Linux: `requirements/linux.txt` + torch), registers the kernels, verifies; `--check` (= `npm run setup:venv:check`, also fails on a pin mismatch), `--dry-run`, `--recreate`, `--no-eda`, `--cpu` / `--cuda` |
| `npm run gpu:check` · `npm run gpu:require` | GPU report (`scripts/gpu_check.py`: nvidia-smi, torch CUDA / MPS, smoke test); `gpu:require` exits 1 when torch sees no CUDA device — use it before `train:full` |

`hf`, `uv` are linked into `/usr/local/bin`; `gh` comes from GitHub's apt repo.

### g6f / gr6f: fractional GPU (vGPU) needs the GRID driver

`g6f` / `gr6f` instances get a **slice of an NVIDIA L4**, exposed as an NVIDIA vGPU (PCI `10de:27b8`, subsystem
`10de:1733`). The GPU memory is only that slice: on `g6f.xlarge` `nvidia-smi` shows `NVIDIA L4-3Q` with **3 GB**
(torch: 2.79 GiB), not 24 GB. Ubuntu's nvidia drivers (`nvidia-driver-*`, `*-open`) refuse a vGPU — dmesg says
`NVRM: The NVIDIA vGPU ... is not supported by open nvidia.ko`, `nvidia-smi` fails and `torch.cuda.is_available()`
is False. These instances need **NVIDIA's GRID guest driver**, which AWS publishes in a public bucket (anonymous, no AWS
CLI or IAM role): `curl 'https://ec2-linux-nvidia-drivers.s3.amazonaws.com/?list-type=2&prefix=latest/'`.

What `bash scripts/ec2_bootstrap.sh` does about it:

- **Detection** (`--driver auto`): a vGPU host is one whose instance type (IMDSv2, 2 s timeout) matches `g6f.` /
  `gr6f.`, or that has an NVIDIA device with a known vGPU subsystem id (`0x1733`), or whose dmesg mentions
  `NVIDIA vGPU`. `--driver grid` / `--driver ubuntu` override the detection.
- **`--check`** shows `GPU kind` (vGPU + why, or full GPU), the installed driver kind (`grid`, `ubuntu`, `other`,
  `none`) and, for GRID, the license status. A vGPU host without a working GRID driver is a failure
  (`vGPU with non-GRID driver`) with the fix: `bash scripts/ec2_bootstrap.sh --install-driver`.
- **`--install-driver`** on a vGPU host: installs `build-essential`, `dkms`, the headers of the running kernel and
  `linux-headers-aws` (DKMS rebuilds the module after kernel updates); purges Ubuntu's versioned nvidia driver
  packages (`nvidia-driver-NNN*`, `nvidia-dkms-NNN*`, `nvidia-utils-NNN*`, `nvidia-compute-utils-NNN`,
  `nvidia-firmware-NNN*`, `libnvidia-*-NNN`, `xserver-xorg-video-nvidia-NNN`; `nvidia-prime`,
  `libnvidia-egl-wayland1` and the container toolkit are kept); unloads leftover nvidia modules; resolves the newest
  `latest/NVIDIA-Linux-<arch>-*-grid-aws.run` from the bucket listing, downloads it to `~/.cache/roadfm-bootstrap/`
  (skipped when a file of the listed size is already there), checks the size against the listing and the runfile's
  own checksum (`sh <file> --check`); installs it with `--silent --dkms --no-questions --ui=none` (the installer's
  default open kernel module accepts the vGPU with the GRID build); blacklists nouveau
  (`/etc/modprobe.d/blacklist-nouveau.conf`, if not already); rebuilds the initramfs with the tool the host uses
  (`update-initramfs` when `initramfs-tools` is installed, else `dracut -f --kver $(uname -r)` — Ubuntu 26.04 uses
  dracut, and the installer may rebuild none when it finds several); then `modprobe nvidia`, `modprobe nvidia-uvm`
  and `nvidia-smi`. If `nvidia-smi` still fails it asks for a reboot and a re-run. The installer also enables the
  `nvidia-persistenced` and `nvidia-gridd` services; on AWS no license configuration is needed
  (`nvidia-smi -q | grep -i license` shows `Licensed`).
- **Idempotent:** a working GRID driver (`nvidia-gridd` present, `nvidia-smi` works) is kept; `--dry-run` prints the
  plan (including the resolved bucket key and size) and changes nothing.

**Never** run `apt install nvidia-driver-*` or `ubuntu-drivers install` on a g6f / gr6f instance afterwards: it
replaces the GRID driver with one that refuses the vGPU. (The script's Ubuntu-driver path only runs on non-vGPU
hosts or with `--driver ubuntu`.)

Check afterwards with `npm run gpu:check` (it also prints a GRID hint when it finds a vGPU without a working driver).
Note that 3 GB of GPU memory is small for batch 256; check `train:smoke` before a long run, or pick a full-GPU
instance (`g6.xlarge`, `g5.xlarge`).

**Tested without AWS** in a simulated fresh instance (Docker is used only for this test, `docker/ec2-sim/`):
`npm run ec2:sim` (root, Ubuntu 26.04) and `npm run ec2:sim:ubuntu` (`ubuntu` user, 24.04) run the bootstrap twice,
then check in a new login shell: npm, `hf`, `gh`, `setup:venv:check`, `.venv` imports, both kernels, `gpu:check`
(exit 0) and `gpu:require` (exit 1, no GPU), `hf:whoami`, `npm test`, `setup:native` (stops at the missing dataset)
and `ec2:check`. GPU parts (driver, cu126 torch) can only be tested on a GPU instance.

The sections below describe the **optional Docker route** (Mac / CPU tests, or EC2 with `--docker`).

## One command

After cloning the repo and downloading the raw data (`npm run data:download` → `data/VeReMi-Dataset/`, 23,048 files),
run from the repo root:

```bash
./scripts/setup.sh                 # EC2 GPU (service gpu); or: npm run setup
./scripts/setup.sh --cpu           # Mac / CPU box (service cpu); or: npm run setup:cpu
#   --skip-build    reuse the existing image      --rebuild-data   re-run both pipeline notebooks
```

It checks Docker, `docker compose`, the GPU (`nvidia-smi`, NVIDIA container runtime) and free disk; builds the image;
checks python / torch / CUDA / Jupyter kernels in the container; checks the raw dataset is there (it never downloads
it — if missing it prints `npm run data:download` and stops); runs `pipeline:prepare` and `pipeline:encoder-input` in
the container if their outputs are missing (timed); runs `train:check`; prints the next commands (`train:smoke`,
`train:full` in tmux, `model:upload`). Safe to re-run. The sections below are the same steps by hand.

## What the image is

| Piece | Value |
|---|---|
| Base | `ubuntu:24.04`, `tini` as entrypoint |
| Python | 3.14.5 via `uv` 0.12.24, venv at `/opt/venv` (`PY=/opt/venv/bin/python`, on `PATH`) |
| Packages | `requirements/linux.txt` (portable pins matching the Mac training env, no mac-only packages; plus `pyarrow` for the legacy parquet tooling and its tests) |
| torch | `torch==${TORCH_VERSION}+${TORCH_VARIANT}` from `https://download.pytorch.org/whl/${TORCH_VARIANT}`; build args `TORCH_VERSION` (default 2.14.1, the Mac pin) and `TORCH_VARIANT`: `cu126` (service `gpu`) or `cpu` (service `cpu`). There is no 2.14.1 `cu128` wheel for Python 3.14 (cu128 stops at 2.11.0); 2.14.1 exists for `cu126` / `cu130` / `cpu`, so `cu126` is the default (driver ≥ 560) |
| Jupyter kernels | `python3` and `roadfm-train`, both pointing at `/opt/venv/bin/python` |
| Node / npm | from Ubuntu apt, so every `npm run …` works inside the container |

Compose services (`docker-compose.yml`):

- **`gpu`** — image `roadfm-lite:gpu`, all NVIDIA GPUs reserved, `shm_size: 8gb`.
- **`cpu`** — image `roadfm-lite:cpu`, no GPU (for testing on a Mac or a CPU instance).

Both bind-mount the repo at `/workspace` (the code is never baked into the image; `.dockerignore` keeps the build
context to `docker/` + `requirements/`) and mount `${HF_HOME_HOST:-~/.cache/huggingface}` at
`/root/.cache/huggingface`, which is where the HF token lives. **No data and no token are ever in the image.**

The host's `.venv` / `.venv-train` folders are visible in `/workspace` but unused: `package.json` runs Python as
`${PY:-.venv-train/bin/python}`, and the image sets `PY=/opt/venv/bin/python`, so the same npm scripts work on the
Mac (venv) and in the container (image venv).

## 0. Once, at home (Mac): put the raw data on the Hub

```bash
npm run hf:whoami        # prints the user name, never the token
npm run data:upload      # private repo, created if missing; writes MANIFEST.json (path -> size, sha256) + card
```

Re-running `data:upload` resumes an interrupted upload (`hf-xet`; already-committed files are skipped).
Nothing is written into `data/`.

## 1. Instance

- **Type:** `g6.xlarge` (NVIDIA L4, 24 GB, 4 vCPU, 16 GB RAM) or `g5.xlarge` (NVIDIA A10G, 24 GB, 4 vCPU, 16 GB RAM)
  is plenty for d = 128 (2.3 M encoder params, batch 256). `g6f` / `gr6f` are fractional L4s (vGPU; `g6f.xlarge`:
  3 GB GPU memory) and need AWS's GRID driver — see "g6f / gr6f" above. The pipeline notebooks use `multiprocessing`; a
  `g6.2xlarge` / `g5.2xlarge` (8 vCPU, 32 GB) makes data preparation faster if you care.
- **AMI, easiest:** "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)" — NVIDIA driver, Docker and the
  NVIDIA Container Toolkit are already installed. Check with `nvidia-smi` and
  `docker run --rm --gpus all ubuntu:24.04 nvidia-smi`, then go to §2.
- **AMI, plain Ubuntu 24.04:** install the three pieces yourself (§1a).
- **Disk:** root EBS volume **≥ 100 GB** gp3: 13 GB raw + ~3 GB prepared data + ~2 GB encoder input + ~10 GB image
  (CUDA torch wheels are large) + Docker build cache + checkpoints, with headroom.
- **Security group:** SSH (22) from your IP only. Nothing else needs to be open.

### 1a. Plain Ubuntu 24.04: driver, Docker, NVIDIA Container Toolkit

```bash
# NVIDIA driver (the cu126 wheels need driver >= 560; ubuntu-drivers picks the recommended server driver)
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install --gpgpu
sudo reboot
# after reconnecting:
nvidia-smi                                  # shows the GPU and "CUDA Version: 12.6" or higher

# Docker Engine + compose plugin (Docker's apt repo)
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker

# NVIDIA Container Toolkit (lets containers see the GPU)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker run --rm --gpus all ubuntu:24.04 nvidia-smi    # must show the GPU
sudo apt-get install -y tmux git
```

## 2. Code and image

```bash
git clone <this repo> sybil-llm && cd sybil-llm
docker compose build gpu                   # or: npm run docker:build (if npm is on the host)
docker compose run --rm gpu nvidia-smi     # GPU visible inside the container
docker compose run --rm gpu python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
```

The build takes several minutes the first time (≈ 3.6 GB of CUDA torch + `nvidia-*` wheels to download; expect an
image of roughly 10 GB); later builds reuse the cache. `npm` is not
needed on the host — every `npm run …` below runs **inside** the container.

## 3. Shell, Hugging Face login and data

```bash
docker compose run --rm gpu bash           # = npm run docker:shell; you are now in /workspace as root
hf auth login                              # paste a token with read (and write, for model:upload) access
npm run hf:whoami
npm run data:download                      # -> data/VeReMi-Dataset/<same tree>, verified (sizes + sha256)
```

- The token is written to the mounted `~/.cache/huggingface` on the host, so it survives the `--rm` container and
  is never in the image. Instead of `hf auth login` you can copy an existing `~/.cache/huggingface/token` to the
  instance (or point `HF_HOME_HOST` at a folder that has one).
- `data:download` refuses to write into a non-empty `data/VeReMi-Dataset` unless `-- --force`
  (`-- --no-hash` checks names + sizes only).
- Files written from the container (data, runs) are owned by root on the host; if that matters,
  `sudo chown -R $USER:$USER .` on the host.

## 4. Rebuild prepared data and encoder input

The derived data is not on the Hub; rebuild it with the two pipeline notebooks, executed in place by
`scripts/run_notebook.py` (each with its own kernel, repo root as cwd). Still inside the container:

```bash
npm run pipeline:prepare          # src/pipeline/input_representation.ipynb            -> src/data/prepared_data/
npm run pipeline:encoder-input    # src/pipeline/benign_gridsybil/encoder_input_T64.ipynb -> src/data/encoder_input/benign_gridsybil/T64/
```

On the Mac they take ≈ 1 min and ≈ 3 min; a 4-vCPU instance is slower (`multiprocessing`). Both outputs are
gitignored. The split is deterministic (seed 0, by sender vehicle), so it is the same split as at home; the
notebooks' own checks must pass (0 vehicles in both splits).

## 5. Train

Short checks, inside the container:

```bash
npm run train:check                  # pre-flight checks only (padding Δ = 0 etc.)
npm run train:one-batch              # optional: one step, run id `one-batch`
npm run train:smoke                  # 30 steps; prints the real seconds/step on this GPU
exit
```

The long run, from the **host**, inside `tmux` so it survives a dropped SSH connection:

```bash
tmux new -s train
docker compose run --rm gpu npm run train:full -- --device cuda --mem-gb 12     # = npm run docker:train (+ the args)
# detach: Ctrl-b d      reattach: tmux attach -t train
```

- **Device:** `--device auto` (default) picks MPS, then CUDA, then CPU (`pick_device` in `train.py`), so in the
  container it picks `cuda`. Pass `--device cuda` to fail loudly instead of silently falling back to CPU.
- **Memory guard:** `--mem-gb` (default 22 GB, sized for the Mac) stops the run cleanly at 95% of it. On CUDA it
  counts only the process's host RAM (RSS; GPU memory is counted on MPS only), so set it below the instance's RAM
  (e.g. 12 on a 16 GB `g5.xlarge` / `g6.xlarge`); a GPU out-of-memory shows up as a CUDA error instead.
  `--no-resource-guard` (or `npm run train:full:no-guard`) switches the guard off entirely (memory is still logged).
- **Watch the GPU:** `docker compose run --rm gpu nvidia-smi` or, on the host, `watch -n 5 nvidia-smi`.
- **Log to a file:** `docker compose run --rm gpu npm run train:full -- --device cuda --mem-gb 12 2>&1 | tee train.log`.
  If you put the log into the run folder as `<run_dir>/*.log`, `model:upload` uploads it too.
- **Interrupted:** `docker compose run --rm gpu npm run train:resume -- src/runs/pretraining/benign_gridsybil/T64/<run_id>`.
- Each run writes `env.json` (Python, torch, CUDA, packages) next to its results (RULE 4), so a run trained in the
  container is distinguishable from a Mac run.

## 6. Ship the run home

On EC2 (inside `docker compose run --rm gpu bash`):

```bash
npm run model:upload -- src/runs/pretraining/benign_gridsybil/T64/<run_id>                  # best.pt + metadata
npm run model:upload -- src/runs/pretraining/benign_gridsybil/T64/<run_id> --include-last   # + last.pt (to resume)
```

One commit per run under `runs/<run_id>/` with a `SHA256SUMS`; it prints the URL and commit id. Without `best.pt`
it uploads `last.pt` and says so.

At home:

```bash
npm run model:list
npm run model:download -- <run_id>     # -> src/runs/pretraining/benign_gridsybil/T64/<run_id>/, sha256-verified
```

The local folder has the same layout as a run trained here, so `pretrain_monitor.ipynb` and `train:resume` work on
it. It refuses to overwrite an existing run folder unless `-- --force`.

## 7. Costs and stopping

- On-demand prices are roughly $0.8 / h (`g6.xlarge`) and $1.0 / h (`g5.xlarge`) in us-east-1; check the current
  price for your region. Spot is cheaper but can be interrupted — then resume with `train:resume` from `last.pt`.
- **Stop the instance as soon as the run is uploaded.** A stopped instance costs nothing for compute, but the EBS
  volume is still billed (gp3 ≈ $0.08 / GB-month, so ≈ $8 / month for 100 GB). Terminate it (and delete the volume)
  when you no longer need the data there — the raw data and the runs are on the Hub.
- Set a billing alarm (AWS Budgets) before a 30-hour run.

## 8. Testing the image on a Mac or a CPU instance

```bash
npm run docker:build:cpu      # docker compose build cpu
npm run docker:check:cpu      # docker compose run --rm cpu npm run train:check
npm run docker:shell:cpu      # docker compose run --rm cpu bash
```

No GPU there, so this proves the environment and the data path, not CUDA speed. Docker Desktop's VM memory limit
applies (lower `--mem-gb` accordingly).

Measured on an M-series Mac (Docker Desktop, arm64, 8 GB VM, 2026-10-08): `docker compose build cpu` ≈ 1.5 min
from scratch, image `roadfm-lite:cpu` 2.8 GB; `train:check` 12 s (all pre-flight checks pass, encoder 2,301,312
params); `python -m pytest -q scripts/tests` 66 passed; `npm run train:smoke -- --max-steps 3` ≈ 10 s/step on CPU,
0.8 GB RSS. The `gpu` image (`cu126`, amd64) was resolved for linux/x86_64 but not built or run there (no NVIDIA
GPU, too little disk) — the first real test of it is `nvidia-smi` + `torch.cuda.is_available()` in §2.

## Appendix — the venv route (Mac)

On the Mac nothing changes: the npm scripts fall back to `.venv-train/bin/python` when `PY` is unset.

```bash
npm run setup:venv                  # .venv-train (requirements-train.txt, MPS) + .venv (requirements.txt) + kernels
npm run setup:venv:check            # report only; fails on a pin mismatch
npm run train:check && npm run train:smoke
```

`requirements-train.txt` is a macOS freeze; on Linux use the Docker image (`requirements/linux.txt` + torch from the
PyTorch index) rather than this file.
