# RoadFM-Lite runtime image (Ubuntu 24.04). The repo is bind-mounted at /workspace; no code, data or secrets are baked in.
#   GPU (EC2, NVIDIA):  docker compose build gpu   (TORCH_VARIANT=cu126)
#   CPU (Mac / CPU box): docker compose build cpu   (TORCH_VARIANT=cpu)
# torch 2.14.1 (the Mac pin) has no cu128 wheel for cp314 on download.pytorch.org (cu128 stops at 2.11.0);
# 2.14.1 exists for cu126 / cu130 / cu132 / cpu, so the GPU default is cu126 (works with NVIDIA drivers >= 560).
FROM ghcr.io/astral-sh/uv:0.12.24 AS uv

FROM ubuntu:24.04

ARG TORCH_VARIANT=cu126
ARG TORCH_VERSION=2.14.1
ARG PYTHON_VERSION=3.14.5

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git build-essential nodejs npm tini \
 && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

RUN uv python install "${PYTHON_VERSION}" \
 && uv venv /opt/venv --python "${PYTHON_VERSION}"

COPY requirements/linux.txt /tmp/requirements-linux.txt
RUN uv pip install --python /opt/venv/bin/python -r /tmp/requirements-linux.txt \
 && uv pip install --python /opt/venv/bin/python \
      --index-url https://pypi.org/simple \
      --extra-index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" \
      "torch==${TORCH_VERSION}+${TORCH_VARIANT}" \
 && rm /tmp/requirements-linux.txt

# Notebooks use both kernel names; both point at /opt/venv/bin/python.
RUN /opt/venv/bin/python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3" \
 && /opt/venv/bin/python -m ipykernel install --sys-prefix --name roadfm-train --display-name "Python (roadfm-train)"

ENV PY=/opt/venv/bin/python \
    PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
ENTRYPOINT ["tini", "--"]
CMD ["bash"]
