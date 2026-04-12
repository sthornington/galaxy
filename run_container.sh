#!/usr/bin/env bash
set -euo pipefail

# Use CDI (Container Device Interface) for GPU injection instead of --gpus all.
# --gpus all uses the nvidia-container-runtime hook which breaks when
# systemctl daemon-reload runs on the host (cgroup driver disruption).
# CDI injects devices via container config, surviving daemon reloads.
#
# Prerequisites (one-time, on the host):
#   1. Generate CDI spec:  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
#   2. Enable CDI in Docker: add to /etc/docker/daemon.json:
#      { "features": { "cdi": true } }
#      Then: sudo systemctl restart docker
#
# Fallback: if CDI is not configured, uncomment the --gpus all line below.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-galaxy}"
IMAGE_NAME="${IMAGE_NAME:-galaxy:latest}"
HOST_GALAXY_DIR="${HOST_GALAXY_DIR:-$SCRIPT_DIR}"
HOST_HTTP_PORT="${HOST_HTTP_PORT:-8080}"
HOST_JUPYTER_PORT="${HOST_JUPYTER_PORT:-8889}"
CONTAINER_HTTP_PORT="${CONTAINER_HTTP_PORT:-8080}"
CONTAINER_JUPYTER_PORT="${CONTAINER_JUPYTER_PORT:-8889}"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run \
    --device nvidia.com/gpu=all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name "$CONTAINER_NAME" \
    -p "${HOST_JUPYTER_PORT}:${CONTAINER_JUPYTER_PORT}" \
    -p "${HOST_HTTP_PORT}:${CONTAINER_HTTP_PORT}" \
    -e SIM_SERVER_ADDR="0.0.0.0:${CONTAINER_HTTP_PORT}" \
    -v "$HOME/.cache:/home/sthornington/.cache" \
    -v "$HOME/.claude:/home/sthornington/.claude" \
    -v "$HOME/.claude.json:/home/sthornington/.claude.json" \
    -v "$HOME/.codex:/home/sthornington/.codex" \
    -v "$HOST_GALAXY_DIR:/galaxy" \
    "$IMAGE_NAME"
