#!/bin/bash
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

docker run --device nvidia.com/gpu=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name galaxy -p 8889:8889 \
    -v $HOME/.cache:/home/sthornington/.cache \
    -v $HOME/.claude:/home/sthornington/.claude \
    -v $HOME/.claude.json:/home/sthornington/.claude.json \
    -v $HOME/.codex:/home/sthornington/.codex \
    -v ~/projects/galaxy:/galaxy galaxy:latest
