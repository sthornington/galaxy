#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RUST_LOG="${RUST_LOG:-sim_server=info,tower_http=info}"
exec cargo run -p sim-server --manifest-path "$ROOT/Cargo.toml"
