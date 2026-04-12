#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/crates/sim-server/static/viewer"

wasm-pack build "$ROOT/crates/sim-viewer" \
  --target web \
  --out-dir "$OUT_DIR" \
  --release
