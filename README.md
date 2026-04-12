# Galaxy

Greenfield workspace for a GPU-first galactic evolution simulator targeting the
NVIDIA GB10 / DGX Spark.

## Workspace layout

- `crates/sim-core`: shared physics units, configuration, presets, preview
  payloads, snapshot manifests, and initial-condition generation.
- `crates/sim-cuda`: CUDA-backed simulation kernel shim exposed through a narrow
  Rust FFI wrapper.
- `crates/sim-server`: `axum` control plane that serves the browser client,
  manages sessions, and streams preview frames.
- `crates/sim-viewer`: Rust/WASM browser client intended to render streamed
  frames with WebGPU.

## Current status

This first implementation pass establishes the full project structure and a
working control-plane contract. The CUDA backend is intentionally a baseline:
it allocates GPU-resident particle buffers, advances particles with a simple
accelerator model, and emits preview samples for the browser. The full TreePM +
weak-field relativistic solver remains the next major implementation step.

## Local development

```bash
cargo check
cargo run -p sim-server
```

The current container image includes the WASM toolchain needed for
`sim-viewer`. If you build in an older image, install `wasm-pack` and the
`wasm32-unknown-unknown` target before building the browser bundle.

## Viewer build and smoke tests

```bash
./scripts/build-viewer.sh
node --test /galaxy/tests/ui-headless.test.mjs
```

For a live end-to-end check without opening a browser, start `sim-server` and
run:

```bash
python ./scripts/live_headless_smoke.py
```
