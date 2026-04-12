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

The WASM toolchain is not present in the current container, so the viewer build
is staged but not included in the default workspace checks. Install
`wasm-pack` and the `wasm32-unknown-unknown` target before building the browser
bundle.
