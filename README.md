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

The CUDA backend is a full TreePM solver:

- **Long range**: FFT particle-mesh with a free-space (isolated) Green's
  function — no periodic-image artifacts — CIC deposit/deconvolution, and a
  Gaussian force split.
- **Short range**: an FMM-style interaction-list walk over a baryon-tight cell
  grid plus a coarsened mass/COM pyramid, giving exact tiling out to the
  erfc-matched cutoff. Dense neighborhoods use exact pair sums for the local
  cell and octant monopoles for neighbors.
- **Integration**: kick-drift-kick leapfrog with merged interior kicks and
  CFL-adaptive substepping; SMBHs are direct point sources with optional 1PN
  corrections.
- **Diagnostics**: kinetic and mesh-sampled potential energy per step; the
  isolated `major-merger-debug` galaxy conserves total energy to ~0.1% over
  tens of Myr and holds its disk structure.

Initial conditions come from Jeans-equilibrium analytic samplers in
`sim-core` (NFW halo, exponential disk with Toomre-Q dispersion and
asymmetric drift, Hernquist bulge).

Solver validation tools live in `crates/sim-cuda/src/bin`:

```bash
# per-particle force accuracy vs a brute-force direct sum
cargo run --release -p sim-cuda --bin force_check -- --samples 300

# time-domain stability / energy-conservation metrics
cargo run --release -p sim-cuda --bin solver_diag -- \
  --preset major-merger-debug --steps 100 --batch 25 --no-relax --analytic-ics
```

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
