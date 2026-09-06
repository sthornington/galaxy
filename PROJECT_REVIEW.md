# Galaxy: project review and improvement plan

Reviewed 2026-09-06 at commit `c4f8468`, on the NVIDIA GB10 host. This plan balances scientific credibility, performance, and the interactive experience. Application source was not changed.

The strongest next version would combine reproducible physics, predictable performance through dense late-stage encounters, and a simulation-first interface that helps people understand what they are seeing. The existing four-crate architecture is a useful foundation: preserve the narrow CUDA interface, persistent GPU state, isolated PM gravity, packed previews, and browser GPU rendering. Focus the next work on correctness contracts, measurable bottlenecks, and presentation.

## Review scope and observed baseline

The review covered all four crates, native CUDA and FFI, initial conditions and all 16 presets, HTTP/WebSocket/session handling, snapshot persistence, all four browser rendering paths, tests, scripts, and container/build configuration. Independent reviews covered CUDA, server, and frontend, with integration checks and targeted reproductions.

| Check | Observed result |
| --- | --- |
| `cargo test --workspace --all-targets` | 16 passed; nine GPU tests ignored by default |
| `cargo test --workspace --all-targets -- --ignored --test-threads=1` | All nine GPU tests passed |
| Node UI and preview-worker tests | All five passed |
| `cargo check -p sim-viewer --target wasm32-unknown-unknown` | Passed |
| `cargo clippy --workspace --all-targets` | Passed with warnings; not warning-clean |
| `cargo fmt --all -- --check` | Failed: existing formatting differences |
| Actual Chromium/SwiftShader WebGL rendering | 512 particles drawn, 12+ rendered frames, 12,994 lit pixels, no browser or GL errors |
| Targeted CPU, GPU, and browser probes | Reproduced the defects described below despite the passing suite |

Browser checks used a separate server on `127.0.0.1:18081` and a 4,098-particle fixture. The test session and server were stopped afterward. The existing server on port 8080 was left undisturbed. Hardware WebGPU/HDR, Safari, and long-horizon convergence were not validated in this review.

Current release measurements:

- `force_check --samples 300`: relative force error **p50 1.89%, p90 6.14%, p99 16.85%, maximum 89.02%**. This is the harness's gravity-only, reorder-disabled, PM-refresh-every-substep configuration. The cause of the large tail remains to be established; these measurements do not prove which approximation is responsible.
- `solver_diag --preset major-merger --steps 12 --batch 4 --no-relax --analytic-ics`: **2,540,002 particles**, with successive four-step advances taking **0.894, 0.833, and 0.817 seconds**, or approximately **204–224 ms/base step**. These are short, early-state measurements; they exclude initial-condition generation and the subsequent full particle download.
- A four-step run with existing stage profiling enabled took 0.906 seconds. Opening/interior kicks accounted for approximately **508 ms**, short-range structure builds **178 ms**, SPH **98 ms**, and PM builds **52 ms**. Nested stage totals must not be added again. Profiling inserts synchronization and therefore changes execution behavior.
- Existing historical `output/knot-gasgrid.log` records a late state at 400 Myr with **~778 ms/step**, including **~723 ms in the opening kick**, versus ~17 ms mesh and ~14 ms SPH. This is historical evidence for where to investigate, not a fresh comparable benchmark.
- An isolated 134,401-particle debug galaxy ran to 20 Myr and retained broadly similar disk size/spin, but its RMS disk height changed from 0.350 to 0.284 kpc and oscillated during the run. The reported energy ratio ended at 0.999193 relative to the first measured batch. Because of the diagnostic limitations below, this is a structural smoke result, not a certified conservation measurement.

## Findings to address first

P1 means a reproducible behavior or clear code defect that can change scientific results, lose work, or misdirect user actions. P2 means important reliability, performance, or product work. Deployment exposure is called out separately because its urgency depends on how the host is accessed.

### 1. P1 — Star formation and feedback depend on API batching and force-build count

The internal simulation clock advances only after the entire `run_steps` loop, while births and young-star feedback read that clock inside the loop. Star formation runs every 16 force builds and estimates elapsed time as `16 * base_timestep * 0.5`, regardless of actual adaptive substeps. The server adaptively changes batch sizes, so this affects a normal execution path.

Evidence: [clock update](/galaxy/crates/sim-cuda/native/src/sim_cuda.cu:4400), [feedback clock](/galaxy/crates/sim-cuda/native/src/sim_cuda.cu:4001), [formation scheduling](/galaxy/crates/sim-cuda/native/src/sim_cuda.cu:4033), [server batching](/galaxy/crates/sim-server/src/session.rs:336).

A controlled 64-gas-particle fixture with negligible gravity and deliberately strong formation produced:

| Same final physical time: 64 Myr | Gas remaining | Stars formed | Stellar age ticks |
| --- | ---: | ---: | --- |
| One `advance(64)` | 21 | 43 | All 10 |
| 64 calls to `advance(1)` | 21 | 43 | 0–7 |
| 64 calls to `step(2)` | 4 | 60 | 0–8 |

The first two also differed in velocity by up to 0.000178613144 km/s because of feedback timing. These exaggerated test parameters isolate bookkeeping defects; they do not quantify default-preset error.

Change: advance physical time at defined integration boundaries; accumulate actual elapsed time for source terms; key stochastic events to persistent particle identity and physical epochs. Keep physical scheduling independent of preview capture and API call boundaries.

Acceptance: gas/SF/feedback runs agree across different batch partitions within declared numerical tolerances; timestep refinement preserves the expected conversion rate statistically. Include late stellar ages and feedback expiration, not just four initial gravity-only steps.

### 2. P1 — Saving can silently fail, overwrite the last save, or terminate the simulation

`write_particle_snapshot` gives a `BufWriter` to `serialize_into` by value; dropping the writer discards final flush errors. An isolated `/dev/full` probe returned success and published a manifest after the particle payload failed to flush. Subsequent saves truncate the same `particles.bin`, so an interrupted replacement can destroy the last valid export. When an I/O failure is detected, the server treats it as a fatal session error and drops the live GPU state.

Evidence: [buffered write](/galaxy/crates/sim-core/src/snapshot.rs:43), [fixed output filenames](/galaxy/crates/sim-server/src/session.rs:687), [snapshot command](/galaxy/crates/sim-server/src/session.rs:534), [fatal command handling](/galaxy/crates/sim-server/src/session.rs:495).

Change: explicitly flush and check errors, sync according to the durability contract, write a new immutable generation, and atomically publish its manifest last. Preserve the previous generation until the new one is complete. Distinguish save errors from fatal CUDA errors.

Acceptance: injected flush/disk-full/permission/interrupted-write failures return errors, leave the live simulation usable, and preserve the last valid saved generation.

### 3. P1 — Stale browser requests can switch controls back to the wrong session

`updateSessionStats` changes `activeSessionId` for every response. Poll and command responses have no session-generation guard. Reproduced: request session A's summary, select B, then resolve A's old response; the active session becomes A again. A subsequent Pause or Stop can target A.

Evidence: [selection overwritten by statistics](/galaxy/crates/sim-server/static/ui-app.mjs:340), [unguarded polling](/galaxy/crates/sim-server/static/ui-app.mjs:371), [control responses](/galaxy/crates/sim-server/static/ui-app.mjs:1015).

Change: separate selected-session identity from statistics; introduce an explicit lifecycle generation and abort obsolete work. Capture and verify identity for every poll, command, renderer boot, and completion callback.

Acceptance: delayed/reordered responses, rapid selection/launch, and stop-during-boot cannot change the chosen session or direct a command to an obsolete one.

### 4. P1 — Physical time is incorrectly used as frame identity

The worker and all renderers discard frames whose simulation time has not increased. A paused preview-budget change correctly publishes a new frame at the same physical time, which clients reject. Real Chromium reproduced **3,074 particles reported by the server/UI while only 512 remained drawn**. Paused delta resync also waits for a future broadcast instead of sending the cached frame immediately.

Evidence: [worker deduplication](/galaxy/crates/sim-server/static/webgl-stream-worker.js:113), [WebGL](/galaxy/crates/sim-server/static/webgl-viewer.mjs:901), [WebGPU](/galaxy/crates/sim-server/static/webgpu-viewer.mjs:1001), [WASM](/galaxy/crates/sim-viewer/src/lib.rs:343), [JSON](/galaxy/crates/sim-server/static/ui-app.mjs:571), [server resync](/galaxy/crates/sim-server/src/app.rs:493).

Change: define a shared packet contract with session/stream generation, monotonic frame sequence, sampling generation, simulation time, and explicit delta-base sequence. Permit authoritative same-time replacement and reset interpolation when particle identity/layout changes. Serve the latest keyframe immediately on resync.

Acceptance: paused increases/decreases, same-time resync, reconnect, changed sampling, and delayed frames work across every renderer; displayed counts describe what was actually rendered.

### 5. P1 — Fallback lifecycle and animation scheduling are incorrect

JSON session attachment does not clear prior render state. Reproduced: switching from a session at 10 Myr to a new one at 0 Myr displays the old particles until the new simulation passes 10 Myr. Its frame handler also calls the render callback directly, clears an existing animation handle without cancelling it, and schedules another callback. Twenty-one incoming frames produced 21 outstanding callbacks instead of one.

After initial rendering, GPU/WASM errors have no persistent listener: the loader removes its only error listener once the first frame arrives. A browser probe disconnected the worker after startup; the viewer did not reconnect or fall back, and still displayed “Streaming.”

Evidence: [JSON attachment](/galaxy/crates/sim-server/static/ui-app.mjs:882), [animation scheduling](/galaxy/crates/sim-server/static/ui-app.mjs:550), [first-frame listener cleanup](/galaxy/crates/sim-server/static/viewer-loader.mjs:43).

Change: one persistent lifecycle controller with explicit reset/dispose, session-scoped events, one outstanding animation callback, reconnect/backoff, and deliberate fallback. Handle WebGL context loss and WebGPU device loss.

Acceptance: session changes clear stale imagery immediately; 1,000 incoming frames never accumulate animation loops; post-startup failures transition through visible recovery states and resume rendering or report a useful error.

### 6. P1 — Initial timestep selection and the SMBH correction need correction

The first adaptive timestep is chosen before the first force build, with acceleration buffers initialized to zero. Initially stationary particles therefore receive one initial substep regardless of gravitational acceleration. Separately, the 1PN correction negates its radial term even though displacement is defined as target minus source. At zero relative velocity, that correction has the opposite direction from the test-particle expression.

Evidence: [initial timestep ordering](/galaxy/crates/sim-cuda/native/src/sim_cuda.cu:4283), [1PN displacement and sign](/galaxy/crates/sim-cuda/native/src/sim_cuda.cu:1448). The sign was checked against the primary [REBOUNDx general-relativity implementation](https://github.com/dtamayo/reboundx/blob/main/src/gr_full.c).

Change: calculate the initial acceleration before choosing the timestep. Add acceleration/softening and gas signal-speed criteria with explicit reporting when configured caps prevent satisfying them. Correct and validate the velocity-dependent relativistic integration; document or replace the test-mass approximation for comparable SMBHs. Check source-velocity reads against concurrent updates using immutable source state or a suitable kernel separation.

Acceptance: cold-collapse first-step refinement, two-body orbital convergence, and periapsis-precession tests; compare PN on/off and comparable-mass cases against an independent reference. The possible concurrent velocity-read issue needs targeted validation; it was not reproduced here.

### 7. P1/P2 — Configuration accepts values that cannot be honored safely or correctly

Validation largely counts particles. A legal JSON `target_fps=u32::MAX` becomes a zero-nanosecond interval and can panic the detached session driver, leaving its summary falsely paused. Huge mesh dimensions can overflow the server's product before CUDA admission. Counts, preview budgets, and local snapshot loads have no unified resource policy.

Several advertised controls are inert or misleading: `opening_angle` is stored but never used; `gas.viscosity_alpha=0` silently selects 1.0; `SnapshotConfig.cadence_steps`, SMBH-specific `substeps`, and weak-field/observer-effect configuration fields have no operative consumers. Existing artistic rendering effects do not establish that those configuration switches work.

Evidence: [validation](/galaxy/crates/sim-core/src/init.rs:1193), [mesh and FPS handling](/galaxy/crates/sim-server/src/session.rs:329), [opening angle](/galaxy/crates/sim-cuda/native/src/sim_cuda.cu:4478), [viscosity fallback](/galaxy/crates/sim-cuda/native/src/sim_cuda.cu:4568), [configuration surface](/galaxy/crates/sim-core/src/config.rs).

Change: one validated configuration type, checked arithmetic, finite/range checks, count/mass consistency, supported-feature reporting, and resource estimates before IC generation or allocation. Supervise driver tasks and expose initialization/failed/stopped states with error details. Reject unsupported settings or implement their documented behavior.

Also fix [UTF-8 log truncation](/galaxy/crates/sim-server/src/app.rs:220): 1,999 ASCII characters followed by `é` reproduces a panic because byte 2000 is not a character boundary.

### 8. P1/P2 — Initial conditions do not consistently match the simulated matter distribution

Three direct inconsistencies were found:

- Uniform-sphere generation ignores configured gas. A small fixture validated as 30 particles generated 17, with zero gas particles. [Sphere generator](/galaxy/crates/sim-core/src/init.rs:269).
- The shared orbital-mass helper omits gas. With actual component masses, the M51 preset starts with approximately **0.553 km/s** net bulk speed; the existing barycenter test repeats the same mass omission. [Mass helper](/galaxy/crates/sim-core/src/preset.rs:163), [test](/galaxy/crates/sim-core/src/lib.rs:276).
- Gas is folded into the stellar disk potential at the stellar scale radius/height, while particles are sampled at a different gas radius/height and reuse the stellar velocity table. Changing the isolated galaxy's gas scale radius from 5.6 to 40 kpc left all non-gas relative velocities unchanged in a deterministic probe. [Potential construction](/galaxy/crates/sim-core/src/init.rs:164), [gas sampling](/galaxy/crates/sim-core/src/init.rs:222).

The stellar disk additionally uses a Miyamoto–Nagai approximation for a sampled tapered exponential disk. Truncation, dispersion floors/caps, and spherical Jeans approximations should be characterized, rather than assumed to create an exact equilibrium. Snapshot-based galaxy initialization also does not apply configured disk tilt.

Change: represent stellar and gas contributions separately, normalize sampled/truncated mass profiles consistently, include gas in all mass/orbit helpers, and either implement or reject gas for spheres. Construct pressure-supported gas kinematics for the implemented equation of state. Cache validated equilibrium states with parameter/build provenance.

Acceptance: component count/mass and profile tests for every preset at small resolution; gas fraction/scale changes affect the expected force and rotation curves; isolated runs preserve agreed density, scale-height, spin, and velocity profiles over multiple dynamical times at increasing resolution.

### 9. P1 for scientific claims — Current diagnostics do not establish the advertised accuracy

Kinetic energy and momentum are sampled with deferred closing kicks still pending; potential energy is refreshed separately and cached. Its mesh estimate is not an independent energy calculation for the full softened force model. `solver_diag` uses the first completed batch as its energy baseline, hiding any earlier change. Gas, viscosity, and feedback also require energy accounting appropriate to their dissipative/source terms.

The force checker prints large errors and still exits successfully. It disables production reordering and PM reuse, and randomly sampling particles can miss rare SMBHs, dense cores, boundaries, and close pairs.

Evidence: [deferred kicks and diagnostics](/galaxy/crates/sim-cuda/native/src/sim_cuda.cu:4364), [potential caching](/galaxy/crates/sim-cuda/native/src/sim_cuda.cu:4400), [diagnostic baseline](/galaxy/crates/sim-cuda/src/bin/solver_diag.rs:89), [force-check setup/output](/galaxy/crates/sim-cuda/src/bin/force_check.rs:55).

Change: add a trusted diagnostic mode with synchronized phase space, timestamps/staleness, a true t=0 baseline, and independent small-N force/energy references. Stratify error reporting by component, radius, density, softening, and acceleration magnitude. Exercise optimized production paths through stable particle identity. Give accuracy tools machine-readable output and failure thresholds.

Acceptance: publish predeclared tolerances and measured convergence for each supported physics mode. Treat the observed 16.85% p99 force error and 89.02% tail as investigation targets before stronger accuracy claims or performance/accuracy tradeoffs.

## Reliability, memory, and maintainability work

| Area | Finding and practical change |
| --- | --- |
| Session ownership | Frame sockets retain a `SessionHandle` containing their own broadcast sender, preventing driver exit from closing the receiver. Add explicit cancellation/terminal-state propagation and keep producer ownership in the driver. [Handle](/galaxy/crates/sim-server/src/session.rs:167), [socket](/galaxy/crates/sim-server/src/app.rs:461). |
| Command semantics | Full fire-and-forget queues return success after dropping commands; acknowledged commands have no deadline. Return overload/failure explicitly, coalesce replaceable budget updates, and use bounded enqueue/ack deadlines. [Queue](/galaxy/crates/sim-server/src/session.rs:188). |
| Preview state | The server can mark a preview pending when the backend schedules none for zero output. Old preview diagnostics can overwrite newer simulation diagnostics. Model pending/empty/ready explicitly and track simulation and displayed-frame times separately. [Request](/galaxy/crates/sim-server/src/session.rs:622), [collection](/galaxy/crates/sim-server/src/session.rs:630). |
| Retained initialization memory | The session owns the original `InitialConditions` vector throughout its lifetime after GPU upload. `Particle` measures 88 bytes here: the current major merger retains approximately **213 MiB**, grand merger approximately **395 MiB**, before other buffers. Explicitly release it after upload and run backend construction off the async worker. [Initialization](/galaxy/crates/sim-server/src/session.rs:318). |
| Browser memory | The 12-frame GPU ring plus four transfer buffers/two worker references can require **768 MiB + 384 MiB** for records alone at 4,194,304 particles. Capacities only grow. Bound queues by bytes, size ring depth by measured jitter, and release oversized buffers after budget reductions. [GPU ring](/galaxy/crates/sim-server/static/webgl-viewer.mjs:22), [worker buffers](/galaxy/crates/sim-server/static/webgl-stream-worker.js:22). |
| Adaptive quality | Recovery requires animation intervals below 12 ms, which a normal 60 Hz display cannot provide even when GPU rendering is fast. Measure actual render/upload cost and display cadence; verify resolution recovery at 60 Hz. Stop unnecessary redraws when paused/static or hidden. [Quality controller](/galaxy/crates/sim-server/static/webgl-viewer.mjs:1172). |
| Restart semantics | Current snapshots are particle exports, not resumable checkpoints: no complete config/seed/birth-time/hydro/integrator state, and load resets time and stellar ages. Define an actual checkpoint contract and validate continuation against uninterrupted runs. [Manifest](/galaxy/crates/sim-core/src/snapshot.rs:15), [particle data](/galaxy/crates/sim-core/src/init.rs:32). |
| Fallback cost/parity | Canvas paths repeatedly allocate/project/sort particles despite additive blending; WASM also makes many string/canvas calls. Compute camera basis once, reuse arrays/sprites, benchmark tier ranking, and define common gas/star/SMBH/style semantics. [JSON draw](/galaxy/crates/sim-server/static/ui-app.mjs:473), [WASM draw](/galaxy/crates/sim-viewer/src/lib.rs:924). |
| Native boundaries | Allocation ownership and exception/error propagation are manual. Introduce RAII buffers and complete C-export exception boundaries; propagate timestep-estimation failures explicitly. Split integration, PM, short-range, hydro/SF, preview, and FFI into focused native units while preserving the public boundary. |
| Duplicated browser behavior | Transport, camera, lifecycle, interpolation, and quality logic are implemented repeatedly. Extract shared mechanics behind renderer adapters after behavioral tests exist. Keep shaders/API calls specific to each renderer. |

Before exposing the application beyond a trusted development environment, address the actual deployment defaults: [server binding](/galaxy/crates/sim-server/src/main.rs:19) is all interfaces, [CORS](/galaxy/crates/sim-server/src/app.rs:245) is permissive, and session controls lack authentication/origin checks. Client-supplied equilibrium manifests and chunks can address local filesystem paths. The [container command](/galaxy/Dockerfile:115) disables Jupyter authentication, while [the launcher](/galaxy/run_container.sh:34) publishes it and mounts developer configuration directories.

Provide explicit local-development and shared-deployment modes. Use loopback/token protection for development access, authenticated ownership and origin policy for shared use, server-owned snapshot IDs/roots, and admission quotas before allocation. A runtime image should start the simulation server and package its assets; keep Jupyter, ML packages, ClickHouse, and development tools in a separate development image. The launcher should make replacing an existing container explicit rather than always removing it first.

## Plan for making it faster

Measure simulation, transport, and rendering separately. Report simulated Myr/second, base-step and substep p50/p95, GPU memory, initialization latency, snapshot stalls, encoded bytes/second, delivery latency, upload/render time, dropped frames, and client memory. Record hardware, build, seed, configuration, environment tuning, particle count, and state age with each result.

1. **Establish representative benchmarks.** Small correctness fixtures; full major merger early and near encounter; a dense late SMBH/gas state; gas-free M87; one WAN client and multiple subscribers. Turn historical logs into reproducible state/config fixtures. Use GPU events and a timeline profiler; existing synchronized stage timers are useful for attribution but distort overlap. This follows NVIDIA's guidance on [profiling and asynchronous transfers](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).

2. **Bound dense-cell interaction work.** The own-cell direct loop remains quadratic even when neighbors use approximations: total work can scale with the sum of squared cell occupancies. Add occupancy/traversal counters, then prototype adaptive subdivision with bounded leaf occupancy and a meaningful accuracy criterion. Compare cooperative/tiled leaf evaluation and load balancing. This is the leading algorithmic candidate given both current kick timings and the historical dense-state result. [Dense-cell path](/galaxy/crates/sim-cuda/native/src/sim_cuda.cu:2335).

3. **Remove avoidable allocation and synchronization.** Drop uploaded ICs; combine six gas-bound reductions into one aggregate; reuse sort/reduction scratch storage; profile radix sorting using only required key bits. Evaluate hot/cold particle fields and reorder traffic before a larger data-layout change. Preserve force equivalence throughout.

4. **Make preview throughput a budgeted service.** At 500,000 particles, a 16-byte-record keyframe is ~8 MB; 30 such frames/second would be ~240 MB/s before overhead. Delta compression helps, but the current one-frame-per-ack protocol also limits throughput to roughly one frame per RTT. Measure first, then consider a small bounded credit window, byte-budgeted ring buffers, camera-aware sampling, and sparse age/component updates that avoid unnecessary full keyframes. Separate global simulation accuracy from per-viewer display quality.

5. **Optimize visible client cost.** Fix accumulated animation loops and quality recovery first. Use actual GPU timings where available, reduce overdraw/bloom passes when they dominate, stop idle work, and adapt particle count/resolution/ring depth together. Consolidate fallback projection and sprite reuse. Budget GPU and worker memory explicitly.

6. **Consider launch-overhead optimizations after attribution.** CUDA Graphs may help repeated short kernel sequences, especially small/debug scenes, but require stable allocation/control paths. They are a later experiment; the measured heavy kicks remain the first target. NVIDIA's [CUDA Graphs example](https://developer.nvidia.com/blog/cuda-graphs/) demonstrates the launch-overhead case.

Acceptance for performance work: compare identical states and declared error budgets; report early and late p95 as well as averages. A useful initial engineering target is a 30% reduction in measured steady-state step time and materially smaller dense-state tails, but this is a target to test, not a promised speedup. Accept no improvement that silently changes source-term timing or exceeds the scientific tolerance.

## Plan for making it more correct

Build three layers of validation:

- **Fast deterministic contracts:** configuration limits; component mass/counts; packet identity and corruption; session state transitions; snapshot failure handling; stale browser responses; one animation loop; every advertised setting either changes behavior or is rejected.
- **Small independent physics references:** softened two-body forces/orbits; force/energy/momentum comparisons; cold collapse; reorder/batch invariance; time-bin and PM-refresh convergence against a synchronized reference; isothermal SPH equilibrium/acoustic/shock tests appropriate to the implemented equation of state; formation-rate and feedback-clock tests; PN precession where enabled.
- **Long-run convergence:** isolated galaxy density/rotation/height/spin, merger trajectories, dense gas structure and star-formation histories across timestep, particle count, mesh resolution, smoothing, and opening tolerance. Dissipative gas runs need an explicit energy/source budget. The GADGET-4 [parameter documentation](https://wwwmpa.mpa-garching.mpg.de/gadget4/05_parameterfile/) is a useful primary reference for separating gravity and hydrodynamic accuracy controls.

Publish a concise physics-methods document stating units, force softening/splitting, approximation domains, isothermal assumptions, pressure-floor behavior, feedback model, numerical tolerances, and limitations of illustrative presets. Distinguish measured stability from equilibrium and scientific fidelity. Persist build/config/seed and resolved tuning values with each run.

CI should run ordinary Rust/Node/wasm checks without a GPU; use a small backend interface to exercise the server state machine in CPU tests. Add serialized GB10 GPU validation and a scheduled longer convergence/benchmark job. Make failures actionable with machine-readable output and stable scene fixtures.

Fix test infrastructure before trusting its coverage: the current API and live “small” smoke fixtures retain 224,000 inherited gas particles; the API fixture's absolute temporary snapshot root is silently rejected, so it writes under project output. The browser smoke resumes an existing session and logs rendering measurements without asserting them. Give each test an isolated bounded fixture, meaningful assertions, injected storage roots, and guaranteed cleanup. Pin browser dependencies and the Rust/build toolchain; build/package WASM assets reproducibly.

## Plan for making it more impressive

The immediate visual improvement is layout and interpretation. All 16 preset cards currently precede the canvas, placing the live simulation well below the initial viewport. Put a large live canvas on the first screen with a compact playback bar and a collapsible preset rail. Use thumbnails, a short scene description, and loading progress. Offer a quick lower-resolution preview or validated cached initial state while preparing a large run.

Then develop a small set of complete experiences:

| Experience | What the user gains | Dependencies |
| --- | --- | --- |
| Curated encounters | Three to five polished scenes with camera bookmarks, tracked galaxy/SMBH centers, scale bar, fullscreen, and restrained overlays | Stable renderer/session lifecycle; tune exposure on SDR/HDR |
| Scientific inspection | Stars/gas/halo/SMBH toggles; density, velocity, and age legends; rotation curves, separation, star-formation history, and trustworthy diagnostics | Correct diagnostics; explicit particle/display semantics |
| Timeline and comparison | Time bookmarks, recorded-preview scrubbing, checkpoint resume, and side-by-side parameter experiments | Persistent IDs, durable checkpoints, coherent clocks |
| Shareable results | Screenshot/video export and a run manifest carrying preset, seed, config, build, time, and camera | Reproducibility metadata and a clear export flow |
| Accessible control | Keyboard and pointer/touch camera controls, labeled controls, readable focus states, and useful loading/error/reconnect feedback | Shared interaction controller and UI tests |

Prioritize visual structure over additional glow: validate exposure/bloom against bright nuclei and faint tidal material at several sampling budgets. The historical root screenshot contains a clipped white nucleus; use it as a regression reference, not as proof about today's renderer. Keep illustrative color mappings and physical diagnostics clearly described.

Preset names also need provenance and generated metadata. The “5.5M” grand-merger preset currently contains **4,710,002** particles; the README's major-merger count predates gas and says 2.24M rather than **2,540,002**. Some comments still say SPH is forthcoming, and README renderer guidance omits WebGPU. Derive counts from config, document the implemented features, and describe Messier scenes as illustrative unless their parameters and comparisons support stronger claims.

## Suggested delivery sequence

These are rough engineering-effort ranges for someone familiar with Rust/CUDA; they are not calendar commitments. Numerical-method changes have the greatest uncertainty. Frontend and server work can proceed alongside the physics track once their regression fixtures exist.

| Stage | Deliverables | Exit condition | Rough effort |
| --- | --- | --- | --- |
| 1. Regression and reliability fixes | Capture reproduced failures; correct formation clocks, durable save/error isolation, stale-session responses, frame identity, JSON scheduling; validate config and supervise tasks | Targeted failures now pass; existing CPU/GPU/browser checks remain green | 1–2 weeks |
| 2. Scientific baseline | Correct initial timestep/PN; synchronized diagnostics; independent small-N references; repair gas IC/count/mass behavior; test all effective controls | Batch/source-term invariance, explicit force-error budgets, measured timestep/resolution convergence | 1–3 weeks |
| 3. Measured performance | Reproducible early/dense-state benchmarks; release ICs; fix client memory/quality; prototype bounded dense-cell work; improve transport only where measured | Comparable faster timings within accuracy limits; bounded host/client memory and delivery queues | 1–3 weeks |
| 4. Persistence and packaging | Resumable checkpoints, state provenance, snapshot history/export; reproducible runtime image/assets; explicit deployment modes | Interrupted-save recovery and resumed-versus-uninterrupted validation; clean-machine startup | 1–2 weeks |
| 5. Showcase experience | Canvas-first UI, curated camera paths, component/field overlays, trustworthy performance indicators, screenshot/video export | A first-time visitor sees and controls a scene immediately and can understand/export a result | 1–2 weeks |

Start by turning the reproduced clock, save, session-selection, and paused-frame failures into regression tests and fixing those behaviors. In parallel, establish the force-accuracy and dense-state benchmark fixtures; those results should decide the next numerical and CUDA changes.
