// WebGPU point-splat renderer with true HDR (extended dynamic range) output.
//
// This is the top rendering tier: identical wire protocol, jitter-buffered
// playback, palette, bloom, and log tonemap as the WebGL tier — but the
// canvas is configured with `toneMapping: { mode: "extended" }` on an
// rgba16float swapchain, so tonemapped values above 1.0 drive the display's
// EDR headroom (real nits on XDR-class panels) instead of clipping at SDR
// white. WebGPU has no point sprites, so particles render as vertex-pulled
// quads (6 vertices per particle) reading the quantized records directly
// from storage buffers.
//
// Everything here degrades gracefully: no adapter, device loss, or a failed
// pipeline dispatches galaxy-viewer-error and the loader falls back to the
// WebGL tier. Failures are also reported to /api/client-log.

const META_BYTES = 72;
const HEADER_BYTES = 80;
const QUANT_BLOCK_BYTES = 56;
const PARTICLE_STRIDE = 16;
const PACKET_MAGIC = 0x54_4b_50_47; // "GPKT" little-endian
const PACKET_VERSION = 2;
const RING_SIZE = 12;
const UNIFORM_BYTES = 256;

// WGSL is exported so tooling (naga) can validate the shaders offline.
export const SHADER_COMMON = /* wgsl */ `
struct Uniforms {
  viewProj: mat4x4f,
  posMin0: vec3f, _a: f32,
  posScale0: vec3f, _b: f32,
  velMin0: vec3f, _c: f32,
  velScale0: vec3f, _d: f32,
  posMin1: vec3f, _e: f32,
  posScale1: vec3f, _f: f32,
  velMin1: vec3f, _g: f32,
  velScale1: vec3f, _h: f32,
  forward: vec3f, alpha: f32,
  massLog: vec2f, style: f32, pointScale: f32,
  sizeBoost: f32, resX: f32, resY: f32, exposureBoost: f32,
  headroom: f32, count: u32, spanMyr: f32, _j: f32,
}
`;

export const ACCUMULATE_SHADER = /* wgsl */ `
${SHADER_COMMON}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> prevRecords: array<vec4u>;
@group(0) @binding(2) var<storage, read> curRecords: array<vec4u>;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) color: vec3f,
  @location(2) marker: f32,
}

fn decodePos(r: vec4u, minV: vec3f, scaleV: vec3f) -> vec3f {
  let q = vec3f(
    f32(r.x & 0xFFFFu),
    f32(r.x >> 16u),
    f32(r.y & 0xFFFFu)
  ) / 65535.0;
  return minV + q * scaleV;
}

fn decodeVel(r: vec4u, minV: vec3f, scaleV: vec3f) -> vec3f {
  let q = vec3f(
    f32(r.y >> 16u),
    f32(r.z & 0xFFFFu),
    f32(r.z >> 16u)
  ) / 65535.0;
  return minV + q * scaleV;
}

fn dopplerShift(color: vec3f, radialVelocity: f32) -> vec3f {
  let shift = clamp(radialVelocity / 700.0, -0.28, 0.28);
  if (shift >= 0.0) {
    return clamp(vec3f(color.r * (1.0 + 0.9 * shift),
                       color.g * (1.0 + 0.2 * shift),
                       color.b * (1.0 - 0.75 * shift)), vec3f(0.0), vec3f(1.0));
  }
  let blue = -shift;
  return clamp(vec3f(color.r * (1.0 - 0.75 * blue),
                     color.g * (1.0 + 0.1 * blue),
                     color.b * (1.0 + 0.95 * blue)), vec3f(0.0), vec3f(1.0));
}

fn degenerate() -> VSOut {
  var out: VSOut;
  out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
  out.uv = vec2f(0.0);
  out.color = vec3f(0.0);
  out.marker = 0.0;
  return out;
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  let particle = vi / 6u;
  let corner = vi % 6u;
  if (particle >= u.count) {
    return degenerate();
  }
  let rc = curRecords[particle];
  let component = (rc.w >> 16u) & 0xFFu;
  // Dark matter is never drawn.
  if (component == 0u) {
    return degenerate();
  }
  let rp = prevRecords[particle];

  let p0 = decodePos(rp, u.posMin0, u.posScale0);
  let p1 = decodePos(rc, u.posMin1, u.posScale1);
  let v0 = decodeVel(rp, u.velMin0, u.velScale0);
  let v1 = decodeVel(rc, u.velMin1, u.velScale1);
  // Cubic Hermite with streamed velocities as tangents: C1-continuous motion
  // across frame boundaries (plain lerp lurches at the frame cadence).
  let a2 = u.alpha * u.alpha;
  let a3 = a2 * u.alpha;
  let spanKpcPerKms = u.spanMyr * 0.0010227;
  let m0 = v0 * spanKpcPerKms;
  let m1 = v1 * spanKpcPerKms;
  let position = (2.0 * a3 - 3.0 * a2 + 1.0) * p0 +
                 (a3 - 2.0 * a2 + u.alpha) * m0 +
                 (-2.0 * a3 + 3.0 * a2) * p1 +
                 (a3 - a2) * m1;
  let velocity = mix(v0, v1, u.alpha);
  let clip = u.viewProj * vec4f(position, 1.0);
  if (clip.w <= 0.1) {
    return degenerate();
  }

  let massQ = f32(rc.w & 0xFFFFu) / 65535.0;
  let logMass = (u.massLog.x + massQ * u.massLog.y) * 0.3010299957;
  let luminosity = clamp((logMass - 3.7) / 2.2, 0.25, 1.8);
  let renderLuminosity = pow(luminosity, 0.58);
  let massBias = clamp((logMass - 4.2) / 1.6, 0.0, 1.0);

  let h = fract(sin(f32(particle) * 12.9898) * 43758.5453);
  let h2 = fract(h * 61.803398875);
  var base: vec3f;
  if (component == 2u) {
    base = mix(vec3f(1.0, 0.70, 0.42), vec3f(1.0, 0.88, 0.70),
               clamp(0.2 + 0.55 * h + 0.15 * massBias, 0.0, 1.0));
  } else {
    let temperature = clamp(0.12 + 0.6 * h + 0.38 * massBias, 0.0, 1.0);
    base = mix(vec3f(1.0, 0.82, 0.58), vec3f(0.60, 0.74, 1.0), temperature);
    base = mix(base, vec3f(1.0, 0.97, 0.93), 0.22);
  }
  let color = dopplerShift(base, dot(velocity, u.forward)) * (0.82 + 0.36 * h2);

  let perspective = clamp((u.pointScale / clip.w) * 0.18, 0.02, 3.5);
  var size: f32;
  var splatColor: vec3f;
  var marker = 0.0;
  if (component == 3u) {
    // SMBH beacon: fixed-size cyan ring + core, unaffected by style.
    marker = 1.0;
    size = clamp(18.0 * u.sizeBoost, 12.0, 40.0);
    splatColor = vec3f(0.45, 1.15, 1.55) * 7.0;
  } else if (u.style > 0.5) {
    size = clamp((1.7 + 1.3 * renderLuminosity) * pow(perspective, 0.35) * u.sizeBoost,
                 1.5, 6.0);
    splatColor = color * (0.05 + 0.11 * renderLuminosity);
  } else {
    size = clamp(2.6 * renderLuminosity * pow(perspective, 0.9) * u.sizeBoost, 1.25, 48.0);
    let alpha = 0.055 * renderLuminosity * pow(perspective, 0.92);
    let area = size * size;
    splatColor = color * clamp(alpha * 52.0 / area, 0.0006, 0.35);
  }

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  let cornerUv = corners[corner];
  let offsetNdc = cornerUv * size / vec2f(u.resX, u.resY);

  var out: VSOut;
  out.pos = vec4f(clip.xy + offsetNdc * clip.w, clip.zw);
  out.uv = cornerUv;
  out.color = splatColor;
  out.marker = marker;
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) {
    discard;
  }
  if (in.marker > 0.5) {
    let ring = smoothstep(0.40, 0.55, r2) * (1.0 - smoothstep(0.75, 1.0, r2));
    let core = 0.55 * exp(-r2 * 42.0);
    return vec4f(in.color * (ring + core), 1.0);
  }
  if (u.style > 0.5) {
    return vec4f(in.color * smoothstep(1.0, 0.7, r2), 1.0);
  }
  let weight = 0.55 * exp(-r2 * 14.0) + 0.45 * exp(-r2 * 3.2);
  return vec4f(in.color * weight, 1.0);
}
`;

export const FULLSCREEN_VS = /* wgsl */ `
struct FSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> FSOut {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let corner = corners[vi];
  var out: FSOut;
  out.pos = vec4f(corner, 0.0, 1.0);
  // Flip Y: WebGPU framebuffer coordinates have y down, uv space y up.
  out.uv = vec2f(corner.x * 0.5 + 0.5, 1.0 - (corner.y * 0.5 + 0.5));
  return out;
}
`;

export const BLIT_SHADER = /* wgsl */ `
${FULLSCREEN_VS}
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs(in: FSOut) -> @location(0) vec4f {
  return vec4f(textureSample(src, samp, in.uv).rgb, 1.0);
}
`;

export const BLOOM_DOWN_SHADER = /* wgsl */ `
${FULLSCREEN_VS}
@group(0) @binding(0) var hdrTex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> exposure: f32;

@fragment
fn fs(in: FSOut) -> @location(0) vec4f {
  let c = textureSampleLevel(hdrTex, samp, in.uv, 0.0).rgb * exposure;
  let lum = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  let threshold = 0.85;
  let knee = 0.45;
  var soft = clamp(lum - threshold + knee, 0.0, 2.0 * knee);
  soft = soft * soft / (4.0 * knee);
  let contribution = max(soft, lum - threshold) / max(lum, 1.0e-4);
  return vec4f(c * contribution, 1.0);
}
`;

export const BLOOM_BLUR_SHADER = /* wgsl */ `
${FULLSCREEN_VS}
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> direction: vec2f;

@fragment
fn fs(in: FSOut) -> @location(0) vec4f {
  var sum = textureSampleLevel(src, samp, in.uv, 0.0).rgb * 0.2270270270;
  sum += textureSampleLevel(src, samp, in.uv + direction * 1.3846153846, 0.0).rgb * 0.3162162162;
  sum += textureSampleLevel(src, samp, in.uv - direction * 1.3846153846, 0.0).rgb * 0.3162162162;
  sum += textureSampleLevel(src, samp, in.uv + direction * 3.2307692308, 0.0).rgb * 0.0702702703;
  sum += textureSampleLevel(src, samp, in.uv - direction * 3.2307692308, 0.0).rgb * 0.0702702703;
  return vec4f(sum, 1.0);
}
`;

export const TONEMAP_SHADER = /* wgsl */ `
${SHADER_COMMON}
${FULLSCREEN_VS}
@group(0) @binding(0) var hdrTex: texture_2d<f32>;
@group(0) @binding(1) var bloomTex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> u: Uniforms;

@fragment
fn fs(in: FSOut) -> @location(0) vec4f {
  // u.exposureBoost carries the fully computed exposure (count-normalized on
  // the CPU — stable under camera motion; see the WebGL tier for rationale).
  let exposure = u.exposureBoost;
  let bloomStrength = select(0.85, 1.1, u.style > 0.5);
  let x = textureSampleLevel(hdrTex, samp, in.uv, 0.0).rgb * exposure +
          textureSampleLevel(bloomTex, samp, in.uv, 0.0).rgb * bloomStrength;
  // Log-luminance compression (see webgl-viewer.mjs for rationale), then, in
  // extended-range mode, the brightest end re-expands into the display's EDR
  // headroom: SDR content is unchanged, galaxy cores actually glow.
  let lum = dot(x, vec3f(0.2126, 0.7152, 0.0722));
  let compressed = min(log2(1.0 + 4.0 * lum) / log2(513.0), 1.0);
  var mapped = vec3f(0.0);
  if (lum > 1.0e-6) {
    mapped = x * (compressed / lum);
  }
  mapped = mix(mapped, vec3f(compressed), smoothstep(0.72, 1.0, compressed) * 0.7);
  mapped = pow(clamp(mapped, vec3f(0.0), vec3f(1.0)), vec3f(0.9));
  mapped *= mix(1.0, u.headroom, smoothstep(0.7, 1.0, compressed));
  let background = vec3f(0.008, 0.031, 0.063);
  return vec4f(max(mapped, background), 1.0);
}
`;

export const LINE_SHADER = /* wgsl */ `
${SHADER_COMMON}
@group(0) @binding(0) var<uniform> u: Uniforms;

struct LineOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec4f,
}

@vertex
fn vs(@location(0) pos: vec3f, @location(1) color: vec4f) -> LineOut {
  var out: LineOut;
  out.pos = u.viewProj * vec4f(pos, 1.0);
  out.color = color;
  return out;
}

@fragment
fn fs(in: LineOut) -> @location(0) vec4f {
  return vec4f(in.color.rgb * in.color.a, 1.0);
}
`;

function reportClient(payload) {
  try {
    void fetch("/api/client-log", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ src: "webgpu-viewer", ...payload }),
    }).catch(() => {});
  } catch {
    // reporting must never break the viewer
  }
}

// --- small math helpers (identical to the WebGL tier) ---

function perspectiveInto(out, fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  out.fill(0);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) / (near - far);
  out[11] = -1;
  out[14] = (2 * far * near) / (near - far);
}

function normalizeInto(v) {
  const length = Math.hypot(v[0], v[1], v[2]);
  if (length > 1e-9) {
    v[0] /= length;
    v[1] /= length;
    v[2] /= length;
  }
}

function crossInto(out, a, b) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

function lookAtInto(out, eye, center, up) {
  const f = [center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]];
  normalizeInto(f);
  const s = [0, 0, 0];
  crossInto(s, f, up);
  normalizeInto(s);
  const u = [0, 0, 0];
  crossInto(u, s, f);
  out[0] = s[0]; out[4] = s[1]; out[8] = s[2];
  out[1] = u[0]; out[5] = u[1]; out[9] = u[2];
  out[2] = -f[0]; out[6] = -f[1]; out[10] = -f[2];
  out[3] = 0; out[7] = 0; out[11] = 0;
  out[12] = -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]);
  out[13] = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
  out[14] = f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2];
  out[15] = 1;
}

function multiplyInto(out, a, b) {
  for (let col = 0; col < 4; col += 1) {
    for (let row = 0; row < 4; row += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) {
        sum += a[k * 4 + row] * b[col * 4 + k];
      }
      out[col * 4 + row] = sum;
    }
  }
}

const WORLD_UP = new Float32Array([0, 0, 1]);

let activeViewer = null;

export function shutdown() {
  if (activeViewer) {
    activeViewer.dispose();
    activeViewer = null;
  }
}

export function boot(canvasId, sessionId) {
  shutdown();
  const restoreCanvas = document.getElementById(canvasId);
  if (!restoreCanvas) {
    throw new Error(`canvas #${canvasId} not found`);
  }
  if (!navigator.gpu) {
    throw new Error("WebGPU unavailable");
  }
  // The async device setup runs behind the loader's first-frame gate; any
  // failure dispatches galaxy-viewer-error, which the loader treats as a
  // fallback signal.
  const canvas = document.createElement("canvas");
  canvas.id = "webgpu-preview-canvas";
  canvas.style.cssText = restoreCanvas.style.cssText;
  canvas.className = restoreCanvas.className;
  canvas.width = restoreCanvas.width;
  canvas.height = restoreCanvas.height;
  restoreCanvas.style.display = "none";
  restoreCanvas.insertAdjacentElement("afterend", canvas);

  const viewer = createViewer(canvas, restoreCanvas, sessionId);
  activeViewer = viewer;
  return viewer;
}

function createViewer(canvas, restoreCanvas, sessionId) {
  const state = {
    disposed: false,
    device: null,
    context: null,
    hdrActive: false,
    sawFirstFrame: false,
    firstFramePending: false,
    renderFailed: false,
    rafHandle: null,
    simTimeMyr: -Infinity,
    playbackSimTime: null,
    lastArrivalWallMs: 0,
    lastRafMs: 0,
    lagIntervals: 2.5,
    underrunStreak: 0,
    cssWidth: 0,
    cssHeight: 0,
    renderScale: 0.75,
    renderScaleLocked: false,
    frameCostEmaMs: 0,
    qualityCooldown: 0,
  };

  // Style precedence: explicit ?style= URL parameter, then the persisted UI
  // setting, then soft glow. The UI toggle dispatches galaxy-render-style,
  // applied live without reloading.
  function resolveDotStyle(params) {
    const fromUrl = params.get("style");
    if (fromUrl === "dots") {
      return true;
    }
    if (fromUrl === "glow") {
      return false;
    }
    try {
      return window.localStorage?.getItem("galaxy-render-style") === "dots";
    } catch {
      return false;
    }
  }

  function onStyleChange() {
    let style = "glow";
    try {
      style = window.localStorage?.getItem("galaxy-render-style") === "dots" ? "dots" : "glow";
    } catch {
      // private mode
    }
    dotStyle = style === "dots";
    if (dotStyle) {
      // Dots have near-zero fill cost and downscaling would blur them away.
      state.renderScale = 1.0;
      state.renderScaleLocked = true;
    } else {
      state.renderScale = 0.75;
      state.renderScaleLocked = false;
    }
  }
  window.addEventListener("galaxy-render-style", onStyleChange);

  let dotStyle = false;
  let exposureBoost = 1.0;
  let headroomOverride = null;
  {
    const params = new URLSearchParams(window.location.search);
    dotStyle = resolveDotStyle(params);
    const quality = Number.parseFloat(params.get("quality") ?? "");
    if (Number.isFinite(quality) && quality > 0.2 && quality <= 1.0) {
      state.renderScale = quality;
      state.renderScaleLocked = true;
    } else if (dotStyle) {
      state.renderScale = 1.0;
      state.renderScaleLocked = true;
    }
    const exposure = Number.parseFloat(params.get("exposure") ?? "");
    if (Number.isFinite(exposure) && exposure > 0.1 && exposure <= 8.0) {
      exposureBoost = exposure;
    }
    const hdrBoost = Number.parseFloat(params.get("hdrboost") ?? "");
    if (Number.isFinite(hdrBoost) && hdrBoost >= 1.0 && hdrBoost <= 16.0) {
      headroomOverride = hdrBoost;
    }
  }

  const camera = {
    yaw: 0.4,
    pitch: 0.9,
    distanceScale: 1.2,
    baseDistance: 120,
    autoFrame: true,
    dragging: false,
    dragMode: "orbit",
    lastX: 0,
    lastY: 0,
    focus: new Float32Array(3),
    sceneRadius: 120,
  };

  // CPU-side ring: raw frame bytes stay on the GPU in storage buffers; the
  // quant ranges and metadata live here.
  const ring = [];
  for (let i = 0; i < RING_SIZE; i += 1) {
    ring.push({
      buffer: null,
      capacityBytes: 0,
      simTime: -Infinity,
      wallMs: 0,
      count: 0,
      posMin: new Float32Array(3),
      posScale: new Float32Array(3),
      velMin: new Float32Array(3),
      velScale: new Float32Array(3),
      massLog: new Float32Array(2),
    });
  }
  let ringHead = -1;
  let ringFrames = 0;

  // Preallocated render scratch.
  const projMatrix = new Float32Array(16);
  const viewMatrix = new Float32Array(16);
  const viewProjMatrix = new Float32Array(16);
  const basis = {
    distance: 0,
    eye: new Float32Array(3),
    forward: new Float32Array(3),
    right: new Float32Array(3),
    up: new Float32Array(3),
  };
  const pairScratch = { previous: null, current: null, alpha: 1 };
  const uniformArray = new ArrayBuffer(UNIFORM_BYTES);
  const uniformF32 = new Float32Array(uniformArray);
  const uniformU32 = new Uint32Array(uniformArray);
  const axesVertices = new Float32Array(42);

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  // --- GPU setup (async) ---

  const gpu = {
    device: null,
    format: "rgba16float",
    accumulatePipeline: null,
    blitPipeline: null,
    bloomDownPipeline: null,
    bloomBlurPipeline: null,
    tonemapPipeline: null,
    linePipeline: null,
    sampler: null,
    uniformBuffer: null,
    exposureBoostBuffer: null,
    blurDirHBuffer: null,
    blurDirVBuffer: null,
    lineVertexBuffer: null,
    // resolution-dependent targets
    hdrTexture: null,
    hdrView: null,
    reduceTextures: [],
    reduceBindGroups: [],
    bloomTexA: null,
    bloomTexB: null,
    bloomViewA: null,
    bloomViewB: null,
    bloomDownBindGroup: null,
    blurBindGroupAtoB: null,
    blurBindGroupBtoA: null,
    tonemapBindGroup: null,
    hdrWidth: 0,
    hdrHeight: 0,
    accumulateBindGroups: new Map(),
    bindGroupEpoch: 0,
  };

  async function initGpu() {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
      throw new Error("no WebGPU adapter");
    }
    const need = 4_194_304 * PARTICLE_STRIDE + 65536;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(
          Math.max(need, 134_217_728),
          adapter.limits.maxStorageBufferBindingSize
        ),
        maxBufferSize: Math.min(
          Math.max(need, 268_435_456),
          adapter.limits.maxBufferSize
        ),
      },
    });
    gpu.device = device;
    device.lost.then((info) => {
      if (!state.disposed) {
        reportClient({ event: "device-lost", reason: info.reason, message: info.message });
        dispatchEvent("galaxy-viewer-error");
      }
    });
    device.onuncapturederror = (event) => {
      failRenderer("uncaptured-error", event.error);
    };

    const context = canvas.getContext("webgpu");
    if (!context) {
      throw new Error("no webgpu canvas context");
    }
    state.context = context;
    // Extended-range HDR: rgba16float swapchain + extended tone mapping lets
    // values above 1.0 reach the display's EDR headroom. Browsers that don't
    // know toneMapping ignore the dictionary member (WebIDL), leaving SDR.
    try {
      context.configure({
        device,
        format: "rgba16float",
        colorSpace: "display-p3",
        toneMapping: { mode: "extended" },
        alphaMode: "opaque",
      });
      gpu.format = "rgba16float";
    } catch {
      context.configure({
        device,
        format: navigator.gpu.getPreferredCanvasFormat(),
        alphaMode: "opaque",
      });
      gpu.format = navigator.gpu.getPreferredCanvasFormat();
    }
    const configured = context.getConfiguration?.();
    state.hdrActive = configured?.toneMapping?.mode === "extended";
    reportClient({
      event: "webgpu-init",
      hdrActive: state.hdrActive,
      format: gpu.format,
      dynamicRange: window.matchMedia?.("(dynamic-range: high)")?.matches ?? null,
    });

    async function makePipeline(label, code, targets, extra = {}) {
      const module = device.createShaderModule({ label, code });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === "error");
      if (errors.length > 0) {
        throw new Error(`${label}: ${errors[0].message}`);
      }
      return device.createRenderPipeline({
        label,
        layout: "auto",
        vertex: { module, entryPoint: "vs", ...(extra.vertexBuffers ? { buffers: extra.vertexBuffers } : {}) },
        fragment: { module, entryPoint: "fs", targets },
        primitive: extra.primitive ?? { topology: "triangle-list" },
      });
    }

    const additive = [{
      format: "rgba16float",
      blend: {
        color: { srcFactor: "one", dstFactor: "one", operation: "add" },
        alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
      },
    }];
    gpu.accumulatePipeline = await makePipeline("accumulate", ACCUMULATE_SHADER, additive);
    gpu.blitPipeline = await makePipeline("blit", BLIT_SHADER, [{ format: "rgba16float" }]);
    gpu.bloomDownPipeline = await makePipeline("bloom-down", BLOOM_DOWN_SHADER, [{ format: "rgba16float" }]);
    gpu.bloomBlurPipeline = await makePipeline("bloom-blur", BLOOM_BLUR_SHADER, [{ format: "rgba16float" }]);
    gpu.tonemapPipeline = await makePipeline("tonemap", TONEMAP_SHADER, [{ format: gpu.format }]);
    gpu.linePipeline = await makePipeline("lines", LINE_SHADER, [{ format: gpu.format }], {
      primitive: { topology: "line-list" },
      vertexBuffers: [{
        arrayStride: 28,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x4" },
        ],
      }],
    });

    gpu.sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
    gpu.uniformBuffer = device.createBuffer({
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    gpu.exposureBoostBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    gpu.blurDirHBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    gpu.blurDirVBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    gpu.lineVertexBuffer = device.createBuffer({
      size: axesVertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
  }

  function destroyTargets() {
    gpu.hdrTexture?.destroy();
    gpu.bloomTexA?.destroy();
    gpu.bloomTexB?.destroy();
    for (const tex of gpu.reduceTextures) {
      tex.destroy();
    }
    gpu.hdrTexture = null;
    gpu.reduceTextures = [];
    gpu.reduceBindGroups = [];
  }

  function ensureTargets() {
    const device = gpu.device;
    const targetWidth = Math.max(64, Math.round(canvas.width * state.renderScale));
    const targetHeight = Math.max(36, Math.round(canvas.height * state.renderScale));
    if (gpu.hdrTexture && gpu.hdrWidth === targetWidth && gpu.hdrHeight === targetHeight) {
      return;
    }
    destroyTargets();
    gpu.hdrWidth = targetWidth;
    gpu.hdrHeight = targetHeight;
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    gpu.hdrTexture = device.createTexture({
      size: [targetWidth, targetHeight],
      format: "rgba16float",
      usage,
    });
    gpu.hdrView = gpu.hdrTexture.createView();

    const bloomWidth = Math.max(32, targetWidth >> 2);
    const bloomHeight = Math.max(18, targetHeight >> 2);
    gpu.bloomTexA = device.createTexture({ size: [bloomWidth, bloomHeight], format: "rgba16float", usage });
    gpu.bloomTexB = device.createTexture({ size: [bloomWidth, bloomHeight], format: "rgba16float", usage });
    gpu.bloomViewA = gpu.bloomTexA.createView();
    gpu.bloomViewB = gpu.bloomTexB.createView();
    gpu.bloomWidth = bloomWidth;
    gpu.bloomHeight = bloomHeight;
    device.queue.writeBuffer(gpu.blurDirHBuffer, 0, new Float32Array([1 / bloomWidth, 0, 0, 0]));
    device.queue.writeBuffer(gpu.blurDirVBuffer, 0, new Float32Array([0, 1 / bloomHeight, 0, 0]));

    gpu.bloomDownBindGroup = device.createBindGroup({
      layout: gpu.bloomDownPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: gpu.hdrView },
        { binding: 1, resource: gpu.sampler },
        { binding: 2, resource: { buffer: gpu.exposureBoostBuffer } },
      ],
    });
    gpu.blurBindGroupAtoB = device.createBindGroup({
      layout: gpu.bloomBlurPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: gpu.bloomViewA },
        { binding: 1, resource: gpu.sampler },
        { binding: 2, resource: { buffer: gpu.blurDirHBuffer } },
      ],
    });
    gpu.blurBindGroupBtoA = device.createBindGroup({
      layout: gpu.bloomBlurPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: gpu.bloomViewB },
        { binding: 1, resource: gpu.sampler },
        { binding: 2, resource: { buffer: gpu.blurDirVBuffer } },
      ],
    });
    gpu.tonemapBindGroup = gpu.device.createBindGroup({
      layout: gpu.tonemapPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: gpu.hdrView },
        { binding: 1, resource: gpu.bloomViewA },
        { binding: 2, resource: gpu.sampler },
        { binding: 3, resource: { buffer: gpu.uniformBuffer } },
      ],
    });
  }

  function accumulateBindGroup(prevSlot, curSlot) {
    const key = `${gpu.bindGroupEpoch}:${ring.indexOf(prevSlot)}:${ring.indexOf(curSlot)}`;
    let cached = gpu.accumulateBindGroups.get(key);
    if (!cached) {
      cached = gpu.device.createBindGroup({
        layout: gpu.accumulatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: gpu.uniformBuffer } },
          { binding: 1, resource: { buffer: prevSlot.buffer } },
          { binding: 2, resource: { buffer: curSlot.buffer } },
        ],
      });
      gpu.accumulateBindGroups.set(key, cached);
    }
    return cached;
  }

  // --- camera / framing (identical logic to the WebGL tier) ---

  function updateBasis() {
    basis.distance = camera.baseDistance * camera.distanceScale;
    const cosPitch = Math.cos(camera.pitch);
    basis.forward[0] = -Math.cos(camera.yaw) * cosPitch;
    basis.forward[1] = -Math.sin(camera.yaw) * cosPitch;
    basis.forward[2] = -Math.sin(camera.pitch);
    normalizeInto(basis.forward);
    basis.eye[0] = camera.focus[0] - basis.forward[0] * basis.distance;
    basis.eye[1] = camera.focus[1] - basis.forward[1] * basis.distance;
    basis.eye[2] = camera.focus[2] - basis.forward[2] * basis.distance;
    crossInto(basis.right, basis.forward, WORLD_UP);
    if (Math.hypot(basis.right[0], basis.right[1], basis.right[2]) < 1e-6) {
      basis.right[0] = 1;
      basis.right[1] = 0;
      basis.right[2] = 0;
    } else {
      normalizeInto(basis.right);
    }
    crossInto(basis.up, basis.right, basis.forward);
  }

  // Robust auto-framing: raw quantization ranges are min/max, which a plume
  // of ejected stars inflates to hundreds of kpc, so bin a strided sample per
  // axis (1024 bins keeps sub-kpc precision even then) and aim the camera at
  // the median star position — the galaxy — with the 3rd..97th percentile
  // extent setting the zoom. The sample is retained so double-click can
  // re-frame against current data at any time.
  const frameHistogram = new Uint32Array(3 * 1024);
  const framingSample = new Uint16Array(40000 * 3);
  let framingSampleCount = 0;

  function captureFramingSample(slot, particleBytes) {
    const stride = Math.max(1, Math.ceil(slot.count / 40000));
    let n = 0;
    for (let i = 0; i < slot.count && n < 40000; i += stride) {
      const base = i * PARTICLE_STRIDE;
      framingSample[n * 3] = particleBytes[base] | (particleBytes[base + 1] << 8);
      framingSample[n * 3 + 1] = particleBytes[base + 2] | (particleBytes[base + 3] << 8);
      framingSample[n * 3 + 2] = particleBytes[base + 4] | (particleBytes[base + 5] << 8);
      n += 1;
    }
    framingSampleCount = n;
  }

  function updateSceneBoundsFromSlot(slot, force = false) {
    if (!camera.autoFrame && !force) {
      return;
    }
    if (framingSampleCount > 0) {
      frameHistogram.fill(0);
      for (let i = 0; i < framingSampleCount; i += 1) {
        frameHistogram[(framingSample[i * 3] >> 6)] += 1;
        frameHistogram[1024 + (framingSample[i * 3 + 1] >> 6)] += 1;
        frameHistogram[2048 + (framingSample[i * 3 + 2] >> 6)] += 1;
      }
      const percentileBin = (axis, fraction) => {
        const cut = framingSampleCount * fraction;
        let acc = 0;
        for (let bin = 0; bin < 1024; bin += 1) {
          acc += frameHistogram[axis * 1024 + bin];
          if (acc >= cut) {
            return bin;
          }
        }
        return 1023;
      };
      let maxExtent = 0;
      for (let axis = 0; axis < 3; axis += 1) {
        const toWorld = (bin) =>
          slot.posMin[axis] + ((bin + 0.5) / 1024) * slot.posScale[axis];
        camera.focus[axis] = toWorld(percentileBin(axis, 0.5));
        maxExtent = Math.max(
          maxExtent,
          toWorld(percentileBin(axis, 0.97)) - toWorld(percentileBin(axis, 0.03))
        );
      }
      camera.sceneRadius = Math.max(1, maxExtent * 0.74);
    } else {
      camera.focus[0] = slot.posMin[0] + slot.posScale[0] * 0.5;
      camera.focus[1] = slot.posMin[1] + slot.posScale[1] * 0.5;
      camera.focus[2] = slot.posMin[2] + slot.posScale[2] * 0.5;
      camera.sceneRadius = Math.max(
        1,
        0.5 * Math.hypot(slot.posScale[0], slot.posScale[1], slot.posScale[2])
      );
    }
    camera.baseDistance = (camera.sceneRadius * 0.9) / Math.tan(Math.PI / 8);
    if (!force && camera.sceneRadius > 0.5) {
      camera.autoFrame = false;
    }
  }

  function refreshCanvasMetrics() {
    const rect = canvas.getBoundingClientRect();
    state.cssWidth = rect.width;
    state.cssHeight = rect.height;
  }
  const resizeObserver =
    typeof ResizeObserver === "function" ? new ResizeObserver(refreshCanvasMetrics) : null;
  resizeObserver?.observe(canvas);
  refreshCanvasMetrics();

  function resizeToDisplay() {
    if (state.cssWidth <= 0 || state.cssHeight <= 0) {
      return;
    }
    const pixelRatio = Math.min(2, window.devicePixelRatio || 1);
    const width = Math.round(state.cssWidth * pixelRatio);
    const height = Math.round(state.cssHeight * pixelRatio);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  }

  function bufferScale() {
    return state.cssWidth > 0 ? canvas.width / state.cssWidth : 1;
  }

  function dispatchEvent(name) {
    window.dispatchEvent(new Event(name));
  }

  function failRenderer(event, error) {
    if (state.disposed || state.renderFailed) {
      return;
    }
    state.renderFailed = true;
    reportClient({ event, message: String(error?.message ?? error).slice(0, 400) });
    dispatchEvent("galaxy-viewer-error");
  }

  // --- frame ingestion ---

  function storeFrame(simTime, count, rangeView, rangeOffset, particleBytes) {
    if (state.disposed || !gpu.device || simTime <= state.simTimeMyr + 1e-9) {
      return;
    }
    ringHead = (ringHead + 1) % RING_SIZE;
    const slot = ring[ringHead];
    for (let i = 0; i < 3; i += 1) {
      slot.posMin[i] = rangeView.getFloat32(rangeOffset + i * 4, true);
      slot.posScale[i] = rangeView.getFloat32(rangeOffset + 12 + i * 4, true);
      slot.velMin[i] = rangeView.getFloat32(rangeOffset + 24 + i * 4, true);
      slot.velScale[i] = rangeView.getFloat32(rangeOffset + 36 + i * 4, true);
    }
    slot.massLog[0] = rangeView.getFloat32(rangeOffset + 48, true);
    slot.massLog[1] = rangeView.getFloat32(rangeOffset + 52, true);

    if (slot.capacityBytes < particleBytes.byteLength) {
      slot.buffer?.destroy();
      slot.buffer = gpu.device.createBuffer({
        size: particleBytes.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      slot.capacityBytes = particleBytes.byteLength;
      gpu.bindGroupEpoch += 1;
      gpu.accumulateBindGroups.clear();
    }
    gpu.device.queue.writeBuffer(slot.buffer, 0, particleBytes, 0, particleBytes.byteLength);
    slot.count = count;
    slot.simTime = simTime;
    slot.wallMs = performance.now();
    ringFrames = Math.min(ringFrames + 1, RING_SIZE);

    if (state.lastArrivalWallMs > 0 && slot.wallMs - state.lastArrivalWallMs > 2500) {
      ringFrames = 1;
      state.playbackSimTime = null;
      state.lagIntervals = 2.5;
      state.underrunStreak = 0;
    }
    state.lastArrivalWallMs = slot.wallMs;

    captureFramingSample(slot, particleBytes);
    updateSceneBoundsFromSlot(slot);
    state.simTimeMyr = simTime;
  }

  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const frameUrl = `${scheme}://${window.location.host}/ws/frames/${sessionId}`;

  let worker = null;
  let inlineSocket = null;

  function connectInline() {
    inlineSocket = new WebSocket(frameUrl);
    inlineSocket.binaryType = "arraybuffer";
    inlineSocket.onmessage = (event) => {
      if (state.disposed || !(event.data instanceof ArrayBuffer)) {
        return;
      }
      const data = event.data;
      if (data.byteLength < HEADER_BYTES + QUANT_BLOCK_BYTES) {
        dispatchEvent("galaxy-viewer-error");
        return;
      }
      const view = new DataView(data);
      if (view.getUint32(0, true) !== PACKET_MAGIC || view.getUint32(4, true) !== PACKET_VERSION) {
        dispatchEvent("galaxy-viewer-error");
        return;
      }
      const count = view.getUint32(16, true);
      if (data.byteLength !== HEADER_BYTES + QUANT_BLOCK_BYTES + count * PARTICLE_STRIDE) {
        dispatchEvent("galaxy-viewer-error");
        return;
      }
      storeFrame(
        view.getFloat64(24, true),
        count,
        view,
        HEADER_BYTES,
        new Uint8Array(data, HEADER_BYTES + QUANT_BLOCK_BYTES, count * PARTICLE_STRIDE)
      );
    };
    inlineSocket.onclose = () => {
      if (!state.disposed) {
        dispatchEvent("galaxy-viewer-error");
      }
    };
    inlineSocket.onerror = () => {
      if (!state.disposed) {
        dispatchEvent("galaxy-viewer-error");
      }
    };
  }

  function connectStream() {
    try {
      worker = new Worker("/webgl-stream-worker.js");
      worker.onmessage = (event) => {
        if (state.disposed) {
          return;
        }
        const message = event.data;
        if (message instanceof ArrayBuffer) {
          const view = new DataView(message);
          const simTime = view.getFloat64(0, true);
          const count = view.getUint32(8, true);
          storeFrame(simTime, count, view, 16, new Uint8Array(message, META_BYTES, count * PARTICLE_STRIDE));
          worker.postMessage(message, [message]);
          return;
        }
        if (message && (message.kind === "error" || message.kind === "closed")) {
          dispatchEvent("galaxy-viewer-error");
        }
      };
      worker.onerror = () => {
        if (!state.disposed) {
          dispatchEvent("galaxy-viewer-error");
        }
      };
      worker.postMessage({ kind: "connect", url: frameUrl });
    } catch (error) {
      console.warn("frame stream worker unavailable; using inline socket", error);
      worker = null;
      connectInline();
    }
  }

  // --- playback (identical logic to the WebGL tier) ---

  function selectFramePair(nowMs) {
    if (ringFrames === 0) {
      return null;
    }
    const dtWall = clamp(nowMs - (state.lastRafMs || nowMs), 0, 100);
    state.lastRafMs = nowMs;

    const latest = ring[ringHead];
    const oldest = ring[(ringHead - (ringFrames - 1) + RING_SIZE * 2) % RING_SIZE];
    const floorSlots = ringFrames > 4 ? 2 : 0;
    const playbackFloor =
      ring[(ringHead - (ringFrames - 1) + floorSlots + RING_SIZE * 2) % RING_SIZE];

    const spanSim = latest.simTime - oldest.simTime;
    const spanWall = Math.max(1, latest.wallMs - oldest.wallMs);
    const playbackRate = spanSim / spanWall;
    const frameIntervalMyr = spanSim / (ringFrames - 1);
    if (ringFrames === 1 || !(playbackRate > 0)) {
      state.playbackSimTime = latest.simTime;
      pairScratch.previous = latest;
      pairScratch.current = latest;
      pairScratch.alpha = 1;
      return pairScratch;
    }

    const target = latest.simTime - state.lagIntervals * frameIntervalMyr;
    if (state.playbackSimTime === null) {
      state.playbackSimTime = target;
    }
    state.playbackSimTime += playbackRate * dtWall;
    state.playbackSimTime += (target - state.playbackSimTime) * 0.08;

    const streamLive = nowMs - state.lastArrivalWallMs < 1500;
    if (state.playbackSimTime >= latest.simTime) {
      if (streamLive) {
        state.underrunStreak += 1;
        if (state.underrunStreak > 8) {
          state.lagIntervals = Math.min(6, state.lagIntervals + 0.5);
          state.underrunStreak = 0;
        }
      }
    } else {
      state.underrunStreak = 0;
      state.lagIntervals = Math.max(2.5, state.lagIntervals - 0.0003);
    }
    state.playbackSimTime = clamp(state.playbackSimTime, playbackFloor.simTime, latest.simTime);

    let previous = oldest;
    let current = latest;
    for (let i = 0; i < ringFrames - 1; i += 1) {
      const a = ring[(ringHead - (ringFrames - 1) + i + RING_SIZE * 2) % RING_SIZE];
      const b = ring[(ringHead - (ringFrames - 1) + i + 1 + RING_SIZE * 2) % RING_SIZE];
      if (state.playbackSimTime >= a.simTime && state.playbackSimTime <= b.simTime) {
        previous = a;
        current = b;
        break;
      }
    }
    if (previous.count !== current.count) {
      pairScratch.previous = current;
      pairScratch.current = current;
      pairScratch.alpha = 1;
      return pairScratch;
    }
    const span = current.simTime - previous.simTime;
    pairScratch.previous = previous;
    pairScratch.current = current;
    pairScratch.alpha = span > 0 ? clamp((state.playbackSimTime - previous.simTime) / span, 0, 1) : 1;
    return pairScratch;
  }

  function updateAutoQuality(nowMs) {
    if (state.renderScaleLocked) {
      return;
    }
    const frameCost = state.lastRafMs > 0 ? Math.min(100, nowMs - state.lastRafMs) : 16.7;
    state.frameCostEmaMs = state.frameCostEmaMs > 0
      ? state.frameCostEmaMs * 0.9 + frameCost * 0.1
      : frameCost;
    if (state.qualityCooldown > 0) {
      state.qualityCooldown -= 1;
      return;
    }
    if (state.frameCostEmaMs > 19 && state.renderScale > 0.35) {
      state.renderScale = Math.max(0.35, state.renderScale * 0.85);
      state.qualityCooldown = 30;
    } else if (state.frameCostEmaMs < 12 && state.renderScale < 1.0) {
      state.renderScale = Math.min(1.0, state.renderScale * 1.08);
      state.qualityCooldown = 60;
    }
  }

  function writeUniforms(pair, focalLength) {
    uniformF32.set(viewProjMatrix, 0);
    const vec = (offset, arr, w) => {
      uniformF32[offset] = arr[0];
      uniformF32[offset + 1] = arr[1];
      uniformF32[offset + 2] = arr[2];
      uniformF32[offset + 3] = w ?? 0;
    };
    vec(16, pair.previous.posMin);
    vec(20, pair.previous.posScale);
    vec(24, pair.previous.velMin);
    vec(28, pair.previous.velScale);
    vec(32, pair.current.posMin);
    vec(36, pair.current.posScale);
    vec(40, pair.current.velMin);
    vec(44, pair.current.velScale);
    vec(48, basis.forward, pair.alpha);
    uniformF32[52] = pair.current.massLog[0];
    uniformF32[53] = pair.current.massLog[1];
    uniformF32[54] = dotStyle ? 1.0 : 0.0;
    uniformF32[55] = focalLength;
    uniformF32[56] = Math.max(1, gpu.hdrWidth / Math.max(1, state.cssWidth));
    uniformF32[57] = gpu.hdrWidth;
    uniformF32[58] = gpu.hdrHeight;
    uniformF32[59] =
      exposureBoost * Math.min(5, Math.max(0.12, 460000 / Math.max(1, pair.current.count)));
    uniformF32[60] = state.hdrActive ? (headroomOverride ?? 3.0) : 1.0;
    uniformU32[61] = pair.current.count;
    uniformF32[62] = Math.max(0, pair.current.simTime - pair.previous.simTime);
    gpu.device.queue.writeBuffer(gpu.uniformBuffer, 0, uniformArray);
    gpu.device.queue.writeBuffer(
      gpu.exposureBoostBuffer,
      0,
      new Float32Array([uniformF32[59], 0, 0, 0])
    );
  }

  function writeAxes() {
    const axisLength = Math.max(1.4, camera.sceneRadius * 0.09);
    axesVertices.fill(0);
    axesVertices[3] = 1.0; axesVertices[4] = 0.43; axesVertices[5] = 0.43; axesVertices[6] = 0.56;
    axesVertices[7] = axisLength;
    axesVertices[10] = 1.0; axesVertices[11] = 0.43; axesVertices[12] = 0.43; axesVertices[13] = 0.56;
    axesVertices[17] = 0.47; axesVertices[18] = 1.0; axesVertices[19] = 0.67; axesVertices[20] = 0.56;
    axesVertices[22] = axisLength;
    axesVertices[24] = 0.47; axesVertices[25] = 1.0; axesVertices[26] = 0.67; axesVertices[27] = 0.56;
    axesVertices[31] = 0.43; axesVertices[32] = 0.67; axesVertices[33] = 1.0; axesVertices[34] = 0.56;
    axesVertices[37] = axisLength;
    axesVertices[38] = 0.43; axesVertices[39] = 0.67; axesVertices[40] = 1.0; axesVertices[41] = 0.56;
    gpu.device.queue.writeBuffer(gpu.lineVertexBuffer, 0, axesVertices);
  }

  function render(nowMs) {
    if (state.disposed || state.renderFailed) {
      return;
    }
    state.rafHandle = window.requestAnimationFrame(render);
    if (!gpu.device || !gpu.accumulatePipeline) {
      return;
    }
    updateAutoQuality(nowMs);
    const pair = selectFramePair(nowMs);
    if (!pair || pair.current.count === 0 || !pair.previous.buffer || !pair.current.buffer) {
      return;
    }

    resizeToDisplay();
    ensureTargets();

    updateBasis();
    perspectiveInto(
      projMatrix,
      Math.PI / 4,
      canvas.width / canvas.height,
      0.1,
      Math.max(10000, camera.sceneRadius * 200)
    );
    lookAtInto(viewMatrix, basis.eye, camera.focus, WORLD_UP);
    multiplyInto(viewProjMatrix, projMatrix, viewMatrix);

    const focalLength = (Math.min(gpu.hdrWidth, gpu.hdrHeight) * 0.5) / Math.tan(Math.PI / 8);
    writeUniforms(pair, focalLength);
    writeAxes();

    const verifyFirstFrame = !state.sawFirstFrame && !state.firstFramePending;
    if (verifyFirstFrame) {
      gpu.device.pushErrorScope("validation");
      gpu.device.pushErrorScope("out-of-memory");
      state.firstFramePending = true;
    }

    let encoder;
    try {
      encoder = gpu.device.createCommandEncoder();

      // 1. Accumulate splats into the HDR target.
      {
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: gpu.hdrView,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          }],
        });
        pass.setPipeline(gpu.accumulatePipeline);
        pass.setBindGroup(0, accumulateBindGroup(pair.previous, pair.current));
        pass.draw(pair.current.count * 6);
        pass.end();
      }

      // 2. Bloom: bright pass at quarter res, then separable blur.
      {
        const pass = encoder.beginRenderPass({
          colorAttachments: [{ view: gpu.bloomViewA, loadOp: "clear", storeOp: "store" }],
        });
        pass.setPipeline(gpu.bloomDownPipeline);
        pass.setBindGroup(0, gpu.bloomDownBindGroup);
        pass.draw(3);
        pass.end();
      }
      {
        const pass = encoder.beginRenderPass({
          colorAttachments: [{ view: gpu.bloomViewB, loadOp: "clear", storeOp: "store" }],
        });
        pass.setPipeline(gpu.bloomBlurPipeline);
        pass.setBindGroup(0, gpu.blurBindGroupAtoB);
        pass.draw(3);
        pass.end();
      }
      {
        const pass = encoder.beginRenderPass({
          colorAttachments: [{ view: gpu.bloomViewA, loadOp: "clear", storeOp: "store" }],
        });
        pass.setPipeline(gpu.bloomBlurPipeline);
        pass.setBindGroup(0, gpu.blurBindGroupBtoA);
        pass.draw(3);
        pass.end();
      }

      // 4. Tonemap to the (extended-range) swapchain + axis marker.
      {
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: state.context.getCurrentTexture().createView(),
            clearValue: { r: 0.008, g: 0.031, b: 0.063, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          }],
        });
        pass.setPipeline(gpu.tonemapPipeline);
        pass.setBindGroup(0, gpu.tonemapBindGroup);
        pass.draw(3);
        pass.setPipeline(gpu.linePipeline);
        pass.setBindGroup(0, gpu.lineBindGroup);
        pass.setVertexBuffer(0, gpu.lineVertexBuffer);
        pass.draw(6);
        pass.end();
      }

      gpu.device.queue.submit([encoder.finish()]);
    } catch (error) {
      if (verifyFirstFrame) {
        void Promise.all([
          gpu.device.popErrorScope(),
          gpu.device.popErrorScope(),
        ]).catch(() => {});
        state.firstFramePending = false;
      }
      failRenderer("render-threw", error);
      return;
    }

    if (verifyFirstFrame) {
      const outOfMemory = gpu.device.popErrorScope();
      const validation = gpu.device.popErrorScope();
      Promise.all([outOfMemory, validation])
        .then((errors) => {
          const error = errors.find(Boolean);
          if (error) {
            failRenderer("first-frame-invalid", error);
            return;
          }
          return gpu.device.queue.onSubmittedWorkDone().then(() => {
            if (!state.disposed && !state.renderFailed && !state.sawFirstFrame) {
              state.sawFirstFrame = true;
              dispatchEvent("galaxy-viewer-frame");
            }
          });
        })
        .catch((error) => failRenderer("first-frame-failed", error))
        .finally(() => {
          state.firstFramePending = false;
        });
    }
  }

  // --- controls (mirrors the other renderers) ---

  const handlers = {
    contextmenu: (event) => event.preventDefault(),
    mousedown: (event) => {
      event.preventDefault();
      camera.autoFrame = false;
      camera.dragging = true;
      camera.dragMode = event.button === 2 || event.shiftKey ? "pan" : "orbit";
      camera.lastX = event.clientX;
      camera.lastY = event.clientY;
    },
    mousemove: (event) => {
      if (!camera.dragging) {
        return;
      }
      const scale = bufferScale();
      const dx = (event.clientX - camera.lastX) * scale;
      const dy = (event.clientY - camera.lastY) * scale;
      if (camera.dragMode === "pan") {
        updateBasis();
        const panScale =
          (basis.distance * Math.tan(Math.PI / 8)) / (Math.min(canvas.width, canvas.height) * 0.5);
        for (let axis = 0; axis < 3; axis += 1) {
          camera.focus[axis] +=
            -dx * panScale * basis.right[axis] + dy * panScale * basis.up[axis];
        }
      } else {
        camera.yaw -= dx * 0.006;
        camera.pitch = clamp(camera.pitch - dy * 0.006, -1.45, 1.45);
      }
      camera.lastX = event.clientX;
      camera.lastY = event.clientY;
    },
    mouseup: () => {
      camera.dragging = false;
    },
    wheel: (event) => {
      event.preventDefault();
      camera.autoFrame = false;
      camera.distanceScale = clamp(
        camera.distanceScale * Math.exp(event.deltaY * 0.0011),
        0.05,
        40
      );
    },
    dblclick: () => {
      camera.autoFrame = true;
      camera.distanceScale = 1.2;
      if (ringFrames > 0) {
        updateSceneBoundsFromSlot(ring[ringHead], true);
        camera.autoFrame = false;
      }
    },
  };
  canvas.addEventListener("contextmenu", handlers.contextmenu);
  canvas.addEventListener("mousedown", handlers.mousedown);
  window.addEventListener("mousemove", handlers.mousemove);
  window.addEventListener("mouseup", handlers.mouseup);
  canvas.addEventListener("wheel", handlers.wheel, { passive: false });
  canvas.addEventListener("dblclick", handlers.dblclick);

  const viewer = {
    dispose() {
      state.disposed = true;
      if (state.rafHandle !== null) {
        window.cancelAnimationFrame(state.rafHandle);
      }
      if (worker) {
        worker.postMessage({ kind: "disconnect" });
        worker.terminate();
        worker = null;
      }
      if (inlineSocket) {
        inlineSocket.onmessage = null;
        inlineSocket.onclose = null;
        inlineSocket.onerror = null;
        try {
          inlineSocket.close();
        } catch {
          // already closed
        }
        inlineSocket = null;
      }
      canvas.removeEventListener("contextmenu", handlers.contextmenu);
      canvas.removeEventListener("mousedown", handlers.mousedown);
      window.removeEventListener("mousemove", handlers.mousemove);
      window.removeEventListener("mouseup", handlers.mouseup);
      canvas.removeEventListener("wheel", handlers.wheel);
      canvas.removeEventListener("dblclick", handlers.dblclick);
      window.removeEventListener("galaxy-render-style", onStyleChange);
      resizeObserver?.disconnect();
      destroyTargets();
      for (const slot of ring) {
        slot.buffer?.destroy();
      }
      gpu.device?.destroy();
      canvas.remove();
      restoreCanvas.style.display = "";
    },
  };

  initGpu()
    .then(() => {
      if (state.disposed) {
        return;
      }
      // Line pass shares the uniform buffer.
      gpu.lineBindGroup = gpu.device.createBindGroup({
        layout: gpu.linePipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: gpu.uniformBuffer } }],
      });
      connectStream();
      state.rafHandle = window.requestAnimationFrame(render);
    })
    .catch((error) => {
      reportClient({
        event: "init-failed",
        error: String((error && error.message) || error).slice(0, 400),
      });
      if (!state.disposed) {
        dispatchEvent("galaxy-viewer-error");
      }
    });

  return viewer;
}
