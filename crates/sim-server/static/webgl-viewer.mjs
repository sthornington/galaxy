// WebGL2 point-sprite renderer with an allocation-free steady state.
//
// - The preview WebSocket lives in a worker; frames arrive as transferable
//   buffers from a fixed pool that ping-pongs between threads, so the ~4 MB
//   per-message receive garbage is confined to the worker and the main thread
//   never triggers large GCs.
// - Frames are u16-quantized (packet v2); dequantization happens in the
//   vertex shader against per-frame range uniforms.
// - Playback runs on a jitter-buffered sim-time clock over a 12-frame GPU
//   ring, interpolating between whichever pair brackets the clock; the lag
//   deepens automatically if delivery ever underruns.
// - The render loop reuses preallocated matrices, vectors, and scratch
//   objects; per-frame allocations are limited to two ~100-byte typed-array
//   views over each arriving transfer (unavoidable across a transfer).

const META_BYTES = 72;
const HEADER_BYTES = 80;
const QUANT_BLOCK_BYTES = 56;
const PARTICLE_STRIDE = 16;
const PACKET_MAGIC = 0x54_4b_50_47; // "GPKT" little-endian
const PACKET_VERSION = 2;
const RING_SIZE = 12;

const VERTEX_SHADER = `#version 300 es
precision highp float;
precision highp int;

uniform mat4 u_viewProj;
uniform vec3 u_forward;
uniform float u_alpha;
uniform float u_pointScale;
uniform float u_sizeBoost;
// Per-endpoint dequantization ranges (frames carry their own quant blocks).
uniform vec3 u_posMin0;
uniform vec3 u_posScale0;
uniform vec3 u_velMin0;
uniform vec3 u_velScale0;
uniform vec3 u_posMin1;
uniform vec3 u_posScale1;
uniform vec3 u_velMin1;
uniform vec3 u_velScale1;
uniform vec2 u_massLog; // (log2 min, log2 scale) of the current frame

layout(location = 0) in vec3 a_prevPos;   // normalized u16
layout(location = 1) in vec3 a_prevVel;   // normalized u16
layout(location = 2) in vec3 a_pos;       // normalized u16
layout(location = 3) in vec3 a_vel;       // normalized u16
layout(location = 4) in float a_mass;     // normalized u16 (log2-encoded)
layout(location = 5) in uint a_component;

out vec3 v_color;

vec3 dopplerShift(vec3 color, float radialVelocity) {
  float shift = clamp(radialVelocity / 700.0, -0.28, 0.28);
  if (shift >= 0.0) {
    return clamp(vec3(color.r * (1.0 + 0.9 * shift),
                      color.g * (1.0 + 0.2 * shift),
                      color.b * (1.0 - 0.75 * shift)), 0.0, 1.0);
  }
  float blue = -shift;
  return clamp(vec3(color.r * (1.0 - 0.75 * blue),
                    color.g * (1.0 + 0.1 * blue),
                    color.b * (1.0 + 0.95 * blue)), 0.0, 1.0);
}

void main() {
  // Dark matter (0) and SMBH markers (3) are not splatted.
  if (a_component == 0u || a_component == 3u) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    gl_PointSize = 0.0;
    return;
  }

  vec3 position = mix(u_posMin0 + a_prevPos * u_posScale0,
                      u_posMin1 + a_pos * u_posScale1,
                      u_alpha);
  vec3 velocity = mix(u_velMin0 + a_prevVel * u_velScale0,
                      u_velMin1 + a_vel * u_velScale1,
                      u_alpha);
  vec4 clip = u_viewProj * vec4(position, 1.0);
  gl_Position = clip;
  if (clip.w <= 0.1) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    gl_PointSize = 0.0;
    return;
  }

  float logMass = (u_massLog.x + a_mass * u_massLog.y) * 0.3010299957;
  float luminosity = clamp((logMass - 3.7) / 2.2, 0.25, 1.8);
  float renderLuminosity = pow(luminosity, 0.58);
  float massBias = clamp((logMass - 4.2) / 1.6, 0.0, 1.0);

  vec3 base;
  if (a_component == 2u) {
    base = mix(vec3(1.0, 0.82, 0.68), vec3(1.0, 0.93, 0.82), 0.35 + 0.2 * massBias);
  } else {
    base = mix(vec3(1.0, 0.92, 0.8), vec3(0.79, 0.88, 1.0), 0.45 + 0.35 * massBias);
  }
  vec3 color = dopplerShift(base, dot(velocity, u_forward));

  float perspective = clamp((u_pointScale / clip.w) * 0.18, 0.02, 3.5);
  float size = 2.6 * renderLuminosity * pow(perspective, 0.9) * u_sizeBoost;
  gl_PointSize = clamp(size, 1.25, 48.0);

  // Fold the per-splat energy into the color (additive accumulation); divide
  // by the sprite area so a splat's total light is size-independent, then the
  // clamp keeps close-up sprites from vanishing entirely.
  float alpha = 0.055 * renderLuminosity * pow(perspective, 0.92);
  float area = gl_PointSize * gl_PointSize;
  v_color = color * clamp(alpha * 52.0 / area, 0.0006, 0.35);
}
`;

const FRAGMENT_SHADER = `#version 300 es
precision mediump float;

in vec3 v_color;
out vec4 fragColor;

void main() {
  vec2 offset = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(offset, offset);
  if (r2 > 1.0) {
    discard;
  }
  // Bright core + soft gaussian skirt.
  float weight = 0.55 * exp(-r2 * 14.0) + 0.45 * exp(-r2 * 3.2);
  fragColor = vec4(v_color * weight, 1.0);
}
`;

const TONEMAP_VERTEX = `#version 300 es
precision highp float;
out vec2 v_uv;
void main() {
  // Fullscreen triangle.
  vec2 corners[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  vec2 corner = corners[gl_VertexID];
  v_uv = corner * 0.5 + 0.5;
  gl_Position = vec4(corner, 0.0, 1.0);
}
`;

const TONEMAP_FRAGMENT = `#version 300 es
precision mediump float;
uniform sampler2D u_hdr;
uniform float u_exposure;
in vec2 v_uv;
out vec4 fragColor;

void main() {
  vec3 hdr = texture(u_hdr, v_uv).rgb;
  vec3 mapped = vec3(1.0) - exp(-hdr * u_exposure);
  // Slight lift so faint structure reads on dark displays.
  mapped = pow(mapped, vec3(0.9));
  vec3 background = vec3(0.008, 0.031, 0.063); // #020810
  fragColor = vec4(max(mapped, background), 1.0);
}
`;

const LINE_VERTEX = `#version 300 es
precision highp float;
uniform mat4 u_viewProj;
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec4 a_color;
out vec4 v_color;
void main() {
  gl_Position = u_viewProj * vec4(a_pos, 1.0);
  v_color = a_color;
}
`;

const LINE_FRAGMENT = `#version 300 es
precision mediump float;
in vec4 v_color;
out vec4 fragColor;
void main() {
  fragColor = v_color;
}
`;

function compileProgram(gl, vertexSource, fragmentSource) {
  const compile = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`shader compile failed: ${log}`);
    }
    return shader;
  };
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`program link failed: ${gl.getProgramInfoLog(program)}`);
  }
  return program;
}

// --- allocation-free mat4/vec3 helpers (column-major, WebGL layout) ---

function perspectiveInto(out, fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  out.fill(0);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) / (near - far);
  out[11] = -1;
  out[14] = (2 * far * near) / (near - far);
}

const lookScratch = {
  x: new Float32Array(3),
  y: new Float32Array(3),
  z: new Float32Array(3),
};

function normalizeInto(v) {
  const length = Math.hypot(v[0], v[1], v[2]) || 1;
  v[0] /= length;
  v[1] /= length;
  v[2] /= length;
}

function crossInto(out, a, b) {
  const x = a[1] * b[2] - a[2] * b[1];
  const y = a[2] * b[0] - a[0] * b[2];
  const z = a[0] * b[1] - a[1] * b[0];
  out[0] = x;
  out[1] = y;
  out[2] = z;
}

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function lookAtInto(out, eye, center, up) {
  const { x, y, z } = lookScratch;
  z[0] = eye[0] - center[0];
  z[1] = eye[1] - center[1];
  z[2] = eye[2] - center[2];
  normalizeInto(z);
  crossInto(x, up, z);
  normalizeInto(x);
  crossInto(y, z, x);
  out[0] = x[0]; out[1] = y[0]; out[2] = z[0]; out[3] = 0;
  out[4] = x[1]; out[5] = y[1]; out[6] = z[1]; out[7] = 0;
  out[8] = x[2]; out[9] = y[2]; out[10] = z[2]; out[11] = 0;
  out[12] = -dot3(x, eye);
  out[13] = -dot3(y, eye);
  out[14] = -dot3(z, eye);
  out[15] = 1;
}

function multiplyInto(out, a, b) {
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) {
        sum += a[k * 4 + row] * b[column * 4 + k];
      }
      out[column * 4 + row] = sum;
    }
  }
}

const WORLD_UP = new Float32Array([0, 0, 1]);

let active = null;

export function shutdown() {
  if (!active) {
    return;
  }
  const viewer = active;
  active = null;
  viewer.dispose();
}

export function boot(canvasId, sessionId) {
  shutdown();

  const baseCanvas = document.getElementById(canvasId);
  if (!baseCanvas) {
    throw new Error(`missing canvas #${canvasId}`);
  }

  // The shared preview canvas may already own a 2d context, which would make
  // getContext("webgl2") fail; render into a sibling canvas instead and hide
  // the original while this renderer is active.
  const canvas = document.createElement("canvas");
  canvas.id = "webgl-preview-canvas";
  canvas.width = baseCanvas.width || 1280;
  canvas.height = baseCanvas.height || 720;
  baseCanvas.insertAdjacentElement("afterend", canvas);
  const previousDisplay = baseCanvas.style.display;
  baseCanvas.style.display = "none";

  const restoreCanvas = () => {
    baseCanvas.style.display = previousDisplay;
    canvas.remove();
  };

  let gl;
  try {
    gl = canvas.getContext("webgl2", { antialias: false, alpha: false, depth: false });
    if (!gl) {
      throw new Error("webgl2 unavailable");
    }
  } catch (error) {
    restoreCanvas();
    throw error;
  }

  try {
    active = createViewer(gl, canvas, restoreCanvas, sessionId);
  } catch (error) {
    restoreCanvas();
    throw error;
  }
}

function createViewer(gl, canvas, restoreCanvas, sessionId) {
  const hdrExtension = gl.getExtension("EXT_color_buffer_float");

  const pointProgram = compileProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);
  const tonemapProgram = hdrExtension
    ? compileProgram(gl, TONEMAP_VERTEX, TONEMAP_FRAGMENT)
    : null;
  const lineProgram = compileProgram(gl, LINE_VERTEX, LINE_FRAGMENT);

  const uniforms = {
    viewProj: gl.getUniformLocation(pointProgram, "u_viewProj"),
    forward: gl.getUniformLocation(pointProgram, "u_forward"),
    alpha: gl.getUniformLocation(pointProgram, "u_alpha"),
    pointScale: gl.getUniformLocation(pointProgram, "u_pointScale"),
    sizeBoost: gl.getUniformLocation(pointProgram, "u_sizeBoost"),
    posMin0: gl.getUniformLocation(pointProgram, "u_posMin0"),
    posScale0: gl.getUniformLocation(pointProgram, "u_posScale0"),
    velMin0: gl.getUniformLocation(pointProgram, "u_velMin0"),
    velScale0: gl.getUniformLocation(pointProgram, "u_velScale0"),
    posMin1: gl.getUniformLocation(pointProgram, "u_posMin1"),
    posScale1: gl.getUniformLocation(pointProgram, "u_posScale1"),
    velMin1: gl.getUniformLocation(pointProgram, "u_velMin1"),
    velScale1: gl.getUniformLocation(pointProgram, "u_velScale1"),
    massLog: gl.getUniformLocation(pointProgram, "u_massLog"),
  };
  const tonemapUniforms = tonemapProgram
    ? {
        hdr: gl.getUniformLocation(tonemapProgram, "u_hdr"),
        exposure: gl.getUniformLocation(tonemapProgram, "u_exposure"),
      }
    : null;
  const lineUniforms = { viewProj: gl.getUniformLocation(lineProgram, "u_viewProj") };

  // GPU jitter ring; the quant arrays are written in place per arrival.
  const ring = [];
  for (let i = 0; i < RING_SIZE; i += 1) {
    ring.push({
      buffer: gl.createBuffer(),
      simTime: -Infinity,
      count: 0,
      capacityBytes: 0,
      posMin: new Float32Array(3),
      posScale: new Float32Array(3),
      velMin: new Float32Array(3),
      velScale: new Float32Array(3),
      massLog: new Float32Array(2),
    });
  }
  let ringHead = -1;
  let ringFrames = 0;
  const lineBuffer = gl.createBuffer();
  let lineBufferCapacity = 0;
  const vao = gl.createVertexArray();
  let boundPrevBuffer = null;
  let boundCurBuffer = null;

  function bindParticleAttributes(prevBuffer, curBuffer) {
    if (boundPrevBuffer === prevBuffer && boundCurBuffer === curBuffer) {
      return;
    }
    boundPrevBuffer = prevBuffer;
    boundCurBuffer = curBuffer;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, prevBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.UNSIGNED_SHORT, true, PARTICLE_STRIDE, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.UNSIGNED_SHORT, true, PARTICLE_STRIDE, 6);
    gl.bindBuffer(gl.ARRAY_BUFFER, curBuffer);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 3, gl.UNSIGNED_SHORT, true, PARTICLE_STRIDE, 0);
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 3, gl.UNSIGNED_SHORT, true, PARTICLE_STRIDE, 6);
    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 1, gl.UNSIGNED_SHORT, true, PARTICLE_STRIDE, 12);
    gl.enableVertexAttribArray(5);
    gl.vertexAttribIPointer(5, 1, gl.UNSIGNED_BYTE, PARTICLE_STRIDE, 14);
    gl.bindVertexArray(null);
  }

  // HDR accumulation target (recreated on resize).
  let hdrFramebuffer = null;
  let hdrTexture = null;
  let hdrWidth = 0;
  let hdrHeight = 0;

  function ensureHdrTarget() {
    if (!hdrExtension) {
      return false;
    }
    if (hdrTexture && hdrWidth === canvas.width && hdrHeight === canvas.height) {
      return true;
    }
    if (hdrTexture) {
      gl.deleteTexture(hdrTexture);
      gl.deleteFramebuffer(hdrFramebuffer);
    }
    hdrWidth = canvas.width;
    hdrHeight = canvas.height;
    hdrTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, hdrTexture);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA16F, hdrWidth, hdrHeight);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    hdrFramebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, hdrFramebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, hdrTexture, 0);
    const complete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return complete;
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

  const state = {
    simTimeMyr: -Infinity,
    sawFirstFrame: false,
    rafHandle: null,
    disposed: false,
    // Playback clock in sim-time units.
    playbackSimTime: null,
    playbackRate: 0, // Myr per wall ms, EMA over arrivals
    frameIntervalMyr: 0, // EMA of sim-time spacing between frames
    lastArrivalWallMs: 0,
    lastRafMs: 0,
    // Self-tuning jitter depth: deepen on underruns, decay very slowly.
    lagIntervals: 2.5,
    underrunStreak: 0,
    // Cached canvas metrics, refreshed by the ResizeObserver instead of
    // querying layout every frame.
    cssWidth: 0,
    cssHeight: 0,
  };

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
  const axesVertices = new Float32Array(42);
  const cursorDirection = new Float32Array(3);

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function updateBasis() {
    basis.distance = Math.max(0.08, camera.baseDistance * camera.distanceScale);
    const cosPitch = Math.cos(camera.pitch);
    basis.eye[0] = camera.focus[0] + basis.distance * cosPitch * Math.cos(camera.yaw);
    basis.eye[1] = camera.focus[1] + basis.distance * cosPitch * Math.sin(camera.yaw);
    basis.eye[2] = camera.focus[2] + basis.distance * Math.sin(camera.pitch);
    basis.forward[0] = camera.focus[0] - basis.eye[0];
    basis.forward[1] = camera.focus[1] - basis.eye[1];
    basis.forward[2] = camera.focus[2] - basis.eye[2];
    normalizeInto(basis.forward);
    crossInto(basis.right, basis.forward, WORLD_UP);
    if (Math.hypot(basis.right[0], basis.right[1], basis.right[2]) <= 1e-6) {
      basis.right[0] = 1;
      basis.right[1] = 0;
      basis.right[2] = 0;
    } else {
      normalizeInto(basis.right);
    }
    crossInto(basis.up, basis.right, basis.forward);
  }

  // Frame bounds come for free from the packet's quantization ranges.
  function updateSceneBoundsFromSlot(slot, force = false) {
    if (!camera.autoFrame && !force) {
      return;
    }
    camera.focus[0] = slot.posMin[0] + slot.posScale[0] * 0.5;
    camera.focus[1] = slot.posMin[1] + slot.posScale[1] * 0.5;
    camera.focus[2] = slot.posMin[2] + slot.posScale[2] * 0.5;
    const radius = Math.max(
      1,
      0.5 * Math.hypot(slot.posScale[0], slot.posScale[1], slot.posScale[2])
    );
    camera.sceneRadius = radius;
    camera.baseDistance = (radius * 0.9) / Math.tan(Math.PI / 8);
    if (!force && radius > 0.5) {
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

  // --- frame ingestion (shared by the worker and inline socket paths) ---

  function storeFrame(simTime, count, rangeView, rangeOffset, particleBytes) {
    if (simTime <= state.simTimeMyr + 1e-9) {
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

    // Stores are preallocated once and refreshed with bufferSubData so the
    // driver never reallocates in the hot path.
    gl.bindBuffer(gl.ARRAY_BUFFER, slot.buffer);
    if (slot.capacityBytes < particleBytes.byteLength) {
      gl.bufferData(gl.ARRAY_BUFFER, particleBytes.byteLength, gl.DYNAMIC_DRAW);
      slot.capacityBytes = particleBytes.byteLength;
    }
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, particleBytes);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    slot.count = count;
    slot.simTime = simTime;
    ringFrames = Math.min(ringFrames + 1, RING_SIZE);

    updateSceneBoundsFromSlot(slot);

    const now = performance.now();
    if (state.lastArrivalWallMs > 0 && state.simTimeMyr > -Infinity) {
      const wallGap = Math.max(1, now - state.lastArrivalWallMs);
      const simGap = simTime - state.simTimeMyr;
      const instantaneousRate = simGap / wallGap;
      state.playbackRate = state.playbackRate > 0
        ? state.playbackRate * 0.75 + instantaneousRate * 0.25
        : instantaneousRate;
      state.frameIntervalMyr = state.frameIntervalMyr > 0
        ? state.frameIntervalMyr * 0.75 + simGap * 0.25
        : simGap;
    }
    state.lastArrivalWallMs = now;
    state.simTimeMyr = simTime;

    if (!state.sawFirstFrame) {
      state.sawFirstFrame = true;
      dispatchEvent("galaxy-viewer-frame");
    }
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
      if (
        view.getUint32(0, true) !== PACKET_MAGIC ||
        view.getUint32(4, true) !== PACKET_VERSION
      ) {
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

  try {
    worker = new Worker("/webgl-stream-worker.js");
    worker.onmessage = (event) => {
      if (state.disposed) {
        return;
      }
      const message = event.data;
      if (message instanceof ArrayBuffer) {
        // Pooled frame: [f64 simTime][u32 count][u32 dropped][56B quant][records].
        const view = new DataView(message);
        const simTime = view.getFloat64(0, true);
        const count = view.getUint32(8, true);
        storeFrame(
          simTime,
          count,
          view,
          16,
          new Uint8Array(message, META_BYTES, count * PARTICLE_STRIDE)
        );
        // Send the buffer home for reuse.
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

  // --- playback ---

  // Chooses the buffered frame pair bracketing the playback clock and the
  // interpolation fraction between them (written into pairScratch).
  function selectFramePair(nowMs) {
    if (ringFrames === 0) {
      return null;
    }
    const dtWall = clamp(nowMs - (state.lastRafMs || nowMs), 0, 100);
    state.lastRafMs = nowMs;

    const latest = ring[ringHead];
    const oldest = ring[(ringHead - (ringFrames - 1) + RING_SIZE * 2) % RING_SIZE];
    if (ringFrames === 1 || state.playbackRate <= 0) {
      state.playbackSimTime = latest.simTime;
      pairScratch.previous = latest;
      pairScratch.current = latest;
      pairScratch.alpha = 1;
      return pairScratch;
    }

    // Advance the clock at the observed sim rate, then ease it toward the
    // target lag point so rate drift never accumulates into a visible jump.
    const target = latest.simTime - state.lagIntervals * state.frameIntervalMyr;
    if (state.playbackSimTime === null) {
      state.playbackSimTime = target;
    }
    state.playbackSimTime += state.playbackRate * dtWall;
    state.playbackSimTime += (target - state.playbackSimTime) * 0.04;

    // Underruns mean the delivery jitter exceeds the current lag: deepen the
    // buffer (more latency, no stutter), and let it creep back down slowly.
    if (state.playbackSimTime >= latest.simTime) {
      state.underrunStreak += 1;
      if (state.underrunStreak > 8) {
        state.lagIntervals = Math.min(6, state.lagIntervals + 0.5);
        state.underrunStreak = 0;
      }
    } else {
      state.underrunStreak = 0;
      state.lagIntervals = Math.max(2.5, state.lagIntervals - 0.0003);
    }
    state.playbackSimTime = clamp(state.playbackSimTime, oldest.simTime, latest.simTime);

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
      // Index-paired interpolation is meaningless across a budget change.
      pairScratch.previous = current;
      pairScratch.current = current;
      pairScratch.alpha = 1;
      return pairScratch;
    }
    const span = current.simTime - previous.simTime;
    pairScratch.previous = previous;
    pairScratch.current = current;
    pairScratch.alpha =
      span > 0 ? clamp((state.playbackSimTime - previous.simTime) / span, 0, 1) : 1;
    return pairScratch;
  }

  function drawAxes() {
    const axisLength = Math.max(1.4, camera.sceneRadius * 0.09);
    // Six vertices of [pos.xyz, color.rgba]; origin vertices carry the axis
    // color too, endpoints carry the axis offset.
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

    gl.useProgram(lineProgram);
    gl.uniformMatrix4fv(lineUniforms.viewProj, false, viewProjMatrix);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuffer);
    if (lineBufferCapacity < axesVertices.byteLength) {
      gl.bufferData(gl.ARRAY_BUFFER, axesVertices.byteLength, gl.DYNAMIC_DRAW);
      lineBufferCapacity = axesVertices.byteLength;
    }
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, axesVertices);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 28, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 28, 12);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.LINES, 0, 6);
    gl.disableVertexAttribArray(0);
    gl.disableVertexAttribArray(1);
  }

  function render(nowMs) {
    if (state.disposed) {
      return;
    }
    state.rafHandle = window.requestAnimationFrame(render);
    const pair = selectFramePair(nowMs);
    if (!pair || pair.current.count === 0) {
      return;
    }
    bindParticleAttributes(pair.previous.buffer, pair.current.buffer);

    resizeToDisplay();
    const width = canvas.width;
    const height = canvas.height;

    updateBasis();
    perspectiveInto(
      projMatrix,
      Math.PI / 4,
      width / height,
      0.1,
      Math.max(10000, camera.sceneRadius * 200)
    );
    lookAtInto(viewMatrix, basis.eye, camera.focus, WORLD_UP);
    multiplyInto(viewProjMatrix, projMatrix, viewMatrix);
    const focalLength = (Math.min(width, height) * 0.5) / Math.tan(Math.PI / 8);

    const useHdr = ensureHdrTarget();
    gl.bindFramebuffer(gl.FRAMEBUFFER, useHdr ? hdrFramebuffer : null);
    gl.viewport(0, 0, width, height);
    gl.clearColor(useHdr ? 0 : 0.008, useHdr ? 0 : 0.031, useHdr ? 0 : 0.063, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(pointProgram);
    gl.uniformMatrix4fv(uniforms.viewProj, false, viewProjMatrix);
    gl.uniform3fv(uniforms.forward, basis.forward);
    gl.uniform1f(uniforms.alpha, pair.alpha);
    gl.uniform1f(uniforms.pointScale, focalLength);
    gl.uniform1f(uniforms.sizeBoost, Math.min(2, window.devicePixelRatio || 1));
    gl.uniform3fv(uniforms.posMin0, pair.previous.posMin);
    gl.uniform3fv(uniforms.posScale0, pair.previous.posScale);
    gl.uniform3fv(uniforms.velMin0, pair.previous.velMin);
    gl.uniform3fv(uniforms.velScale0, pair.previous.velScale);
    gl.uniform3fv(uniforms.posMin1, pair.current.posMin);
    gl.uniform3fv(uniforms.posScale1, pair.current.posScale);
    gl.uniform3fv(uniforms.velMin1, pair.current.velMin);
    gl.uniform3fv(uniforms.velScale1, pair.current.velScale);
    gl.uniform2fv(uniforms.massLog, pair.current.massLog);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.POINTS, 0, pair.current.count);
    gl.bindVertexArray(null);

    if (useHdr) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, width, height);
      gl.disable(gl.BLEND);
      gl.useProgram(tonemapProgram);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, hdrTexture);
      gl.uniform1i(tonemapUniforms.hdr, 0);
      gl.uniform1f(tonemapUniforms.exposure, 1.15);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    drawAxes();
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
      camera.dragMode = "orbit";
    },
    mouseleave: () => {
      camera.dragging = false;
      camera.dragMode = "orbit";
    },
    wheel: (event) => {
      event.preventDefault();
      camera.autoFrame = false;
      const factor = event.deltaY < 0 ? 0.78 : 1 / 0.78;
      const previousScale = camera.distanceScale;
      camera.distanceScale = clamp(camera.distanceScale * factor, 0.003, 20);

      // Zoom toward the cursor.
      if (state.cssWidth > 0 && state.cssHeight > 0) {
        const rect = canvas.getBoundingClientRect();
        const scale = bufferScale();
        const cursorX = (event.clientX - rect.left) * scale;
        const cursorY = (event.clientY - rect.top) * scale;
        updateBasis();
        const focal = (Math.min(canvas.width, canvas.height) * 0.5) / Math.tan(Math.PI / 8);
        const rightAmount = (cursorX - canvas.width * 0.5) / focal;
        const upAmount = -(cursorY - canvas.height * 0.5) / focal;
        for (let axis = 0; axis < 3; axis += 1) {
          cursorDirection[axis] =
            basis.forward[axis] +
            basis.right[axis] * rightAmount +
            basis.up[axis] * upAmount;
        }
        const length =
          Math.hypot(cursorDirection[0], cursorDirection[1], cursorDirection[2]) || 1;
        const along = dot3(cursorDirection, basis.forward) / length;
        if (along > 1e-6) {
          const range = basis.distance / along;
          const pull = 1 - camera.distanceScale / previousScale;
          for (let axis = 0; axis < 3; axis += 1) {
            const target = basis.eye[axis] + (cursorDirection[axis] / length) * range;
            camera.focus[axis] += (target - camera.focus[axis]) * pull;
          }
        }
      }
    },
    dblclick: () => {
      camera.yaw = 0.4;
      camera.pitch = 0.9;
      camera.distanceScale = 1.2;
      camera.autoFrame = true;
      if (ringFrames > 0) {
        updateSceneBoundsFromSlot(ring[ringHead], true);
      }
    },
  };
  for (const [name, handler] of Object.entries(handlers)) {
    canvas.addEventListener(name, handler);
  }
  canvas.style.cursor = "grab";

  state.rafHandle = window.requestAnimationFrame(render);

  return {
    dispose() {
      state.disposed = true;
      if (state.rafHandle !== null) {
        window.cancelAnimationFrame(state.rafHandle);
      }
      resizeObserver?.disconnect();
      if (worker) {
        worker.postMessage({ kind: "disconnect" });
        worker.terminate();
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
      }
      for (const [name, handler] of Object.entries(handlers)) {
        canvas.removeEventListener(name, handler);
      }
      const extension = gl.getExtension("WEBGL_lose_context");
      restoreCanvas();
      if (extension) {
        extension.loseContext();
      }
    },
  };
}
