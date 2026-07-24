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
uniform float u_spanMyr; // sim-time gap between the interpolation endpoints
uniform float u_simTimeMyr;
uniform mediump float u_style;  // 0 = soft glow sprites, 1 = small crisp dots

layout(location = 0) in vec3 a_prevPos;   // normalized u16
layout(location = 1) in vec3 a_prevVel;   // normalized u16
layout(location = 2) in vec3 a_pos;       // normalized u16
layout(location = 3) in vec3 a_vel;       // normalized u16
layout(location = 4) in float a_mass;     // normalized u16 (log2-encoded)
layout(location = 5) in uint a_component;
layout(location = 6) in float a_age; // stellar age, 6.4 Myr per 1/255 (1.0 = old)

out vec3 v_color;
out float v_marker;

void main() {
  v_marker = 0.0;
  // Dark matter is never drawn.
  if (a_component == 0u) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    gl_PointSize = 0.0;
    return;
  }

  vec3 p0 = u_posMin0 + a_prevPos * u_posScale0;
  vec3 p1 = u_posMin1 + a_pos * u_posScale1;
  vec3 v0 = u_velMin0 + a_prevVel * u_velScale0;
  vec3 v1 = u_velMin1 + a_vel * u_velScale1;
  // Cubic Hermite interpolation using the streamed velocities as tangents:
  // C1-continuous motion across frame boundaries, where plain lerp lurches
  // at the frame cadence on fast movers (SMBH cores, slingshot stars).
  float a2 = u_alpha * u_alpha;
  float a3 = a2 * u_alpha;
  float spanKpcPerKms = u_spanMyr * 0.0010227;
  vec3 m0 = v0 * spanKpcPerKms;
  vec3 m1 = v1 * spanKpcPerKms;
  vec3 position = (2.0 * a3 - 3.0 * a2 + 1.0) * p0 +
                  (a3 - 2.0 * a2 + u_alpha) * m0 +
                  (-2.0 * a3 + 3.0 * a2) * p1 +
                  (a3 - a2) * m1;
  vec3 velocity = mix(v0, v1, u_alpha);
  vec4 clip = u_viewProj * vec4(position, 1.0);
  gl_Position = clip;
  if (clip.w <= 0.1) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    gl_PointSize = 0.0;
    return;
  }

  // SMBHs render as fixed-size beacon markers (cyan ring + core), unaffected
  // by style, luminosity, or doppler.
  if (a_component == 3u) {
    v_marker = 1.0;
    gl_PointSize = clamp(18.0 * u_sizeBoost, 12.0, 40.0);
    v_color = vec3(0.45, 1.15, 1.55) * 7.0;
    return;
  }

  // Stable per-particle hash (preview slots map to fixed physical particles):
  // temperature spread, IMF magnitude spread, and brightness jitter.
  float h = fract(sin(float(gl_VertexID) * 12.9898) * 43758.5453);
  float h2 = fract(h * 61.803398875);
  // Newborn clusters blaze blue-hot then fade toward the old population;
  // the first ~30 Myr get an extra flash while their massive stars die.
  float youth = a_component == 5u ? 1.0 - a_age : 0.0;
  float flash = a_component == 5u ? exp(-a_age * 42.0) : 0.0;
  float logMass = (u_massLog.x + a_mass * u_massLog.y) * 0.3010299957;
  float luminosity = a_component == 4u
      ? 0.55
      : clamp((logMass - 3.7) / 2.2, 0.25, 1.8);
  // IMF-like magnitude spread: a heavy faint tail with rare bright outliers,
  // so the stellar field sparkles instead of rendering uniform points.
  if (a_component == 1u || a_component == 2u || a_component == 5u) {
    float imf = 0.42 + 2.2 * pow(h2, 3.0);
    luminosity *= imf;
  }
  luminosity *= 1.0 + 2.0 * youth + 6.0 * flash;
  float renderLuminosity = pow(luminosity, 0.58);
  float massBias = clamp((logMass - 4.2) / 1.6, 0.0, 1.0);

  vec3 base;
  if (a_component == 4u) {
    // Gas: cool teal-cyan fluid, hue drifting with the per-particle hash.
    base = mix(vec3(0.30, 0.80, 0.72), vec3(0.50, 0.92, 1.05), h);
  } else if (a_component == 5u) {
    base = mix(vec3(1.0, 0.94, 0.80), vec3(0.62, 0.78, 1.35), 0.35 + 0.65 * youth);
  } else if (a_component == 2u) {
    // Bulge: old stars — K/M giants, warm gold through deep orange.
    base = mix(vec3(1.0, 0.70, 0.42), vec3(1.0, 0.88, 0.70),
               clamp(0.2 + 0.55 * h + 0.15 * massBias, 0.0, 1.0));
  } else {
    // Disk: mixed population — warm G dwarfs through hot blue-white OB
    // stars, biased bluer for the more massive (brighter) samples.
    float temperature = clamp(0.12 + 0.6 * h + 0.38 * massBias, 0.0, 1.0);
    base = mix(vec3(1.0, 0.82, 0.58), vec3(0.60, 0.74, 1.0), temperature);
    base = mix(base, vec3(1.0, 0.97, 0.93), 0.22);
  }
  vec3 color = base * (0.82 + 0.36 * h2);

  // Supernovae: clusters younger than ~40 Myr host core-collapse SNe; a few
  // percent flare at any moment (re-rolled every ~0.8 Myr) and render as
  // oversized white-hot sprites that bypass the area normalization — pushed
  // through the regular luminosity path, the tonemap compresses them into
  // invisibility.
  if (a_component == 5u && a_age < 0.013) {
    // Supernova light curve: sharp rise to a brief white-blue peak, then an
    // exponential decay cooling through orange to a faint ember. Per-particle
    // phase offsets desynchronize the flares; the peak is bright but lives
    // only ~6% of the flare window, so the population reads as scattered
    // transient flashes rather than steady beacons.
    float t = u_simTimeMyr * 4.0 + h * 37.0;
    float epoch = floor(t);
    float roll = fract(h * 977.31 + epoch * 0.618034);
    if (roll > 0.97) {
      float p = fract(t);
      float curve = smoothstep(0.0, 0.06, p) * min(1.0, exp(-(p - 0.06) * 7.0));
      if (curve > 0.02) {
        // Lens-flare fragment path (marker 2): the sprite canvas is mostly
        // transparent — a pinpoint HDR core the bloom halos, plus thin
        // diffraction spikes. Subtle by area, bright by intensity.
        v_marker = 2.0;
        gl_PointSize = clamp((10.0 + 14.0 * curve) * u_sizeBoost, 8.0, 26.0);
        v_color = mix(vec3(1.8, 0.8, 0.4), vec3(1.0, 1.05, 1.3), curve) * (3.0 + 52.0 * curve);
        return;
      }
    }
  }

  float perspective = clamp((u_pointScale / clip.w) * 0.18, 0.02, 3.5);
  // Gas is a fluid: always the soft-glow path (even in dots mode) with a
  // broader footprint so it reads as a continuous medium between the stars.
  float gasBoost = a_component == 4u ? 1.6 : 1.0;
  if (u_style > 0.5 && a_component != 4u) {
    // Crisp dots: tiny fixed-ish footprint (near-zero fill cost), mild
    // distance attenuation, energy carried by the dot itself.
    gl_PointSize = clamp((1.7 + 1.3 * renderLuminosity) * pow(perspective, 0.35) * u_sizeBoost,
                         1.5, 6.0);
    v_color = color * (0.05 + 0.11 * renderLuminosity);
    return;
  }
  float size = 2.6 * gasBoost * renderLuminosity * pow(perspective, 0.9) * u_sizeBoost;
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

uniform mediump float u_style;
in vec3 v_color;
in float v_marker;
out vec4 fragColor;

void main() {
  vec2 offset = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(offset, offset);
  if (r2 > 1.0) {
    discard;
  }
  if (v_marker > 1.5) {
    // Supernova lens flare: pinpoint core + thin diffraction spikes.
    float ax = abs(offset.x);
    float ay = abs(offset.y);
    float core = exp(-r2 * 55.0);
    float spikes = exp(-ay * ay * 260.0) * max(0.0, 1.0 - ax)
                 + exp(-ax * ax * 260.0) * max(0.0, 1.0 - ay);
    fragColor = vec4(v_color * (core + 0.22 * spikes), 1.0);
    return;
  }
  if (v_marker > 0.5) {
    float ring = smoothstep(0.40, 0.55, r2) * (1.0 - smoothstep(0.75, 1.0, r2));
    float core = 0.55 * exp(-r2 * 42.0);
    fragColor = vec4(v_color * (ring + core), 1.0);
    return;
  }
  if (u_style > 0.5) {
    // Crisp dot: solid disc with a barely softened rim.
    fragColor = vec4(v_color * smoothstep(1.0, 0.7, r2), 1.0);
    return;
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
uniform sampler2D u_bloom;
uniform float u_bloomStrength;
// Exposure is computed CPU-side from the drawn particle count — screen
// brightness scales linearly with sample count, and count (unlike the
// frame-average luminance) does not change when the camera moves, so the
// image never re-exposes while orbiting or zooming.
uniform float u_exposure;
in vec2 v_uv;
out vec4 fragColor;

void main() {
  float exposure = u_exposure;
  vec3 x = texture(u_hdr, v_uv).rgb * exposure +
           texture(u_bloom, v_uv).rgb * u_bloomStrength;
  // Log-luminance compression: galaxy cores span ~3 decades of brightness,
  // so spread the decades across the display range instead of clipping at a
  // white point — dense cores keep radial structure instead of posterizing
  // into a flat white disc. Hue is preserved by scaling the color by the
  // luminance ratio; the smoothstep desaturates only the very top end so the
  // brightest core still rolls to white like film.
  float lum = dot(x, vec3(0.2126, 0.7152, 0.0722));
  float compressed = min(log2(1.0 + 4.0 * lum) / log2(513.0), 1.0);
  vec3 mapped = lum > 1.0e-6 ? x * (compressed / lum) : vec3(0.0);
  mapped = mix(mapped, vec3(compressed), smoothstep(0.72, 1.0, compressed) * 0.7);
  // Slight lift so faint structure reads on dark displays.
  mapped = pow(clamp(mapped, 0.0, 1.0), vec3(0.9));
  vec3 background = vec3(0.008, 0.031, 0.063); // #020810
  fragColor = vec4(max(mapped, background), 1.0);
}
`;

// Bright-pass + 4x downsample into the bloom chain. Threshold works on
// pre-tonemap HDR energy with a soft quadratic knee so bloom fades in.
const BLOOM_DOWN_FRAGMENT = `#version 300 es
precision mediump float;
uniform sampler2D u_hdr;
uniform float u_exposure;
uniform vec2 u_texel; // source texel size
in vec2 v_uv;
out vec4 fragColor;

void main() {
  float exposure = u_exposure;
  vec3 c = 0.25 * (textureLod(u_hdr, v_uv + u_texel * vec2(-1.0, -1.0), 0.0).rgb +
                   textureLod(u_hdr, v_uv + u_texel * vec2(1.0, -1.0), 0.0).rgb +
                   textureLod(u_hdr, v_uv + u_texel * vec2(-1.0, 1.0), 0.0).rgb +
                   textureLod(u_hdr, v_uv + u_texel * vec2(1.0, 1.0), 0.0).rgb) * exposure;
  float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
  const float threshold = 0.85;
  const float knee = 0.45;
  float soft = clamp(lum - threshold + knee, 0.0, 2.0 * knee);
  soft = soft * soft / (4.0 * knee);
  float contribution = max(soft, lum - threshold) / max(lum, 1.0e-4);
  fragColor = vec4(c * contribution, 1.0);
}
`;

// Separable gaussian (5 linear-filtered taps ~ 9-tap kernel).
const BLOOM_BLUR_FRAGMENT = `#version 300 es
precision mediump float;
uniform sampler2D u_source;
uniform vec2 u_direction; // texel-scaled blur axis
in vec2 v_uv;
out vec4 fragColor;

void main() {
  vec3 sum = texture(u_source, v_uv).rgb * 0.2270270270;
  sum += texture(u_source, v_uv + u_direction * 1.3846153846).rgb * 0.3162162162;
  sum += texture(u_source, v_uv - u_direction * 1.3846153846).rgb * 0.3162162162;
  sum += texture(u_source, v_uv + u_direction * 3.2307692308).rgb * 0.0702702703;
  sum += texture(u_source, v_uv - u_direction * 3.2307692308).rgb * 0.0702702703;
  fragColor = vec4(sum, 1.0);
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
  const bloomDownProgram = hdrExtension
    ? compileProgram(gl, TONEMAP_VERTEX, BLOOM_DOWN_FRAGMENT)
    : null;
  const bloomBlurProgram = hdrExtension
    ? compileProgram(gl, TONEMAP_VERTEX, BLOOM_BLUR_FRAGMENT)
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
    spanMyr: gl.getUniformLocation(pointProgram, "u_spanMyr"),
    simTimeMyr: gl.getUniformLocation(pointProgram, "u_simTimeMyr"),
    style: gl.getUniformLocation(pointProgram, "u_style"),
  };
  const tonemapUniforms = tonemapProgram
    ? {
        hdr: gl.getUniformLocation(tonemapProgram, "u_hdr"),
        bloom: gl.getUniformLocation(tonemapProgram, "u_bloom"),
        bloomStrength: gl.getUniformLocation(tonemapProgram, "u_bloomStrength"),
        exposure: gl.getUniformLocation(tonemapProgram, "u_exposure"),
      }
    : null;
  const bloomDownUniforms = bloomDownProgram
    ? {
        hdr: gl.getUniformLocation(bloomDownProgram, "u_hdr"),
        exposure: gl.getUniformLocation(bloomDownProgram, "u_exposure"),
        texel: gl.getUniformLocation(bloomDownProgram, "u_texel"),
      }
    : null;
  const bloomBlurUniforms = bloomBlurProgram
    ? {
        source: gl.getUniformLocation(bloomBlurProgram, "u_source"),
        direction: gl.getUniformLocation(bloomBlurProgram, "u_direction"),
      }
    : null;
  const lineUniforms = { viewProj: gl.getUniformLocation(lineProgram, "u_viewProj") };

  // GPU jitter ring; the quant arrays are written in place per arrival.
  const ring = [];
  for (let i = 0; i < RING_SIZE; i += 1) {
    ring.push({
      buffer: gl.createBuffer(),
      simTime: -Infinity,
      wallMs: 0,
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
    gl.enableVertexAttribArray(6);
    gl.vertexAttribPointer(6, 1, gl.UNSIGNED_BYTE, true, PARTICLE_STRIDE, 15);
    gl.bindVertexArray(null);
  }

  // HDR accumulation target (recreated on resize).
  let hdrFramebuffer = null;
  let hdrTexture = null;
  let hdrWidth = 0;
  let hdrHeight = 0;
  // Quarter-resolution bloom ping-pong chain.
  let bloomTexA = null;
  let bloomTexB = null;
  let bloomFboA = null;
  let bloomFboB = null;
  let bloomWidth = 0;
  let bloomHeight = 0;

  function makeBloomTarget(width, height) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA16F, width, height);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { texture, framebuffer };
  }

  function ensureHdrTarget() {
    if (!hdrExtension) {
      return false;
    }
    const targetWidth = Math.max(64, Math.round(canvas.width * state.renderScale));
    const targetHeight = Math.max(36, Math.round(canvas.height * state.renderScale));
    if (hdrTexture && hdrWidth === targetWidth && hdrHeight === targetHeight) {
      return true;
    }
    if (hdrTexture) {
      gl.deleteTexture(hdrTexture);
      gl.deleteFramebuffer(hdrFramebuffer);
    }
    hdrWidth = targetWidth;
    hdrHeight = targetHeight;
    hdrTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, hdrTexture);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA16F, hdrWidth, hdrHeight);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    hdrFramebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, hdrFramebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, hdrTexture, 0);
    const complete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (complete) {
      if (bloomTexA) {
        gl.deleteTexture(bloomTexA);
        gl.deleteTexture(bloomTexB);
        gl.deleteFramebuffer(bloomFboA);
        gl.deleteFramebuffer(bloomFboB);
      }
      bloomWidth = Math.max(32, hdrWidth >> 2);
      bloomHeight = Math.max(18, hdrHeight >> 2);
      const targetA = makeBloomTarget(bloomWidth, bloomHeight);
      const targetB = makeBloomTarget(bloomWidth, bloomHeight);
      bloomTexA = targetA.texture;
      bloomFboA = targetA.framebuffer;
      bloomTexB = targetB.texture;
      bloomFboB = targetB.framebuffer;
    }
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
    lastArrivalWallMs: 0,
    lastRafMs: 0,
    // Self-tuning jitter depth: deepen on underruns, decay very slowly.
    lagIntervals: 2.5,
    underrunStreak: 0,
    // Cached canvas metrics, refreshed by the ResizeObserver instead of
    // querying layout every frame.
    cssWidth: 0,
    cssHeight: 0,
    // Internal resolution of the HDR glow pass relative to the canvas. The
    // sprites are soft by design, so accumulating them at reduced resolution
    // and upscaling in the tonemap is visually free while the fill cost drops
    // with the square of the scale. Auto-tuned against the frame budget.
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
  {
    const params = new URLSearchParams(window.location.search);
    dotStyle = resolveDotStyle(params);
    const exposure = Number.parseFloat(params.get("exposure") ?? "");
    if (Number.isFinite(exposure) && exposure > 0.1 && exposure <= 8.0) {
      exposureBoost = exposure;
    }
    const quality = Number.parseFloat(params.get("quality") ?? "");
    if (Number.isFinite(quality) && quality > 0.2 && quality <= 1.0) {
      state.renderScale = quality;
      state.renderScaleLocked = true;
    } else if (dotStyle) {
      // Dots have near-zero fill cost and downscaling would blur them away:
      // pin the glow pass to full resolution.
      state.renderScale = 1.0;
      state.renderScaleLocked = true;
    }
  }

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
    slot.wallMs = performance.now();
    ringFrames = Math.min(ringFrames + 1, RING_SIZE);

    // A long silence (pause, reconnect) is a stream discontinuity: frames on
    // either side of it must not be interpolated across or fed into the rate
    // window, so restart playback from this frame alone.
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
    worker = new Worker("/webgl-stream-worker.js?v=20260722-24");
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
    worker.postMessage({ kind: "connect", url: `${frameUrl}?delta=1` });
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
    // Playback floor: keep two slots of headroom above the oldest entry so an
    // arriving frame never overwrites an actively-bound interpolation
    // endpoint (the write head reuses the oldest slot in place).
    const floorSlots = ringFrames > 4 ? 2 : 0;
    const playbackFloor =
      ring[(ringHead - (ringFrames - 1) + floorSlots + RING_SIZE * 2) % RING_SIZE];

    // Estimate the sim rate over the whole buffered window rather than from
    // per-arrival gaps: TCP delivery bursts (a long gap followed by a
    // back-to-back catch-up frame) make instantaneous gap ratios wildly wrong
    // and a poisoned rate estimate pins the clock against the newest frame.
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

    // Advance the clock at the observed sim rate, then ease it toward the
    // target lag point so rate drift never accumulates into a visible jump.
    const target = latest.simTime - state.lagIntervals * frameIntervalMyr;
    if (state.playbackSimTime === null) {
      state.playbackSimTime = target;
    }
    state.playbackSimTime += playbackRate * dtWall;
    state.playbackSimTime += (target - state.playbackSimTime) * 0.08;

    // Underruns mean the delivery jitter exceeds the current lag: deepen the
    // buffer (more latency, no stutter), and let it creep back down slowly.
    // Only while frames are actually flowing — an idle stream (paused
    // session) is supposed to sit pinned at the newest frame.
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

  // Frame-budget feedback: when rAF intervals persistently exceed 60 fps the
  // glow pass drops internal resolution (fill-rate is the dominant cost of
  // the soft sprites); with ample headroom it recovers toward full quality.
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

  function render(nowMs) {
    if (state.disposed) {
      return;
    }
    state.rafHandle = window.requestAnimationFrame(render);
    updateAutoQuality(nowMs);
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

    const useHdr = ensureHdrTarget();
    const passWidth = useHdr ? hdrWidth : width;
    const passHeight = useHdr ? hdrHeight : height;
    const focalLength = (Math.min(passWidth, passHeight) * 0.5) / Math.tan(Math.PI / 8);

    gl.bindFramebuffer(gl.FRAMEBUFFER, useHdr ? hdrFramebuffer : null);
    gl.viewport(0, 0, passWidth, passHeight);
    gl.clearColor(useHdr ? 0 : 0.008, useHdr ? 0 : 0.031, useHdr ? 0 : 0.063, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(pointProgram);
    gl.uniformMatrix4fv(uniforms.viewProj, false, viewProjMatrix);
    gl.uniform3fv(uniforms.forward, basis.forward);
    gl.uniform1f(uniforms.alpha, pair.alpha);
    gl.uniform1f(uniforms.pointScale, focalLength);
    gl.uniform1f(uniforms.sizeBoost, Math.max(1, passWidth / Math.max(1, state.cssWidth)));
    gl.uniform3fv(uniforms.posMin0, pair.previous.posMin);
    gl.uniform3fv(uniforms.posScale0, pair.previous.posScale);
    gl.uniform3fv(uniforms.velMin0, pair.previous.velMin);
    gl.uniform3fv(uniforms.velScale0, pair.previous.velScale);
    gl.uniform3fv(uniforms.posMin1, pair.current.posMin);
    gl.uniform3fv(uniforms.posScale1, pair.current.posScale);
    gl.uniform3fv(uniforms.velMin1, pair.current.velMin);
    gl.uniform3fv(uniforms.velScale1, pair.current.velScale);
    gl.uniform2fv(uniforms.massLog, pair.current.massLog);
    gl.uniform1f(
      uniforms.spanMyr,
      Math.max(0, pair.current.simTime - pair.previous.simTime)
    );
    // The interpolated playback clock, not the frame stamp: flare light
    // curves animate continuously instead of stepping once per frame.
    gl.uniform1f(uniforms.simTimeMyr, state.playbackSimTime ?? pair.current.simTime);
    gl.uniform1f(uniforms.style, dotStyle ? 1.0 : 0.0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.POINTS, 0, pair.current.count);
    gl.bindVertexArray(null);

    if (useHdr) {
      gl.disable(gl.BLEND);

      // Screen brightness scales linearly with the drawn sample count, so
      // exposure normalizes by it: stable under camera motion, calibrated
      // across preview budgets (the slider), tunable via ?exposure=.
      const exposure =
        exposureBoost * Math.min(5, Math.max(0.12, 460000 / Math.max(1, pair.current.count)));

      // Bloom: bright-pass downsample to quarter res, then a separable blur.
      gl.bindFramebuffer(gl.FRAMEBUFFER, bloomFboA);
      gl.viewport(0, 0, bloomWidth, bloomHeight);
      gl.useProgram(bloomDownProgram);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, hdrTexture);
      gl.uniform1i(bloomDownUniforms.hdr, 0);
      gl.uniform1f(bloomDownUniforms.exposure, exposure);
      gl.uniform2f(bloomDownUniforms.texel, 1 / hdrWidth, 1 / hdrHeight);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      gl.useProgram(bloomBlurProgram);
      gl.uniform1i(bloomBlurUniforms.source, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, bloomFboB);
      gl.bindTexture(gl.TEXTURE_2D, bloomTexA);
      gl.uniform2f(bloomBlurUniforms.direction, 1 / bloomWidth, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindFramebuffer(gl.FRAMEBUFFER, bloomFboA);
      gl.bindTexture(gl.TEXTURE_2D, bloomTexB);
      gl.uniform2f(bloomBlurUniforms.direction, 0, 1 / bloomHeight);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, width, height);
      gl.useProgram(tonemapProgram);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, hdrTexture);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, bloomTexA);
      gl.activeTexture(gl.TEXTURE0);
      gl.uniform1i(tonemapUniforms.hdr, 0);
      gl.uniform1i(tonemapUniforms.bloom, 1);
      // Dots carry less raw HDR energy per pixel, so give them a slightly
      // stronger halo to read as stars rather than pixels.
      gl.uniform1f(tonemapUniforms.bloomStrength, dotStyle ? 1.1 : 0.85);
      gl.uniform1f(tonemapUniforms.exposure, exposure);
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
      window.removeEventListener("galaxy-render-style", onStyleChange);
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
