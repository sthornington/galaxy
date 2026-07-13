// WebGL2 instanced point-sprite renderer: binary preview packets go straight
// from the WebSocket into GPU vertex buffers (no per-particle JS objects), and
// projection, frame interpolation, mass->luminosity, and Doppler tinting all
// run in the vertex shader. Additive blending is order-independent, so there
// is no CPU depth sort; splats accumulate into an RGBA16F target and a
// tonemap pass maps the HDR sum to the screen.

const HEADER_BYTES = 80;
const QUANT_BLOCK_BYTES = 56;
const PARTICLE_STRIDE = 16;
const PACKET_MAGIC = 0x54_4b_50_47; // "GPKT" little-endian
const PACKET_VERSION = 2;

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
  // Bright core + soft gaussian skirt, matching the two-arc Canvas2D look.
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

// --- minimal mat4 helpers (column-major, WebGL layout) ---

function perspectiveMatrix(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) / (near - far);
  out[11] = -1;
  out[14] = (2 * far * near) / (near - far);
  return out;
}

function lookAtMatrix(eye, center, up) {
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const normalize = (v) => {
    const length = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / length, v[1] / length, v[2] / length];
  };
  const zAxis = normalize(sub(eye, center));
  const xAxis = normalize(cross(up, zAxis));
  const yAxis = cross(zAxis, xAxis);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  return new Float32Array([
    xAxis[0], yAxis[0], zAxis[0], 0,
    xAxis[1], yAxis[1], zAxis[1], 0,
    xAxis[2], yAxis[2], zAxis[2], 0,
    -dot(xAxis, eye), -dot(yAxis, eye), -dot(zAxis, eye), 1,
  ]);
}

function multiplyMat4(a, b) {
  const out = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) {
        sum += a[k * 4 + row] * b[column * 4 + k];
      }
      out[column * 4 + row] = sum;
    }
  }
  return out;
}

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
    active = createViewer(gl, canvas, baseCanvas, restoreCanvas, sessionId);
  } catch (error) {
    restoreCanvas();
    throw error;
  }
}

function createViewer(gl, canvas, baseCanvas, restoreCanvas, sessionId) {
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

  // Jitter buffer: physics frames arrive at an irregular ~5 Hz (the substep
  // count varies step to step), so playback runs against a smoothed sim-time
  // clock that lags the newest frame by ~2.5 frame intervals and interpolates
  // inside whichever pair of buffered frames brackets it. Roughly half a
  // second of latency in exchange for stutter-free motion.
  const RING_SIZE = 12;
  const ring = [];
  for (let i = 0; i < RING_SIZE; i += 1) {
    ring.push({
      buffer: gl.createBuffer(),
      simTime: -Infinity,
      count: 0,
      capacityBytes: 0,
      quant: null,
    });
  }
  let ringHead = -1;
  let ringFrames = 0;
  const lineBuffer = gl.createBuffer();
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
    focus: [0, 0, 0],
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
    latestQuant: null,
  };

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function cameraBasis() {
    const distance = Math.max(0.08, camera.baseDistance * camera.distanceScale);
    const eye = [
      camera.focus[0] + distance * Math.cos(camera.pitch) * Math.cos(camera.yaw),
      camera.focus[1] + distance * Math.cos(camera.pitch) * Math.sin(camera.yaw),
      camera.focus[2] + distance * Math.sin(camera.pitch),
    ];
    const normalize = (v) => {
      const length = Math.hypot(v[0], v[1], v[2]) || 1;
      return [v[0] / length, v[1] / length, v[2] / length];
    };
    const forward = normalize([
      camera.focus[0] - eye[0],
      camera.focus[1] - eye[1],
      camera.focus[2] - eye[2],
    ]);
    let right = [
      forward[1] * 1 - forward[2] * 0,
      forward[2] * 0 - forward[0] * 1,
      forward[0] * 0 - forward[1] * 0,
    ];
    const rightLength = Math.hypot(right[0], right[1], right[2]);
    right = rightLength <= 1e-6 ? [1, 0, 0] : right.map((v) => v / rightLength);
    const up = [
      right[1] * forward[2] - right[2] * forward[1],
      right[2] * forward[0] - right[0] * forward[2],
      right[0] * forward[1] - right[1] * forward[0],
    ];
    return { distance, eye, forward, right, up };
  }

  // Frame bounds come for free from the packet's quantization ranges.
  function updateSceneBoundsFromQuant(quant, force = false) {
    if (!camera.autoFrame && !force) {
      return;
    }
    camera.focus = [
      quant.posMin[0] + quant.posScale[0] * 0.5,
      quant.posMin[1] + quant.posScale[1] * 0.5,
      quant.posMin[2] + quant.posScale[2] * 0.5,
    ];
    const radius = Math.max(
      1,
      0.5 * Math.hypot(quant.posScale[0], quant.posScale[1], quant.posScale[2])
    );
    camera.sceneRadius = radius;
    camera.baseDistance = (radius * 0.9) / Math.tan(Math.PI / 8);
    if (!force && radius > 0.5) {
      camera.autoFrame = false;
    }
  }

  function resizeToDisplay() {
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return;
    }
    const pixelRatio = Math.min(2, window.devicePixelRatio || 1);
    const width = Math.round(rect.width * pixelRatio);
    const height = Math.round(rect.height * pixelRatio);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  }

  function drawAxes(viewProj) {
    const axisLength = Math.max(1.4, camera.sceneRadius * 0.09);
    const vertices = new Float32Array([
      // x axis: red
      0, 0, 0, 1.0, 0.43, 0.43, 0.56, axisLength, 0, 0, 1.0, 0.43, 0.43, 0.56,
      // y axis: green
      0, 0, 0, 0.47, 1.0, 0.67, 0.56, 0, axisLength, 0, 0.47, 1.0, 0.67, 0.56,
      // z axis: blue
      0, 0, 0, 0.43, 0.67, 1.0, 0.56, 0, 0, axisLength, 0.43, 0.67, 1.0, 0.56,
    ]);
    gl.useProgram(lineProgram);
    gl.uniformMatrix4fv(lineUniforms.viewProj, false, viewProj);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STREAM_DRAW);
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

  // Chooses the buffered frame pair bracketing the playback clock and the
  // interpolation fraction between them.
  function selectFramePair(nowMs) {
    if (ringFrames === 0) {
      return null;
    }
    const dtWall = clamp(nowMs - (state.lastRafMs || nowMs), 0, 100);
    state.lastRafMs = nowMs;

    const ordered = [];
    for (let i = ringFrames - 1; i >= 0; i -= 1) {
      ordered.push(ring[(ringHead - i + RING_SIZE * 2) % RING_SIZE]);
    }
    const latest = ordered[ordered.length - 1];
    if (ordered.length === 1 || state.playbackRate <= 0) {
      state.playbackSimTime = latest.simTime;
      return { previous: latest, current: latest, alpha: 1 };
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
    state.playbackSimTime = clamp(
      state.playbackSimTime,
      ordered[0].simTime,
      latest.simTime
    );

    let previous = ordered[0];
    let current = latest;
    for (let i = 0; i < ordered.length - 1; i += 1) {
      if (
        state.playbackSimTime >= ordered[i].simTime &&
        state.playbackSimTime <= ordered[i + 1].simTime
      ) {
        previous = ordered[i];
        current = ordered[i + 1];
        break;
      }
    }
    if (previous.count !== current.count) {
      // Index-paired interpolation is meaningless across a budget change.
      return { previous: current, current, alpha: 1 };
    }
    const span = current.simTime - previous.simTime;
    const alpha = span > 0
      ? clamp((state.playbackSimTime - previous.simTime) / span, 0, 1)
      : 1;
    return { previous, current, alpha };
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
    const alpha = pair.alpha;

    const basis = cameraBasis();
    const projection = perspectiveMatrix(
      Math.PI / 4,
      width / height,
      0.1,
      Math.max(10000, camera.sceneRadius * 200)
    );
    const view = lookAtMatrix(basis.eye, camera.focus, [0, 0, 1]);
    const viewProj = multiplyMat4(projection, view);
    const focalLength = (Math.min(width, height) * 0.5) / Math.tan(Math.PI / 8);

    const useHdr = ensureHdrTarget();
    gl.bindFramebuffer(gl.FRAMEBUFFER, useHdr ? hdrFramebuffer : null);
    gl.viewport(0, 0, width, height);
    gl.clearColor(useHdr ? 0 : 0.008, useHdr ? 0 : 0.031, useHdr ? 0 : 0.063, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(pointProgram);
    gl.uniformMatrix4fv(uniforms.viewProj, false, viewProj);
    gl.uniform3f(uniforms.forward, basis.forward[0], basis.forward[1], basis.forward[2]);
    gl.uniform1f(uniforms.alpha, alpha);
    gl.uniform1f(uniforms.pointScale, focalLength);
    gl.uniform1f(uniforms.sizeBoost, Math.min(2, window.devicePixelRatio || 1));
    const prevQuant = pair.previous.quant;
    const curQuant = pair.current.quant;
    gl.uniform3fv(uniforms.posMin0, prevQuant.posMin);
    gl.uniform3fv(uniforms.posScale0, prevQuant.posScale);
    gl.uniform3fv(uniforms.velMin0, prevQuant.velMin);
    gl.uniform3fv(uniforms.velScale0, prevQuant.velScale);
    gl.uniform3fv(uniforms.posMin1, curQuant.posMin);
    gl.uniform3fv(uniforms.posScale1, curQuant.posScale);
    gl.uniform3fv(uniforms.velMin1, curQuant.velMin);
    gl.uniform3fv(uniforms.velScale1, curQuant.velScale);
    gl.uniform2f(uniforms.massLog, curQuant.massLogMin, curQuant.massLogScale);
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

    drawAxes(viewProj);
  }

  // --- WebSocket ---
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}/ws/frames/${sessionId}`);
  socket.binaryType = "arraybuffer";

  function dispatchEvent(name) {
    window.dispatchEvent(new Event(name));
  }

  socket.onmessage = (event) => {
    if (state.disposed || !(event.data instanceof ArrayBuffer)) {
      return;
    }
    const view = new DataView(event.data);
    if (event.data.byteLength < HEADER_BYTES ||
        view.getUint32(0, true) !== PACKET_MAGIC ||
        view.getUint32(4, true) !== PACKET_VERSION) {
      dispatchEvent("galaxy-viewer-error");
      return;
    }
    const previewCount = view.getUint32(16, true);
    const simTime = view.getFloat64(24, true);
    if (
      event.data.byteLength !==
      HEADER_BYTES + QUANT_BLOCK_BYTES + previewCount * PARTICLE_STRIDE
    ) {
      dispatchEvent("galaxy-viewer-error");
      return;
    }
    if (simTime <= state.simTimeMyr + 1e-9) {
      return;
    }

    const quant = {
      posMin: [
        view.getFloat32(80, true),
        view.getFloat32(84, true),
        view.getFloat32(88, true),
      ],
      posScale: [
        view.getFloat32(92, true),
        view.getFloat32(96, true),
        view.getFloat32(100, true),
      ],
      velMin: [
        view.getFloat32(104, true),
        view.getFloat32(108, true),
        view.getFloat32(112, true),
      ],
      velScale: [
        view.getFloat32(116, true),
        view.getFloat32(120, true),
        view.getFloat32(124, true),
      ],
      massLogMin: view.getFloat32(128, true),
      massLogScale: view.getFloat32(132, true),
    };
    updateSceneBoundsFromQuant(quant);

    const particleBytes = new Uint8Array(
      event.data,
      HEADER_BYTES + QUANT_BLOCK_BYTES
    );

    // Write into the next ring slot; the playback clock lags far enough
    // behind that the overwritten slot is never part of the active pair.
    // Stores are preallocated once and updated with bufferSubData so the
    // driver never reallocates in the hot path.
    ringHead = (ringHead + 1) % RING_SIZE;
    const slot = ring[ringHead];
    gl.bindBuffer(gl.ARRAY_BUFFER, slot.buffer);
    if (slot.capacityBytes < particleBytes.byteLength) {
      gl.bufferData(gl.ARRAY_BUFFER, particleBytes.byteLength, gl.DYNAMIC_DRAW);
      slot.capacityBytes = particleBytes.byteLength;
    }
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, particleBytes);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    slot.simTime = simTime;
    slot.count = previewCount;
    slot.quant = quant;
    ringFrames = Math.min(ringFrames + 1, RING_SIZE);
    state.latestQuant = quant;

    // Track the sim-time playback rate and frame spacing (EMAs) that drive
    // the jitter-buffered clock in render().
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
  };
  socket.onclose = () => {
    if (!state.disposed) {
      dispatchEvent("galaxy-viewer-error");
    }
  };
  socket.onerror = () => {
    if (!state.disposed) {
      dispatchEvent("galaxy-viewer-error");
    }
  };

  // --- controls (mirrors the other renderers) ---
  function bufferScale() {
    const rect = canvas.getBoundingClientRect();
    return rect.width > 0 ? canvas.width / rect.width : 1;
  }

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
        const basis = cameraBasis();
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
      const rect = canvas.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        const scale = bufferScale();
        const cursorX = (event.clientX - rect.left) * scale;
        const cursorY = (event.clientY - rect.top) * scale;
        const basis = cameraBasis();
        const focal = (Math.min(canvas.width, canvas.height) * 0.5) / Math.tan(Math.PI / 8);
        const rightAmount = (cursorX - canvas.width * 0.5) / focal;
        const upAmount = -(cursorY - canvas.height * 0.5) / focal;
        const direction = [
          basis.forward[0] + basis.right[0] * rightAmount + basis.up[0] * upAmount,
          basis.forward[1] + basis.right[1] * rightAmount + basis.up[1] * upAmount,
          basis.forward[2] + basis.right[2] * rightAmount + basis.up[2] * upAmount,
        ];
        const length = Math.hypot(direction[0], direction[1], direction[2]) || 1;
        const along =
          (direction[0] * basis.forward[0] +
            direction[1] * basis.forward[1] +
            direction[2] * basis.forward[2]) /
          length;
        if (along > 1e-6) {
          const range = basis.distance / along;
          const pull = 1 - camera.distanceScale / previousScale;
          for (let axis = 0; axis < 3; axis += 1) {
            const target = basis.eye[axis] + (direction[axis] / length) * range;
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
      if (state.latestQuant) {
        updateSceneBoundsFromQuant(state.latestQuant, true);
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
      socket.onmessage = null;
      socket.onclose = null;
      socket.onerror = null;
      try {
        socket.close();
      } catch {
        // already closed
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
