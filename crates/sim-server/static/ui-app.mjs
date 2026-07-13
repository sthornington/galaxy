export function createUiApp({
  document,
  window,
  fetchImpl = fetch,
  WebSocketImpl = WebSocket,
  tryBootRustViewer,
  setIntervalImpl = setInterval,
  clearIntervalImpl = clearInterval,
}) {
  const nodes = {
    presetList: document.getElementById("preset-list"),
    sessionId: document.getElementById("session-id"),
    sessionState: document.getElementById("session-state"),
    simTime: document.getElementById("sim-time"),
    particleCount: document.getElementById("particle-count"),
    previewCount: document.getElementById("preview-count"),
    viewerStatus: document.getElementById("viewer-status"),
    canvas: document.getElementById("preview-canvas"),
    refresh: document.getElementById("refresh-presets"),
    pause: document.getElementById("pause-btn"),
    resume: document.getElementById("resume-btn"),
    stop: document.getElementById("stop-btn"),
    snapshot: document.getElementById("snapshot-btn"),
    step: document.getElementById("step-btn"),
  };
  const context = nodes.canvas.getContext("2d");
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
    focusX: 0,
    focusY: 0,
    focusZ: 0,
    sceneRadius: 120,
    lastFrame: null,
  };
  const renderState = {
    previousFrame: null,
    targetFrame: null,
    blendStartMs: 0,
    blendDurationMs: 90,
    rafHandle: null,
  };
  const requestAnimationFrameImpl =
    typeof window.requestAnimationFrame === "function"
      ? window.requestAnimationFrame.bind(window)
      : (callback) => globalThis.setTimeout(() => callback(Date.now()), 16);

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  // Frames from the binary decoder always carry `component`; synthetic frames
  // (tests, tools) may omit it. Default to disk so such particles render.
  function componentOf(particle) {
    return particle.component ?? 1;
  }

  function isLuminous(particle) {
    const component = componentOf(particle);
    return component !== 0 && component !== 3;
  }

  function normalize3([x, y, z]) {
    const length = Math.hypot(x, y, z);
    if (length <= Number.EPSILON) {
      return [0, 0, 0];
    }
    return [x / length, y / length, z / length];
  }

  function dot3([ax, ay, az], [bx, by, bz]) {
    return ax * bx + ay * by + az * bz;
  }

  function cross3([ax, ay, az], [bx, by, bz]) {
    return [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx];
  }

  function updateSceneBounds(frame, force = false) {
    if (!camera.autoFrame && !force) {
      return;
    }
    const luminousParticles = frame.particles.filter(isLuminous);
    const particles = luminousParticles.length ? luminousParticles : frame.particles;

    if (!particles.length) {
      camera.focusX = 0;
      camera.focusY = 0;
      camera.focusZ = 0;
      camera.sceneRadius = 1;
      return;
    }

    let sumX = 0;
    let sumY = 0;
    let sumZ = 0;
    for (const particle of particles) {
      sumX += particle.position_kpc[0];
      sumY += particle.position_kpc[1];
      sumZ += particle.position_kpc[2];
    }
    camera.focusX = sumX / particles.length;
    camera.focusY = sumY / particles.length;
    camera.focusZ = sumZ / particles.length;

    let maxRadius = 1;
    for (const particle of particles) {
      const dx = particle.position_kpc[0] - camera.focusX;
      const dy = particle.position_kpc[1] - camera.focusY;
      const dz = particle.position_kpc[2] - camera.focusZ;
      maxRadius = Math.max(maxRadius, Math.hypot(dx, dy, dz));
    }
    camera.sceneRadius = maxRadius;
    camera.baseDistance = (camera.sceneRadius * 0.9) / Math.tan(Math.PI / 8);
    if (!force && particles.length >= 32 && maxRadius > 0.5) {
      camera.autoFrame = false;
    }
  }

  function projectParticle(particle, width, height) {
    const projected = projectPoint(particle.position_kpc, width, height);
    if (!projected) {
      return null;
    }

    return {
      ...projected,
      radialVelocityKms: dot3(particle.velocity_kms, projected.forward),
      particle,
    };
  }

  function projectPoint(position, width, height) {
    const fov = Math.PI / 4;
    const { cameraPosition, forward, right, up } = getCameraBasis();
    const relative = [
      position[0] - cameraPosition[0],
      position[1] - cameraPosition[1],
      position[2] - cameraPosition[2],
    ];
    const depth = dot3(relative, forward);
    if (depth <= 0.1) {
      return null;
    }

    const focalLength = (Math.min(width, height) * 0.5) / Math.tan(fov * 0.5);
    const x = width * 0.5 + (dot3(relative, right) * focalLength) / depth;
    const y = height * 0.5 - (dot3(relative, up) * focalLength) / depth;
    const perspective = clamp((focalLength / depth) * 0.18, 0.08, 3.5);

    return { x, y, depth, perspective, forward };
  }

  function getCameraBasis() {
    const distance = Math.max(0.08, camera.baseDistance * camera.distanceScale);
    const cameraPosition = [
      camera.focusX + distance * Math.cos(camera.pitch) * Math.cos(camera.yaw),
      camera.focusY + distance * Math.cos(camera.pitch) * Math.sin(camera.yaw),
      camera.focusZ + distance * Math.sin(camera.pitch),
    ];
    const forward = normalize3([
      camera.focusX - cameraPosition[0],
      camera.focusY - cameraPosition[1],
      camera.focusZ - cameraPosition[2],
    ]);
    let right = cross3(forward, [0, 0, 1]);
    if (Math.hypot(...right) <= 1.0e-6) {
      right = [1, 0, 0];
    } else {
      right = normalize3(right);
    }
    const up = normalize3(cross3(right, forward));
    return { distance, cameraPosition, forward, right, up };
  }

  function mixColor(a, b, t) {
    return [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t,
    ];
  }

  function clampColor([r, g, b]) {
    return [clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1)];
  }

  function stellarBaseColor(particle) {
    const component = componentOf(particle);
    const massMsun = Math.max(1, particle.mass_msun ?? 1);
    const massBias = clamp((Math.log10(massMsun) - 4.2) / 1.6, 0, 1);
    if (component === 3 || component === 0) {
      return null;
    }
    if (component === 2) {
      return mixColor([1.0, 0.82, 0.68], [1.0, 0.93, 0.82], 0.35 + 0.2 * massBias);
    }
    return mixColor([1.0, 0.92, 0.8], [0.79, 0.88, 1.0], 0.45 + 0.35 * massBias);
  }

  function applyDopplerShift(color, radialVelocityKms) {
    const shift = clamp(radialVelocityKms / 700, -0.28, 0.28);
    if (shift >= 0) {
      return clampColor([
        color[0] * (1 + 0.9 * shift),
        color[1] * (1 + 0.2 * shift),
        color[2] * (1 - 0.75 * shift),
      ]);
    }
    const blue = -shift;
    return clampColor([
      color[0] * (1 - 0.75 * blue),
      color[1] * (1 + 0.1 * blue),
      color[2] * (1 + 0.95 * blue),
    ]);
  }

  function stellarRenderStyle(projected) {
    const particle = projected.particle;
    const baseColor = stellarBaseColor(particle);
    if (!baseColor) {
      return null;
    }

    const luminosity = clamp(
      (Math.log10(Math.max(1, particle.mass_msun ?? 1)) - 3.7) / 2.2,
      0.25,
      1.8,
    );
    const renderLuminosity = Math.pow(luminosity, 0.58);
    const color = applyDopplerShift(baseColor, projected.radialVelocityKms);
    const sizeScale = Math.pow(projected.perspective, 1.18);
    const alphaScale = Math.pow(projected.perspective, 0.92);

    return {
      x: projected.x,
      y: projected.y,
      depth: projected.depth,
      glowRadius: Math.max(0.08, 0.52 * renderLuminosity * sizeScale),
      coreRadius: Math.max(0.03, 0.12 * renderLuminosity * sizeScale),
      glowAlpha: clamp(0.0032 * renderLuminosity * alphaScale, 0.0012, 0.012),
      coreAlpha: clamp(0.072 * renderLuminosity * alphaScale, 0.012, 0.18),
      color,
    };
  }

  let activeSessionId = null;
  let frameSocket = null;
  let sessionPoll = null;
  let attachGeneration = 0;

  const PREVIEW_BUDGETS = { webgl: 262_144, wasm: 32_768, json: 12_288 };

  function preferredPreviewBudget() {
    const params = new URLSearchParams(window.location.search);
    const renderer = params.get("renderer");
    if (renderer === "json" || renderer === "wasm") {
      return PREVIEW_BUDGETS[renderer];
    }
    return PREVIEW_BUDGETS.webgl;
  }

  // Sends a one-shot preview-budget renegotiation over the control socket,
  // used when the attach falls back to a renderer that cannot sustain the
  // WebGL-sized stream.
  function sendPreviewBudget(sessionId, particleBudget) {
    try {
      const scheme = window.location.protocol === "https:" ? "wss" : "ws";
      const socket = new WebSocketImpl(
        `${scheme}://${window.location.host}/ws/control/${sessionId}`
      );
      socket.onopen = () => {
        socket.send(
          JSON.stringify({ kind: "set_preview_budget", particle_budget: particleBudget })
        );
        socket.close();
      };
      socket.onerror = () => {};
    } catch {
      // Best effort; oversized frames still render, just slowly.
    }
  }

  async function fetchJson(path, options = {}) {
    const response = await fetchImpl(path, {
      headers: { "content-type": "application/json" },
      ...options,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error ?? `Request failed: ${response.status}`);
    }
    return response.json();
  }

  function updateSessionStats(session) {
    activeSessionId = session.id;
    nodes.sessionId.textContent = session.id;
    nodes.sessionState.textContent = session.state;
    nodes.simTime.textContent = `${session.sim_time_myr.toFixed(2)} Myr`;
    nodes.particleCount.textContent = session.particle_count.toLocaleString();
    nodes.previewCount.textContent =
      session.diagnostics.preview_count.toLocaleString();
  }

  function closeFrameSocket() {
    if (frameSocket) {
      frameSocket.close();
      frameSocket = null;
    }
  }

  function stopSessionPolling() {
    if (sessionPoll !== null) {
      clearIntervalImpl(sessionPoll);
      sessionPoll = null;
    }
  }

  function clearActiveSession(sessionId = activeSessionId) {
    activeSessionId = null;
    nodes.sessionId.textContent = sessionId ?? "none";
    nodes.sessionState.textContent = "stopped";
  }

  async function refreshSessionSummary(sessionId = activeSessionId) {
    if (!sessionId) {
      return null;
    }
    const session = await fetchJson(`/api/session/${sessionId}`);
    updateSessionStats(session);
    return session;
  }

  function startSessionPolling(sessionId) {
    stopSessionPolling();
    let consecutiveFailures = 0;
    sessionPoll = setIntervalImpl(() => {
      refreshSessionSummary(sessionId)
        .then(() => {
          consecutiveFailures = 0;
        })
        .catch((error) => {
          consecutiveFailures += 1;
          nodes.viewerStatus.textContent = error.message;
          // A vanished session (or persistent failure) should not be polled
          // forever at 4 requests per second.
          if (
            consecutiveFailures >= 20 ||
            String(error.message).includes("unknown session")
          ) {
            stopSessionPolling();
          }
        });
    }, 250);
    return sessionPoll;
  }

  // Pre-rendered radial sprites per quantized color: one drawImage per splat
  // instead of two path+fill calls, roughly an order of magnitude faster at
  // full preview budgets.
  const spriteCache = new Map();
  const SPRITE_SIZE = 32;

  function particleSprite(r, g, b) {
    const key = (r << 16) | (g << 8) | b;
    let sprite = spriteCache.get(key);
    if (sprite !== undefined) {
      return sprite;
    }
    sprite = null;
    try {
      const surface = document.createElement("canvas");
      surface.width = SPRITE_SIZE;
      surface.height = SPRITE_SIZE;
      const surfaceContext = surface.getContext("2d");
      if (surfaceContext?.createRadialGradient) {
        const half = SPRITE_SIZE / 2;
        const gradient = surfaceContext.createRadialGradient(half, half, 0, half, half, half);
        gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, 1)`);
        gradient.addColorStop(0.55, `rgba(${r}, ${g}, ${b}, 0.45)`);
        gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
        surfaceContext.fillStyle = gradient;
        surfaceContext.fillRect(0, 0, SPRITE_SIZE, SPRITE_SIZE);
        sprite = surface;
      }
    } catch {
      sprite = null;
    }
    spriteCache.set(key, sprite);
    return sprite;
  }

  function drawFrameInternal(frameA, frameB = null, alpha = 1) {
    const targetFrame = frameB ?? frameA;
    camera.lastFrame = targetFrame;
    const width = nodes.canvas.width;
    const height = nodes.canvas.height;
    context.clearRect(0, 0, width, height);

    // Match the WASM viewer's deep-space backdrop so switching renderers is
    // not visually jarring.
    context.fillStyle = "#020810";
    context.fillRect(0, 0, width, height);

    const stellarProjected = [];
    const count = frameB
      ? Math.min(frameA.particles.length, frameB.particles.length)
      : frameA.particles.length;
    for (let index = 0; index < count; index += 1) {
      const particle = frameA.particles[index];
      const targetParticle = frameB ? frameB.particles[index] : particle;
      if (!isLuminous(targetParticle)) {
        continue;
      }
      const point = frameB
        ? projectInterpolatedParticle(particle, targetParticle, alpha, width, height)
        : projectParticle(particle, width, height);
      if (!point) {
        continue;
      }
      const style = stellarRenderStyle(point);
      if (style) {
        stellarProjected.push(style);
      }
    }

    stellarProjected.sort((left, right) => right.depth - left.depth);
    context.globalCompositeOperation = "lighter";

    const canDrawSprites = typeof context.drawImage === "function";
    for (const point of stellarProjected) {
      const [r, g, b] = point.color;
      // Quantize to 16 levels per channel so the sprite cache stays small.
      const red = Math.floor(r * 15) * 17;
      const green = Math.floor(g * 15) * 17;
      const blue = Math.floor(b * 15) * 17;
      const sprite = canDrawSprites ? particleSprite(red, green, blue) : null;
      if (sprite) {
        context.globalAlpha = clamp(point.glowAlpha * 2.2, 0, 1);
        context.drawImage(
          sprite,
          point.x - point.glowRadius,
          point.y - point.glowRadius,
          point.glowRadius * 2,
          point.glowRadius * 2
        );
        context.globalAlpha = clamp(point.coreAlpha, 0, 1);
        context.drawImage(
          sprite,
          point.x - point.coreRadius,
          point.y - point.coreRadius,
          point.coreRadius * 2,
          point.coreRadius * 2
        );
        continue;
      }
      context.fillStyle = `rgba(${red}, ${green}, ${blue}, ${point.glowAlpha})`;
      context.beginPath();
      context.arc(point.x, point.y, point.glowRadius, 0, Math.PI * 2);
      context.fill();
      context.fillStyle = `rgba(${red}, ${green}, ${blue}, ${point.coreAlpha})`;
      context.beginPath();
      context.arc(point.x, point.y, point.coreRadius, 0, Math.PI * 2);
      context.fill();
    }
    context.globalAlpha = 1;
    context.globalCompositeOperation = "source-over";
    if (camera.dragging) {
      drawXyPlaneGrid(width, height);
    }
    drawOriginAxes(width, height);
  }

  function drawFrame(frame) {
    updateSceneBounds(frame);
    drawFrameInternal(frame);
  }

  function projectInterpolatedParticle(particleA, particleB, alpha, width, height) {
    if (componentOf(particleA) !== componentOf(particleB)) {
      return projectParticle(particleB, width, height);
    }
    const position = [
      particleA.position_kpc[0] + (particleB.position_kpc[0] - particleA.position_kpc[0]) * alpha,
      particleA.position_kpc[1] + (particleB.position_kpc[1] - particleA.position_kpc[1]) * alpha,
      particleA.position_kpc[2] + (particleB.position_kpc[2] - particleA.position_kpc[2]) * alpha,
    ];
    const projected = projectPoint(position, width, height);
    if (!projected) {
      return null;
    }
    const velocity = [
      particleA.velocity_kms[0] + (particleB.velocity_kms[0] - particleA.velocity_kms[0]) * alpha,
      particleA.velocity_kms[1] + (particleB.velocity_kms[1] - particleA.velocity_kms[1]) * alpha,
      particleA.velocity_kms[2] + (particleB.velocity_kms[2] - particleA.velocity_kms[2]) * alpha,
    ];
    return {
      ...projected,
      radialVelocityKms: dot3(velocity, projected.forward),
      particle: particleB,
    };
  }

  function renderLatestFrame(now = window.performance?.now?.() ?? Date.now()) {
    renderState.rafHandle = null;
    if (!renderState.targetFrame) {
      return;
    }
    if (!renderState.previousFrame) {
      drawFrame(renderState.targetFrame);
      return;
    }
    const alpha = clamp(
      (now - renderState.blendStartMs) / renderState.blendDurationMs,
      0,
      1
    );
    drawFrameInternal(renderState.previousFrame, renderState.targetFrame, alpha);
    if (alpha < 1) {
      renderState.rafHandle = requestAnimationFrameImpl(renderLatestFrame);
    }
  }

  function queueFrame(frame) {
    if (
      renderState.targetFrame &&
      Number.isFinite(renderState.targetFrame.sim_time_myr) &&
      Number.isFinite(frame.sim_time_myr) &&
      frame.sim_time_myr <= renderState.targetFrame.sim_time_myr + 1e-9
    ) {
      return;
    }
    updateSceneBounds(frame);
    camera.lastFrame = frame;
    // Interpolation pairs particles by index, which only holds while the
    // preview budget is unchanged between frames.
    if (
      renderState.targetFrame &&
      renderState.targetFrame.particles.length === frame.particles.length
    ) {
      renderState.previousFrame = renderState.targetFrame;
    } else {
      renderState.previousFrame = frame;
    }
    renderState.targetFrame = frame;
    renderState.blendStartMs = window.performance?.now?.() ?? Date.now();
    renderLatestFrame(renderState.blendStartMs);
    if (renderState.rafHandle === null) {
      renderState.rafHandle = requestAnimationFrameImpl(renderLatestFrame);
    }
  }

  function niceGridSpacing(target) {
    const normalized = Math.max(target, 0.25);
    const exponent = Math.floor(Math.log10(normalized));
    const base = 10 ** exponent;
    const scaled = normalized / base;
    if (scaled <= 1) {
      return base;
    }
    if (scaled <= 2) {
      return 2 * base;
    }
    if (scaled <= 5) {
      return 5 * base;
    }
    return 10 * base;
  }

  function drawProjectedSegment(a, b, color, width) {
    if (!a || !b) {
      return;
    }
    context.beginPath();
    context.strokeStyle = color;
    context.lineWidth = width;
    context.moveTo(a.x, a.y);
    context.lineTo(b.x, b.y);
    context.stroke();
  }

  function drawXyPlaneGrid(width, height) {
    const extent = Math.max(10, camera.sceneRadius * 0.9);
    const spacing = niceGridSpacing(extent / 6);
    const lineCount = Math.max(2, Math.min(10, Math.ceil(extent / spacing)));

    context.save?.();
    for (let i = -lineCount; i <= lineCount; i += 1) {
      const axisOffset = i * spacing;
      const major = i % 5 === 0;
      const color = major
        ? "rgba(210, 224, 245, 0.11)"
        : "rgba(210, 224, 245, 0.055)";
      const strokeWidth = major ? 0.9 : 0.65;

      drawProjectedSegment(
        projectPoint([axisOffset, -extent, 0], width, height),
        projectPoint([axisOffset, extent, 0], width, height),
        color,
        strokeWidth
      );
      drawProjectedSegment(
        projectPoint([-extent, axisOffset, 0], width, height),
        projectPoint([extent, axisOffset, 0], width, height),
        color,
        strokeWidth
      );
    }
    context.restore?.();
  }

  function drawOriginAxes(width, height) {
    const axisLength = Math.max(1.4, camera.sceneRadius * 0.09);
    const origin = projectPoint([0, 0, 0], width, height);
    if (!origin) {
      return;
    }

    const axes = [
      { label: "X", color: "rgba(255, 110, 110, 0.56)", endpoint: [axisLength, 0, 0] },
      { label: "Y", color: "rgba(120, 255, 170, 0.56)", endpoint: [0, axisLength, 0] },
      { label: "Z", color: "rgba(110, 170, 255, 0.56)", endpoint: [0, 0, axisLength] },
    ];

    context.save?.();
    context.lineWidth = clamp(origin.perspective * 0.75, 0.8, 1.4);
    context.font = '500 10px "IBM Plex Sans", sans-serif';
    context.textAlign = "center";
    context.textBaseline = "middle";

    for (const axis of axes) {
      const projected = projectPoint(axis.endpoint, width, height);
      if (!projected) {
        continue;
      }
      context.strokeStyle = axis.color;
      context.beginPath();
      context.moveTo(origin.x, origin.y);
      context.lineTo(projected.x, projected.y);
      context.stroke();
      context.fillStyle = axis.color;
      context.fillText(axis.label, projected.x, projected.y);
    }

    context.beginPath();
    context.fillStyle = "rgba(245, 248, 255, 0.6)";
    context.arc(origin.x, origin.y, clamp(origin.perspective * 0.9, 1.1, 2.2), 0, Math.PI * 2);
    context.fill();
    context.fillStyle = "rgba(245, 248, 255, 0.45)";
    context.fillText("O", origin.x, origin.y - 9);
    context.restore?.();
  }

  function redrawFrame() {
    if (renderState.targetFrame) {
      renderLatestFrame(window.performance?.now?.() ?? Date.now());
    } else if (camera.lastFrame) {
      drawFrame(camera.lastFrame);
    }
  }

  function cssToBufferScale() {
    const rect = nodes.canvas.getBoundingClientRect?.();
    if (rect && rect.width > 0) {
      return nodes.canvas.width / rect.width;
    }
    return 1;
  }

  function zoomAt(clientX, clientY, nextDistanceScale) {
    camera.autoFrame = false;
    const previousScale = camera.distanceScale;
    camera.distanceScale = clamp(nextDistanceScale, 0.003, 20);

    // Slide the focus toward the world point under the cursor so it stays
    // roughly fixed on screen while zooming.
    const rect = nodes.canvas.getBoundingClientRect?.();
    if (rect && rect.width > 0 && rect.height > 0 && previousScale > 0) {
      const bufferScale = cssToBufferScale();
      const cursorX = (clientX - rect.left) * bufferScale;
      const cursorY = (clientY - rect.top) * bufferScale;
      const width = nodes.canvas.width;
      const height = nodes.canvas.height;
      const fov = Math.PI / 4;
      const { distance, cameraPosition, forward, right, up } = getCameraBasis();
      const focalLength = (Math.min(width, height) * 0.5) / Math.tan(fov * 0.5);
      const rightAmount = (cursorX - width * 0.5) / focalLength;
      const upAmount = -(cursorY - height * 0.5) / focalLength;
      const direction = normalize3([
        forward[0] + right[0] * rightAmount + up[0] * upAmount,
        forward[1] + right[1] * rightAmount + up[1] * upAmount,
        forward[2] + right[2] * rightAmount + up[2] * upAmount,
      ]);
      const along = dot3(direction, forward);
      if (along > 1e-6) {
        const range = distance / along;
        const pull = 1 - camera.distanceScale / previousScale;
        camera.focusX += (cameraPosition[0] + direction[0] * range - camera.focusX) * pull;
        camera.focusY += (cameraPosition[1] + direction[1] * range - camera.focusY) * pull;
        camera.focusZ += (cameraPosition[2] + direction[2] * range - camera.focusZ) * pull;
      }
    }
    redrawFrame();
  }

  function bindCanvasControls() {
    nodes.canvas.addEventListener("contextmenu", (event) => {
      event.preventDefault?.();
    });

    nodes.canvas.addEventListener("mousedown", (event) => {
      camera.autoFrame = false;
      camera.dragging = true;
      camera.dragMode =
        event.button === 2 || event.shiftKey ? "pan" : "orbit";
      camera.lastX = event.clientX;
      camera.lastY = event.clientY;
    });

    nodes.canvas.addEventListener("mousemove", (event) => {
      if (!camera.dragging) {
        return;
      }
      const bufferScale = cssToBufferScale();
      const dx = (event.clientX - camera.lastX) * bufferScale;
      const dy = (event.clientY - camera.lastY) * bufferScale;
      if (camera.dragMode === "pan") {
        const { distance, right, up } = getCameraBasis();
        const fov = Math.PI / 4;
        const panScale =
          (distance * Math.tan(fov * 0.5)) / (Math.min(nodes.canvas.width, nodes.canvas.height) * 0.5);
        camera.focusX += (-dx * panScale) * right[0] + (dy * panScale) * up[0];
        camera.focusY += (-dx * panScale) * right[1] + (dy * panScale) * up[1];
        camera.focusZ += (-dx * panScale) * right[2] + (dy * panScale) * up[2];
      } else {
        camera.yaw -= dx * 0.006;
        camera.pitch = clamp(camera.pitch - dy * 0.006, -1.45, 1.45);
      }
      camera.lastX = event.clientX;
      camera.lastY = event.clientY;
      redrawFrame();
    });

    const stopDragging = () => {
      camera.dragging = false;
      camera.dragMode = "orbit";
    };
    nodes.canvas.addEventListener("mouseup", stopDragging);
    nodes.canvas.addEventListener("mouseleave", stopDragging);

    nodes.canvas.addEventListener("wheel", (event) => {
      event.preventDefault?.();
      const factor = event.deltaY < 0 ? 0.78 : 1.0 / 0.78;
      zoomAt(event.clientX, event.clientY, camera.distanceScale * factor);
    });

    nodes.canvas.addEventListener("dblclick", () => {
      camera.yaw = 0.4;
      camera.pitch = 0.9;
      camera.distanceScale = 1.2;
      camera.autoFrame = true;
      if (camera.lastFrame) {
        updateSceneBounds(camera.lastFrame, true);
      }
      redrawFrame();
    });
  }

  function openFrameSocket(sessionId, attempt = 0) {
    closeFrameSocket();
    // The JSON stream carries preview data only; keep polling the session
    // summary so State/Particles stay live in this mode too.
    startSessionPolling(sessionId);

    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocketImpl(
      `${scheme}://${window.location.host}/ws/frames/${sessionId}?format=json`
    );
    frameSocket = socket;
    socket.onopen = () => {
      attempt = 0;
      nodes.viewerStatus.textContent = "Streaming JSON preview frames.";
    };
    socket.onclose = () => {
      if (frameSocket !== socket || activeSessionId !== sessionId) {
        return;
      }
      // Transient drops should not permanently kill the preview: reconnect
      // with capped exponential backoff while the session is still active.
      const delayMs = Math.min(8000, 500 * 2 ** Math.min(attempt, 4));
      nodes.viewerStatus.textContent = `Preview stream closed; reconnecting in ${(delayMs / 1000).toFixed(1)}s...`;
      globalThis.setTimeout(() => {
        if (frameSocket === socket && activeSessionId === sessionId) {
          openFrameSocket(sessionId, attempt + 1);
        }
      }, delayMs);
    };
    socket.onerror = () => {};
    socket.onmessage = (event) => {
      const frame = JSON.parse(event.data);
      queueFrame(frame);
      nodes.previewCount.textContent =
        frame.diagnostics.preview_count.toLocaleString();
      nodes.simTime.textContent = `${frame.sim_time_myr.toFixed(2)} Myr`;
    };
    return socket;
  }

  async function attachToSession(session) {
    // A second Launch while the viewer boot below is still awaiting must win;
    // the stale attach would otherwise clobber the new session's socket.
    const generation = ++attachGeneration;
    updateSessionStats(session);
    closeFrameSocket();
    const tier = await tryBootRustViewer(session.id);
    if (generation !== attachGeneration) {
      return session;
    }
    if (tier) {
      startSessionPolling(session.id);
      if (tier === "webgl") {
        nodes.viewerStatus.textContent =
          "Streaming binary preview frames into the WebGL renderer.";
      } else {
        nodes.viewerStatus.textContent =
          "Streaming binary preview frames into the Rust/WASM viewer.";
        if (session.preview_particle_budget > PREVIEW_BUDGETS.wasm) {
          sendPreviewBudget(session.id, PREVIEW_BUDGETS.wasm);
        }
      }
      return session;
    }
    if (session.preview_particle_budget > PREVIEW_BUDGETS.json) {
      sendPreviewBudget(session.id, PREVIEW_BUDGETS.json);
    }
    openFrameSocket(session.id);
    return session;
  }

  async function stopActiveSession() {
    if (!activeSessionId) {
      return null;
    }
    const sessionId = activeSessionId;
    stopSessionPolling();
    closeFrameSocket();
    const session = await fetchJson(`/api/session/${sessionId}/stop`, {
      method: "POST",
      body: JSON.stringify({}),
    });
    clearActiveSession(session.id);
    nodes.viewerStatus.textContent = `Stopped session ${session.id}.`;
    return session;
  }

  async function createSession(presetId) {
    nodes.viewerStatus.textContent = "Creating session on the GPU backend...";
    if (activeSessionId) {
      await stopActiveSession().catch(() => null);
    }
    const session = await fetchJson("/api/session", {
      method: "POST",
      body: JSON.stringify({
        preset_id: presetId,
        seed: 42,
        preview_particle_budget: preferredPreviewBudget(),
      }),
    });
    return attachToSession(session);
  }

  async function control(action, payload = {}) {
    if (!activeSessionId) {
      return null;
    }
    if (action === "stop") {
      return stopActiveSession();
    }
    const session = await fetchJson(`/api/session/${activeSessionId}/${action}`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    updateSessionStats(session);
    return session;
  }

  function buildPresetCard(preset) {
    const card = document.createElement("article");
    card.className = "preset-card";

    const title = document.createElement("h3");
    title.textContent = preset.title;
    const summary = document.createElement("p");
    summary.textContent = preset.summary;
    const meta = document.createElement("div");
    meta.className = "preset-meta";
    const presetId = document.createElement("span");
    presetId.textContent = preset.id;
    const galaxyCount = document.createElement("span");
    galaxyCount.textContent = `${preset.config.galaxies.length} galaxies`;
    meta.appendChild(presetId);
    meta.appendChild(galaxyCount);
    const button = document.createElement("button");
    button.dataset.preset = preset.id;
    button.textContent = "Launch";
    button.addEventListener("click", () => {
      createSession(preset.id).catch((error) => {
        nodes.viewerStatus.textContent = error.message;
      });
    });

    card.appendChild(title);
    card.appendChild(summary);
    card.appendChild(meta);
    card.appendChild(button);
    return card;
  }

  async function loadPresets() {
    nodes.presetList.innerHTML = "";
    const presets = await fetchJson("/api/presets");
    for (const preset of presets) {
      nodes.presetList.appendChild(buildPresetCard(preset));
    }
    return presets;
  }

  async function restoreLatestSession() {
    const sessions = await fetchJson("/api/sessions");
    const session = sessions[0] ?? null;
    if (!session) {
      return null;
    }
    nodes.viewerStatus.textContent = `Reattached to session ${session.id}.`;
    return attachToSession(session);
  }

  function bindControls() {
    bindCanvasControls();
    nodes.refresh.addEventListener("click", () => {
      loadPresets().catch((error) => {
        nodes.viewerStatus.textContent = error.message;
      });
    });
    const reportControlError = (error) => {
      nodes.viewerStatus.textContent = error.message;
    };
    nodes.pause.addEventListener("click", () => control("pause").catch(reportControlError));
    nodes.resume.addEventListener("click", () => control("resume").catch(reportControlError));
    nodes.stop.addEventListener("click", () => control("stop").catch(reportControlError));
    nodes.snapshot.addEventListener("click", () =>
      control("snapshot").catch(reportControlError)
    );
    nodes.step.addEventListener("click", () =>
      control("step", { substeps: 1 }).catch(reportControlError)
    );
  }

  function resizeCanvasToDisplay() {
    const rect = nodes.canvas.getBoundingClientRect?.();
    if (!rect || rect.width <= 0 || rect.height <= 0) {
      return;
    }
    const pixelRatio = Math.min(2, window.devicePixelRatio ?? 1);
    const width = Math.round(rect.width * pixelRatio);
    const height = Math.round(rect.height * pixelRatio);
    if (nodes.canvas.width !== width || nodes.canvas.height !== height) {
      nodes.canvas.width = width;
      nodes.canvas.height = height;
      redrawFrame();
    }
  }

  async function boot() {
    bindControls();
    resizeCanvasToDisplay();
    window.addEventListener?.("resize", resizeCanvasToDisplay);
    try {
      await loadPresets();
      await restoreLatestSession();
    } catch (error) {
      nodes.viewerStatus.textContent = error.message;
      throw error;
    }
  }

  return {
    boot,
    bindControls,
    control,
    createSession,
    drawFrame,
    attachToSession,
    fetchJson,
    loadPresets,
    openFrameSocket,
    refreshSessionSummary,
    restoreLatestSession,
    stopActiveSession,
    startSessionPolling,
    stopSessionPolling,
    updateSessionStats,
    zoomAt,
    getActiveSessionId: () => activeSessionId,
    getFrameSocket: () => frameSocket,
    getCamera: () => camera,
    nodes,
  };
}
