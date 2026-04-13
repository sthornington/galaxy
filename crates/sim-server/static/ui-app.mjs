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

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
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
    const luminousParticles = frame.particles.filter(
      (particle) => particle.component !== 0 && particle.component !== 3
    );
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
    const perspective = clamp((focalLength / depth) * 0.18, 0.35, 3.5);

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
    const component = particle.component ?? 1;
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

    return {
      x: projected.x,
      y: projected.y,
      depth: projected.depth,
      glowRadius: Math.max(0.6, 1.3 * renderLuminosity * projected.perspective),
      coreRadius: Math.max(0.32, 0.42 * renderLuminosity * projected.perspective),
      glowAlpha: clamp(0.012 * renderLuminosity * projected.perspective, 0.003, 0.03),
      coreAlpha: clamp(0.032 * renderLuminosity * projected.perspective, 0.01, 0.08),
      color,
    };
  }

  let activeSessionId = null;
  let frameSocket = null;
  let sessionPoll = null;

  function preferredPreviewBudget() {
    const params = new URLSearchParams(window.location.search);
    return params.get("renderer") === "json" ? 12_288 : 32_768;
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
    sessionPoll = setIntervalImpl(() => {
      refreshSessionSummary(sessionId).catch((error) => {
        nodes.viewerStatus.textContent = error.message;
      });
    }, 250);
    return sessionPoll;
  }

  function drawFrame(frame) {
    camera.lastFrame = frame;
    updateSceneBounds(frame);
    const width = nodes.canvas.width;
    const height = nodes.canvas.height;
    context.clearRect(0, 0, width, height);

    const gradient = context.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, "rgba(76, 201, 240, 0.06)");
    gradient.addColorStop(1, "rgba(242, 193, 75, 0.06)");
    context.fillStyle = gradient;
    context.fillRect(0, 0, width, height);

    const projected = [];
    for (const particle of frame.particles) {
      const point = projectParticle(particle, width, height);
      if (point) {
        projected.push(point);
      }
    }
    projected.sort((left, right) => right.depth - left.depth);
    context.globalCompositeOperation = "lighter";

    for (const point of projected.map(stellarRenderStyle).filter(Boolean)) {
      const [r, g, b] = point.color;
      context.fillStyle = `rgba(${Math.floor(
        r * 255
      )}, ${Math.floor(g * 255)}, ${Math.floor(b * 255)}, ${point.glowAlpha})`;
      context.beginPath();
      context.arc(point.x, point.y, point.glowRadius, 0, Math.PI * 2);
      context.fill();

      context.fillStyle = `rgba(${Math.floor(r * 255)}, ${Math.floor(
        g * 255
      )}, ${Math.floor(b * 255)}, ${point.coreAlpha})`;
      context.beginPath();
      context.arc(point.x, point.y, point.coreRadius, 0, Math.PI * 2);
      context.fill();
    }
    context.globalCompositeOperation = "source-over";
    if (camera.dragging) {
      drawXyPlaneGrid(width, height);
    }
    drawOriginAxes(width, height);
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
    if (camera.lastFrame) {
      drawFrame(camera.lastFrame);
    }
  }

  function zoomAt(_clientX, _clientY, nextDistanceScale) {
    camera.autoFrame = false;
    camera.distanceScale = clamp(nextDistanceScale, 0.003, 20);
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
      const dx = event.clientX - camera.lastX;
      const dy = event.clientY - camera.lastY;
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

  function openFrameSocket(sessionId) {
    stopSessionPolling();
    closeFrameSocket();

    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    frameSocket = new WebSocketImpl(
      `${scheme}://${window.location.host}/ws/frames/${sessionId}?format=json`
    );
    frameSocket.onopen = () => {
      nodes.viewerStatus.textContent = "Streaming JSON preview frames.";
    };
    frameSocket.onclose = () => {
      nodes.viewerStatus.textContent = "Preview stream closed.";
    };
    frameSocket.onmessage = (event) => {
      const frame = JSON.parse(event.data);
      drawFrame(frame);
      nodes.previewCount.textContent =
        frame.diagnostics.preview_count.toLocaleString();
      nodes.simTime.textContent = `${frame.sim_time_myr.toFixed(2)} Myr`;
    };
    return frameSocket;
  }

  async function attachToSession(session) {
    updateSessionStats(session);
    closeFrameSocket();
    const usingRustViewer = await tryBootRustViewer(session.id);
    if (usingRustViewer) {
      startSessionPolling(session.id);
      nodes.viewerStatus.textContent =
        "Streaming binary preview frames into the Rust/WASM viewer.";
      return session;
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
    nodes.pause.addEventListener("click", () => control("pause"));
    nodes.resume.addEventListener("click", () => control("resume"));
    nodes.stop.addEventListener("click", () => control("stop"));
    nodes.snapshot.addEventListener("click", () => control("snapshot"));
    nodes.step.addEventListener("click", () => control("step", { substeps: 1 }));
  }

  async function boot() {
    bindControls();
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
