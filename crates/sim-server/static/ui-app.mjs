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
    const sizeScale = Math.pow(projected.perspective, 1.28);
    const alphaScale = Math.pow(projected.perspective, 1.42);

    return {
      x: projected.x,
      y: projected.y,
      depth: projected.depth,
      glowRadius: Math.max(0.08, 0.52 * renderLuminosity * sizeScale),
      coreRadius: Math.max(0.03, 0.12 * renderLuminosity * sizeScale),
      glowAlpha: clamp(0.0024 * renderLuminosity * alphaScale, 0.00045, 0.009),
      coreAlpha: clamp(0.06 * renderLuminosity * alphaScale, 0.006, 0.16),
      color,
    };
  }

  function buildHaloField(width, height, haloPoints) {
    const gridWidth = Math.max(48, Math.floor(width / 10));
    const gridHeight = Math.max(27, Math.floor(height / 10));
    const density = new Float32Array(gridWidth * gridHeight);
    let maxDensity = 0;

    function accumulate(ix, iy, weight) {
      if (ix < 0 || iy < 0 || ix >= gridWidth || iy >= gridHeight || weight <= 0) {
        return;
      }
      const slot = iy * gridWidth + ix;
      density[slot] += weight;
      maxDensity = Math.max(maxDensity, density[slot]);
    }

    for (const point of haloPoints) {
      const gx = clamp((point.x / width) * gridWidth, 0, gridWidth - 1.0001);
      const gy = clamp((point.y / height) * gridHeight, 0, gridHeight - 1.0001);
      const ix = Math.floor(gx);
      const iy = Math.floor(gy);
      const tx = gx - ix;
      const ty = gy - iy;
      const massWeight = clamp(Math.log10(Math.max(1, point.particle.mass_msun)) / 6.5, 0.18, 1.0);
      const perspectiveWeight = Math.pow(Math.max(0.08, point.perspective), 0.85);
      const weight = massWeight * perspectiveWeight;
      accumulate(ix, iy, weight * (1 - tx) * (1 - ty));
      accumulate(ix + 1, iy, weight * tx * (1 - ty));
      accumulate(ix, iy + 1, weight * (1 - tx) * ty);
      accumulate(ix + 1, iy + 1, weight * tx * ty);
    }

    return {
      density,
      gridWidth,
      gridHeight,
      cellWidth: width / gridWidth,
      cellHeight: height / gridHeight,
      maxDensity,
    };
  }

  function drawHaloContours(field) {
    if (field.maxDensity <= 0) {
      return;
    }

    const thresholds = [
      { level: 0.16, color: "rgba(110, 170, 255, 0.12)", width: 0.8 },
      { level: 0.32, color: "rgba(130, 205, 255, 0.18)", width: 0.95 },
      { level: 0.54, color: "rgba(185, 235, 255, 0.26)", width: 1.05 },
    ];

    function edgePoint(x0, y0, v0, x1, y1, v1, threshold) {
      const dv = v1 - v0;
      if (Math.abs(dv) <= 1.0e-8) {
        return null;
      }
      const t = (threshold - v0) / dv;
      if (t < 0 || t > 1) {
        return null;
      }
      return [x0 + (x1 - x0) * t, y0 + (y1 - y0) * t];
    }

    for (const threshold of thresholds) {
      context.save?.();
      context.strokeStyle = threshold.color;
      context.lineWidth = threshold.width;
      for (let y = 0; y < field.gridHeight - 1; y += 1) {
        for (let x = 0; x < field.gridWidth - 1; x += 1) {
          const v00 = field.density[y * field.gridWidth + x] / field.maxDensity;
          const v10 = field.density[y * field.gridWidth + x + 1] / field.maxDensity;
          const v01 = field.density[(y + 1) * field.gridWidth + x] / field.maxDensity;
          const v11 = field.density[(y + 1) * field.gridWidth + x + 1] / field.maxDensity;
          const minValue = Math.min(v00, v10, v01, v11);
          const maxValue = Math.max(v00, v10, v01, v11);
          if (minValue > threshold.level || maxValue < threshold.level) {
            continue;
          }

          const x0 = x * field.cellWidth;
          const x1 = (x + 1) * field.cellWidth;
          const y0 = y * field.cellHeight;
          const y1 = (y + 1) * field.cellHeight;
          const points = [];
          const top = edgePoint(x0, y0, v00, x1, y0, v10, threshold.level);
          const right = edgePoint(x1, y0, v10, x1, y1, v11, threshold.level);
          const bottom = edgePoint(x1, y1, v11, x0, y1, v01, threshold.level);
          const left = edgePoint(x0, y1, v01, x0, y0, v00, threshold.level);
          if (top) points.push(top);
          if (right) points.push(right);
          if (bottom) points.push(bottom);
          if (left) points.push(left);
          if (points.length < 2) {
            continue;
          }
          context.beginPath();
          context.moveTo(points[0][0], points[0][1]);
          context.lineTo(points[1][0], points[1][1]);
          context.stroke();
          if (points.length === 4) {
            context.beginPath();
            context.moveTo(points[2][0], points[2][1]);
            context.lineTo(points[3][0], points[3][1]);
            context.stroke();
          }
        }
      }
      context.restore?.();
    }
  }

  function drawHaloFogAndContours(width, height, haloPoints) {
    if (!haloPoints.length) {
      return;
    }
    const field = buildHaloField(width, height, haloPoints);
    if (!(field.maxDensity > 0)) {
      return;
    }

    context.save?.();
    for (let y = 0; y < field.gridHeight; y += 1) {
      for (let x = 0; x < field.gridWidth; x += 1) {
        const density = field.density[y * field.gridWidth + x] / field.maxDensity;
        if (density < 0.03) {
          continue;
        }
        const fog = Math.pow(density, 0.62);
        const alpha = clamp(0.015 + 0.11 * fog, 0.0, 0.12);
        const red = Math.floor(36 + 28 * fog);
        const green = Math.floor(88 + 72 * fog);
        const blue = Math.floor(132 + 108 * fog);
        context.fillStyle = `rgba(${red}, ${green}, ${blue}, ${alpha})`;
        context.fillRect(
          x * field.cellWidth,
          y * field.cellHeight,
          field.cellWidth + 0.7,
          field.cellHeight + 0.7
        );
      }
    }
    context.restore?.();
    drawHaloContours(field);
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

    const stellarProjected = [];
    for (const particle of frame.particles) {
      const point = projectParticle(particle, width, height);
      if (!point) {
        continue;
      }
      if ((particle.component ?? 0) === 0) continue;
      const style = stellarRenderStyle(point);
      if (style) {
        stellarProjected.push(style);
      }
    }

    stellarProjected.sort((left, right) => right.depth - left.depth);
    context.globalCompositeOperation = "lighter";

    for (const point of stellarProjected) {
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
