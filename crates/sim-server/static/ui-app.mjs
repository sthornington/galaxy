export function createUiApp({
  document,
  window,
  fetchImpl = fetch,
  WebSocketImpl = WebSocket,
  tryBootRustViewer,
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
    snapshot: document.getElementById("snapshot-btn"),
    step: document.getElementById("step-btn"),
  };
  const context = nodes.canvas.getContext("2d");

  let activeSessionId = null;
  let frameSocket = null;

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

  function drawFrame(frame) {
    const width = nodes.canvas.width;
    const height = nodes.canvas.height;
    context.clearRect(0, 0, width, height);

    const gradient = context.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, "rgba(76, 201, 240, 0.06)");
    gradient.addColorStop(1, "rgba(242, 193, 75, 0.06)");
    context.fillStyle = gradient;
    context.fillRect(0, 0, width, height);

    const scale = 2.6;
    for (const particle of frame.particles) {
      const x = width * 0.5 + particle.position_kpc[0] * scale;
      const y = height * 0.5 + particle.position_kpc[1] * scale;
      const radius = Math.max(0.5, 1.6 * particle.intensity);
      context.fillStyle = `rgba(${Math.floor(
        particle.color_rgba[0] * 255
      )}, ${Math.floor(particle.color_rgba[1] * 255)}, ${Math.floor(
        particle.color_rgba[2] * 255
      )}, ${Math.max(0.18, particle.intensity)})`;
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.fill();
    }
  }

  function openFrameSocket(sessionId) {
    if (frameSocket) {
      frameSocket.close();
    }

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

  async function createSession(presetId) {
    nodes.viewerStatus.textContent = "Creating session on the GPU backend...";
    const session = await fetchJson("/api/session", {
      method: "POST",
      body: JSON.stringify({
        preset_id: presetId,
        seed: 42,
      }),
    });
    updateSessionStats(session);
    const usingRustViewer = await tryBootRustViewer(session.id);
    if (usingRustViewer) {
      nodes.viewerStatus.textContent =
        "Streaming binary preview frames into the Rust/WASM viewer.";
      return session;
    }
    openFrameSocket(session.id);
    return session;
  }

  async function control(action, payload = {}) {
    if (!activeSessionId) {
      return null;
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

  function bindControls() {
    nodes.refresh.addEventListener("click", () => {
      loadPresets().catch((error) => {
        nodes.viewerStatus.textContent = error.message;
      });
    });
    nodes.pause.addEventListener("click", () => control("pause"));
    nodes.resume.addEventListener("click", () => control("resume"));
    nodes.snapshot.addEventListener("click", () => control("snapshot"));
    nodes.step.addEventListener("click", () => control("step", { substeps: 1 }));
  }

  async function boot() {
    bindControls();
    try {
      await loadPresets();
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
    fetchJson,
    loadPresets,
    openFrameSocket,
    updateSessionStats,
    getActiveSessionId: () => activeSessionId,
    getFrameSocket: () => frameSocket,
    nodes,
  };
}
