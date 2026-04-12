const presetList = document.getElementById("preset-list");
const sessionIdNode = document.getElementById("session-id");
const sessionStateNode = document.getElementById("session-state");
const simTimeNode = document.getElementById("sim-time");
const particleCountNode = document.getElementById("particle-count");
const previewCountNode = document.getElementById("preview-count");
const viewerStatusNode = document.getElementById("viewer-status");
const canvas = document.getElementById("preview-canvas");
const context = canvas.getContext("2d");

let activeSessionId = null;
let frameSocket = null;

async function fetchJson(path, options = {}) {
  const response = await fetch(path, {
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
  sessionIdNode.textContent = session.id;
  sessionStateNode.textContent = session.state;
  simTimeNode.textContent = `${session.sim_time_myr.toFixed(2)} Myr`;
  particleCountNode.textContent = session.particle_count.toLocaleString();
  previewCountNode.textContent = session.diagnostics.preview_count.toLocaleString();
}

function openFrameSocket(sessionId) {
  if (frameSocket) {
    frameSocket.close();
  }

  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  frameSocket = new WebSocket(`${scheme}://${window.location.host}/ws/frames/${sessionId}?format=json`);
  frameSocket.onopen = () => {
    viewerStatusNode.textContent = "Streaming JSON preview frames.";
  };
  frameSocket.onclose = () => {
    viewerStatusNode.textContent = "Preview stream closed.";
  };
  frameSocket.onmessage = (event) => {
    const frame = JSON.parse(event.data);
    drawFrame(frame);
    previewCountNode.textContent = frame.diagnostics.preview_count.toLocaleString();
    simTimeNode.textContent = `${frame.sim_time_myr.toFixed(2)} Myr`;
  };
}

function drawFrame(frame) {
  const width = canvas.width;
  const height = canvas.height;
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
    context.fillStyle = `rgba(${Math.floor(particle.color_rgba[0] * 255)}, ${Math.floor(
      particle.color_rgba[1] * 255
    )}, ${Math.floor(particle.color_rgba[2] * 255)}, ${Math.max(0.18, particle.intensity)})`;
    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    context.fill();
  }
}

async function createSession(presetId) {
  viewerStatusNode.textContent = "Creating session on the GPU backend...";
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
    viewerStatusNode.textContent = "Streaming binary preview frames into the Rust/WASM viewer.";
    return;
  }
  openFrameSocket(session.id);
}

async function control(action, payload = {}) {
  if (!activeSessionId) {
    return;
  }
  const session = await fetchJson(`/api/session/${activeSessionId}/${action}`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  updateSessionStats(session);
}

async function loadPresets() {
  presetList.innerHTML = "";
  const presets = await fetchJson("/api/presets");
  for (const preset of presets) {
    const card = document.createElement("article");
    card.className = "preset-card";
    card.innerHTML = `
      <h3>${preset.title}</h3>
      <p>${preset.summary}</p>
      <div class="preset-meta">
        <span>${preset.id}</span>
        <span>${preset.config.galaxies.length} galaxies</span>
      </div>
      <button data-preset="${preset.id}">Launch</button>
    `;
    card.querySelector("button").addEventListener("click", () => {
      createSession(preset.id).catch((error) => {
        viewerStatusNode.textContent = error.message;
      });
    });
    presetList.appendChild(card);
  }
}

document.getElementById("refresh-presets").addEventListener("click", () => {
  loadPresets().catch((error) => {
    viewerStatusNode.textContent = error.message;
  });
});
document.getElementById("pause-btn").addEventListener("click", () => control("pause"));
document.getElementById("resume-btn").addEventListener("click", () => control("resume"));
document.getElementById("snapshot-btn").addEventListener("click", () => control("snapshot"));
document.getElementById("step-btn").addEventListener("click", () =>
  control("step", { substeps: 1 })
);

loadPresets().catch((error) => {
  viewerStatusNode.textContent = error.message;
});
import { tryBootRustViewer } from "./viewer-loader.js";
