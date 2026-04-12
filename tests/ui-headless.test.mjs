import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createUiApp } from "../crates/sim-server/static/ui-app.mjs";

class FakeElement {
  constructor(tagName, id = null) {
    this.tagName = tagName.toUpperCase();
    this.id = id;
    this.textContent = "";
    this.className = "";
    this.dataset = {};
    this.children = [];
    this.listeners = new Map();
    this.width = 0;
    this.height = 0;
    this._innerHTML = "";
  }

  appendChild(child) {
    this.children.push(child);
    child.parentNode = this;
    return child;
  }

  addEventListener(type, handler) {
    this.listeners.set(type, handler);
  }

  click() {
    const handler = this.listeners.get("click");
    if (handler) {
      handler({ currentTarget: this });
    }
  }

  set innerHTML(value) {
    this._innerHTML = value;
    if (value === "") {
      this.children = [];
    }
  }

  get innerHTML() {
    return this._innerHTML;
  }
}

class FakeGradient {
  constructor() {
    this.stops = [];
  }

  addColorStop(offset, color) {
    this.stops.push({ offset, color });
  }
}

class FakeCanvasContext {
  constructor() {
    this.calls = [];
    this.fillStyle = null;
    this.strokeStyle = null;
    this.lineWidth = 1;
    this.font = "";
    this.textAlign = "start";
    this.textBaseline = "alphabetic";
  }

  clearRect(...args) {
    this.calls.push({ kind: "clearRect", args });
  }

  createLinearGradient(...args) {
    this.calls.push({ kind: "createLinearGradient", args });
    return new FakeGradient();
  }

  fillRect(...args) {
    this.calls.push({ kind: "fillRect", args, fillStyle: this.fillStyle });
  }

  beginPath() {
    this.calls.push({ kind: "beginPath" });
  }

  moveTo(...args) {
    this.calls.push({ kind: "moveTo", args });
  }

  lineTo(...args) {
    this.calls.push({ kind: "lineTo", args });
  }

  arc(...args) {
    this.calls.push({ kind: "arc", args, fillStyle: this.fillStyle });
  }

  fill() {
    this.calls.push({ kind: "fill", fillStyle: this.fillStyle });
  }

  stroke() {
    this.calls.push({
      kind: "stroke",
      strokeStyle: this.strokeStyle,
      lineWidth: this.lineWidth,
    });
  }

  fillText(...args) {
    this.calls.push({ kind: "fillText", args, fillStyle: this.fillStyle });
  }

  save() {
    this.calls.push({ kind: "save" });
  }

  restore() {
    this.calls.push({ kind: "restore" });
  }
}

class FakeCanvasElement extends FakeElement {
  constructor(id) {
    super("canvas", id);
    this.width = 1280;
    this.height = 720;
    this.context = new FakeCanvasContext();
  }

  getContext(kind) {
    assert.equal(kind, "2d");
    return this.context;
  }
}

class FakeDocument {
  constructor() {
    this.elements = new Map();
    this.register(new FakeElement("div", "preset-list"));
    this.register(new FakeElement("div", "session-id"));
    this.register(new FakeElement("div", "session-state"));
    this.register(new FakeElement("div", "sim-time"));
    this.register(new FakeElement("div", "particle-count"));
    this.register(new FakeElement("div", "preview-count"));
    this.register(new FakeElement("div", "viewer-status"));
    this.register(new FakeCanvasElement("preview-canvas"));
    this.register(new FakeElement("button", "refresh-presets"));
    this.register(new FakeElement("button", "pause-btn"));
    this.register(new FakeElement("button", "resume-btn"));
    this.register(new FakeElement("button", "stop-btn"));
    this.register(new FakeElement("button", "snapshot-btn"));
    this.register(new FakeElement("button", "step-btn"));
  }

  register(element) {
    if (element.id) {
      this.elements.set(element.id, element);
    }
    return element;
  }

  getElementById(id) {
    return this.elements.get(id);
  }

  createElement(tagName) {
    return new FakeElement(tagName);
  }
}

class FakeWebSocket {
  static instances = [];

  constructor(url) {
    this.url = url;
    this.closed = false;
    FakeWebSocket.instances.push(this);
  }

  close() {
    this.closed = true;
    if (this.onclose) {
      this.onclose();
    }
  }

  emitOpen() {
    if (this.onopen) {
      this.onopen();
    }
  }

  emitMessage(data) {
    if (this.onmessage) {
      this.onmessage({ data });
    }
  }
}

function createHarness({ useRustViewer = false, existingSessions = [] } = {}) {
  FakeWebSocket.instances = [];
  const document = new FakeDocument();
  const fetchCalls = [];
  const intervals = new Map();
  let nextIntervalId = 1;
  const window = {
    location: {
      protocol: "http:",
      host: "127.0.0.1:8080",
    },
    setInterval(handler) {
      const id = nextIntervalId++;
      intervals.set(id, handler);
      return id;
    },
    clearInterval(id) {
      intervals.delete(id);
    },
  };

  async function fetchImpl(path, options = {}) {
    fetchCalls.push({ path, options });
    if (path === "/api/presets") {
      return okJson([
        {
          id: "minor-merger",
          title: "Minor Merger",
          summary: "A compact smoke test preset.",
          config: { galaxies: [{}, {}] },
        },
      ]);
    }
    if (path === "/api/session") {
      return okJson({
        id: "session-1",
        state: "paused",
        sim_time_myr: 0,
        particle_count: 208,
        diagnostics: { preview_count: 0 },
      });
    }
    if (path === "/api/sessions") {
      return okJson(existingSessions);
    }
    if (path === "/api/session/session-1/step") {
      return okJson({
        id: "session-1",
        state: "paused",
        sim_time_myr: 0.05,
        particle_count: 208,
        diagnostics: { preview_count: 32 },
      });
    }
    if (path === "/api/session/session-1") {
      return okJson({
        id: "session-1",
        state: "running",
        sim_time_myr: 2.5,
        particle_count: 208,
        diagnostics: { preview_count: 64 },
      });
    }
    if (path === "/api/session/session-1/stop") {
      return okJson({
        id: "session-1",
        state: "paused",
        sim_time_myr: 2.5,
        particle_count: 208,
        diagnostics: { preview_count: 64 },
      });
    }
    throw new Error(`unexpected fetch path: ${path}`);
  }

  const app = createUiApp({
    document,
    window,
    fetchImpl,
    WebSocketImpl: FakeWebSocket,
    tryBootRustViewer: async () => useRustViewer,
    setIntervalImpl: window.setInterval.bind(window),
    clearIntervalImpl: window.clearInterval.bind(window),
  });

  return { app, document, fetchCalls, intervals };
}

function okJson(value) {
  return {
    ok: true,
    status: 200,
    async json() {
      return value;
    },
  };
}

test("headless UI renders presets and falls back to JSON websocket streaming", async () => {
  const { app, document, fetchCalls } = createHarness();
  await app.boot();

  const presetList = document.getElementById("preset-list");
  assert.equal(presetList.children.length, 1);
  const launchButton = presetList.children[0].children[3];
  launchButton.click();
  await new Promise((resolve) => setImmediate(resolve));

  assert.deepEqual(
    fetchCalls.map((call) => call.path),
    ["/api/presets", "/api/sessions", "/api/session"]
  );
  assert.deepEqual(JSON.parse(fetchCalls[2].options.body), {
    preset_id: "minor-merger",
    seed: 42,
    preview_particle_budget: 32768,
  });
  assert.equal(document.getElementById("session-id").textContent, "session-1");
  assert.equal(FakeWebSocket.instances.length, 1);
  assert.equal(
    FakeWebSocket.instances[0].url,
    "ws://127.0.0.1:8080/ws/frames/session-1?format=json"
  );

  FakeWebSocket.instances[0].emitOpen();
  assert.equal(
    document.getElementById("viewer-status").textContent,
    "Streaming JSON preview frames."
  );

  FakeWebSocket.instances[0].emitMessage(
    JSON.stringify({
      sim_time_myr: 1.25,
      diagnostics: { preview_count: 1 },
      particles: [
        {
          position_kpc: [10, -5, 0],
          velocity_kms: [0, 0, 0],
          color_rgba: [1, 0.5, 0.25, 1],
          intensity: 0.7,
        },
      ],
    })
  );
  assert.equal(document.getElementById("sim-time").textContent, "1.25 Myr");
  assert.equal(document.getElementById("preview-count").textContent, "1");
  const canvasContext = document.getElementById("preview-canvas").context;
  assert.ok(canvasContext.calls.some((call) => call.kind === "arc"));
});

test("headless UI prefers the Rust viewer path when it is available", async () => {
  const { app, document, intervals, fetchCalls } = createHarness({
    useRustViewer: true,
  });
  await app.boot();
  const launchButton = document.getElementById("preset-list").children[0].children[3];
  launchButton.click();
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(FakeWebSocket.instances.length, 0);
  assert.deepEqual(JSON.parse(fetchCalls[2].options.body), {
    preset_id: "minor-merger",
    seed: 42,
    preview_particle_budget: 32768,
  });
  assert.equal(intervals.size, 1);
  assert.equal(
    document.getElementById("viewer-status").textContent,
    "Streaming binary preview frames into the Rust/WASM viewer."
  );

  const poll = [...intervals.values()][0];
  poll();
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(document.getElementById("sim-time").textContent, "2.50 Myr");
  assert.equal(document.getElementById("preview-count").textContent, "64");
});

test("headless UI reattaches to an existing session and can stop it", async () => {
  const existingSession = {
    id: "session-1",
    state: "running",
    sim_time_myr: 1.75,
    particle_count: 208,
    diagnostics: { preview_count: 32 },
  };
  const { app, document, fetchCalls } = createHarness({
    existingSessions: [existingSession],
  });

  await app.boot();

  assert.equal(document.getElementById("session-id").textContent, "session-1");
  assert.equal(FakeWebSocket.instances.length, 1);
  assert.deepEqual(
    fetchCalls.map((call) => call.path),
    ["/api/presets", "/api/sessions"]
  );

  document.getElementById("stop-btn").click();
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(
    fetchCalls.map((call) => call.path).at(-1),
    "/api/session/session-1/stop"
  );
  assert.equal(document.getElementById("session-state").textContent, "stopped");
  assert.equal(
    document.getElementById("viewer-status").textContent,
    "Stopped session session-1."
  );
});

test("index shell references the module entrypoint expected by the headless tests", async () => {
  const html = await readFile(
    new URL("../crates/sim-server/static/index.html", import.meta.url),
    "utf8"
  );
  assert.match(html, /<script type="module" src="\/fallback\.mjs"><\/script>/);
  assert.match(html, /id="preview-canvas"/);
  assert.match(html, /Left-drag orbits around the current focus/);
  assert.match(html, /The axis marker shows the simulation origin/);
});
