const VIEWER_BUNDLE_VERSION = "20260724-44";

// Client-side viewer failures are invisible from the server; report tier
// outcomes so `grep client sim-server.log` shows exactly what each browser
// did (and why a tier fell back).
function reportTier(payload) {
  try {
    void fetch("/api/client-log", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        v: VIEWER_BUNDLE_VERSION,
        ua: navigator.userAgent,
        gpu: typeof navigator !== "undefined" && !!navigator.gpu,
        hdrDisplay: window.matchMedia?.("(dynamic-range: high)")?.matches ?? null,
        ...payload,
      }),
    }).catch(() => {});
  } catch {
    // reporting must never break the viewer
  }
}

// A freshly created session may take a moment before its first frame is
// published; give the GPU-backed paths a generous window before demoting to
// the heavier JSON fallback (an explicit galaxy-viewer-error still fails fast).
const FIRST_FRAME_TIMEOUT_MS = 15000;

function waitForFirstFrame() {
  return new Promise((resolve) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      resolve(false);
    }, FIRST_FRAME_TIMEOUT_MS);
    const onFrame = () => {
      cleanup();
      resolve(true);
    };
    const onError = () => {
      cleanup();
      resolve(false);
    };
    const cleanup = () => {
      window.clearTimeout(timeout);
      window.removeEventListener("galaxy-viewer-frame", onFrame);
      window.removeEventListener("galaxy-viewer-error", onError);
    };
    window.addEventListener("galaxy-viewer-frame", onFrame);
    window.addEventListener("galaxy-viewer-error", onError);
  });
}

// Tries the renderer tiers in order — WebGL2 point sprites, then the
// Rust/WASM Canvas2D viewer — returning the tier name that produced a frame,
// or false to let the caller fall back to the JSON renderer. `?renderer=`
// (webgl | wasm | json) pins a tier.
export async function tryBootRustViewer(sessionId) {
  const params = new URLSearchParams(window.location.search);
  const renderer = params.get("renderer");
  if (renderer === "json") {
    return false;
  }

  // WebGPU tier: true HDR (extended dynamic range) output. Tried first when
  // pinned, or by default on HDR-capable displays with WebGPU available;
  // any failure falls through to the proven WebGL tier below.
  const wantWebgpu =
    renderer === "webgpu" ||
    (renderer === null &&
      typeof navigator !== "undefined" &&
      !!navigator.gpu &&
      window.matchMedia?.("(dynamic-range: high)")?.matches === true);
  if (wantWebgpu) {
    try {
      const module = await import(`/webgpu-viewer.mjs?v=${VIEWER_BUNDLE_VERSION}`);
      const firstFrame = waitForFirstFrame();
      module.boot("preview-canvas", sessionId);
      if (await firstFrame) {
        reportTier({ tier: "webgpu", ok: true });
        return "webgpu";
      }
      module.shutdown();
      reportTier({
        tier: "webgpu",
        ok: false,
        error: "no first frame (timeout or galaxy-viewer-error)",
      });
    } catch (error) {
      console.error("WebGPU viewer boot failed", error);
      reportTier({
        tier: "webgpu",
        ok: false,
        error: String((error && error.message) || error),
        stack: String((error && error.stack) || "").slice(0, 600),
      });
    }
  }

  if (renderer !== "wasm") {
    try {
      const module = await import(`/webgl-viewer.mjs?v=${VIEWER_BUNDLE_VERSION}`);
      const firstFrame = waitForFirstFrame();
      module.boot("preview-canvas", sessionId);
      if (await firstFrame) {
        reportTier({ tier: "webgl", ok: true });
        return "webgl";
      }
      module.shutdown();
      reportTier({
        tier: "webgl",
        ok: false,
        error: "no first frame (timeout or galaxy-viewer-error)",
      });
    } catch (error) {
      console.error("WebGL viewer boot failed", error);
      reportTier({
        tier: "webgl",
        ok: false,
        error: String((error && error.message) || error),
        stack: String((error && error.stack) || "").slice(0, 600),
      });
    }
    if (renderer === "webgl") {
      return false;
    }
  }

  try {
    const module = await import(
      `/viewer/sim_viewer.js?v=${VIEWER_BUNDLE_VERSION}`
    );
    if (typeof module.default === "function") {
      await module.default({
        module_or_path: `/viewer/sim_viewer_bg.wasm?v=${VIEWER_BUNDLE_VERSION}`,
      });
    }
    if (typeof module.boot === "function") {
      const firstFrame = waitForFirstFrame();
      module.boot("preview-canvas", sessionId);
      const ok = await firstFrame;
      reportTier({ tier: "wasm", ok });
      if (!ok && typeof module.shutdown === "function") {
        module.shutdown();
      }
      return ok ? "wasm" : false;
    }
  } catch (error) {
    console.error("Rust/WASM viewer boot failed", error);
    return false;
  }

  return false;
}
