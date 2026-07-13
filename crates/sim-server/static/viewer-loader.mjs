const VIEWER_BUNDLE_VERSION = "20260713-9";

// A freshly created session may take a moment before its first frame is
// published; give the GPU-backed paths a generous window before demoting to
// the heavier JSON fallback (an explicit galaxy-viewer-error still fails fast).
const FIRST_FRAME_TIMEOUT_MS = 4000;

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

  if (renderer !== "wasm") {
    try {
      const module = await import(`/webgl-viewer.mjs?v=${VIEWER_BUNDLE_VERSION}`);
      const firstFrame = waitForFirstFrame();
      module.boot("preview-canvas", sessionId);
      if (await firstFrame) {
        return "webgl";
      }
      module.shutdown();
    } catch (error) {
      console.error("WebGL viewer boot failed", error);
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
