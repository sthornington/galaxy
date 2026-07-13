const VIEWER_BUNDLE_VERSION = "20260713-1";

// A freshly created session may take a moment before its first frame is
// published; give the WASM path a generous window before demoting to the
// heavier JSON fallback (an explicit galaxy-viewer-error still fails fast).
const FIRST_FRAME_TIMEOUT_MS = 4000;

export async function tryBootRustViewer(sessionId) {
  const params = new URLSearchParams(window.location.search);
  if (params.get("renderer") === "json") {
    return false;
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
      const firstFrame = new Promise((resolve) => {
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
      module.boot("preview-canvas", sessionId);
      const ok = await firstFrame;
      if (!ok && typeof module.shutdown === "function") {
        module.shutdown();
      }
      return ok;
    }
  } catch (error) {
    console.error("Rust/WASM viewer boot failed", error);
    return false;
  }

  return false;
}
