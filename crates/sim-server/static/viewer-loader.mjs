export async function tryBootRustViewer(sessionId) {
  try {
    const module = await import("/viewer/sim_viewer.js");
    if (typeof module.default === "function") {
      await module.default();
    }
    if (typeof module.boot === "function") {
      module.boot("preview-canvas", sessionId);
      return true;
    }
  } catch (_) {
    return false;
  }

  return false;
}
