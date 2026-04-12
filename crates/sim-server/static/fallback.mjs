import { createUiApp } from "./ui-app.mjs";
import { tryBootRustViewer } from "./viewer-loader.mjs";

const app = createUiApp({
  document,
  window,
  fetchImpl: fetch,
  WebSocketImpl: WebSocket,
  tryBootRustViewer,
});

app.boot().catch((error) => {
  app.nodes.viewerStatus.textContent = error.message;
});
