// Frame-stream worker: owns the preview WebSocket so the per-message
// ArrayBuffer garbage (one ~4 MB buffer per physics frame) is allocated and
// collected on this thread, never pausing the renderer. Validated frames are
// repacked into a small pool of transferable buffers that ping-pong with the
// main thread — steady state allocates nothing on either side beyond the
// browser's own socket receive.
//
// Pooled buffer layout (little-endian):
//   [0..8)    f64 sim_time_myr
//   [8..12)   u32 preview_count
//   [12..16)  u32 reserved
//   [16..72)  quant block (56 bytes, verbatim from the wire)
//   [72..)    particle records (16 bytes each, verbatim from the wire)
const META_BYTES = 72;
const HEADER_BYTES = 80;
const QUANT_BLOCK_BYTES = 56;
const PARTICLE_STRIDE = 16;
const PACKET_MAGIC = 0x54_4b_50_47; // "GPKT"
const PACKET_VERSION = 2;
const POOL_SIZE = 4;

let socket = null;
const pool = [];
let poolBufferBytes = 0;
let lastSimTime = -Infinity;
let dropped = 0;

function ensurePool(payloadBytes) {
  // 16 bytes of frame meta, then the quant block + particle records verbatim.
  const needed = 16 + payloadBytes;
  if (needed > poolBufferBytes) {
    pool.length = 0;
    poolBufferBytes = needed;
    for (let i = 0; i < POOL_SIZE; i += 1) {
      pool.push(new ArrayBuffer(poolBufferBytes));
    }
  }
}

function handlePacket(data) {
  if (!(data instanceof ArrayBuffer) || data.byteLength < HEADER_BYTES) {
    postMessage({ kind: "error", detail: "short packet" });
    return;
  }
  const view = new DataView(data);
  if (
    view.getUint32(0, true) !== PACKET_MAGIC ||
    view.getUint32(4, true) !== PACKET_VERSION
  ) {
    postMessage({ kind: "error", detail: "bad magic/version" });
    return;
  }
  const previewCount = view.getUint32(16, true);
  const simTime = view.getFloat64(24, true);
  const payloadBytes = QUANT_BLOCK_BYTES + previewCount * PARTICLE_STRIDE;
  if (data.byteLength !== HEADER_BYTES + payloadBytes) {
    postMessage({ kind: "error", detail: "size mismatch" });
    return;
  }
  if (simTime <= lastSimTime + 1e-9) {
    return;
  }

  ensurePool(payloadBytes);
  const buffer = pool.pop();
  if (buffer === undefined) {
    // Main thread hasn't returned a buffer yet; drop this frame — the
    // renderer's jitter buffer interpolates straight across the gap.
    dropped += 1;
    return;
  }
  lastSimTime = simTime;

  const out = new DataView(buffer);
  out.setFloat64(0, simTime, true);
  out.setUint32(8, previewCount, true);
  out.setUint32(12, dropped, true);
  // Quant block lands at offset 16, particle records at META_BYTES (72).
  new Uint8Array(buffer, 16, payloadBytes).set(
    new Uint8Array(data, HEADER_BYTES, payloadBytes)
  );
  postMessage(buffer, [buffer]);
}

onmessage = (event) => {
  const message = event.data;
  if (message instanceof ArrayBuffer) {
    // A pooled buffer coming home; only readmit current-generation sizes.
    if (message.byteLength === poolBufferBytes) {
      pool.push(message);
    }
    return;
  }
  if (message && message.kind === "connect") {
    lastSimTime = -Infinity;
    socket = new WebSocket(message.url);
    socket.binaryType = "arraybuffer";
    socket.onmessage = (frame) => handlePacket(frame.data);
    socket.onclose = () => postMessage({ kind: "closed" });
    socket.onerror = () => postMessage({ kind: "error", detail: "socket error" });
    return;
  }
  if (message && message.kind === "disconnect" && socket) {
    socket.onmessage = null;
    socket.onclose = null;
    socket.onerror = null;
    try {
      socket.close();
    } catch {
      // already closed
    }
    socket = null;
  }
};
