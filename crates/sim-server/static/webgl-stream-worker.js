// Frame-stream worker: owns the preview WebSocket and reconstructs compact
// delta packets away from the renderer. Validated full frames are repacked
// into a small pool of transferable buffers that ping-pong with the main
// thread; steady state allocates nothing beyond the browser's socket receive.
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
const PACKET_PREFIX_BYTES = HEADER_BYTES + QUANT_BLOCK_BYTES;
const DELTA_MAGIC = 0x4c_44_50_47; // "GPDL"
const DELTA_VERSION = 1;
const DELTA_HEADER_BYTES = 32;
const POOL_SIZE = 4;

let socket = null;
const pool = [];
let poolBufferBytes = 0;
let lastSimTime = -Infinity;
let lastDecodedSimTime = -Infinity;
let dropped = 0;
let referenceFrames = [null, null];
let referenceIndex = 0;
let referenceBytes = 0;
let referenceReady = false;

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

function reportPacketError(detail) {
  postMessage({ kind: "error", detail });
}

function ensureReferenceFrames(frameBytes) {
  if (frameBytes !== referenceBytes) {
    referenceFrames = [new ArrayBuffer(frameBytes), new ArrayBuffer(frameBytes)];
    referenceIndex = 0;
    referenceBytes = frameBytes;
    referenceReady = false;
  }
}

function validateFullFrame(data) {
  if (!(data instanceof ArrayBuffer) || data.byteLength < PACKET_PREFIX_BYTES) {
    return null;
  }
  const view = new DataView(data);
  if (
    view.getUint32(0, true) !== PACKET_MAGIC ||
    view.getUint32(4, true) !== PACKET_VERSION
  ) {
    return null;
  }
  const previewCount = view.getUint32(16, true);
  const simTime = view.getFloat64(24, true);
  const payloadBytes = QUANT_BLOCK_BYTES + previewCount * PARTICLE_STRIDE;
  if (data.byteLength !== HEADER_BYTES + payloadBytes) {
    return null;
  }
  return { previewCount, simTime, payloadBytes };
}

function forwardFrame(data, frame) {
  const { previewCount, simTime, payloadBytes } = frame;
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

function acceptKeyframe(data) {
  const frame = validateFullFrame(data);
  if (frame === null) {
    reportPacketError("bad keyframe");
    return false;
  }
  if (frame.simTime <= lastDecodedSimTime + 1e-9) {
    return true;
  }

  ensureReferenceFrames(data.byteLength);
  const nextIndex = referenceReady ? referenceIndex ^ 1 : referenceIndex;
  new Uint8Array(referenceFrames[nextIndex]).set(new Uint8Array(data));
  referenceIndex = nextIndex;
  referenceReady = true;
  lastDecodedSimTime = frame.simTime;
  forwardFrame(referenceFrames[referenceIndex], frame);
  return true;
}

function acceptDelta(data) {
  if (data.byteLength < DELTA_HEADER_BYTES + PACKET_PREFIX_BYTES) {
    reportPacketError("short delta packet");
    return false;
  }
  const header = new DataView(data);
  const previewCount = header.getUint32(8, true);
  const positionBits = header.getUint8(12);
  const velocityBits = header.getUint8(13);
  const baseSimTime = header.getFloat64(16, true);
  const fullFrameBytes = header.getUint32(24, true);
  const payloadBytes = header.getUint32(28, true);
  const expectedBits = previewCount * 3 * (positionBits + velocityBits);
  const expectedPayloadBytes = Math.ceil(expectedBits / 8);
  if (
    header.getUint32(4, true) !== DELTA_VERSION ||
    positionBits < 1 ||
    positionBits > 17 ||
    velocityBits < 1 ||
    velocityBits > 17 ||
    fullFrameBytes !== PACKET_PREFIX_BYTES + previewCount * PARTICLE_STRIDE ||
    payloadBytes !== expectedPayloadBytes ||
    data.byteLength !== DELTA_HEADER_BYTES + PACKET_PREFIX_BYTES + payloadBytes
  ) {
    reportPacketError("invalid delta header");
    return false;
  }
  if (!referenceReady || referenceBytes !== fullFrameBytes) {
    reportPacketError("delta without matching keyframe");
    return false;
  }

  const previous = referenceFrames[referenceIndex];
  const previousView = new DataView(previous);
  if (previousView.getFloat64(24, true) !== baseSimTime) {
    reportPacketError("delta base mismatch");
    return false;
  }

  const prefixOffset = DELTA_HEADER_BYTES;
  const prefix = new DataView(data, prefixOffset, PACKET_PREFIX_BYTES);
  if (
    prefix.getUint32(0, true) !== PACKET_MAGIC ||
    prefix.getUint32(4, true) !== PACKET_VERSION ||
    prefix.getUint32(16, true) !== previewCount
  ) {
    reportPacketError("invalid delta keyframe prefix");
    return false;
  }
  const simTime = prefix.getFloat64(24, true);
  if (simTime <= lastDecodedSimTime + 1e-9) {
    return true;
  }

  const nextIndex = referenceIndex ^ 1;
  const next = referenceFrames[nextIndex];
  new Uint8Array(next, 0, PACKET_PREFIX_BYTES).set(
    new Uint8Array(data, prefixOffset, PACKET_PREFIX_BYTES)
  );

  // Particle records are eight little-endian u16 words. The first six are
  // signed deltas; mass, component, and the reserved byte remain unchanged.
  const previousWords = new Uint16Array(
    previous,
    PACKET_PREFIX_BYTES,
    previewCount * (PARTICLE_STRIDE / 2)
  );
  const nextWords = new Uint16Array(
    next,
    PACKET_PREFIX_BYTES,
    previewCount * (PARTICLE_STRIDE / 2)
  );
  const packed = new Uint8Array(
    data,
    DELTA_HEADER_BYTES + PACKET_PREFIX_BYTES,
    payloadBytes
  );
  const positionMask = (1 << positionBits) - 1;
  const positionSign = 1 << (positionBits - 1);
  const positionRange = 1 << positionBits;
  const velocityMask = (1 << velocityBits) - 1;
  const velocitySign = 1 << (velocityBits - 1);
  const velocityRange = 1 << velocityBits;
  let packedIndex = 0;
  let bitBuffer = 0;
  let bufferedBits = 0;

  for (let particle = 0; particle < previewCount; particle += 1) {
    const record = particle * (PARTICLE_STRIDE / 2);
    for (let field = 0; field < 6; field += 1) {
      const bits = field < 3 ? positionBits : velocityBits;
      while (bufferedBits < bits) {
        bitBuffer |= packed[packedIndex] << bufferedBits;
        packedIndex += 1;
        bufferedBits += 8;
      }
      const mask = field < 3 ? positionMask : velocityMask;
      const sign = field < 3 ? positionSign : velocitySign;
      const range = field < 3 ? positionRange : velocityRange;
      const encoded = bitBuffer & mask;
      const delta = encoded & sign ? encoded - range : encoded;
      const value = previousWords[record + field] + delta;
      if (value < 0 || value > 0xffff) {
        reportPacketError("delta value overflow");
        return false;
      }
      nextWords[record + field] = value;
      bitBuffer >>>= bits;
      bufferedBits -= bits;
    }
    nextWords[record + 6] = previousWords[record + 6];
    nextWords[record + 7] = previousWords[record + 7];
  }

  referenceIndex = nextIndex;
  lastDecodedSimTime = simTime;
  forwardFrame(next, {
    previewCount,
    simTime,
    payloadBytes: QUANT_BLOCK_BYTES + previewCount * PARTICLE_STRIDE,
  });
  return true;
}

function handlePacket(data) {
  if (!(data instanceof ArrayBuffer) || data.byteLength < 8) {
    reportPacketError("short packet");
    return;
  }
  const magic = new DataView(data).getUint32(0, true);
  let accepted = false;
  if (magic === PACKET_MAGIC) {
    accepted = acceptKeyframe(data);
  } else if (magic === DELTA_MAGIC) {
    accepted = acceptDelta(data);
  } else {
    reportPacketError("bad magic");
  }
  if (!socket) {
    return;
  }
  if (accepted) {
    socket.send("ready");
  } else {
    // A rejected packet must still answer the server's flow control or the
    // stream freezes forever waiting for an ack. "resync" makes the server
    // drop its delta reference and send a fresh keyframe.
    referenceReady = false;
    socket.send("resync");
  }
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
    lastDecodedSimTime = -Infinity;
    dropped = 0;
    referenceReady = false;
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
