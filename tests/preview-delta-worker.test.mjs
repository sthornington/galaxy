import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const HEADER_BYTES = 80;
const QUANT_BLOCK_BYTES = 56;
const PREFIX_BYTES = HEADER_BYTES + QUANT_BLOCK_BYTES;
const PARTICLE_BYTES = 16;
const META_BYTES = 72;

function fullFrame(count, step) {
  const buffer = new ArrayBuffer(PREFIX_BYTES + count * PARTICLE_BYTES);
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  bytes.set(new TextEncoder().encode("GPKT"), 0);
  view.setUint32(4, 2, true);
  view.setUint32(16, count, true);
  view.setFloat64(24, step, true);
  for (let particle = 0; particle < count; particle += 1) {
    const offset = PREFIX_BYTES + particle * PARTICLE_BYTES;
    const base = 20_000 + particle * 16;
    const values = [
      base + step,
      base + 1 + step * 2,
      base + 2 - step,
      base + 3 + step * 4,
      base + 4 - step * 3,
      base + 5 + step * 2,
    ];
    values.forEach((value, field) => view.setUint16(offset + field * 2, value, true));
    view.setUint16(offset + 12, 500 + particle + step * 3, true);
    view.setUint8(offset + 14, particle % 4);
  }
  return buffer;
}

function deltaFrame(previous, current, positionBits = 4, velocityBits = 5, massBits = 4) {
  const previousView = new DataView(previous);
  const currentView = new DataView(current);
  const count = currentView.getUint32(16, true);
  const payloadBits = count * (3 * (positionBits + velocityBits) + massBits);
  const payloadBytes = Math.ceil(payloadBits / 8);
  const buffer = new ArrayBuffer(32 + PREFIX_BYTES + payloadBytes);
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  bytes.set(new TextEncoder().encode("GPDL"), 0);
  view.setUint32(4, 2, true);
  view.setUint32(8, count, true);
  view.setUint8(12, positionBits);
  view.setUint8(13, velocityBits);
  view.setUint8(14, massBits);
  view.setFloat64(16, previousView.getFloat64(24, true), true);
  view.setUint32(24, current.byteLength, true);
  view.setUint32(28, payloadBytes, true);
  bytes.set(new Uint8Array(current, 0, PREFIX_BYTES), 32);

  let bitOffset = 0;
  for (let particle = 0; particle < count; particle += 1) {
    const record = PREFIX_BYTES + particle * PARTICLE_BYTES;
    for (let field = 0; field < 7; field += 1) {
      const bits = field < 3 ? positionBits : field < 6 ? velocityBits : massBits;
      const previousValue = previousView.getUint16(record + field * 2, true);
      const currentValue = currentView.getUint16(record + field * 2, true);
      const encoded = (currentValue - previousValue) & ((1 << bits) - 1);
      for (let bit = 0; bit < bits; bit += 1) {
        if (encoded & (1 << bit)) {
          const targetBit = bitOffset + bit;
          bytes[32 + PREFIX_BYTES + Math.floor(targetBit / 8)] |= 1 << (targetBit % 8);
        }
      }
      bitOffset += bits;
    }
  }
  return buffer;
}

test("delta worker preserves its decode chain while renderer buffers are exhausted", async () => {
  const source = await readFile(
    new URL("../crates/sim-server/static/webgl-stream-worker.js", import.meta.url),
    "utf8"
  );
  const posted = [];
  const sockets = [];
  const acknowledgements = [];
  class FakeWebSocket {
    constructor(url) {
      this.url = url;
      sockets.push(this);
    }
    send(message) {
      acknowledgements.push(message);
    }
    close() {}
  }
  const context = vm.createContext({
    ArrayBuffer,
    DataView,
    Uint8Array,
    Uint16Array,
    WebSocket: FakeWebSocket,
    postMessage(message) {
      posted.push(message);
    },
  });
  vm.runInContext(source, context);
  context.onmessage({ data: { kind: "connect", url: "ws://example/frames?delta=1" } });
  assert.equal(sockets.length, 1);

  const frames = Array.from({ length: 8 }, (_, step) => fullFrame(8, step));
  sockets[0].onmessage({ data: frames[0] });
  for (let step = 1; step <= 6; step += 1) {
    sockets[0].onmessage({ data: deltaFrame(frames[step - 1], frames[step]) });
  }

  const rendered = posted.filter((message) => message instanceof ArrayBuffer);
  const errors = posted.filter((message) => message?.kind === "error");
  assert.equal(errors.length, 0);
  assert.equal(rendered.length, 4, "the four-buffer renderer pool should be exhausted");

  // Return one renderer buffer, then send a delta based on a frame that was
  // decoded but never forwarded. A stale decode base would fail this packet.
  context.onmessage({ data: rendered[0] });
  sockets[0].onmessage({ data: deltaFrame(frames[6], frames[7]) });
  assert.equal(posted.filter((message) => message instanceof ArrayBuffer).length, 5);
  assert.equal(posted.filter((message) => message?.kind === "error").length, 0);
  assert.deepEqual(acknowledgements, Array(8).fill("ready"));

  const latest = posted.filter((message) => message instanceof ArrayBuffer).at(-1);
  const latestView = new DataView(latest);
  assert.equal(latestView.getFloat64(0, true), 7);
  assert.equal(latestView.getUint16(META_BYTES, true), 20_007);

  // Every rejected packet must still answer flow control ("resync"), or the
  // server waits for an ack forever and the stream freezes: a packet too
  // short for a magic word, garbage with an unknown magic, and a delta whose
  // base no longer matches after the forced resync.
  const ackCountBefore = acknowledgements.length;
  sockets[0].onmessage({ data: new ArrayBuffer(4) });
  assert.equal(acknowledgements.at(-1), "resync", "short packet must request resync");
  sockets[0].onmessage({ data: fullFrame(8, 9).slice(4) });
  assert.equal(acknowledgements.at(-1), "resync", "bad magic must request resync");
  sockets[0].onmessage({ data: deltaFrame(frames[0], frames[1]) });
  assert.equal(acknowledgements.at(-1), "resync", "stale delta must request resync");
  assert.equal(acknowledgements.length, ackCountBefore + 3);

  // A fresh keyframe recovers the stream.
  sockets[0].onmessage({ data: fullFrame(8, 10) });
  assert.equal(acknowledgements.at(-1), "ready");
});
