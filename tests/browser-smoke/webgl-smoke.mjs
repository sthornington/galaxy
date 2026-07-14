// Real-browser reproduction: attach the actual UI to a live session, count
// lit pixels every frame via a hooked drawArrays, and compare the paused
// (single-buffer) phase against the streaming (two-buffer interpolation)
// phase.
import { chromium } from "playwright";

const BASE = "http://127.0.0.1:8080";

async function api(path, body) {
  const res = await fetch(BASE + path, {
    method: body !== undefined ? "POST" : "GET",
    headers: { "content-type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  return res.json();
}

const browser = await chromium.launch({
  args: [
    "--no-sandbox",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
  ],
});
const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
page.on("console", (m) => {
  const text = m.text();
  if (!text.includes("[vite]")) console.log("[page]", text.slice(0, 200));
});
page.on("pageerror", (e) => console.log("[pageerror]", String(e).slice(0, 300)));

// Hook drawArrays before any page script runs: after each LINES draw (the
// axes pass, last draw of a frame), read back a centered region of the
// default framebuffer and count pixels brighter than the background.
await page.addInitScript(() => {
  const proto = WebGL2RenderingContext.prototype;
  const original = proto.drawArrays;
  window.__frames = [];
  proto.drawArrays = function (mode, first, count) {
    original.call(this, mode, first, count);
    if (mode === this.LINES) {
      window.__lastPointCount = window.__pendingPointCount ?? 0;
      if (window.__frames.length % 5 === 0 || window.__frames.length < 400) {
        const gl = this;
        const w = 400, h = 300;
        const x = Math.max(0, (gl.drawingBufferWidth - w) >> 1);
        const y = Math.max(0, (gl.drawingBufferHeight - h) >> 1);
        const buf = (window.__px ||= new Uint8Array(w * h * 4));
        gl.readPixels(x, y, w, h, gl.RGBA, gl.UNSIGNED_BYTE, buf);
        let lit = 0;
        for (let i = 0; i < buf.length; i += 4) {
          if (buf[i] > 22 || buf[i + 1] > 30 || buf[i + 2] > 40) lit += 1;
        }
        window.__frames.push({ t: performance.now(), lit, pts: window.__pendingPointCount ?? -1 });
      } else {
        window.__frames.push({ t: performance.now(), lit: -1, pts: window.__pendingPointCount ?? -1 });
      }
    } else if (mode === this.POINTS) {
      window.__pendingPointCount = count;
    }
  };
});

await page.goto(BASE + "/?renderer=webgl");
await page.waitForFunction(() => window.__frames?.length > 30, null, { timeout: 30000 });
console.log("viewer rendering; sampling PAUSED phase...");
await page.waitForTimeout(3000);
const paused = await page.evaluate(() => {
  const s = window.__frames.filter((f) => f.lit >= 0).slice(-20);
  window.__frames.length = 0;
  return s;
});

const sessions = await api("/api/sessions");
const sid = sessions[0].id;
console.log("resuming", sid.slice(0, 8));
await api(`/api/session/${sid}/resume`, {});
await page.waitForTimeout(9000);
const streaming = await page.evaluate(() => window.__frames.filter((f) => f.lit >= 0));

const avg = (a) => Math.round(a.reduce((s, f) => s + f.lit, 0) / Math.max(1, a.length));
console.log(`PAUSED:    frames=${paused.length} avg lit px=${avg(paused)} pts=${paused.at(-1)?.pts}`);
console.log(`STREAMING: frames=${streaming.length} avg lit px=${avg(streaming)} pts=${streaming.at(-1)?.pts}`);
// timeline in 1s buckets to see the transition
const t0 = streaming[0]?.t ?? 0;
const buckets = new Map();
for (const f of streaming) {
  const b = Math.floor((f.t - t0) / 1000);
  if (!buckets.has(b)) buckets.set(b, []);
  buckets.get(b).push(f.lit);
}
for (const [b, vals] of buckets) {
  console.log(`  t+${b}s: lit=${Math.round(vals.reduce((s, v) => s + v, 0) / vals.length)}`);
}
await page.screenshot({ path: "streaming.png" });
await browser.close();
