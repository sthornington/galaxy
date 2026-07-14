# Real-browser WebGL smoke test

Runs the actual WebGL viewer in headless Chromium (SwiftShader) against the
live server, hooks `drawArrays`, and verifies the paused -> streaming
transition renders the full particle count. This catches whole-tier failures
(e.g. shader link errors on strict GLSL linkers) that node-mocked tests
cannot see — the 2026-07-14 `u_style` uniform-precision mismatch killed the
entire WebGL tier on ANGLE/Metal and was only caught this way.

Setup (once):
    cd tests/browser-smoke && npm install playwright && npx playwright install chromium
    # container lacks nspr/nss; fetch locally, no root needed:
    mkdir -p libs && cd libs && apt-get download libnspr4 libnss3 && for f in *.deb; do dpkg-deb -x "$f" extracted/; done

Run (server on :8080 with a session available; the script creates none):
    LD_LIBRARY_PATH=$PWD/libs/extracted/usr/lib/aarch64-linux-gnu \
    PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS=1 node webgl-smoke.mjs

Expect: "PAUSED"/"STREAMING" lit-pixel counts of similar magnitude and
pts equal to the session's full preview count while streaming. A tier
fallback (WASM/JSON) or a shader link failure shows up as a page error and
a small pts value.
