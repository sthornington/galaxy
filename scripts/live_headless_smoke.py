#!/usr/bin/env python3
import argparse
import asyncio
import json
from copy import deepcopy
from pathlib import Path

import aiohttp
import websockets


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exercise the live sim-server stack without opening a browser."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080",
        help="HTTP base URL for sim-server",
    )
    parser.add_argument(
        "--preview-budget",
        type=int,
        default=64,
        help="Preview particle budget for the smoke session",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Random seed for the smoke session",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="/tmp/galaxy-live-headless",
        help="Directory for smoke-test snapshots",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    ws_base = base_url.replace("http://", "ws://").replace("https://", "wss://")

    async with aiohttp.ClientSession() as session:
        await assert_static_assets(session, base_url)
        preset = await fetch_minor_merger(session, base_url)
        config = shrink_config(
            preset["config"],
            preview_budget=args.preview_budget,
            snapshot_dir=args.snapshot_dir,
        )
        session_id = await create_session(
            session,
            base_url,
            config=config,
            seed=args.seed,
            preview_budget=args.preview_budget,
        )
        await step_session(session, base_url, session_id)
        await assert_json_frame(ws_base, session_id)
        await assert_binary_frame(ws_base, session_id)
        await snapshot_session(session, base_url, session_id)

    print("live headless smoke: ok")


async def assert_static_assets(session: aiohttp.ClientSession, base_url: str) -> None:
    for path in (
        "/",
        "/fallback.mjs",
        "/viewer/sim_viewer.js",
        "/viewer/sim_viewer_bg.wasm",
    ):
        async with session.get(base_url + path) as response:
            body = await response.read()
            print(f"{path}: {response.status} ({len(body)} bytes)")
            if response.status != 200 or not body:
                raise RuntimeError(f"asset check failed for {path}: {response.status}")


async def fetch_minor_merger(
    session: aiohttp.ClientSession, base_url: str
) -> dict:
    async with session.get(base_url + "/api/presets") as response:
        presets = await response.json()
        if response.status != 200:
            raise RuntimeError(f"/api/presets failed with {response.status}")
    return next(preset for preset in presets if preset["id"] == "minor-merger")


def shrink_config(config: dict, preview_budget: int, snapshot_dir: str) -> dict:
    config = deepcopy(config)
    config["name"] = "live-headless-smoke"
    config["preview"]["particle_budget"] = preview_budget
    config["snapshots"]["directory"] = snapshot_dir
    for galaxy in config["galaxies"]:
        galaxy["halo_particle_count"] = 96
        galaxy["disk_particle_count"] = 48
        galaxy["bulge_particle_count"] = 12
    return config


async def create_session(
    session: aiohttp.ClientSession,
    base_url: str,
    config: dict,
    seed: int,
    preview_budget: int,
) -> str:
    async with session.post(
        base_url + "/api/session",
        json={
            "config": config,
            "seed": seed,
            "preview_particle_budget": preview_budget,
        },
    ) as response:
        payload = await response.json()
        if response.status != 200:
            raise RuntimeError(f"session creation failed: {payload}")
        print(
            f"session: {payload['id']} ({payload['particle_count']} particles, state={payload['state']})"
        )
        return payload["id"]


async def step_session(
    session: aiohttp.ClientSession, base_url: str, session_id: str
) -> None:
    async with session.post(
        base_url + f"/api/session/{session_id}/step",
        json={"substeps": 2},
    ) as response:
        payload = await response.json()
        if response.status != 200:
            raise RuntimeError(f"step failed: {payload}")
        print(
            f"step: sim_time={payload['sim_time_myr']} preview={payload['diagnostics']['preview_count']}"
        )


async def assert_json_frame(ws_base: str, session_id: str) -> None:
    async with websockets.connect(
        ws_base + f"/ws/frames/{session_id}?format=json"
    ) as websocket:
        payload = json.loads(await asyncio.wait_for(websocket.recv(), timeout=5))
        if payload["diagnostics"]["preview_count"] <= 0:
            raise RuntimeError("json frame had no preview particles")
        print(
            f"json frame: time={payload['sim_time_myr']} particles={len(payload['particles'])}"
        )


async def assert_binary_frame(ws_base: str, session_id: str) -> None:
    async with websockets.connect(ws_base + f"/ws/frames/{session_id}") as websocket:
        payload = await asyncio.wait_for(websocket.recv(), timeout=5)
        if not isinstance(payload, (bytes, bytearray)) or not payload:
            raise RuntimeError("binary frame payload was empty")
        print(f"binary frame: {len(payload)} bytes")


async def snapshot_session(
    session: aiohttp.ClientSession, base_url: str, session_id: str
) -> None:
    async with session.post(
        base_url + f"/api/session/{session_id}/snapshot",
        json={},
    ) as response:
        payload = await response.json()
        if response.status != 200:
            raise RuntimeError(f"snapshot failed: {payload}")
        manifest_path = Path(payload["latest_snapshot"])
        particles_path = manifest_path.parent / "particles.bin"
        if not manifest_path.exists() or not particles_path.exists():
            raise RuntimeError("snapshot files were not written")
        print(f"snapshot: {manifest_path}")


if __name__ == "__main__":
    asyncio.run(main())
