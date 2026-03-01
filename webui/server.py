"""
WebUI server for Pfobos SDR — FastAPI + WebSocket spectrum streaming.

Usage:
    uv run python -m webui.server --stub          # C stub (no hardware)
    uv run python -m webui.server                  # real device
    uv run python -m webui.server --host 0.0.0.0   # accessible on LAN
"""

import argparse
import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
STUB_LIB     = os.path.join(PROJECT_ROOT, "tests", "stub", "libfobos.so")

sys.path.insert(0, PROJECT_ROOT)

from webui.sdr_worker import SDRWorker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App factory — accepts lib_path so we can pass --stub from CLI
# ---------------------------------------------------------------------------

def create_app(lib_path: str = None) -> FastAPI:
    # Shared asyncio queue — sdr_worker pushes, WebSocket handlers pop
    frame_queue = asyncio.Queue(maxsize=8)
    worker_ref = [None]  # mutable cell so lifespan closure can write it
    connected_clients = []

    async def _broadcaster():
        """Read frames from the shared queue and broadcast to all WS clients."""
        while True:
            frame = await frame_queue.get()
            dead = []
            for ws in list(connected_clients):
                try:
                    await ws.send_json(frame)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                if ws in connected_clients:
                    connected_clients.remove(ws)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_event_loop()
        worker = SDRWorker(loop=loop, queue=frame_queue, lib_path=lib_path)
        worker_ref[0] = worker
        worker.start()
        asyncio.create_task(_broadcaster())
        yield
        worker.stop()

    app = FastAPI(title="Pfobos WebUI", version="0.1.0", lifespan=lifespan)

    # -----------------------------------------------------------------------
    # Static files
    # -----------------------------------------------------------------------

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    # -----------------------------------------------------------------------
    # REST API
    # -----------------------------------------------------------------------

    @app.get("/api/config")
    async def get_config():
        worker = worker_ref[0]
        if worker is None:
            return {}
        return {
            "center_freq": worker.center_freq,
            "sample_rate": worker.sample_rate,
            "lna_gain":    worker.lna_gain,
            "vga_gain":    worker.vga_gain,
            "fft_size":    worker.fft_size,
            "buf_length":  worker.buf_length,
        }

    @app.post("/api/config")
    async def post_config(cfg: dict):
        worker = worker_ref[0]
        if worker:
            worker.apply_config(cfg)
        return {"status": "ok"}

    @app.get("/api/devices")
    async def get_devices():
        try:
            from shared.fwrapper import FobosSDR
            sdr = FobosSDR(lib_path=lib_path)
            return {"devices": sdr.list_devices(), "count": sdr.get_device_count()}
        except Exception as e:
            return {"error": str(e), "devices": [], "count": 0}

    # -----------------------------------------------------------------------
    # WebSocket — broadcaster pushes to all connected clients
    # -----------------------------------------------------------------------

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        connected_clients.append(ws)
        logger.info("WebSocket client connected (%d total)", len(connected_clients))
        try:
            while True:
                # Keep connection alive; data is pushed by _broadcaster task
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            if ws in connected_clients:
                connected_clients.remove(ws)
            logger.info("WebSocket client disconnected (%d total)", len(connected_clients))

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Pfobos WebUI server")
    p.add_argument("--stub",   action="store_true", help="Use C stub library instead of real device")
    p.add_argument("--host",   default="127.0.0.1",  help="Bind host (default: 127.0.0.1)")
    p.add_argument("--port",   type=int, default=8000, help="Bind port (default: 8000)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    lib = STUB_LIB if args.stub else None

    if args.stub and not os.path.exists(STUB_LIB):
        print(f"Stub library not found: {STUB_LIB}")
        print("Build it first: uv run python tests/stub/build.py")
        sys.exit(1)

    app = create_app(lib_path=lib)

    print(f"Starting Pfobos WebUI on http://{args.host}:{args.port}")
    print(f"Mode: {'stub library (synthetic signals)' if args.stub else 'real Fobos SDR device'}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
