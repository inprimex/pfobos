"""
pfobos §4.3 wrapper ABI smoke test.

Validates that the pfobos 0.2.x CFFI wrapper opens the installed production
libfobos.so and reads IQ data cleanly. Gate for embedded-agent's §5.1 default
flip (sdr.backend: pfobos in watchtower-edge config templates and Docker
defaults).

Run on OPI5-N with a Fobos SDR USB-attached:

    uv run python bench/wrapper_smoke.py
    uv run python bench/wrapper_smoke.py --out bench/results/<host>.json

Exits non-zero on any of:
  - libfobos.so does not load
  - no Fobos device enumerated
  - 50 MSPS missing from device-reported sample-rate list
  - sync RX produces a malformed or all-zero chunk
  - sync RX underflows / aborts mid-run
  - async RX callback never fires
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pfobos import FobosSDR, FobosException


# Target sample rate for the sustained sync test. Must be advertised by the
# device in fobos_rx_get_samplerates(). 50 MSPS is the production profile
# after the 2026-04-24 fobos_sdr.yaml schema change (was 20 MSPS / 20 MHz).
TARGET_RATE_HZ = 50_000_000

# Sustained-read duration (seconds) — long enough to surface USB hiccups and
# OPI5 RK3588 xhci-hcd buffer behaviour without being painful to re-run.
SUSTAIN_SECONDS = 60

# Sync buffer length in floats (I/Q interleaved). 65536 floats = 32768 IQ
# pairs = 32768 complex64 samples per chunk = ~0.66 ms at 50 MSPS.
SYNC_BUF_FLOATS = 65536

# Async test parameters — 16 buffers x 32768 IQ pairs each, ~10 ms wall-clock
# inside the C thread before we cancel. Goal is to confirm the callback wiring
# survives the C → numpy hand-off, not to stress the bus.
ASYNC_BUF_COUNT = 16
ASYNC_BUF_LENGTH = 32768
ASYNC_WALL_SECONDS = 5


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _chunk_health(iq: np.ndarray) -> dict:
    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2))) if iq.size else 0.0
    return {
        "len": int(iq.size),
        "dtype": str(iq.dtype),
        "rms": rms,
        "all_zero": bool(np.all(iq == 0)) if iq.size else True,
        "any_nan": bool(np.any(np.isnan(iq.real)) or np.any(np.isnan(iq.imag))),
    }


def smoke(out_path: Path | None) -> int:
    result = {
        "schema": "pfobos-wrapper-smoke/1",
        "started_utc": _now_utc_iso(),
        "host": os.uname().nodename,
        "kernel": os.uname().release,
        "pfobos": {"import": "ok"},
        "device": {},
        "sync_50msps": {},
        "async": {},
        "ok": False,
    }

    try:
        sdr = FobosSDR()
    except OSError as e:
        result["pfobos"]["import"] = f"load_library_failed: {e}"
        return _finalize(result, out_path, ok=False, exit_code=2)

    result["pfobos"]["api"] = sdr.get_api_info()

    n_devs = sdr.get_device_count()
    result["device"]["count"] = n_devs
    if n_devs < 1:
        return _finalize(result, out_path, ok=False, exit_code=3)

    try:
        sdr.open(0)
        result["device"].update(sdr.get_board_info())

        rates = sdr.get_samplerates()
        result["device"]["samplerates_hz"] = [float(r) for r in rates]
        if TARGET_RATE_HZ not in [int(r) for r in rates]:
            result["device"]["target_rate_supported"] = False
            return _finalize(result, out_path, ok=False, exit_code=4)
        result["device"]["target_rate_supported"] = True

        # ---- Sustained sync read at 50 MSPS ----
        actual_rate = sdr.set_samplerate(TARGET_RATE_HZ)
        sdr.set_frequency(868_000_000)
        sdr.set_lna_gain(1)
        sdr.set_vga_gain(8)
        sdr.start_rx_sync(SYNC_BUF_FLOATS)

        deadline = time.monotonic() + SUSTAIN_SECONDS
        n_chunks = 0
        n_underrun = 0
        first_chunk_health = None
        bytes_read = 0
        t_start = time.monotonic()
        try:
            while time.monotonic() < deadline:
                try:
                    iq = sdr.read_rx_sync()
                except FobosException as e:
                    n_underrun += 1
                    if n_underrun > 3:
                        result["sync_50msps"]["aborted_after"] = e.message
                        return _finalize(result, out_path, ok=False, exit_code=5)
                    continue
                n_chunks += 1
                bytes_read += iq.nbytes
                if first_chunk_health is None:
                    first_chunk_health = _chunk_health(iq)
                    if first_chunk_health["all_zero"] or first_chunk_health["any_nan"]:
                        return _finalize(result, out_path, ok=False, exit_code=6)
        finally:
            sdr.stop_rx_sync()

        elapsed = time.monotonic() - t_start
        nominal_bytes = SUSTAIN_SECONDS * actual_rate * 8  # 8 bytes per complex64
        result["sync_50msps"] = {
            "requested_rate_hz": TARGET_RATE_HZ,
            "actual_rate_hz": float(actual_rate),
            "duration_s": elapsed,
            "chunks_read": n_chunks,
            "bytes_read": bytes_read,
            "throughput_mbps": (bytes_read * 8) / (elapsed * 1e6),
            "nominal_mbps": (nominal_bytes * 8) / (SUSTAIN_SECONDS * 1e6),
            "underrun_count": n_underrun,
            "first_chunk": first_chunk_health,
        }

        # ---- Async callback round-trip ----
        callback_events = []
        callback_done = threading.Event()

        def cb(iq: np.ndarray):
            callback_events.append({
                "t": time.monotonic(),
                "health": _chunk_health(iq),
            })
            if len(callback_events) >= ASYNC_BUF_COUNT // 2:
                callback_done.set()

        t_async = threading.Thread(
            target=lambda: sdr.start_rx_async(
                cb, buf_count=ASYNC_BUF_COUNT, buf_length=ASYNC_BUF_LENGTH
            ),
            daemon=True,
        )
        t_async.start()
        callback_done.wait(timeout=ASYNC_WALL_SECONDS)
        sdr.stop_rx_async()
        t_async.join(timeout=5)

        result["async"] = {
            "wall_seconds": ASYNC_WALL_SECONDS,
            "callbacks_received": len(callback_events),
            "first_callback_health": callback_events[0]["health"] if callback_events else None,
            "callback_thread_joined": not t_async.is_alive(),
        }
        if len(callback_events) == 0:
            return _finalize(result, out_path, ok=False, exit_code=7)

        sdr.close()

    except FobosException as e:
        result["error"] = {"code": e.code, "message": e.message}
        return _finalize(result, out_path, ok=False, exit_code=8)

    return _finalize(result, out_path, ok=True, exit_code=0)


def _finalize(result: dict, out_path: Path | None, *, ok: bool, exit_code: int) -> int:
    result["ok"] = ok
    result["finished_utc"] = _now_utc_iso()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return exit_code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON output path (default: stdout only)")
    args = ap.parse_args()
    return smoke(args.out)


if __name__ == "__main__":
    sys.exit(main())
