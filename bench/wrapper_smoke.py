"""
pfobos §4.3 wrapper ABI smoke test.

Validates that the pfobos 0.2.x CFFI wrapper opens the installed production
libfobos.so and reads IQ data cleanly. Gate for embedded-agent's §5.1 default
flip (sdr.backend: pfobos in watchtower-edge config templates and Docker
defaults).

The goal is ABI compatibility, not sustained-throughput stress. Long sustained
reads at high rates are easy to write but hard to interpret — `read_rx_sync`
can block indefinitely without raising on USB hiccups, so a multi-minute
naive loop doesn't validate ABI, it just stress-tests the bus. We do two
short scoped checks instead:

  1. Sustained sync at SUSTAIN_RATE_HZ (default 10 MSPS — embedded-agent's
     validated rate from 2026-06-10 OPI5-1 bring-up) for SUSTAIN_SECONDS
     (default 5 s). Counts chunks, measures throughput, validates first-chunk
     health (dtype / shape / not-all-zero / no-NaN).
  2. Brief negotiation + 10-chunk read at HIGH_RATE_HZ (default 50 MSPS — the
     fobos-sdr-profile spec target) to confirm the device accepts the rate
     and produces well-formed chunks. Does NOT do sustained reads at this
     rate; that's a separate fobos-sdr-profile validation, not §4.3.

Run on OPI5-N with a Fobos SDR USB-attached:

    uv run python bench/wrapper_smoke.py
    uv run python bench/wrapper_smoke.py --out bench/results/<host>.json

Exits non-zero on any of:
  - libfobos.so does not load
  - no Fobos device enumerated
  - HIGH_RATE_HZ missing from device-reported sample-rate list
  - sync RX produces a malformed or all-zero chunk
  - sync RX underruns 4+ times mid-run
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


# Sustained sync test — embedded-agent's validated rate from the 2026-06-10
# OPI5-1 bring-up (Fobos serial A1D610000964, FW 2.2.0, 16384 complex64
# samples at 10 MSPS clean). Short SUSTAIN_SECONDS — §4.3 is an ABI smoke,
# not a throughput benchmark.
SUSTAIN_RATE_HZ = 10_000_000
SUSTAIN_SECONDS = 5

# High-rate sanity — fobos-sdr-profile spec target (post 2026-04-24 schema
# change to 50 MSPS / 50 MHz BW from 20 / 20). We only confirm the device
# advertises and accepts this rate and produces good chunks; we do NOT
# sustained-read at this rate (that's a separate fobos-sdr-profile
# validation, not §4.3 ABI smoke).
HIGH_RATE_HZ = 50_000_000
HIGH_RATE_CHUNKS = 10

# Sync buffer length in floats (I/Q interleaved). 65536 floats = 32768 IQ
# pairs = 32768 complex64 samples per chunk.
SYNC_BUF_FLOATS = 65536

# Async test parameters — 16 buffers x 32768 IQ pairs each. Goal is to
# confirm the callback wiring survives the C → numpy hand-off, not to
# stress the bus.
ASYNC_BUF_COUNT = 16
ASYNC_BUF_LENGTH = 32768
ASYNC_WALL_SECONDS = 3


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


def _sustained_read(sdr: FobosSDR, rate_hz: int, seconds: int,
                    label: str, freq_hz: int = 868_000_000) -> dict:
    actual_rate = sdr.set_samplerate(rate_hz)
    sdr.set_frequency(freq_hz)
    sdr.set_lna_gain(1)
    sdr.set_vga_gain(8)
    sdr.start_rx_sync(SYNC_BUF_FLOATS)

    deadline = time.monotonic() + seconds
    n_chunks = 0
    n_underrun = 0
    first_chunk_health: dict | None = None
    bytes_read = 0
    t_start = time.monotonic()
    aborted = None
    try:
        while time.monotonic() < deadline:
            try:
                iq = sdr.read_rx_sync()
            except FobosException as e:
                n_underrun += 1
                if n_underrun > 3:
                    aborted = e.message
                    break
                continue
            n_chunks += 1
            bytes_read += iq.nbytes
            if first_chunk_health is None:
                first_chunk_health = _chunk_health(iq)
    finally:
        sdr.stop_rx_sync()

    elapsed = time.monotonic() - t_start
    nominal_bytes = seconds * actual_rate * 8  # 8 bytes per complex64
    return {
        "label": label,
        "requested_rate_hz": rate_hz,
        "actual_rate_hz": float(actual_rate),
        "duration_s": elapsed,
        "chunks_read": n_chunks,
        "bytes_read": bytes_read,
        "throughput_mbps": (bytes_read * 8) / (elapsed * 1e6) if elapsed > 0 else 0.0,
        "nominal_mbps": (nominal_bytes * 8) / (seconds * 1e6) if seconds > 0 else 0.0,
        "underrun_count": n_underrun,
        "aborted_after": aborted,
        "first_chunk": first_chunk_health,
    }


def _bounded_chunk_read(sdr: FobosSDR, rate_hz: int, n_chunks: int,
                        freq_hz: int = 2_400_000_000) -> dict:
    """Read a fixed number of chunks at rate_hz — bounded work, no wall-clock loop."""
    actual_rate = sdr.set_samplerate(rate_hz)
    sdr.set_frequency(freq_hz)
    sdr.set_lna_gain(1)
    sdr.set_vga_gain(8)
    sdr.start_rx_sync(SYNC_BUF_FLOATS)

    chunks: list[dict] = []
    underrun = 0
    t_start = time.monotonic()
    try:
        for _ in range(n_chunks):
            t_a = time.monotonic()
            try:
                iq = sdr.read_rx_sync()
            except FobosException as e:
                underrun += 1
                chunks.append({"underrun": e.message})
                continue
            t_b = time.monotonic()
            chunks.append({
                "read_ms": (t_b - t_a) * 1000.0,
                "health": _chunk_health(iq),
            })
    finally:
        sdr.stop_rx_sync()
    elapsed = time.monotonic() - t_start

    return {
        "requested_rate_hz": rate_hz,
        "actual_rate_hz": float(actual_rate),
        "n_requested": n_chunks,
        "n_received": sum(1 for c in chunks if "health" in c),
        "underrun_count": underrun,
        "elapsed_s": elapsed,
        "chunks": chunks,
    }


def smoke(out_path: Path | None,
          sustain_rate_hz: int,
          sustain_seconds: int,
          high_rate_hz: int,
          high_rate_chunks: int) -> int:
    result = {
        "schema": "pfobos-wrapper-smoke/2",
        "started_utc": _now_utc_iso(),
        "host": os.uname().nodename,
        "kernel": os.uname().release,
        "pfobos": {"import": "ok"},
        "device": {},
        "sustained_sync": {},
        "high_rate_check": {},
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
        result["device"]["high_rate_supported"] = (
            high_rate_hz in [int(r) for r in rates]
        )
        result["device"]["sustain_rate_supported"] = (
            sustain_rate_hz in [int(r) for r in rates]
        )
        if not result["device"]["high_rate_supported"]:
            return _finalize(result, out_path, ok=False, exit_code=4)
        if not result["device"]["sustain_rate_supported"]:
            return _finalize(result, out_path, ok=False, exit_code=4)

        result["sustained_sync"] = _sustained_read(
            sdr, sustain_rate_hz, sustain_seconds,
            label=f"{sustain_rate_hz//1_000_000}MSPS sustained {sustain_seconds}s",
        )
        if result["sustained_sync"]["aborted_after"]:
            return _finalize(result, out_path, ok=False, exit_code=5)
        if not result["sustained_sync"]["first_chunk"]:
            return _finalize(result, out_path, ok=False, exit_code=5)
        fc = result["sustained_sync"]["first_chunk"]
        if fc["all_zero"] or fc["any_nan"]:
            return _finalize(result, out_path, ok=False, exit_code=6)

        result["high_rate_check"] = _bounded_chunk_read(
            sdr, high_rate_hz, high_rate_chunks,
        )
        if result["high_rate_check"]["n_received"] == 0:
            return _finalize(result, out_path, ok=False, exit_code=6)

        # ---- Async callback round-trip ----
        callback_events: list[dict] = []
        callback_done = threading.Event()

        sdr.set_samplerate(sustain_rate_hz)
        sdr.set_frequency(868_000_000)

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
            "first_callback_health": (
                callback_events[0]["health"] if callback_events else None
            ),
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
    ap.add_argument("--sustain-rate", type=int, default=SUSTAIN_RATE_HZ,
                    help=f"Sustained sync sample rate Hz (default {SUSTAIN_RATE_HZ})")
    ap.add_argument("--sustain-seconds", type=int, default=SUSTAIN_SECONDS,
                    help=f"Sustained sync duration seconds (default {SUSTAIN_SECONDS})")
    ap.add_argument("--high-rate", type=int, default=HIGH_RATE_HZ,
                    help=f"High-rate negotiation check Hz (default {HIGH_RATE_HZ})")
    ap.add_argument("--high-rate-chunks", type=int, default=HIGH_RATE_CHUNKS,
                    help=f"Number of chunks to read at the high rate (default {HIGH_RATE_CHUNKS})")
    args = ap.parse_args()
    return smoke(
        args.out,
        sustain_rate_hz=args.sustain_rate,
        sustain_seconds=args.sustain_seconds,
        high_rate_hz=args.high_rate,
        high_rate_chunks=args.high_rate_chunks,
    )


if __name__ == "__main__":
    sys.exit(main())
