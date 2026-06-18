"""
Effective sample-rate profiler for FobosSDR sync vs async paths.

Surfaced the 2026-06-18 finding that pfobos sync at default chunk size
delivers 99.4-99.8% of nominal sample rate when uncontended, falsifying
the earlier "wrapper has 12.5% overhead" hypothesis (the gap was actually
the watchtower-edge container's CPU/USB contention).

Use this script when:
  - Investigating an effective-rate regression
  - Comparing wrapper behavior pre/post a refactor
  - Auditing whether a workload contends with libfobos's USB pipeline

Modes:
  sync             buf_length=32768   FobosSDR.read_rx_sync(), single buffer
  sync-large       buf_length=131072  same, with the 4x larger buffer
  async            buf_length=32768   FobosSDR.start_rx_async() callback path
  async-large      buf_length=131072  same, with the 4x larger buffer

Usage:
  uv run python bench/profile_wrapper.py --mode sync --rate 8e6 --duration 10
  uv run python bench/profile_wrapper.py --sweep --duration 10 --out profile.json
  uv run python bench/profile_wrapper.py --sweep --pause-docker watchtower-edge-edge-1

The --pause-docker flag pauses the named container before the profile and
unpauses after, isolating the wrapper from CPU/USB contention. Without it
results can vary 20-30% depending on what else is running on the host.

Output: stdout (always) + JSON if --out is given.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from pfobos import FobosSDR


def run_sync(rate_hz: float, duration_s: float, buf_length: int,
             freq_hz: float = 868e6) -> dict:
    sdr = FobosSDR()
    sdr.open(0)
    actual_rate = sdr.set_samplerate(rate_hz)
    sdr.set_frequency(freq_hz)
    sdr.set_lna_gain(1)
    sdr.set_vga_gain(8)
    sdr.start_rx_sync(buf_length)

    deadline = time.monotonic() + duration_s
    total_samples = 0
    n_chunks = 0
    read_times_ms: list[float] = []
    t_start = time.monotonic()
    try:
        while time.monotonic() < deadline:
            t_a = time.perf_counter()
            iq = sdr.read_rx_sync()
            t_b = time.perf_counter()
            n_chunks += 1
            total_samples += iq.size
            read_times_ms.append((t_b - t_a) * 1000.0)
    finally:
        sdr.stop_rx_sync()
        sdr.close()

    elapsed = time.monotonic() - t_start
    sorted_times = sorted(read_times_ms)
    return {
        "mode": "sync",
        "buf_length": buf_length,
        "requested_rate_hz": rate_hz,
        "actual_rate_hz": float(actual_rate),
        "duration_s": round(elapsed, 3),
        "total_samples": total_samples,
        "n_chunks": n_chunks,
        "eff_rate_MSPS": round(total_samples / elapsed / 1e6, 3),
        "eff_rate_pct_of_actual": round(total_samples / (actual_rate * elapsed) * 100, 2),
        "chunks_per_s": round(n_chunks / elapsed, 1),
        "samples_per_chunk_avg": round(total_samples / max(n_chunks, 1), 1),
        "read_ms_p50": round(statistics.median(read_times_ms), 3) if read_times_ms else None,
        "read_ms_p99": round(sorted_times[int(0.99 * len(sorted_times))], 3) if sorted_times else None,
        "read_ms_max": round(max(read_times_ms), 3) if read_times_ms else None,
    }


def run_async(rate_hz: float, duration_s: float, buf_length: int,
              freq_hz: float = 868e6, buf_count: int = 16) -> dict:
    sdr = FobosSDR()
    sdr.open(0)
    actual_rate = sdr.set_samplerate(rate_hz)
    sdr.set_frequency(freq_hz)
    sdr.set_lna_gain(1)
    sdr.set_vga_gain(8)

    samples_lock = threading.Lock()
    total_samples = [0]
    n_callbacks = [0]

    def cb(iq: np.ndarray) -> None:
        with samples_lock:
            total_samples[0] += iq.size
            n_callbacks[0] += 1

    t_thread = threading.Thread(
        target=lambda: sdr.start_rx_async(
            cb, buf_count=buf_count, buf_length=buf_length
        ),
        daemon=True,
    )
    t_start = time.monotonic()
    t_thread.start()
    time.sleep(duration_s)
    t_stop = time.monotonic()
    sdr.stop_rx_async()
    t_thread.join(timeout=5)
    sdr.close()

    elapsed = t_stop - t_start
    return {
        "mode": "async",
        "buf_length": buf_length,
        "buf_count": buf_count,
        "requested_rate_hz": rate_hz,
        "actual_rate_hz": float(actual_rate),
        "duration_s": round(elapsed, 3),
        "total_samples": total_samples[0],
        "n_chunks": n_callbacks[0],
        "eff_rate_MSPS": round(total_samples[0] / elapsed / 1e6, 3),
        "eff_rate_pct_of_actual": round(total_samples[0] / (actual_rate * elapsed) * 100, 2),
        "chunks_per_s": round(n_callbacks[0] / elapsed, 1),
        "samples_per_chunk_avg": round(total_samples[0] / max(n_callbacks[0], 1), 1),
    }


@contextmanager
def _pause_container(container_name: str | None):
    if not container_name:
        yield
        return
    print(f"[profile] docker pause {container_name}", file=sys.stderr)
    subprocess.run(["docker", "pause", container_name], check=True, capture_output=True)
    try:
        yield
    finally:
        print(f"[profile] docker unpause {container_name}", file=sys.stderr)
        subprocess.run(["docker", "unpause", container_name], check=False, capture_output=True)


def _print(result: dict) -> None:
    print()
    for k, v in result.items():
        print(f"  {k}: {v}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mode", choices=["sync", "sync-large", "async", "async-large"],
                   help="Single mode to run")
    g.add_argument("--sweep", action="store_true",
                   help="Sweep sync + async at rates 4/8/16 MSPS, default chunk size")
    ap.add_argument("--rate", type=float, default=8e6,
                    help="Sample rate Hz (single-mode only; default 8e6)")
    ap.add_argument("--duration", type=float, default=10.0,
                    help="Test duration seconds (default 10)")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON output path (default: stdout only)")
    ap.add_argument("--pause-docker", type=str, default=None,
                    help="Docker container name to pause for the duration of the test")
    args = ap.parse_args()

    sweep_specs: list[tuple[str, float, int]] = []
    if args.sweep:
        for rate in (4e6, 8e6, 16e6):
            sweep_specs.append(("sync", rate, 32768))
            sweep_specs.append(("async", rate, 32768))
    else:
        buf_length = 131072 if "large" in args.mode else 32768
        mode_key = "sync" if args.mode.startswith("sync") else "async"
        sweep_specs.append((mode_key, args.rate, buf_length))

    results: list[dict] = []
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _pause_container(args.pause_docker):
        for mode_key, rate, buf_length in sweep_specs:
            print(f"==================== {mode_key} @ {rate/1e6:g} MSPS ====================",
                  file=sys.stderr)
            if mode_key == "sync":
                r = run_sync(rate, args.duration, buf_length)
            else:
                r = run_async(rate, args.duration, buf_length)
            results.append(r)
            _print(r)

    if args.out:
        payload = {
            "schema": "pfobos-profile-wrapper/1",
            "started_utc": started_utc,
            "host": os.uname().nodename,
            "kernel": os.uname().release,
            "pause_docker": args.pause_docker,
            "duration_s": args.duration,
            "results": results,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
