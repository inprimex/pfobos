"""
observation-mode-fast-scan-detection-research E1.2 task 4.6 —
quiet-spectrum capture for embedded-agent's PFA-calibration corpus.

Captures `--duration` seconds of raw IQ at the device's native rate, splits
into 500k-IQ-sample files (per embedded-agent's bus 20260611T121717 spec),
and writes a sibling JSON sidecar with all metadata.

Schema produced: `pfobos-noise-capture/1`. Consumer-side parser:
`watchtower-edge/bench/observation_mode_research/noise_capture_ingest.py`
(embedded-agent, watchtower-edge PR #15) validated end-to-end against the
2026-06-16 partial captures — 15 unit tests + a CLI smoke that recovers
RMS p50 = 0.0005 from the 22 s 868 MHz quiet-lab capture, consistent with
the LNA=1/VGA=8 noise floor.

## Background writer architecture (since 2026-06-17 refactor)

The original tight loop wrote each 500k-sample .iq file with a synchronous
`chunk.tofile()` call inside the read loop. At 10 MSPS that's a ~4 MB
write every ~50 ms; the filesystem syscall stalled the inner loop long
enough that libfobos's USB ring buffer eventually overflowed and the read
returned `Fobos error -9: libusb error` at ~10–22 s (finding §5.3 of the
2026-06-16 session memo).

This version moves the disk write off the read thread. The main thread:

  - Reads from the SDR.
  - Accumulates samples up to SAMPLES_PER_CHUNK_FILE.
  - Posts a (file_idx, numpy chunk) tuple to a bounded queue.

A dedicated writer thread consumes the queue and writes each chunk to
disk. Backpressure: if the queue is full the main thread blocks on
`put` — which would re-introduce the original failure mode, so the queue
is generously bounded (QUEUE_MAX_CHUNKS) and the writer logs a warning if
it ever fills (data-loss diagnostic).

Metadata `files[]` accumulates on the WRITER side and is merged into the
main metadata dict at shutdown, so there's no shared-state race.

## Run

    uv run python bench/observation_mode_research/capture_noise.py \\
        --freq 868e6 \\
        --rate 16e6 \\
        --duration 60 \\
        --env quiet_lab \\
        --antenna "868MHz dipole, terminated" \\
        --out-dir bench/observation_mode_research/noise_captures

## Filename layout (per embedded-agent)

    <band>_<rate>MSPS_<env>_<utc>_part0001.iq
    <band>_<rate>MSPS_<env>_<utc>_part0002.iq
    ...
    <band>_<rate>MSPS_<env>_<utc>.json
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pfobos import FobosSDR, FobosException


SAMPLES_PER_CHUNK_FILE = 500_000   # IQ samples per .iq file (embedded-agent's chunk-size)

# Sync buffer length in floats (I/Q interleaved). 65536 floats = 32768 IQ
# pairs per chunk. At 16 MSPS that's ~2.0 ms per chunk; we accumulate
# 500k IQ pairs across ~16 chunks before each .iq file is queued.
SYNC_BUF_FLOATS = 65536

# Writer-queue bound. At ~4 MB per chunk × 64 chunks ≈ 256 MB in flight
# max. Should never get close; if the writer keeps up there'll be ≤1 chunk
# in the queue at any time. The bound exists to fail loudly rather than
# OOM if disk catastrophically lags behind USB.
QUEUE_MAX_CHUNKS = 64


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _band_label(freq_hz: float) -> str:
    if freq_hz >= 1e9:
        return f"{freq_hz/1e9:g}GHz"
    return f"{int(round(freq_hz/1e6))}MHz"


# Sentinel posted on the queue to signal end-of-stream cleanly.
_POISON = object()


def _writer_loop(
    q: queue.Queue,
    out_dir: Path,
    base: str,
    files_record: list[dict],
    writer_error: list[Exception | None],
    writer_died: threading.Event,
) -> None:
    """Consume (file_idx, chunk) tuples from the queue and write .iq files.

    Runs in a dedicated thread so the main read loop is never blocked on
    filesystem syscalls. Exits on the _POISON sentinel or on the first
    write error (records the exception so the main thread can surface
    it in the metadata and abort cleanly).
    """
    try:
        while True:
            item = q.get()
            if item is _POISON:
                q.task_done()
                return
            file_idx, chunk = item
            part_path = out_dir / f"{base}_part{file_idx:04d}.iq"
            chunk.tofile(part_path)
            files_record.append({
                "path": part_path.name,
                "samples": int(chunk.size),
                "bytes": int(chunk.size * 8),
            })
            q.task_done()
    except Exception as e:
        writer_error[0] = e
        writer_died.set()


def capture(args: argparse.Namespace) -> int:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    utc_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"{_band_label(args.freq)}_{args.rate/1e6:g}MSPS_{args.env}_{utc_stamp}"
    meta_path = out_dir / f"{base}.json"

    sdr = FobosSDR()
    sdr.open(0)

    board = sdr.get_board_info()
    api = sdr.get_api_info()
    advertised_rates = [float(r) for r in sdr.get_samplerates()]
    if int(args.rate) not in [int(r) for r in advertised_rates]:
        print(
            f"WARNING: requested rate {args.rate} Hz not in advertised list "
            f"{advertised_rates}. Continuing — Fobos will negotiate the closest "
            f"supported rate.",
            file=sys.stderr,
        )

    actual_rate = sdr.set_samplerate(args.rate)
    actual_freq = sdr.set_frequency(args.freq)
    sdr.set_lna_gain(args.lna_gain)
    sdr.set_vga_gain(args.vga_gain)

    metadata = {
        "schema": "pfobos-noise-capture/1",
        "base": base,
        "started_utc": _now_utc_iso(),
        "host": os.uname().nodename,
        "kernel": os.uname().release,
        "fobos": {
            **board,
            "api": api,
            "advertised_samplerates_hz": advertised_rates,
        },
        "config": {
            "requested_center_freq_hz": args.freq,
            "actual_center_freq_hz": float(actual_freq),
            "requested_sample_rate_hz": args.rate,
            "actual_sample_rate_hz": float(actual_rate),
            "lna_gain": args.lna_gain,
            "vga_gain": args.vga_gain,
            "antenna": args.antenna,
            "environment_tag": args.env,
            "requested_duration_s": args.duration,
        },
        "files": [],
        "samples_per_file": SAMPLES_PER_CHUNK_FILE,
        "dtype": "complex64",
        "byte_order": sys.byteorder,
        "finished_utc": None,
        "actual_duration_s": None,
        "total_samples": 0,
        "writer": {
            "queue_max_chunks": QUEUE_MAX_CHUNKS,
            "queue_full_blocks": 0,
        },
        "ok": False,
    }

    # Write metadata stub up-front so even a hard-aborted capture leaves
    # a parseable JSON sidecar that embedded-agent can inspect.
    meta_path.write_text(json.dumps(metadata, indent=2))

    # Writer-thread setup: bounded queue + side-channel for the files list
    # (avoids cross-thread mutation of metadata). writer_died fires if the
    # writer hits an exception (e.g. disk full); main thread checks it
    # before every put so we don't block forever on a dead consumer.
    chunk_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX_CHUNKS)
    files_record: list[dict] = []
    queue_full_count = [0]
    writer_error: list[Exception | None] = [None]
    writer_died = threading.Event()
    writer = threading.Thread(
        target=_writer_loop,
        args=(chunk_queue, out_dir, base, files_record, writer_error, writer_died),
        name="capture-writer",
        daemon=False,
    )
    writer.start()

    sdr.start_rx_sync(SYNC_BUF_FLOATS)
    interrupted = False

    def _on_signal(signum, _frame):
        nonlocal interrupted
        interrupted = True
        print(f"\n[capture] caught signal {signum}; flushing current chunk file...",
              file=sys.stderr, flush=True)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    deadline = time.monotonic() + args.duration
    file_idx = 0
    accum: list[np.ndarray] = []
    accum_samples = 0
    total_samples = 0
    t_start = time.monotonic()

    try:
        while not interrupted and time.monotonic() < deadline:
            try:
                iq = sdr.read_rx_sync()
            except FobosException as e:
                # pfobos 0.3.0 contract: sync mode survives errors. We
                # still abort the capture session on first error because
                # noise-capture cares about contiguous integrity per
                # segment, not best-effort retry — embedded-agent's
                # ingest is gap-tolerant via the sidecar files[] array,
                # so a clean abort + new run is cleaner than an in-band
                # gap that the JSON can't describe.
                print(f"[capture] read error: {e.message}; aborting", file=sys.stderr)
                metadata["error"] = {"code": e.code, "message": e.message}
                break
            accum.append(iq)
            accum_samples += iq.size
            total_samples += iq.size

            while accum_samples >= SAMPLES_PER_CHUNK_FILE:
                if writer_died.is_set():
                    print(
                        f"[capture] writer thread died: {writer_error[0]}; aborting",
                        file=sys.stderr,
                    )
                    metadata["error"] = {
                        "code": 0,
                        "message": f"writer thread died: {writer_error[0]}",
                    }
                    interrupted = True
                    break
                stitched = np.concatenate(accum) if len(accum) > 1 else accum[0]
                file_idx += 1
                chunk = stitched[:SAMPLES_PER_CHUNK_FILE].copy()
                remainder = stitched[SAMPLES_PER_CHUNK_FILE:]
                # Non-blocking put first so we can detect queue-full
                # situations as a diagnostic signal (writer can't keep
                # up with the read pipeline — re-introduces the original
                # failure mode if it persists).
                try:
                    chunk_queue.put_nowait((file_idx, chunk))
                except queue.Full:
                    queue_full_count[0] += 1
                    print(
                        f"[capture] WARNING: writer queue full ({QUEUE_MAX_CHUNKS} "
                        f"chunks pending); blocking read until disk catches up. "
                        f"This may stress the USB pipeline.",
                        file=sys.stderr, flush=True,
                    )
                    chunk_queue.put((file_idx, chunk))
                if remainder.size:
                    accum = [remainder.copy()]
                    accum_samples = int(remainder.size)
                else:
                    accum = []
                    accum_samples = 0
    finally:
        # Stop the SDR before the writer drains so the device session is
        # released even if the writer takes a while on the trailing
        # chunks.
        try:
            sdr.stop_rx_sync()
        except Exception:
            pass
        try:
            sdr.close()
        except Exception:
            pass

    # Flush trailing partial chunk so no samples are dropped (skip if the
    # writer died — would block forever on a dead consumer).
    if accum_samples > 0 and not writer_died.is_set():
        stitched = np.concatenate(accum) if len(accum) > 1 else accum[0]
        file_idx += 1
        chunk_queue.put((file_idx, stitched.copy()))

    # Signal writer to exit + wait for it to drain. If the writer is
    # already dead, the poison is a no-op queue.put against the bounded
    # queue (capped at QUEUE_MAX_CHUNKS); only blocks if the queue is
    # full, which can't happen by construction since the writer's queue
    # consumption was the only reason chunks were being removed — if it
    # died at chunk N, queue size is at most N pending. Cap with a
    # timeout for safety.
    try:
        chunk_queue.put(_POISON, timeout=5.0)
    except queue.Full:
        # Writer dead and queue stayed full; nothing we can do but proceed
        # to record the failure in metadata.
        pass
    writer.join(timeout=10.0)
    if writer.is_alive():
        print(
            "[capture] WARNING: writer thread did not join within 10 s; "
            "metadata may be incomplete.",
            file=sys.stderr,
        )

    metadata["files"] = files_record
    metadata["writer"]["queue_full_blocks"] = queue_full_count[0]
    metadata["finished_utc"] = _now_utc_iso()
    metadata["actual_duration_s"] = time.monotonic() - t_start
    metadata["total_samples"] = total_samples
    metadata["ok"] = not interrupted and "error" not in metadata
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(
        f"\n[capture] wrote {len(files_record)} .iq files + 1 .json sidecar to {out_dir}\n"
        f"  base={base}\n"
        f"  total_samples={total_samples}   duration={metadata['actual_duration_s']:.2f} s\n"
        f"  writer queue-full blocks={queue_full_count[0]}",
        flush=True,
    )
    return 0 if metadata["ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=float, required=True,
                    help="Center frequency in Hz (e.g. 868e6)")
    ap.add_argument("--rate", type=float, default=16e6,
                    help="Sample rate in Hz (default 16 MSPS; closest Fobos rate to embedded-agent's 15.36 spec)")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="Capture duration in seconds (default 60; embedded-agent's PD@PFA target needs ≥1200 windows × 500k samples per band)")
    ap.add_argument("--env", type=str, required=True,
                    choices=["urban", "field", "quiet_lab"],
                    help="Environment tag (per embedded-agent format spec)")
    ap.add_argument("--antenna", type=str, required=True,
                    help="Antenna description for the metadata sidecar")
    ap.add_argument("--lna-gain", type=int, default=1,
                    help="LNA gain stage 0..2 (default 1)")
    ap.add_argument("--vga-gain", type=int, default=8,
                    help="VGA gain stage 0..15 (default 8)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("bench/observation_mode_research/noise_captures"),
                    help="Output directory (default: bench/observation_mode_research/noise_captures)")
    args = ap.parse_args()
    return capture(args)


if __name__ == "__main__":
    sys.exit(main())
