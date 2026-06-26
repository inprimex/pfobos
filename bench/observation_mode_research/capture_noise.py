"""
observation-mode-fast-scan-detection-research E1.2 task 4.6 + Phase 3 §5.1 —
quiet-spectrum + drone-linked corpus capture.

Captures `--duration` seconds of raw IQ at the device's native rate, splits
into 500k-IQ-sample files (per embedded-agent's bus 20260611T121717 spec),
and writes a sibling JSON sidecar with all metadata.

Schema produced: `pfobos-noise-capture/2`. Consumer-side parser:
`watchtower-edge/bench/observation_mode_research/noise_capture_ingest.py`
(embedded-agent, watchtower-edge PR #15) validated end-to-end against the
2026-06-16 partial captures — 15 unit tests + a CLI smoke that recovers
RMS p50 = 0.0005 from the 22 s 868 MHz quiet-lab capture, consistent with
the LNA=1/VGA=8 noise floor.

## Library API (since the 2026-06-26 refactor)

The capture engine is now library-first. Embed it in your own pipelines
via `CaptureConfig` + `CaptureSession.run()` (or the convenience
`run_capture(config)` function). The CLI in `main()` is a thin shim that
builds a `CaptureConfig` from argparse and invokes the session.

```python
from bench.observation_mode_research.capture_noise import (
    CaptureConfig, run_capture,
)

def my_metric_writer(chunk):
    # Called synchronously in the writer thread AFTER the .iq write.
    # Exceptions are caught + recorded in metadata['fan_out']['errors'].
    rms = float(np.abs(chunk.samples).mean())
    metrics_jsonl.write(json.dumps({"file_idx": chunk.file_idx, "rms": rms}) + "\\n")

config = CaptureConfig(
    freq_hz=2.4e9, rate_hz=16e6, duration_s=30.0,
    env="field", antenna="2.4 GHz dipole",
    out_dir=Path("/mnt/nvme/corpus/run-001"),
    fan_out=[my_metric_writer],
)
result = run_capture(config)
print(result.ok, result.total_samples, result.metadata_path)
```

## fan_out hook contract

Each fan_out consumer is a `Callable[[CaptureChunk], None]` invoked in the
writer thread immediately AFTER the .iq file lands on disk. The writer
loop catches and logs any exception so a misbehaving consumer cannot
break the primary IQ-write path. Consumers see the same chunk objects in
the same order as the .iq files were written.

`CaptureChunk` fields:
  - `file_idx`: 1-based file index matching the `_partNNNN.iq` filename
  - `samples`: `numpy.ndarray[complex64]` — the chunk that was just written
  - `path`: `Path` — the .iq file that was just written
  - `band_label`: e.g. `2.4GHz`, `868MHz`
  - `base`: filename base (band_rate_env_utc)

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

## Operational notes for sustained captures on OPI5

  - `docker pause watchtower-edge-edge-1` BEFORE invoking — edge container
    contention causes the libusb-9 cliff at 8+ MSPS sustained.
  - Use the IRQ 87 USB port (`xhci-hcd:usb3`); bus number rotates across
    reboots, IRQ name is stable. 2-4× sustained-throughput delta vs.
    the other USB 3.0 port.
  - For multi-phase sessions in the same process, call
    `sdr.get_board_info()` between `open()` and `set_samplerate()` to
    clear residual device state (libfobos quirk).
  - Target `out_dir` on NVMe (e.g. `/mnt/nvme/corpus/`), never SD card.

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from pfobos import FobosSDR, FobosException

from bench.observation_mode_research.hwtel import HwTelemetry


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


@dataclass
class CaptureChunk:
    """A single 500k-IQ-sample chunk that was just written to disk.

    Passed to each fan_out consumer in the writer thread, after the .iq
    file lands. Consumers can read `samples` for in-memory analysis or
    re-open `path` for downstream tools that expect a file.
    """
    file_idx: int
    samples: np.ndarray
    path: Path
    band_label: str
    base: str


FanOutConsumer = Callable[[CaptureChunk], None]


@dataclass
class CaptureConfig:
    """Configuration for a single capture session.

    Required: freq_hz, rate_hz, duration_s, env, antenna, out_dir.
    The rest carry defaults that match the embedded-agent corpus spec.
    """
    freq_hz: float
    rate_hz: float
    duration_s: float
    env: str  # "urban" | "field" | "quiet_lab"
    antenna: str
    out_dir: Path

    lna_gain: int = 1
    vga_gain: int = 8
    hwtel_interval_s: float = 1.0

    samples_per_file: int = SAMPLES_PER_CHUNK_FILE
    sync_buf_floats: int = SYNC_BUF_FLOATS
    queue_max_chunks: int = QUEUE_MAX_CHUNKS

    fan_out: list[FanOutConsumer] = field(default_factory=list)

    # Optional lib_path override — primarily for tests using the stub
    # libfobos.so. Production callers leave this None and let pfobos's
    # system library discovery find the real libfobos.so.
    lib_path: str | None = None


@dataclass
class CaptureResult:
    """Outcome of a CaptureSession.run() call."""
    ok: bool
    metadata: dict
    metadata_path: Path
    iq_paths: list[Path]
    total_samples: int
    actual_duration_s: float
    error: Exception | None = None


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
    band_label: str,
    files_record: list[dict],
    fan_out_consumers: list[FanOutConsumer],
    fan_out_errors: list[dict],
    writer_error: list[Exception | None],
    writer_died: threading.Event,
) -> None:
    """Consume (file_idx, chunk) tuples from the queue and write .iq files.

    Runs in a dedicated thread so the main read loop is never blocked on
    filesystem syscalls. Exits on the _POISON sentinel or on the first
    write error (records the exception so the main thread can surface
    it in the metadata and abort cleanly).

    After each successful .iq write, invokes every fan_out consumer in
    order. Consumer exceptions are caught + appended to `fan_out_errors`
    so a buggy consumer cannot break the IQ-write path.
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
            if fan_out_consumers:
                capture_chunk = CaptureChunk(
                    file_idx=file_idx,
                    samples=chunk,
                    path=part_path,
                    band_label=band_label,
                    base=base,
                )
                for consumer in fan_out_consumers:
                    try:
                        consumer(capture_chunk)
                    except Exception as e:  # noqa: BLE001 — fan_out isolation is the point
                        fan_out_errors.append({
                            "file_idx": file_idx,
                            "consumer": getattr(consumer, "__name__", repr(consumer)),
                            "error": f"{type(e).__name__}: {e}",
                        })
            q.task_done()
    except Exception as e:
        writer_error[0] = e
        writer_died.set()


class CaptureSession:
    """One-shot capture session: open SDR → stream → close.

    Usage:

        session = CaptureSession(config)
        result = session.run()

    Or use the convenience function `run_capture(config)`.

    Not re-entrant: construct a new instance per capture.
    """

    def __init__(self, config: CaptureConfig):
        self.config = config

    def run(self) -> CaptureResult:
        cfg = self.config
        out_dir = cfg.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        utc_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        band_label = _band_label(cfg.freq_hz)
        base = f"{band_label}_{cfg.rate_hz/1e6:g}MSPS_{cfg.env}_{utc_stamp}"
        meta_path = out_dir / f"{base}.json"

        sdr = FobosSDR(lib_path=cfg.lib_path) if cfg.lib_path else FobosSDR()
        sdr.open(0)

        board = sdr.get_board_info()
        api = sdr.get_api_info()
        advertised_rates = [float(r) for r in sdr.get_samplerates()]
        if int(cfg.rate_hz) not in [int(r) for r in advertised_rates]:
            print(
                f"WARNING: requested rate {cfg.rate_hz} Hz not in advertised list "
                f"{advertised_rates}. Continuing — Fobos will negotiate the closest "
                f"supported rate.",
                file=sys.stderr,
            )

        actual_rate = sdr.set_samplerate(cfg.rate_hz)
        actual_freq = sdr.set_frequency(cfg.freq_hz)
        sdr.set_lna_gain(cfg.lna_gain)
        sdr.set_vga_gain(cfg.vga_gain)

        hwtel = HwTelemetry(interval_s=cfg.hwtel_interval_s)
        metadata = {
            "schema": "pfobos-noise-capture/2",
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
                "requested_center_freq_hz": cfg.freq_hz,
                "actual_center_freq_hz": float(actual_freq),
                "requested_sample_rate_hz": cfg.rate_hz,
                "actual_sample_rate_hz": float(actual_rate),
                "lna_gain": cfg.lna_gain,
                "vga_gain": cfg.vga_gain,
                "antenna": cfg.antenna,
                "environment_tag": cfg.env,
                "requested_duration_s": cfg.duration_s,
            },
            "hwtel": {
                "interval_s": cfg.hwtel_interval_s,
                "zones": hwtel.zones,
                "cpus": hwtel.cpus,
                "samples": [],
                "summary": {},
            },
            "files": [],
            "samples_per_file": cfg.samples_per_file,
            "dtype": "complex64",
            "byte_order": sys.byteorder,
            "finished_utc": None,
            "actual_duration_s": None,
            "total_samples": 0,
            "writer": {
                "queue_max_chunks": cfg.queue_max_chunks,
                "queue_full_blocks": 0,
            },
            "fan_out": {
                "consumers": [
                    getattr(c, "__name__", repr(c)) for c in cfg.fan_out
                ],
                "errors": [],
            },
            "ok": False,
        }

        # Write metadata stub up-front so even a hard-aborted capture leaves
        # a parseable JSON sidecar that embedded-agent can inspect.
        meta_path.write_text(json.dumps(metadata, indent=2))

        chunk_queue: queue.Queue = queue.Queue(maxsize=cfg.queue_max_chunks)
        files_record: list[dict] = []
        fan_out_errors: list[dict] = []
        queue_full_count = [0]
        writer_error: list[Exception | None] = [None]
        writer_died = threading.Event()
        writer = threading.Thread(
            target=_writer_loop,
            args=(
                chunk_queue, out_dir, base, band_label,
                files_record, list(cfg.fan_out), fan_out_errors,
                writer_error, writer_died,
            ),
            name="capture-writer",
            daemon=False,
        )
        writer.start()

        sdr.start_rx_sync(cfg.sync_buf_floats)
        hwtel.start()
        interrupted = False

        def _on_signal(signum, _frame):
            nonlocal interrupted
            interrupted = True
            print(f"\n[capture] caught signal {signum}; flushing current chunk file...",
                  file=sys.stderr, flush=True)

        # Signal handlers only install if we're on the main thread —
        # library users embedding CaptureSession in a worker thread will
        # raise ValueError from signal.signal(). Skip silently in that case;
        # they're responsible for their own shutdown signalling.
        installed_handlers = False
        try:
            signal.signal(signal.SIGINT, _on_signal)
            signal.signal(signal.SIGTERM, _on_signal)
            installed_handlers = True
        except ValueError:
            pass

        deadline = time.monotonic() + cfg.duration_s
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
                    # pfobos 0.3.0+ contract: sync mode survives errors. We
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

                while accum_samples >= cfg.samples_per_file:
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
                    chunk = stitched[:cfg.samples_per_file].copy()
                    remainder = stitched[cfg.samples_per_file:]
                    try:
                        chunk_queue.put_nowait((file_idx, chunk))
                    except queue.Full:
                        queue_full_count[0] += 1
                        print(
                            f"[capture] WARNING: writer queue full ({cfg.queue_max_chunks} "
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
            try:
                sdr.stop_rx_sync()
            except Exception:
                pass
            try:
                sdr.close()
            except Exception:
                pass
            # Best-effort restore default signal handlers so we don't
            # leave the caller's process in an altered state.
            if installed_handlers:
                try:
                    signal.signal(signal.SIGINT, signal.SIG_DFL)
                    signal.signal(signal.SIGTERM, signal.SIG_DFL)
                except (ValueError, OSError):
                    pass

        # Flush trailing partial chunk so no samples are dropped (skip if the
        # writer died — would block forever on a dead consumer).
        if accum_samples > 0 and not writer_died.is_set():
            stitched = np.concatenate(accum) if len(accum) > 1 else accum[0]
            file_idx += 1
            chunk_queue.put((file_idx, stitched.copy()))

        try:
            chunk_queue.put(_POISON, timeout=5.0)
        except queue.Full:
            pass
        writer.join(timeout=10.0)
        if writer.is_alive():
            print(
                "[capture] WARNING: writer thread did not join within 10 s; "
                "metadata may be incomplete.",
                file=sys.stderr,
            )

        metadata["hwtel"]["samples"] = hwtel.stop()
        metadata["hwtel"]["summary"] = hwtel.summary()
        metadata["files"] = files_record
        metadata["writer"]["queue_full_blocks"] = queue_full_count[0]
        metadata["fan_out"]["errors"] = fan_out_errors
        metadata["finished_utc"] = _now_utc_iso()
        metadata["actual_duration_s"] = time.monotonic() - t_start
        metadata["total_samples"] = total_samples
        metadata["ok"] = not interrupted and "error" not in metadata
        meta_path.write_text(json.dumps(metadata, indent=2))

        return CaptureResult(
            ok=metadata["ok"],
            metadata=metadata,
            metadata_path=meta_path,
            iq_paths=[out_dir / f["path"] for f in files_record],
            total_samples=total_samples,
            actual_duration_s=metadata["actual_duration_s"],
            error=writer_error[0],
        )


def run_capture(config: CaptureConfig) -> CaptureResult:
    """Convenience wrapper: construct + run a CaptureSession."""
    return CaptureSession(config).run()


def _print_summary(result: CaptureResult, out_dir: Path) -> None:
    """Console summary for the CLI path (operator-facing)."""
    metadata = result.metadata
    hwtel_summary = metadata["hwtel"]["summary"]
    thermal_summary = " ".join(
        f"{zone}:{stats['min']:.0f}→{stats['max']:.0f}°C"
        for zone, stats in hwtel_summary.get("thermal_C", {}).items()
    )
    fan_out_errors = metadata["fan_out"]["errors"]
    fan_out_line = (
        f"  fan_out errors: {len(fan_out_errors)}\n" if fan_out_errors else ""
    )
    print(
        f"\n[capture] wrote {len(result.iq_paths)} .iq files + 1 .json sidecar to {out_dir}\n"
        f"  base={metadata['base']}\n"
        f"  total_samples={result.total_samples}   duration={result.actual_duration_s:.2f} s\n"
        f"  writer queue-full blocks={metadata['writer']['queue_full_blocks']}\n"
        f"{fan_out_line}"
        f"  hwtel thermal: {thermal_summary or '(no zones detected)'}",
        flush=True,
    )


def capture(args: argparse.Namespace) -> int:
    """CLI shim: build a CaptureConfig from argparse and run the session."""
    config = CaptureConfig(
        freq_hz=args.freq,
        rate_hz=args.rate,
        duration_s=args.duration,
        env=args.env,
        antenna=args.antenna,
        out_dir=args.out_dir,
        lna_gain=args.lna_gain,
        vga_gain=args.vga_gain,
        hwtel_interval_s=args.hwtel_interval,
    )
    result = run_capture(config)
    _print_summary(result, args.out_dir)
    return 0 if result.ok else 1


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
    ap.add_argument("--hwtel-interval", type=float, default=1.0,
                    help="Hardware telemetry sample interval in seconds (default 1.0; "
                         "set to 0.5 for finer correlation with libusb-9 cliff timing)")
    args = ap.parse_args()
    return capture(args)


if __name__ == "__main__":
    sys.exit(main())
