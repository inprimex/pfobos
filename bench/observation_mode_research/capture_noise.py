"""
observation-mode-fast-scan-detection-research E1.2 task 4.6 starter —
quiet-spectrum capture for embedded-agent's PFA-calibration corpus.

Captures `--duration` seconds of raw IQ at the device's native rate, splits
into 500k-IQ-sample files (per embedded-agent's bus 20260611T121717 spec),
and writes a sibling JSON sidecar with all metadata. Default tuned to land
ONE band quickly so embedded-agent's WST-bench ingest path unblocks; bulk
multi-band sessions extend the same script with a sweep wrapper.

Run on OPI5-N with a Fobos SDR USB-attached:

    uv run python bench/observation_mode_research/capture_noise.py \
        --freq 868e6 \
        --rate 15.36e6 \
        --duration 30 \
        --env quiet_lab \
        --antenna "868MHz dipole, terminated" \
        --out-dir bench/observation_mode_research/noise_captures

Filename layout (per embedded-agent):
    <band>_<rate>MSPS_<env>_<utc>_part0001.iq
    <band>_<rate>MSPS_<env>_<utc>_part0002.iq
    ...
    <band>_<rate>MSPS_<env>_<utc>.json
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pfobos import FobosSDR, FobosException


SAMPLES_PER_CHUNK_FILE = 500_000   # IQ samples per .iq file (embedded-agent's chunk-size)

# Sync buffer length in floats (I/Q interleaved). 65536 floats = 32768 IQ
# pairs per chunk. At 15.36 MSPS that's ~2.1 ms per chunk; we accumulate
# 500k IQ pairs across ~16 chunks before each .iq file is written.
SYNC_BUF_FLOATS = 65536

# Magic-bytes header on each .iq file. complex64 native byte order; no
# endian conversion. Consumers should be on the same byte order or do
# their own swap on ingest.
# Keep simple: no header — pure interleaved-float32 (I0 Q0 I1 Q1 ...) as
# numpy.complex64.tobytes() emits, so np.fromfile(..., dtype=np.complex64)
# round-trips losslessly. This matches what wst_filter_bank ingests.


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _band_label(freq_hz: float) -> str:
    if freq_hz >= 1e9:
        return f"{freq_hz/1e9:g}GHz"
    return f"{int(round(freq_hz/1e6))}MHz"


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
        "ok": False,
    }

    # Write metadata stub up-front so even a hard-aborted capture leaves
    # a parseable JSON sidecar that embedded-agent can inspect.
    meta_path.write_text(json.dumps(metadata, indent=2))

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
    accum = []
    accum_samples = 0
    total_samples = 0
    t_start = time.monotonic()

    try:
        while not interrupted and time.monotonic() < deadline:
            try:
                iq = sdr.read_rx_sync()
            except FobosException as e:
                print(f"[capture] read error: {e.message}; aborting", file=sys.stderr)
                metadata["error"] = {"code": e.code, "message": e.message}
                break
            accum.append(iq)
            accum_samples += iq.size
            total_samples += iq.size

            while accum_samples >= SAMPLES_PER_CHUNK_FILE:
                stitched = np.concatenate(accum) if len(accum) > 1 else accum[0]
                file_idx += 1
                chunk = stitched[:SAMPLES_PER_CHUNK_FILE]
                remainder = stitched[SAMPLES_PER_CHUNK_FILE:]
                part_path = out_dir / f"{base}_part{file_idx:04d}.iq"
                chunk.tofile(part_path)
                metadata["files"].append({
                    "path": part_path.name,
                    "samples": int(SAMPLES_PER_CHUNK_FILE),
                    "bytes": int(SAMPLES_PER_CHUNK_FILE * 8),
                })
                if remainder.size:
                    accum = [remainder]
                    accum_samples = int(remainder.size)
                else:
                    accum = []
                    accum_samples = 0
    finally:
        sdr.stop_rx_sync()
        sdr.close()

    # Flush trailing partial chunk so no samples are dropped.
    if accum_samples > 0:
        stitched = np.concatenate(accum) if len(accum) > 1 else accum[0]
        file_idx += 1
        part_path = out_dir / f"{base}_part{file_idx:04d}.iq"
        stitched.tofile(part_path)
        metadata["files"].append({
            "path": part_path.name,
            "samples": int(stitched.size),
            "bytes": int(stitched.size * 8),
        })

    metadata["finished_utc"] = _now_utc_iso()
    metadata["actual_duration_s"] = time.monotonic() - t_start
    metadata["total_samples"] = total_samples
    metadata["ok"] = not interrupted and "error" not in metadata
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(
        f"\n[capture] wrote {file_idx} .iq files + 1 .json sidecar to {out_dir}\n"
        f"  base={base}\n"
        f"  total_samples={total_samples} "
        f"  duration={metadata['actual_duration_s']:.2f} s",
        flush=True,
    )
    return 0 if metadata["ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=float, required=True,
                    help="Center frequency in Hz (e.g. 868e6)")
    ap.add_argument("--rate", type=float, default=15.36e6,
                    help="Sample rate in Hz (default 15.36 MSPS for narrowband)")
    ap.add_argument("--duration", type=float, default=30.0,
                    help="Capture duration in seconds (default 30)")
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
