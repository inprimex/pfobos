"""
observation-mode-fast-scan-detection-research E0.1 task 2.4 —
classic-FW retune latency on OPI5 with current production Fobos FW 2.2.0.

Methodology (also recorded in the bench writeup at
watchtower-edge/bench/observation_mode_research/retune_classic_vs_agile.md):

  - For each target frequency in the BANDS sweep (100 MHz - 6 GHz), perform
    N_RETUNES_PER_BAND retunes.
  - One retune cycle:
      t0       = time.perf_counter()
      sdr.set_frequency(target_hz)
      t_set    = time.perf_counter()
      iq_0     = sdr.read_rx_sync()          # first chunk available after retune
      t_first  = time.perf_counter()
      [stream chunks until RMS stabilises within STABLE_RMS_TOL_DB of the
       trailing-window steady-state]
      t_stable = time.perf_counter()
  - Two latency metrics emitted per retune:
      first_chunk_ms  = (t_first  - t0) * 1000
      stable_ms       = (t_stable - t0) * 1000
    Both are reported because the spec text in fobos-sdr-profile claims
    `scan.retune_time_ms: 3.5` without explicitly defining "stable". 2.7 picks
    the canonical one for the writeup once we see the data; the JSON preserves
    both so re-analysis is cheap.
  - Between retunes, the device is parked on a far-away frequency so two
    consecutive retunes to the same band still involve a real PLL lock
    (otherwise the inner-loop measurement degenerates to chunk-transit time).

Output:
  - JSON per-band latency arrays + summary stats (p50/p90/p99/p99.9).
  - Markdown table for the writeup.

Run on OPI5-N with a Fobos SDR USB-attached:

    uv run python bench/observation_mode_research/retune_classic.py \
        --out bench/observation_mode_research/results/retune_classic_<host>.json \
        --md  bench/observation_mode_research/results/retune_classic_<host>.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from pfobos import FobosSDR, FobosException


# Sweep representative frequencies across the device's full range (10 MHz –
# 6 GHz). Log-ish spacing in the lower bands (where ISM clutter sits) and
# linear in the upper bands. ISM-anchored: 433, 868, 915, 2400, 5800.
BANDS_HZ = [
    100_000_000,
    200_000_000,
    433_000_000,
    868_000_000,
    915_000_000,
    1_575_000_000,    # GPS L1
    2_400_000_000,
    3_000_000_000,
    4_000_000_000,
    5_800_000_000,
]
# Far-away "park" frequency we hop to between every measured retune so each
# inner-loop retune is a true PLL relock, not a same-band micro-tune.
PARK_FREQ_HZ = 50_000_000

N_RETUNES_PER_BAND = 100

# Sample rate for the measurement. Moderate rate (not 50 MSPS) keeps each
# chunk small so chunk-transit time doesn't dominate the latency budget;
# we want retune-dominated latencies, not USB-buffering ones.
SAMPLE_RATE_HZ = 20_000_000

# Sync buffer length in floats (I/Q interleaved). 8192 floats = 4096 IQ
# pairs = ~205 us at 20 MSPS — fine-grained enough to land "first stable
# chunk" near the actual PLL-lock instant.
SYNC_BUF_FLOATS = 8192

# "Stable" definition: chunk RMS within STABLE_RMS_TOL_DB of the trailing
# STABLE_WINDOW chunks' median. Caps at STABLE_MAX_CHUNKS so a never-
# stabilising retune fails fast instead of hanging the sweep.
STABLE_RMS_TOL_DB = 1.0
STABLE_WINDOW = 5
STABLE_MAX_CHUNKS = 200

LNA_GAIN = 1
VGA_GAIN = 8


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rms_db(iq: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
    return 20 * np.log10(max(rms, 1e-12))


def _percentile(arr: list[float], q: float) -> float:
    return float(np.percentile(np.array(arr), q)) if arr else float("nan")


def _ensure_sync_started(sdr: FobosSDR) -> None:
    """Restart sync mode if it's not active (belt-and-suspenders).

    Originally added as a workaround for the pre-0.3.0 pfobos auto-stop trap:
    `read_rx_sync` used to call `stop_rx_sync()` on any C-side error, killing
    sync mode and making the next read raise `RuntimeError: Synchronous mode
    not started`. The auto-stop was removed in pfobos 0.3.0 via spec change
    `pfobos-read-rx-sync-no-auto-stop-on-error` — see the change folder in
    watchtower-specs for the full contract.

    Kept here as defensive code because the sweep does its own
    `try/except FobosException; continue`, and if any future change to the
    wrapper or a caller-side path (`stop_rx_sync` in a `finally`, manual
    restart on certain error codes) drops sync mode we'd rather re-arm than
    fail the rest of the sweep.
    """
    if not sdr._sync_mode:
        sdr.start_rx_sync(SYNC_BUF_FLOATS)


def measure_band(sdr: FobosSDR, target_hz: int) -> dict:
    first_ms: list[float] = []
    stable_ms: list[float] = []
    failures = 0

    for _ in range(N_RETUNES_PER_BAND):
        try:
            _ensure_sync_started(sdr)
            # Park away so the next set_frequency triggers a real relock.
            sdr.set_frequency(PARK_FREQ_HZ)
            # Drain a chunk at the park freq so the buffer pipeline is fresh.
            _ = sdr.read_rx_sync()

            t0 = time.perf_counter()
            sdr.set_frequency(target_hz)
            iq0 = sdr.read_rx_sync()
            t_first = time.perf_counter()
            first_ms.append((t_first - t0) * 1000.0)
        except FobosException:
            failures += 1
            # Drop this retune. Sync mode survives errors as of pfobos 0.3.0
            # (no auto-stop); _ensure_sync_started next iter is a no-op in the
            # happy path but stays as defensive code (see its docstring).
            stable_ms.append(float("nan"))
            continue

        window: list[float] = [_rms_db(iq0)]
        stable_reached = False
        n_chunks = 1
        while n_chunks < STABLE_MAX_CHUNKS:
            try:
                iq = sdr.read_rx_sync()
            except FobosException:
                failures += 1
                break
            n_chunks += 1
            window.append(_rms_db(iq))
            if len(window) >= STABLE_WINDOW + 1:
                trail = window[-(STABLE_WINDOW + 1):-1]
                median = float(np.median(trail))
                if abs(window[-1] - median) <= STABLE_RMS_TOL_DB:
                    stable_reached = True
                    break
        t_stable = time.perf_counter()
        if stable_reached:
            stable_ms.append((t_stable - t0) * 1000.0)
        else:
            stable_ms.append(float("nan"))

    return {
        "target_hz": target_hz,
        "n_retunes": N_RETUNES_PER_BAND,
        "failures": failures,
        "first_chunk_ms": {
            "samples": first_ms,
            "p50": _percentile(first_ms, 50),
            "p90": _percentile(first_ms, 90),
            "p99": _percentile(first_ms, 99),
            "p99_9": _percentile(first_ms, 99.9),
        },
        "stable_ms": {
            "samples": [v for v in stable_ms if not np.isnan(v)],
            "n_unstabilised": sum(1 for v in stable_ms if np.isnan(v)),
            "p50": _percentile([v for v in stable_ms if not np.isnan(v)], 50),
            "p90": _percentile([v for v in stable_ms if not np.isnan(v)], 90),
            "p99": _percentile([v for v in stable_ms if not np.isnan(v)], 99),
            "p99_9": _percentile([v for v in stable_ms if not np.isnan(v)], 99.9),
        },
    }


def _markdown(result: dict) -> str:
    lines = [
        "# E0.1 task 2.4 — classic-FW retune latency",
        "",
        f"- host: `{result['host']}`",
        f"- fobos serial: `{result['device']['serial']}`",
        f"- fw version: `{result['device']['fw_version']}`",
        f"- sample rate: {result['sample_rate_hz']/1e6:.2f} MSPS",
        f"- retunes per band: {result['n_retunes_per_band']}",
        f"- stable window / tol: {result['stable_window']} chunks / {result['stable_rms_tol_db']} dB",
        f"- started: {result['started_utc']}",
        f"- finished: {result['finished_utc']}",
        "",
        "## first_chunk_ms (set_frequency → first chunk in hand)",
        "",
        "| band (MHz) | p50 | p90 | p99 | p99.9 | failures |",
        "|------------|-----|-----|-----|-------|----------|",
    ]
    for band in result["bands"]:
        f = band["first_chunk_ms"]
        lines.append(
            f"| {band['target_hz']/1e6:>10.1f} | "
            f"{f['p50']:.2f} | {f['p90']:.2f} | {f['p99']:.2f} | "
            f"{f['p99_9']:.2f} | {band['failures']} |"
        )
    lines += [
        "",
        f"## stable_ms (set_frequency → first chunk with RMS within "
        f"{result['stable_rms_tol_db']} dB of trailing median)",
        "",
        "| band (MHz) | p50 | p90 | p99 | p99.9 | unstabilised |",
        "|------------|-----|-----|-----|-------|--------------|",
    ]
    for band in result["bands"]:
        s = band["stable_ms"]
        lines.append(
            f"| {band['target_hz']/1e6:>10.1f} | "
            f"{s['p50']:.2f} | {s['p90']:.2f} | {s['p99']:.2f} | "
            f"{s['p99_9']:.2f} | {s['n_unstabilised']} |"
        )
    lines.append("")
    lines.append(
        f"Spec-paper claim being benchmarked: "
        f"`scan.retune_time_ms: 3.5` (fobos-sdr-profile)."
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="JSON output path")
    ap.add_argument("--md", type=Path, default=None,
                    help="Optional markdown summary output path")
    ap.add_argument("--bands", type=int, nargs="*", default=BANDS_HZ,
                    help=f"Frequencies (Hz) to sweep. Default: {len(BANDS_HZ)} bands")
    args = ap.parse_args()

    result: dict = {
        "schema": "pfobos-retune-classic/1",
        "started_utc": _now_utc_iso(),
        "host": os.uname().nodename,
        "kernel": os.uname().release,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "n_retunes_per_band": N_RETUNES_PER_BAND,
        "stable_window": STABLE_WINDOW,
        "stable_rms_tol_db": STABLE_RMS_TOL_DB,
        "park_freq_hz": PARK_FREQ_HZ,
        "lna_gain": LNA_GAIN,
        "vga_gain": VGA_GAIN,
        "bands": [],
    }

    sdr = FobosSDR()
    sdr.open(0)
    fatal_error: BaseException | None = None
    try:
        result["device"] = sdr.get_board_info()
        result["api"] = sdr.get_api_info()
        result["device"]["samplerates_hz"] = [float(r) for r in sdr.get_samplerates()]

        sdr.set_samplerate(SAMPLE_RATE_HZ)
        sdr.set_lna_gain(LNA_GAIN)
        sdr.set_vga_gain(VGA_GAIN)
        sdr.set_frequency(PARK_FREQ_HZ)
        sdr.start_rx_sync(SYNC_BUF_FLOATS)
        try:
            for freq in args.bands:
                print(f"  measuring {freq/1e6:.1f} MHz ...", flush=True)
                try:
                    band_result = measure_band(sdr, int(freq))
                except BaseException as e:
                    # Preserve partial results from earlier bands; surface the
                    # failure mode in the JSON for the writeup so the bench
                    # crash itself is part of the dataset.
                    result.setdefault("incomplete_at_hz", int(freq))
                    result["error"] = {
                        "type": type(e).__name__,
                        "message": str(e),
                    }
                    fatal_error = e
                    break
                result["bands"].append(band_result)
                f = band_result["first_chunk_ms"]
                print(
                    f"    first_chunk_ms p50={f['p50']:.2f} p99={f['p99']:.2f}  "
                    f"failures={band_result['failures']}",
                    flush=True,
                )
        finally:
            try:
                sdr.stop_rx_sync()
            except Exception:
                pass
    finally:
        try:
            sdr.close()
        except Exception:
            pass

    result["finished_utc"] = _now_utc_iso()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")

    if args.md and result["bands"]:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(_markdown(result))
        print(f"wrote {args.md}")

    if fatal_error is not None:
        raise fatal_error
    return 0


if __name__ == "__main__":
    sys.exit(main())
