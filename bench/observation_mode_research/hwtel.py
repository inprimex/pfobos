"""Lightweight hardware telemetry sampler for capture-side benches.

Samples thermal zones + per-CPU frequencies + load average on a background
thread so capture scripts can record the hardware environment alongside
the IQ data. Each sample is a small dict; full sample series + summary
land in the capture's sidecar JSON.

Designed for OPI5 Max / RK3588 (7 thermal zones — soc, bigcore0,
bigcore1, littlecore, center, gpu, npu) but auto-detects whatever
sysfs interfaces are present, so it's portable to other Linux SBCs
and degrades gracefully on systems without /sys/class/thermal.

## Why this exists

The libusb-9 sustained-streaming cliff on OPI5+Fobos shows large run-to-
run variance — 22 s at 10 MSPS one day, 4 s the next, with the same
usbreset pre-flight. The OPI5 in operator's lab ships without an active
heatsink; RK3588 thermal throttling under load is a strong candidate
for the variance. Without temperature + CPU-freq data captured
alongside the IQ stream we can't tell environmental from intrinsic
device behaviour.

## Usage

    from bench.observation_mode_research.hwtel import HwTelemetry

    tel = HwTelemetry(interval_s=1.0)
    tel.start()
    try:
        # ... do capture / bench work ...
    finally:
        samples = tel.stop()
    summary = tel.summary()

    # samples: list of {"t_s": float, "thermal_C": {...}, "cpu_freq_MHz": {...}, "loadavg": [...]}
    # summary: {"thermal_C": {zone: {min, max, last}}, "cpu_freq_MHz": {cpu: {min, max, last}}}

## Overhead

At interval_s=1.0 the cost is ~7 sysfs reads + ~8 cpufreq reads per
second on a separate thread = ~15 small file reads / s. Negligible
versus the SDR read pipeline.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path


THERMAL_DIR = Path("/sys/class/thermal")
CPUFREQ_DIR = Path("/sys/devices/system/cpu")


class HwTelemetry:
    """Background sampler for thermal zones + per-CPU frequency + load.

    Sysfs interfaces are discovered at construction time; if a zone or
    CPU disappears mid-run (hotplug etc.) its samples drop out gracefully.
    """

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self._samples: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._t_start: float | None = None
        self._zones = self._discover_thermal_zones()
        self._cpus = self._discover_cpus()

    def _discover_thermal_zones(self) -> dict[str, Path]:
        zones: dict[str, Path] = {}
        if not THERMAL_DIR.exists():
            return zones
        for zone_path in sorted(THERMAL_DIR.glob("thermal_zone*")):
            type_file = zone_path / "type"
            temp_file = zone_path / "temp"
            if type_file.exists() and temp_file.exists():
                try:
                    name = type_file.read_text().strip()
                except OSError:
                    continue
                zones[name] = temp_file
        return zones

    def _discover_cpus(self) -> dict[str, Path]:
        cpus: dict[str, Path] = {}
        for cpu_path in sorted(CPUFREQ_DIR.glob("cpu[0-9]*")):
            freq_file = cpu_path / "cpufreq" / "scaling_cur_freq"
            if freq_file.exists():
                cpus[cpu_path.name] = freq_file
        return cpus

    @property
    def zones(self) -> list[str]:
        return list(self._zones)

    @property
    def cpus(self) -> list[str]:
        return list(self._cpus)

    def _read_int(self, path: Path) -> int | None:
        try:
            return int(path.read_text().strip())
        except (OSError, ValueError):
            return None

    def _sample(self) -> dict:
        sample: dict = {
            "t_s": time.monotonic() - (self._t_start or 0.0),
            "thermal_C": {},
            "cpu_freq_MHz": {},
        }
        for name, temp_file in self._zones.items():
            milli_c = self._read_int(temp_file)
            if milli_c is not None:
                # Thermal-zone temp is reported in milli-Celsius on Linux
                sample["thermal_C"][name] = round(milli_c / 1000.0, 1)
        for cpu, freq_file in self._cpus.items():
            khz = self._read_int(freq_file)
            if khz is not None:
                # scaling_cur_freq is in kHz
                sample["cpu_freq_MHz"][cpu] = round(khz / 1000.0, 1)
        try:
            sample["loadavg"] = list(os.getloadavg())
        except OSError:
            sample["loadavg"] = []
        return sample

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._samples.append(self._sample())
            except Exception:
                # Telemetry must NEVER crash the bench. If sysfs glitches,
                # drop the sample and keep going.
                pass
            self._stop_event.wait(self.interval_s)

    def start(self) -> None:
        self._t_start = time.monotonic()
        self._samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="hwtel-sampler", daemon=True,
        )
        self._thread.start()

    def stop(self) -> list[dict]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return list(self._samples)

    def summary(self) -> dict:
        """Quick min/max/last aggregate for at-a-glance reading."""
        out: dict = {"thermal_C": {}, "cpu_freq_MHz": {}}
        if not self._samples:
            return out
        for name in self._zones:
            vals = [
                s["thermal_C"][name]
                for s in self._samples
                if name in s.get("thermal_C", {})
            ]
            if vals:
                out["thermal_C"][name] = {
                    "min": min(vals), "max": max(vals), "last": vals[-1],
                    "delta": round(max(vals) - min(vals), 1),
                }
        for cpu in self._cpus:
            vals = [
                s["cpu_freq_MHz"][cpu]
                for s in self._samples
                if cpu in s.get("cpu_freq_MHz", {})
            ]
            if vals:
                out["cpu_freq_MHz"][cpu] = {
                    "min": min(vals), "max": max(vals), "last": vals[-1],
                    "delta": round(max(vals) - min(vals), 1),
                }
        if self._samples:
            last_load = self._samples[-1].get("loadavg", [])
            if last_load:
                out["loadavg_last"] = last_load
        return out
