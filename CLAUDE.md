# PFobos — Claude Code Context

## Project Summary

Python wrapper for the **Fobos SDR** (Software Defined Radio) C library (`libfobos`).
Provides a Pythonic, NumPy-integrated API for spectrum analysis and FM demodulation.
Platform: Linux (WSL2), Python 3.7+, hardware USB SDR device.

## Architecture

```
pfobos/
├── shared/fwrapper.py        # Core: FobosSDR class (CFFI bindings to libfobos)
├── shared/__init__.py        # Exports FobosSDR, FobosException
├── rtanalyzer/rtanalyzer.py  # Real-time spectrum analyzer (matplotlib, sync mode)
├── rtanalyzer/__init__.py
├── fmreceiver/
│   ├── fobos_fm_receiver.py         # FM demodulation + sounddevice audio
│   └── fobos_fm_receiver_paplay.py  # FM receiver via PulseAudio (paplay)
├── tests/
│   ├── test_mock_fobos.py        # Unit tests — no hardware (uses unittest.mock)
│   ├── test_wrapper_logic.py     # Wrapper logic / error-handling tests
│   ├── test_stub_integration.py  # End-to-end tests via C stub library (no hardware)
│   ├── test_integration.py       # Hardware integration tests (requires device)
│   ├── test_performance.py       # Performance / timing tests (requires device)
│   ├── benchmark.py              # Detailed benchmark tool (FobosSDRBenchmark class)
│   ├── benchmark_analyze.py      # Plot/compare saved benchmark results
│   ├── stub/                     # C stub library for hardware-free testing
│   │   ├── libfobos_stub.c       # All 16 fobos_rx_* API functions
│   │   ├── signals.json          # Signal config (noise, tone, FM)
│   │   ├── build.py              # gcc build script → libfobos.so
│   │   └── .gitignore            # excludes libfobos.so (built artifact)
│   └── __main__.py               # python -m tests entry point
├── webui/                    # Web-based spectrum viewer (FastAPI + WebSocket)
│   ├── server.py             # FastAPI app: REST + WebSocket /ws
│   ├── sdr_worker.py         # Background IQ reader + FFT → asyncio queue
│   ├── static/
│   │   ├── index.html        # SPA: spectrum, waterfall, IQ constellation
│   │   └── app.js            # WebSocket client + Chart.js + WebFFT
│   └── __main__.py
├── webui/                    # Web-based spectrum viewer (FastAPI + WebSocket)
│   ├── server.py             # FastAPI app: REST + WebSocket /ws
│   ├── sdr_worker.py         # Background IQ reader + FFT → asyncio queue
│   ├── static/
│   │   ├── index.html        # SPA: spectrum, waterfall, IQ constellation
│   │   └── app.js            # WebSocket client + Chart.js + WebFFT
│   └── __main__.py
├── scripts/                  # Dev/debug helper scripts
├── setup/setup-fobos-sdr.sh  # udev rules setup (Linux)
├── doc/                      # Markdown docs per component
├── run_tests.py              # Main test runner (argparse CLI)
├── run_rtanalyzer.py         # Launch spectrum analyzer
├── run_setup.py              # Setup verification
└── requirements.txt          # numpy, scipy, matplotlib, cffi, pandas, tabulate
```

## Core Class: FobosSDR (`shared/fwrapper.py`)

- Uses **CFFI** (`cffi.FFI`) to load `libfobos.so` (Linux) / `fobos.dll` (Windows)
- Library loaded via `ffi.dlopen()` — must be in system path or same directory
- All C errors converted to `FobosException(code, message)`
- IQ data returned as `np.ndarray` of `complex64` (interleaved float I/Q → complex)

### Key state flags
- `self.dev` — CFFI device pointer (None if not opened)
- `self._sync_mode` / `self._async_mode` — active reception mode
- `self._callback` — kept alive to prevent GC during async mode

### Synchronous reception pattern
```python
sdr.start_rx_sync(buf_length)  # allocates buffer
iq = sdr.read_rx_sync()         # returns complex64 ndarray
sdr.stop_rx_sync()
```

### Asynchronous reception pattern
```python
def cb(iq_samples): ...  # called from C thread
sdr.start_rx_async(cb, buf_count=16, buf_length=32768)
# blocks until fobos_rx_cancel_async is called
sdr.stop_rx_async()
```

### Gain ranges
- LNA gain: 0–2
- VGA gain: 0–15

### Frequency ranges
- General: 10 MHz – 6 GHz (hardware dependent)
- MAX2830: 2350–2550 MHz
- RFFC507x LO: 25 MHz – 5400 MHz

## Running Tests

```bash
# No-hardware tests
uv run python run_tests.py               # mock + logic tests
uv run python run_tests.py --verbose

# Stub integration tests (no hardware — builds C stub first)
uv run python tests/stub/build.py        # compile tests/stub/libfobos.so
uv run pytest tests/test_stub_integration.py -v  # 25 tests

# Hardware tests
uv run python run_tests.py --integration        # requires device
uv run python run_tests.py --performance-only   # requires device
uv run python run_tests.py --benchmark
```

Default timeout: 30s per test. Hardware tests skip automatically when device absent.

## Stub Library (`tests/stub/`)

A real compiled C `.so` loaded via CFFI — no mocks — for end-to-end testing without hardware.

**Build:**
```bash
uv run python tests/stub/build.py   # produces tests/stub/libfobos.so
```

**Usage:**
```python
from shared.fwrapper import FobosSDR
sdr = FobosSDR(lib_path="tests/stub/libfobos.so")
```

**Signal configuration** (`tests/stub/signals.json`):
```json
{ "signals": [
    {"type": "noise",  "amplitude": 0.05},
    {"type": "fm",     "audio_hz": 1000, "deviation": 75000, "amplitude": 0.8},
    {"type": "tone",   "freq_hz": 100000, "amplitude": 0.3}
]}
```
Override path via env var: `FOBOS_STUB_SIGNALS=/path/to/signals.json`

**What the stub covers:**
- CFFI loading and all 16 API functions
- Synchronous and asynchronous IQ reception (configurable count)
- FFT peak detection with a known tone frequency
- FM demodulation DSP pipeline

**Isolation from production:** The stub `.so` is a gitignored build artifact; `FobosSDR()` with no `lib_path` always uses system library discovery unchanged.

## Running Applications

```bash
python run_rtanalyzer.py          # Spectrum analyzer (TkAgg matplotlib)
python -m fmreceiver.fobos_fm_receiver -f 95.5 -g 12
```

## Key Conventions

- **Imports**: All apps import from `shared.fwrapper` (not a legacy `fobos_wrapper`)
  - Note: `fmreceiver/fobos_fm_receiver_paplay.py` still uses `from fobos_wrapper import` (legacy, needs fix)
- **Error handling**: Always catch `FobosException`; never swallow in finally (log only)
- **Buffer sizes**: Must be even (I/Q pairs); minimum 1024 floats
- **Context manager**: `FobosSDR` supports `with` statement → auto `close()`
- **No legacy methods**: `start_rx_async_legacy`, `stop_rx_async_legacy*` have been removed

## Known Issues / Tech Debt

- `fmreceiver/fobos_fm_receiver_paplay.py` imports from `fobos_wrapper` (old name) instead of `shared.fwrapper`
- `stop_rx_async` has a 5s timeout + polling loop; can block on USB cancel
- `read_rx_sync` and async callbacks use `np.frombuffer(ffi.buffer(...)).copy()` for fast C→numpy transfer
- `rtanalyzer.py` saves to `spectrum_plots/` directory (excluded from git)

## Package Manager: uv

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync                  # install all dependencies (creates .venv)
uv sync --extra audio    # also install sounddevice + pulsectl
uv run python run_tests.py        # run with managed environment
uv run python run_rtanalyzer.py   # run spectrum analyzer
uv add <package>         # add a new dependency
```

`uv.lock` is committed — ensures reproducible installs on all machines.
`.venv/` is gitignored — recreated locally by `uv sync`.

## Dependencies

| Package    | Group    | Purpose                               |
|------------|----------|---------------------------------------|
| numpy      | core     | IQ array processing                   |
| scipy      | core     | Signal processing (FM demod, FFT windows) |
| matplotlib | core     | Spectrum visualisation (TkAgg)        |
| cffi       | core     | C library FFI bindings                |
| pandas     | core     | Benchmark result analysis             |
| tabulate   | core     | Benchmark result formatting           |
| sounddevice| audio    | Audio output (FM receiver)            |
| pulsectl   | audio    | PulseAudio control (optional)         |
| pytest     | dev      | Test runner                           |

## Hardware Setup (Linux)

```bash
sudo ./setup/setup-fobos-sdr.sh   # installs udev rules
python run_setup.py               # verifies environment
```

USB access requires udev rules or running as root. WSL2 needs `usbipd` to forward USB.

## WebUI ()

Browser-based SDR spectrum viewer — works with stub (no hardware) and real device.

**Run:**
```bash
uv sync --extra webui                              # install FastAPI + uvicorn
uv run python -m webui.server --stub               # stub mode, http://localhost:8000
uv run python -m webui.server --host 0.0.0.0       # accessible from Windows browser
```

**Endpoints:**
- `GET /` — SPA frontend
- `WS /ws` — live spectrum frames (JSON: freqs, spectrum, waterfall_row, iq_raw)
- `GET /api/config` — current SDR config
- `POST /api/config` — update center_freq, sample_rate, lna_gain, vga_gain, fft_size
- `GET /api/devices` — enumerate connected devices

**Frontend visualisations (CDN, no build step):**
- Spectrum: Chart.js line chart (dB vs Hz)
- Waterfall: Canvas 2D heatmap (plasma colormap, scrolling)
- IQ Constellation: Canvas 2D scatter (last 2048 IQ pairs)
- WebFFT: client-side FFT via webfft npm package (toggle on/off)
