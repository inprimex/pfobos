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
│   ├── test_mock_fobos.py    # Unit tests — no hardware required (uses unittest.mock)
│   ├── test_wrapper_logic.py # Wrapper logic / error-handling tests
│   ├── test_integration.py   # Hardware integration tests (requires device)
│   ├── test_performance.py   # Performance / timing tests (requires device)
│   ├── benchmark.py          # Detailed benchmark tool (FobosSDRBenchmark class)
│   ├── benchmark_analyze.py  # Plot/compare saved benchmark results
│   └── __main__.py           # python -m tests entry point
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
python run_tests.py                      # mock + logic tests (no hardware)
python run_tests.py --verbose
python run_tests.py --integration        # requires hardware
python run_tests.py --performance-only   # requires hardware
python run_tests.py --benchmark          # benchmark tool
python -m tests                          # alternative entry point
```

Default timeout: 30s per test. Hardware tests skip automatically when device absent.

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
