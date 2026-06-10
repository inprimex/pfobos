# PFobos — Python wrapper for Fobos SDR

Python wrapper for the [Fobos SDR](https://github.com/rigexpert/libfobos) C library (`libfobos`).
Provides a Pythonic, NumPy-integrated API for spectrum analysis and FM demodulation,
plus a browser-based spectrum viewer that runs without physical hardware.

## Features

- Full Python interface to Fobos SDR via CFFI bindings
- Synchronous and asynchronous IQ sample collection
- NumPy `complex64` output — plug directly into scipy/numpy signal processing
- Real-time spectrum analyzer (matplotlib, requires X server / TkAgg)
- **Browser-based WebUI** — spectrum, waterfall, IQ constellation; works in WSL2 without a display
- FM radio demodulation with audio output
- C stub library for hardware-free development and testing (no USB device needed)
- 25 stub integration tests + mock unit tests

## Quick Start

### Package manager

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync              # core deps
uv sync --extra webui   # + FastAPI / uvicorn for WebUI
uv sync --extra audio   # + sounddevice / pulsectl for FM receiver
```

### Hardware setup (Linux)

```bash
sudo ./setup/setup-fobos-sdr.sh   # install udev rules
uv run python run_setup.py        # verify environment
```

WSL2: use `usbipd` to forward the USB device from Windows to WSL.

### Run without hardware (stub mode)

```bash
# Build the C stub library once
uv run python tests/stub/build.py

# Run tests
uv run pytest tests/test_stub_integration.py -v   # 25 integration tests

# Launch WebUI (no hardware needed)
uv run python -m webui.server --stub
# Open http://localhost:8000 in any browser
```

## Applications

### WebUI — browser spectrum viewer

```bash
uv sync --extra webui
uv run python -m webui.server --stub               # stub (no hardware)
uv run python -m webui.server                      # real Fobos SDR device
uv run python -m webui.server --host 0.0.0.0       # accessible from Windows browser / LAN
```

Open `http://localhost:8000`. Live panels:

| Panel | Description |
|-------|-------------|
| Spectrum | FFT power vs frequency (dB), absolute axis labels |
| Waterfall | Scrolling heatmap (plasma colormap) |
| IQ Constellation | I/Q scatter for last frame |

Controls: center frequency, sample rate, LNA/VGA gain, FFT size.
Toggle between server-side FFT (numpy) and client-side FFT ([WebFFT](https://github.com/IQEngine/WebFFT)).

See [doc/webui.md](doc/webui.md) for full details.

### Spectrum Analyzer (desktop)

Requires X server (not needed in WSL2 without VcXsrv/WSLg).

```bash
uv run python run_rtanalyzer.py
```

### FM Receiver

```bash
uv run python -m fmreceiver.fobos_fm_receiver -f 95.5
```

## API Example

```python
from pfobos import FobosSDR

with FobosSDR() as sdr:
    sdr.open(0)
    sdr.set_frequency(100e6)
    sdr.set_samplerate(2.048e6)
    sdr.set_lna_gain(1)
    sdr.set_vga_gain(10)
    sdr.start_rx_sync(32768)
    iq = sdr.read_rx_sync()   # numpy complex64 array
    sdr.stop_rx_sync()
    print(f"Got {len(iq)} IQ samples")
```

## Testing

```bash
# No hardware
uv run python run_tests.py               # mock + logic tests
uv run python tests/stub/build.py        # build stub once
uv run pytest tests/test_stub_integration.py -v

# Hardware required
uv run python run_tests.py --integration
uv run python run_tests.py --performance-only
uv run python run_tests.py --benchmark
```

See [doc/tests.md](doc/tests.md) for full testing documentation.

## Project Structure

```
pfobos/
├── pfobos/fwrapper.py        # Core: FobosSDR class (CFFI bindings) — shipped in wheel
├── webui/                    # Browser spectrum viewer (FastAPI + WebSocket)
│   ├── server.py             # FastAPI app + REST + WebSocket /ws
│   ├── sdr_worker.py         # Background IQ reader + FFT → asyncio queue
│   └── static/               # index.html, app.js (Canvas 2D + WebFFT)
├── rtanalyzer/               # Real-time spectrum analyzer (matplotlib)
├── fmreceiver/               # FM demodulation + audio output
├── tests/                    # Tests, benchmarks, C stub library
│   └── stub/                 # libfobos_stub.c → libfobos.so (no hardware)
├── scripts/                  # Dev/debug helpers
├── setup/                    # udev rules installer
└── doc/                      # Per-component documentation
```

## Documentation

| Doc | Description |
|-----|-------------|
| [doc/webui.md](doc/webui.md) | WebUI server, API, frontend architecture |
| [doc/fwrapper.md](doc/fwrapper.md) | FobosSDR API reference |
| [doc/rtanalyzer.md](doc/rtanalyzer.md) | Desktop spectrum analyzer |
| [doc/tests.md](doc/tests.md) | Testing infrastructure |
| [doc/benchmark.md](doc/benchmark.md) | Benchmark tool |
| [doc/structure.md](doc/structure.md) | Full project structure |
| [doc/aarch64-build.md](doc/aarch64-build.md) | aarch64 (OPI5 / Jetson) libfobos + SoapyFobosSDR builder |
| [doc/edge-runtime-deployment.md](doc/edge-runtime-deployment.md) | `libfobos.so` install paths for edge runtime images (watchtower-edge contract) |

## License

MIT — see [LICENSE](LICENSE).
