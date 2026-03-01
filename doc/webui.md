# WebUI — Browser Spectrum Viewer

Browser-based real-time SDR spectrum viewer built with FastAPI and WebSocket.
Runs in WSL2 without a display server — open the UI from any browser on Windows or LAN.

## Quick Start

```bash
# Install dependencies (once)
uv sync --extra webui

# Build the C stub library (once, for hardware-free mode)
uv run python tests/stub/build.py

# Run in stub mode (no Fobos SDR device needed)
uv run python -m webui.server --stub

# Run with real device
uv run python -m webui.server

# Accessible from Windows browser or LAN
uv run python -m webui.server --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in any browser.

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--stub` | off | Load `tests/stub/libfobos.so` instead of real device |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8000` | Bind port |

## REST API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | SPA frontend (`static/index.html`) |
| `GET` | `/api/config` | Current SDR configuration (JSON) |
| `POST` | `/api/config` | Apply new configuration |
| `GET` | `/api/devices` | List connected Fobos SDR devices |
| `WS` | `/ws` | Live spectrum stream |

### GET /api/config response

```json
{
  "center_freq": 100000000.0,
  "sample_rate": 2048000.0,
  "lna_gain": 1,
  "vga_gain": 10,
  "fft_size": 1024,
  "buf_length": 32768
}
```

### POST /api/config request

Send any subset of fields — only provided keys are applied:

```json
{
  "center_freq": 433920000.0,
  "sample_rate": 2048000.0,
  "lna_gain": 2,
  "vga_gain": 12,
  "fft_size": 2048
}
```

## WebSocket Frame Format

Each frame pushed to `/ws` is a JSON object:

```json
{
  "freqs":         [-1024000.0, ..., 1024000.0],
  "spectrum":      [-80.2, ..., -40.1],
  "iq_raw":        [0.01, -0.02, ...],
  "waterfall_row": [[r,g,b], [r,g,b], ...],
  "center_freq":   100000000.0,
  "sample_rate":   2048000.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `freqs` | `float[]` | Frequency offsets from DC, Hz (length = fft_size) |
| `spectrum` | `float[]` | Power in dB, length = fft_size |
| `iq_raw` | `float[]` | Interleaved I,Q float32 — exactly fft_size pairs |
| `waterfall_row` | `[[r,g,b]]` | Plasma-mapped color row, length = fft_size |
| `center_freq` | `float` | Current center frequency in Hz |
| `sample_rate` | `float` | Current sample rate in Hz |

## Architecture

### Backend

```
webui/
├── server.py       # FastAPI app: lifespan, routes, WebSocket broadcaster
├── sdr_worker.py   # Background thread: IQ → FFT → asyncio queue
└── __main__.py     # python -m webui.server entry point
```

**Data flow:**

```
FobosSDR.read_rx_sync()
    → SDRWorker._compute_frame()   [background thread]
    → asyncio.Queue (maxsize=8)    [via call_soon_threadsafe]
    → _broadcaster() task          [event loop]
    → ws.send_json(frame)          [all connected clients]
```

- The worker thread never blocks: it uses `call_soon_threadsafe` + `put_nowait`.
- If the queue is full (no WebSocket clients), the oldest frame is silently dropped.
- Config changes (`POST /api/config`) are applied at the start of the next read iteration via a thread-safe pending-config mechanism.
- Changing `sample_rate` triggers `stop_rx_sync → set_samplerate → start_rx_sync` in the worker thread.

### Frontend

```
webui/static/
├── index.html   # Bootstrap 5 SPA (CDN, no build step)
└── app.js       # WebSocket client + Canvas 2D rendering + WebFFT
```

**Rendering:**

| Panel | Implementation |
|-------|---------------|
| Spectrum | Canvas 2D — custom line + fill, grid, absolute Hz X-axis |
| Waterfall | Canvas 2D — shift rows down by 1, draw new row at top (plasma colormap) |
| IQ Constellation | Canvas 2D — scatter plot of last fft_size IQ pairs |

**FFT modes** (toggle in UI):

- **Server FFT** (default fallback): numpy `fftshift(fft(iq * hanning))`, normalized by N
- **WebFFT** (client-side): [`webfft@0.0.15`](https://github.com/IQEngine/WebFFT) loaded from CDN; receives raw `iq_raw` and computes FFT in the browser

Both modes normalize by N so the dB range is consistent (`-80` to `-10` dB typical).

### FFT Normalization

```python
# server (sdr_worker.py)
power_db = 20.0 * np.log10(np.abs(fft_out) / n + 1e-10)
```

```javascript
// client (app.js)
powerDb[i] = 20 * Math.log10(Math.sqrt(re*re + im*im) / n + 1e-10);
```

### Colormap

The waterfall uses a precomputed 256-entry plasma colormap embedded in `sdr_worker.py`.
Color mapping: `vmin = -90 dB`, `vmax = -20 dB`.

## Dependencies

Added to `pyproject.toml` as an optional group:

```toml
[project.optional-dependencies]
webui = ["fastapi>=0.100.0", "uvicorn[standard]>=0.22.0"]
```

Frontend dependencies are loaded from CDN — no build step required:

| Library | Source |
|---------|--------|
| Bootstrap 5.3 | `cdn.jsdelivr.net` |
| WebFFT 0.0.15 | `cdn.jsdelivr.net/npm/webfft` |

## Stub Mode

In stub mode (`--stub`), the server loads `tests/stub/libfobos.so` instead of the
system `libfobos.so`. The stub generates synthetic signals:

- Background noise (amplitude 0.05)
- FM signal at baseband (1 kHz audio, 75 kHz deviation, amplitude 0.8)
- CW tone at 100 kHz offset (amplitude 0.3)

Signal content is fixed regardless of `center_freq` or gain settings.
The X-axis labels do update to reflect the configured center frequency.

Build the stub:

```bash
uv run python tests/stub/build.py   # → tests/stub/libfobos.so
```
