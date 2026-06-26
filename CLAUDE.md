# PFobos — Claude Code Context

## Project Summary

Python wrapper for the **Fobos SDR** (Software Defined Radio) C library (`libfobos`).
Provides a Pythonic, NumPy-integrated API for spectrum analysis and FM demodulation.
Platform: Linux (WSL2), Python 3.7+, hardware USB SDR device.

## Architecture

```
pfobos/
├── pfobos/fwrapper.py        # Core: FobosSDR class (CFFI bindings to libfobos) — shipped in wheel
├── pfobos/__init__.py        # Exports FobosSDR, FobosException, FobosError
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
│   │   └── app.js            # WebSocket client + Canvas 2D + WebFFT
│   └── __main__.py
├── scripts/                  # Dev/debug helper scripts
├── setup/setup-fobos-sdr.sh  # udev rules setup (Linux)
├── doc/                      # Markdown docs per component
├── run_tests.py              # Main test runner (argparse CLI)
├── run_rtanalyzer.py         # Launch spectrum analyzer
├── run_setup.py              # Setup verification
└── pyproject.toml            # core: numpy + cffi only; extras: [apps], [audio], [webui]
```

## Core Class: FobosSDR (`pfobos/fwrapper.py`)

Importable as `from pfobos import FobosSDR, FobosException, FobosError`.

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
from pfobos import FobosSDR
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

- **Imports**: All consumers import from `pfobos` (`from pfobos import FobosSDR, FobosException`). Legacy `shared.fwrapper` and `fobos_wrapper` paths are gone.
- **Error handling**: Always catch `FobosException`; never swallow in finally (log only)
- **Buffer sizes**: Must be even (I/Q pairs); minimum 1024 floats
- **Context manager**: `FobosSDR` supports `with` statement → auto `close()`
- **No legacy methods**: `start_rx_async_legacy`, `stop_rx_async_legacy*` have been removed

## Known Issues / Tech Debt

- `stop_rx_async` has a 5s timeout + polling loop; can block on USB cancel
- `read_rx_sync` and async callbacks use `np.frombuffer(ffi.buffer(...)).copy()` for fast C→numpy transfer
- `rtanalyzer.py` saves to `spectrum_plots/` directory (excluded from git)

## Package Manager: uv

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync                  # core deps only: numpy + cffi (slim runtime profile)
uv sync --extra apps     # add scipy/matplotlib/pandas/tabulate for rtanalyzer + benchmarks
uv sync --extra audio    # add sounddevice + pulsectl for fmreceiver
uv sync --extra webui    # add fastapi + uvicorn for webui/
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

## WebUI (`webui/`)

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
- Spectrum: Canvas 2D line chart (dB vs Hz, no external deps)
- Waterfall: Canvas 2D heatmap (plasma colormap, scrolling)
- IQ Constellation: Canvas 2D scatter (last 2048 IQ pairs)
- WebFFT: client-side FFT via webfft npm package (toggle on/off)


<!-- otaman:begin -->
## Otaman Orchestration Rules

**You are `hardware-agent`**. You own this repository: **pfobos**.

Otaman folder: `../watchtower-otaman/` (contains `.agents/`, `platform.yaml`, bus messages)

### First Session Checklist
1. Run `otaman check` (Bash) — see pending bus messages. The CLI auto-detects project root, your agent identity, and ack status. No MCP tool-loading needed for this hot path; pre-allowed in `.claude/settings.local.json`.
2. Read `../watchtower-otaman/.agents/queue/hardware-agent.md` — see your active/queued/blocked tasks
3. Read specs relevant to your repo (specs_dir paths below)
4. Run `git log --oneline -10` — understand recent changes
5. If `../watchtower-otaman/.agents/knowledge/` exists, check for tech docs relevant to your work
6. Then: resume active task, or pick highest-priority queued task, or act on bus messages

### Ownership
- This repo (`../pfobos`) is YOURS — you may read and write freely here
- Other repos (READ-ONLY, do not write to them):
  - detectmod (../detectmod) — owned by **classification-agent** (READ-ONLY)
  - watchtower-edge (../watchtower-edge) — owned by **embedded-agent** (READ-ONLY)
  - watchtower-sdr-probe (../watchtower-sdr-probe) — owned by **probe-agent** (READ-ONLY)
  - watchtower-fusion (../watchtower-fusion) — owned by **fusion-agent** (READ-ONLY)
  - watchtower-specs (../watchtower-specs) — owned by **spec-agent** (READ-ONLY)
  - watchtower-synthetic (../watchtower-synthetic) — owned by **synthetic-agent** (READ-ONLY)
  - watchtower-tactiq (../watchtower-tactiq) — owned by **tactiq-agent** (READ-ONLY)
  - watchtower-e2e (../watchtower-e2e) — owned by **e2e-agent** (READ-ONLY)
  - watchtower-train (../watchtower-train) — owned by **train-agent** (READ-ONLY)
  - watchtower-citadel (../watchtower-citadel) — owned by **citadel-agent** (READ-ONLY)
  - watchtower-grants (../watchtower-grants) — owned by **grants-agent** (READ-ONLY)
  - watchtower-image-builds (../watchtower-image-builds) — owned by **image-build-agent** (READ-ONLY)
- You may read other repos' source code, configs, and CLAUDE.md to understand their APIs
- If you need a change in another repo, send a `task-assignment` or `question` message to its owner


### Communication — Bash CLI for hot path, MCP for richer ops

Hot-path commands (frequent, read-mostly) — use the `otaman` Bash CLI, pre-allowed in this repo's settings:
- `otaman check` — list pending messages for you (auto-detects identity)
- `otaman ack <msg-stem>` — acknowledge a message (default: resolved; `--read` keeps it visible)
- `otaman status` — project-wide summary
- `otaman complete <change-name> --all` — mark OpenSpec tasks complete + broadcast task-complete
- `otaman propose <title>` — propose a spec change (pending human approval)
- Read `.agents/queue/<your-agent>.md` directly for your task queue (no CLI subcommand needed)
- Read `.agents/blocked/<your-agent>.md` directly for blocked-task tracking

Richer / less-frequent ops — use MCP tools (load schemas with ToolSearch first when calling directly):
- `otaman_send(cwd, to, subject, body)` — send a message to another agent
- `otaman_read_message(cwd, message_stem)` — read full message content programmatically
- `otaman_propose(cwd, title, what_needs_to_change, why_needed)` — propose a spec change
- `otaman_complete(cwd, change_name, tasks)` — report task completion
- `otaman_read_spec(cwd, spec_path)` — read spec files
- `otaman_list_agents(cwd)`, `otaman_set_agent(cwd, name)`, `otaman_cleanup(cwd)` — agent registry / housekeeping

Why the split: bus checks happen dozens of times per session, and the MCP-via-instruction path proved unreliable across model variants (2026-04-29 incident — see plugin CLAUDE.md). The Bash CLI is deterministic. Heavier write operations stay on MCP because their structured payload is worth the schema-load overhead.

### Bus Awareness (CRITICAL)
- **Check the bus proactively** — do NOT wait for the human to tell you:
  - After completing each task (feature done, test passing)
  - Before starting a new task from your queue
  - When idle or waiting for anything
  - After every 3-5 tool calls during active work
- **Never let pending messages exceed 3 without acting**
- When you change an API or shared type: send `contract-change` via `otaman_send` BEFORE committing
- Message handling while busy: ack as `read`, add to queue, finish current task first
- Urgent messages: pause current work, inform the human immediately

### Outcome Proposals (business-impact ideas)

When you spot a business-impact idea — a pricing change, a process change, a
new outcome the program should pursue — send it as an **outcome-proposal**,
not as `info`:

```
otaman send --type outcome-proposal --to human --subject "<short hook>"
```

Strategic agents (cofounder-agent, cpo-agent, and any others named in the
project's `bus.routing_rules`) are auto-notified via CC — you do not list
them manually. The primary delivery stays addressed to `human` for sign-off.

- Use this type whenever your subject mentions business impact, a proposed
  outcome, a market move, or a structural change to how the program is run.
- Do **not** use `--type info` for outcome statements; they get lost in the
  general bus noise and skip the strategic CC fan-out.
- Implementation tasks, status updates, and routine FYIs stay on `info` /
  `question` / `task-complete` as before.

### Agent Status (REQUIRED)

Before writing any code for a specific task, call:
```
otaman set-status working --task "<N.M task description>" --change <change-name>
```

When waiting on another agent or a dependency:
```
otaman set-status waiting --task "<N.M ...>" --change <change-name>
```

When done with all current tasks:
```
otaman set-status idle
```

This is a single CLI call — no file editing, no token overhead. It lets the human see live fleet state in `otaman status` and in `otaman check`. Per `agent-status-presence` design Q3.

### Task Queue
- Your queue file: `../watchtower-otaman/.agents/queue/hardware-agent.md`
- Max 1 active task at a time — finish or pause before switching
- When a `task-assignment` arrives while you're busy: ack as `read`, add to Queued section
- When you finish a task: check bus, then pick highest-priority queued item
- Urgent messages override: pause active task, handle urgent item

### Task Completion Reporting (CRITICAL)
- When you finish tasks from a `task-assignment`, you MUST report completion:
  - `otaman complete <change-name> --tasks "2.1, 2.3"` (specific tasks)
  - `otaman complete <change-name> --all` (all tasks for that change)
- This updates `tasks.md` checkboxes in the specs repo and sends a `task-complete` bus message
- **Lifecycle**: task-assignment received -> ack "read" -> implement -> `otaman complete` -> ack "resolved"
- NEVER ack a task-assignment as "resolved" without first running `otaman complete`

### Specs (OpenSpec)
- Specs repo: `../watchtower-specs` (READ-ONLY)
- Your spec area is not yet mapped — check `../watchtower-specs/openspec/specs/` for relevant folders
- **Shared contracts**: `../watchtower-specs/openspec/specs/shared-contracts/spec.md` — message schemas, signal classes, security contracts
- **Active changes for you**: scan `../watchtower-specs/openspec/changes/` for folders whose `tasks.md` references your repo or domain. Read `proposal.md` → `design.md` → `tasks.md` in each.
- **All accumulated specs**: `../watchtower-specs/openspec/specs/`
- To propose a spec change, use `/otaman:propose` — do NOT modify specs directly

### Spec Change Rules (CRITICAL)
- If you discover a missing endpoint, contract gap, or any spec change needed: run `/otaman:propose`, then **STOP** working on that feature
- **Never implement against a spec that doesn't exist yet** — wait for human approval + spec commit
- After proposing, switch to other tasks. Run `/otaman:check` periodically to see if your proposal was approved
- Resume the blocked task only after you see BOTH `spec-change-approved` AND `spec-change` messages
- Check `../watchtower-otaman/.agents/blocked/hardware-agent.md` for your currently blocked tasks

- **Branching**: gitflow
- **Commits**: conventional format



### Git Workflow
- Work in branches: `agent/hardware-agent/{feature-name}`
- All changes go through PRs
- Write clear commit messages for the audit trail
<!-- otaman:end -->
