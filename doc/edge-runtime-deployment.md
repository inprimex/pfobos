# Edge Runtime Deployment — libfobos.so install paths

How the `pfobos` Python package finds `libfobos.so` at runtime, and where
the shared library must live inside the OPI5 + Jetson edge Docker images.

This is the contract between **pfobos** (this repo) and **watchtower-edge**
(the consumer that wraps `FobosSDR` in `FobosIQSource(IQSource)` per
OpenSpec change `pfobos-as-primary-sdr-backend` §2).

## TL;DR for the edge Dockerfile

Place `libfobos.so*` somewhere `ldconfig` indexes (default: `/usr/local/lib/`),
run `ldconfig` once during image build, and `pfobos.FobosSDR()` will load
it without any environment variables.

```dockerfile
# Stage: SDR runtime libs (from the pfobos aarch64 builder artifact)
COPY --from=pfobos-artifact /lib/libfobos.so*       /usr/local/lib/
COPY --from=pfobos-artifact /lib/libfobos_sdr.so*   /usr/local/lib/
COPY --from=pfobos-artifact /udev/fobos-sdr.rules   /etc/udev/rules.d/
RUN ldconfig
```

That's it. No `LD_LIBRARY_PATH`, no `FOBOS_LIB_PATH` env var.

## Discovery order inside `pfobos.FobosSDR`

`pfobos/fwrapper.py::FobosSDR._load_library()` tries, in order:

1. **Explicit path override** — when constructed as `FobosSDR(lib_path=...)`.
   Used by the test stub (`tests/stub/libfobos.so`). **Not used in production.**
2. **System loader** — `ffi.dlopen("libfobos.so")`. This goes through
   `dlopen(3)` which consults:
   - `LD_LIBRARY_PATH` (if set)
   - the cache populated by `ldconfig` (`/etc/ld.so.cache`)
   - standard system paths (`/lib`, `/usr/lib`, `/usr/local/lib`,
     and arch-specific `*/aarch64-linux-gnu/` variants on arm64)
3. **Adjacent to the package** — `<site-packages>/pfobos/libfobos.so`.
   Last-resort fallback for self-contained installs that ship the `.so`
   inside the wheel. The standard pfobos wheel does **not** ship `libfobos.so`
   (it is platform/USB-specific), so this path is normally empty.

Discovery fails with a single `OSError` naming both attempted paths.

## Where to put `libfobos.so` in the edge image

### Recommended: `/usr/local/lib/` (matches the aarch64 builder layout)

The pfobos aarch64 builder (`setup/aarch64/Dockerfile.build-aarch64`) installs
libfobos via `cmake --install`, which lands here. Mirroring this in the edge
image keeps the source-of-truth path consistent across build and runtime.

```
/usr/local/lib/libfobos.so          -> libfobos.so.0.3.x   (symlink, ABI major)
/usr/local/lib/libfobos.so.0        -> libfobos.so.0.3.x   (symlink, ABI minor)
/usr/local/lib/libfobos.so.0.3.x                            (real SONAME file)
/usr/local/lib/libfobos_sdr.so*                             (agile API, optional)
```

Run `ldconfig` after `COPY` so the `.so.0` SONAME is wired up.

### Acceptable alternatives

| Path | When to use |
|------|-------------|
| `/usr/lib/aarch64-linux-gnu/` | If you package via `dpkg-deb`; arch-specific multiarch layout |
| `/opt/fobos/lib/` (with `LD_LIBRARY_PATH=/opt/fobos/lib` or `/etc/ld.so.conf.d/fobos.conf`) | If the image needs a vendor-isolated dir |

### Not recommended

- **Bundling `libfobos.so` inside the wheel** (path 3 above) — would force
  per-platform wheels and complicates ABI tracking. Keep the wheel pure
  Python; let the OS package the `.so`.
- **Setting `LD_LIBRARY_PATH` globally** in the systemd unit — works but
  obscures the load path. Prefer `ldconfig` + standard dirs.

## udev rule (USB permissions)

Without `udev/fobos-sdr.rules` installed and udev reloaded, the container
(if running non-root) cannot open the USB device. Copy and reload:

```dockerfile
COPY --from=pfobos-artifact /udev/fobos-sdr.rules /etc/udev/rules.d/
# Inside the running container (or host udev) at first boot:
#   udevadm control --reload-rules && udevadm trigger
```

The container needs to either run as root, be in the `plugdev` group, or
have explicit `--device=/dev/bus/usb/...` exposure from the host.

## Verifying inside the running edge container

```bash
# 1. Confirm the loader can find the library
ldconfig -p | grep libfobos
#   libfobos.so.0 (libc6,AArch64) => /usr/local/lib/libfobos.so.0
#   libfobos.so   (libc6,AArch64) => /usr/local/lib/libfobos.so

# 2. Confirm the Python wrapper can open it
python -c "from pfobos import FobosSDR; FobosSDR()" \
    && echo "OK: pfobos loaded libfobos.so"

# 3. Confirm hardware is reachable (requires Fobos USB device attached)
python -c "
from pfobos import FobosSDR
sdr = FobosSDR()
print('devices:', sdr.get_device_count())
"
```

If step 2 fails with `cannot open shared object file`, the `.so` is not
on the loader path or `ldconfig` was not run after `COPY`. If step 3
returns `0` devices, the udev rule is missing or USB is not exposed
to the container.

## Source-of-truth manifest

The aarch64 builder writes a `VERSIONS` manifest naming the `libfobos`,
`libfobos-sdr-agile`, and `SoapyFobosSDR` commits used. Copy it into the
edge image (e.g. to `/usr/local/share/pfobos/VERSIONS`) so the running
container can self-report its native-lib provenance — useful for the
`pfobos backend init failure` scenario in §3.1 of the spec change
(CRITICAL log line includes `libfobos.so` path + detected FW version).

## Runtime gotchas surfaced from real deployments

The following are field-observed pitfalls from the 2026-06-16 OPI5-1
session that touched libfobos, the pfobos wrapper, and SoapySDR's Python
bindings end-to-end. None of them block deployment; all of them
are silent until they're not, so they live here rather than as a
support ticket later.

### Device degrades on uncaught crash → `usbreset 16d0:132e` to recover

When a process holding the Fobos exits uncleanly (uncaught Python
exception, SIGKILL, etc.), the device can drift into a state where the
next `FobosSDR().open(0)` succeeds but `get_board_info()` returns
garbled bytes: `hw_revision` regresses to `2.0.0`, `manufacturer` /
`product` / `serial` go blank, `fw_version` becomes `" unknown"`.
Subsequent calls flood `fobos_i2c_write() err -4` /
`fobos_max2830_write_reg() err -4` — internal RF-chip register writes
are failing.

`sdr.reset()` (which calls `fobos_rx_reset` under the hood) does NOT
recover. `usbreset 16d0:132e` (Linux user-space bus-level reset via
libusb; available as `/usr/bin/usbreset` from the `usbutils` package)
DOES.

```bash
# After an unclean exit, before any open():
usbreset 16d0:132e
sleep 2  # give libusb time to re-enumerate the device
```

dmesg shows clean SuperSpeed USB enumeration during the degraded
period, so this is a libfobos / firmware-side state-corruption issue,
not a kernel USB driver issue. The systemd unit for any edge service
that drives the Fobos should run a `usbreset` pre-start step on any
restart-after-failure path.

### Sustained sync streaming has a libusb-9 cliff (chunk-size-dependent)

`fobos_rx_read_sync` returns error `-9` (libusb error) after sustained
streaming. The cliff timing is chunk-size-dependent:

- ~10 s at 16 MSPS with `start_rx_sync(65536)` (32 768 IQ pairs per chunk)
- ~22 s at 10 MSPS with `start_rx_sync(65536)`
- ~2.5 s at 50 MSPS with `start_rx_sync(131072)` (65 536 IQ pairs)
  — observed by embedded-agent on watchtower-edge's `FobosIQSource`
  sustained-read bench, bus `20260616T222139`

Larger per-call chunks trigger the cliff sooner. Suspect: the device's
internal USB ring buffer overruns faster at larger per-call payload, OR
libfobos's buffer wrap-around hits an edge case earlier.

No corresponding kernel-side USB errors in dmesg — user-space only, in
libfobos's internal libusb bulk-transfer state machine.

Mitigations available today:

- **Background writer queue** for consumers that write IQ to disk
  while reading. `bench/observation_mode_research/capture_noise.py`
  uses this pattern — the read loop posts numpy arrays into a bounded
  `queue.Queue` and a dedicated writer thread drains it onto disk. The
  bench's wrapper smoke (`bench/wrapper_smoke.py`, no inside-loop
  writes) does NOT hit the cliff at 10 MSPS / 5 s, supporting the
  I/O-choking-USB hypothesis.
- **Short sustained windows** of <10 s if no disk write is involved.
- **Multi-segment captures spliced via the consumer-side sidecar
  `files[]` array** — embedded-agent's
  `noise_capture_ingest.py` is gap-tolerant per segment.

Not yet diagnosed at the libfobos level. A parameter sweep across
chunk_size + sample_rate + duration is queued for a future §5.3
diagnostic deep-dive if the background-writer mitigation doesn't fully
dodge the cliff.

### Post-libusb-9, libfobos internally NULLs the device handle

When `fobos_rx_read_sync` returns `-9`, libfobos's internal state
machine sets its `dev` pointer to NULL while pfobos's Python-side
`_sync_mode` flag stays `True` (correctly, per the 0.3.0 contract —
sync mode is consumer-managed). A caller that retries `read_rx_sync`
after the `-9` then gets:

    FobosException(code=-1, message="No device spesified, dev == NUL")

This is a REAL FobosException with a real C error code; it's NOT the
pre-0.3.0 spurious `RuntimeError: Synchronous mode not started`. It IS
a libfobos behavior worth understanding, because:

- The error message is confusing — the device IS specified at the
  Python level; libfobos's internal `dev` got nulled.
- A caller that wanted to ride out libusb hiccups can't, because the
  device handle is gone. The recovery path is `stop_rx_sync()` +
  `close()` + `usbreset 16d0:132e` + `open()` + `start_rx_sync()`.
- pfobos does not currently expose a "rebind dev after libfobos
  internal teardown" path. If one is needed, file a proposal.

Found 2026-06-16 by embedded-agent during the
`pfobos-read-rx-sync-no-auto-stop-on-error` §3.4 integration smoke
(bus `20260616T222139`). Not a 0.3.0 contract violation — banked as
a libfobos behavior note.

### `read_rx_sync` error-handling contract (pfobos 0.3.0+)

The pre-0.3.0 wrapper auto-stopped sync mode on any C error inside
`read_rx_sync`, then re-raised as `FobosException`. A consumer that
caught the exception and tried to keep reading hit
`RuntimeError: Synchronous mode not started` on the next call — sync
state had been torn down inside the exception handler.

**Fixed in pfobos 0.3.0** via OpenSpec change
`pfobos-read-rx-sync-no-auto-stop-on-error`. The new contract:

- `read_rx_sync` does NOT auto-stop sync mode on errors.
- The original `FobosException` propagates with its libfobos error
  code intact (no more `-1` wrapping; you see `.code == -9` for
  `LIBUSB`, etc.).
- The caller decides: retry, restart, close, or escalate.

Recommended pattern for sustained readers:

```python
sdr.start_rx_sync(buf_length)
try:
    while keep_going:
        try:
            iq = sdr.read_rx_sync()
        except FobosException as e:
            if e.code == FobosError.LIBUSB:  # -9, transient USB hiccup
                continue
            raise
        process(iq)
finally:
    sdr.stop_rx_sync()
```

If you depend on the pre-0.3.0 auto-stop side effect (you skipped
calling `stop_rx_sync` explicitly because you assumed the wrapper did
it), update your error path to call `stop_rx_sync` explicitly OR
adopt the pattern above. The migration is detailed in
`CHANGELOG.md`.

### SoapySDR.Device({"driver": "fobos"}) dict-form fails on Debian 13 SWIG bindings

If you fall back to `sdr.backend: soapy` for non-Fobos hardware OR for
debugging libfobos behind the SoapyFobosSDR module, the SoapySDR
Python binding's `Device()` constructor has a SWIG-binding bug on
Debian 13 / SoapySDR 0.8.1:

```python
# Fails on Debian 13 + python3-soapysdr 0.8.1 with:
#   SoapySDR::Device::make() no match
SoapySDR.Device({"driver": "fobos"})

# Works (kwargs-string form):
SoapySDR.Device(f"driver=fobos")
```

`SoapySDRUtil --make="driver=fobos"` (the C++ utility) opens the same
device fine, so the bug is specific to the SWIG dict→Kwargs conversion
path in `_SoapySDR.cpython-313-aarch64-linux-gnu.so`, not the underlying
driver.

Both forms are documented as valid upstream. Use the kwargs-string form
in any new code; if you maintain code that uses the dict form, switch
it on Debian 13 deployments. Reported by embedded-agent during the
§3.5 soapy-fallback smoke on 2026-06-16.

## See also

- `setup/aarch64/Dockerfile.build-aarch64` — builder image that produces the artifact set referenced above
- `setup/aarch64/build-docker.sh` — script that runs the build and exports `lib/`, `include/`, `udev/`, `VERSIONS`
- `pfobos/fwrapper.py::FobosSDR._load_library` — the discovery code itself
- OpenSpec change `pfobos-as-primary-sdr-backend` (in `watchtower-specs`) — §4.2 of `tasks.md` is this document
- Findings memo `bench/observation_mode_research/findings/findings-2026-06-16-session-1.md` — full context for the runtime gotchas above
