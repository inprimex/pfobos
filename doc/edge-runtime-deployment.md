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

### Sustained sync streaming has a libusb-9 cliff around 10–22 s

`fobos_rx_read_sync` returns error `-9` (libusb error) after sustained
streaming for ~10 s at 16 MSPS or ~22 s at 10 MSPS on the OPI5 +
xhci-hcd combo. No corresponding kernel-side USB errors in dmesg —
this is at user-space in libfobos's internal libusb bulk-transfer
state machine.

Suspect cause: synchronous file I/O inside the inner read loop chokes
the USB pipeline. A consumer that writes to disk while reading IQ
should use a background writer thread with a queue to decouple disk
syscalls from the USB read path. The wrapper smoke test in
`bench/wrapper_smoke.py` (no inside-loop writes) does NOT hit this at
the same rates and durations, lending support to the I/O-choking-USB
hypothesis.

Not yet diagnosed at the libfobos level. Workarounds for now: keep
sustained reads under ~20 s, or run a background writer queue, or
both. A multi-band bulk capture session needs this resolved before it
runs.

### `pfobos/fwrapper.py:read_rx_sync` auto-stops sync mode on transient errors

When the underlying C call returns a non-zero status, the wrapper
calls `self.stop_rx_sync()` (setting `_sync_mode = False`) BEFORE
re-raising as `FobosException`. A consumer that catches the
`FobosException` and tries to keep reading will then get
`RuntimeError: Synchronous mode not started` on the next call — sync
state has already been torn down inside the exception handler.

Workaround pattern:

```python
try:
    iq = sdr.read_rx_sync()
except FobosException:
    # Wrapper auto-stopped sync; re-arm if you want to continue
    if not sdr._sync_mode:
        sdr.start_rx_sync(BUF_FLOATS)
    continue
```

A real fix that drops the auto-stop semantics would be a breaking API
change for existing consumers (rtanalyzer, fmreceiver, FobosIQSource);
see the 2026-06-16 hardware-agent findings memo if you need to file
the proposal.

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
