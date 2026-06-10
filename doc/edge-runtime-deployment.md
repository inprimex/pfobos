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

## See also

- `setup/aarch64/Dockerfile.build-aarch64` — builder image that produces the artifact set referenced above
- `setup/aarch64/build-docker.sh` — script that runs the build and exports `lib/`, `include/`, `udev/`, `VERSIONS`
- `pfobos/fwrapper.py::FobosSDR._load_library` — the discovery code itself
- OpenSpec change `pfobos-as-primary-sdr-backend` (in `watchtower-specs`) — §4.2 of `tasks.md` is this document
