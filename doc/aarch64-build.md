# Fobos SDR — aarch64 build (OrangePi 5 Max / RK3588)

## What this produces

`libfobos.so` and `libfobos_sdr.so` built for `linux/arm64` (always). Optionally
`FobosSDRSupport.so` (SoapySDR `driver=fobos`) when built with `--with-soapy`.
Artifact lives in `dist/fobos-aarch64/` after either build method.

After the `pfobos-as-primary-sdr-backend` cutover, `watchtower-edge` uses the
native `pfobos` CFFI wrapper and no longer ships SoapySDR — so the default build
is **WITH_SOAPY=0**. Opt into the SoapyFobosSDR module only for the
`sdr.backend: soapy` fallback path, the §3.5 regression smoke, or external
Soapy-based tooling.

## Build methods

### 1. Docker (recommended — no cross-toolchain needed)

```bash
# One-time QEMU setup (if not already done):
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

# Default — pfobos native backend only (no SoapyFobosSDR):
./setup/aarch64/build-docker.sh

# With SoapyFobosSDR module (for soapy fallback / §3.5 regression smoke):
./setup/aarch64/build-docker.sh --with-soapy
```

Takes ~5–10 min on first run (QEMU-emulated arm64). Subsequent runs use layer cache.
The `--with-soapy` build adds ~1–2 min for the extra SoapyFobosSDR cmake step and
pulls `libsoapysdr-dev` + `soapysdr-tools` into the builder image (does not affect
the final artifact size — only `FobosSDRSupport.so` is exported).

### 2. Cross-compile (faster, no Docker required)

```bash
# Install multiarch deps:
sudo dpkg --add-architecture arm64
sudo apt-get update
sudo apt-get install -y \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
    libusb-1.0-0-dev:arm64 \
    libsoapysdr-dev:arm64

# Build:
./setup/aarch64/build-cross.sh
```

## Artifact layout

WITH_SOAPY=0 (default):
```
dist/fobos-aarch64/
├── lib/
│   ├── libfobos.so → libfobos.so.X.Y.Z
│   └── libfobos_sdr.so → libfobos_sdr.so.X.Y.Z
├── include/
│   ├── fobos.h
│   └── fobos_sdr.h
├── udev/
│   └── fobos-sdr.rules
└── VERSIONS
```

WITH_SOAPY=1 adds:
```
├── lib/
│   └── SoapySDR/modules0.8/FobosSDRSupport.so
└── soapy-probe.log
```

The `VERSIONS` manifest always records the `WITH_SOAPY` flag and SoapyFobosSDR
commit (when built) so downstream consumers can verify what's in the artifact.

## Embedded-agent Dockerfile integration

The `build-docker.sh` script prints the matching Dockerfile snippet for the
build mode at the end of its run. Quick reference:

### WITH_SOAPY=0 (pfobos native, default)
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY dist/fobos-aarch64/lib/ /usr/local/lib/
COPY dist/fobos-aarch64/include/ /usr/local/include/
RUN ldconfig
```

### WITH_SOAPY=1 (SoapyFobosSDR fallback included)
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsoapysdr0.8 \
        soapysdr-tools \
        libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY dist/fobos-aarch64/lib/ /usr/local/lib/
COPY dist/fobos-aarch64/include/ /usr/local/include/
RUN ldconfig

# Verify module registration (no hardware needed)
RUN SoapySDRUtil --find="driver=fobos" 2>&1 || true
```

See `doc/edge-runtime-deployment.md` for the full deployment contract.

## Board-specific kernel notes (OrangePi 5 Max / RK3588)

### USB subsystem
- Fobos USB VID:PID is `16d0:132e` (RigExpert)
- The RK3588 USB 3.0 host controller (`xhci-hcd`) supports the required bulk transfer rate
- At 50 MSPS / 14-bit ADC: wire throughput ≈ **700 Mbit/s** — within USB 3.0 spec but close
  to practical limit; USB 2.0 fallback (~480 Mbit/s) is **insufficient** at 50 MSPS

### udev rules (bare-metal only — not needed inside Docker)
```bash
sudo cp dist/fobos-aarch64/udev/fobos-sdr.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```
Rule grants access to `plugdev` group: `SUBSYSTEMS=="usb", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="132e", MODE="0666", GROUP="plugdev"`

### Kernel version
Tested target: Ubuntu 22.04 arm64 (kernel 5.15+, as shipped on OrangePi 5 Max).
The libusb-1.0 backend uses the kernel's USB device filesystem (`usbfs`), no custom driver needed.

### Docker USB passthrough
To use real hardware inside Docker on the OrangePi:
```bash
docker run --device /dev/bus/usb/... ...
# or pass the entire USB bus:
docker run --privileged -v /dev/bus/usb:/dev/bus/usb ...
```

## Validation steps (requires hardware)

```bash
# On OrangePi with Fobos connected:
SoapySDRUtil --find="driver=fobos"
SoapySDRUtil --probe="driver=fobos"

# Confirm 50 MSPS is in sample_rate range:
SoapySDRUtil --probe="driver=fobos" 2>&1 | grep -i "sample"

# Python smoke-test via pfobos wrapper:
python3 -c "
from pfobos import FobosSDR
sdr = FobosSDR()
sdr.open(0)
rates = sdr.get_samplerates()
print('sample rates:', rates)
assert 50e6 in rates, f'50 MSPS not supported: {rates}'
sdr.close()
print('OK')
"
```
