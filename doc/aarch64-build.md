# Fobos SDR — aarch64 build (OrangePi 5 Max / RK3588)

## What this produces

`libfobos.so`, `libfobos_sdr.so`, and `FobosSDRSupport.so` (SoapySDR `driver=fobos`)
built for `linux/arm64`. Artifact lives in `dist/fobos-aarch64/` after either build method.

## Build methods

### 1. Docker (recommended — no cross-toolchain needed)

```bash
# One-time QEMU setup (if not already done):
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

# Build:
./setup/aarch64/build-docker.sh
```

Takes ~5–10 min on first run (QEMU-emulated arm64). Subsequent runs use layer cache.

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

```
dist/fobos-aarch64/
├── lib/
│   ├── libfobos.so → libfobos.so.X.Y.Z
│   ├── libfobos_sdr.so → libfobos_sdr.so.X.Y.Z
│   └── SoapySDR/modules0.8/FobosSDRSupport.so
├── include/
│   ├── fobos.h
│   └── fobos_sdr.h
├── udev/
│   └── fobos-sdr.rules
└── VERSIONS
```

## Embedded-agent Dockerfile integration

Add to `watchtower-edge/Dockerfile` (OrangePi target stage):

```dockerfile
# SoapySDR runtime + USB
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsoapysdr0.8 \
        soapysdr-tools \
        libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Fobos libs + SoapySDR module
COPY dist/fobos-aarch64/lib/ /usr/local/lib/
COPY dist/fobos-aarch64/include/ /usr/local/include/
RUN ldconfig

# Verify module registration (no hardware needed)
RUN SoapySDRUtil --find="driver=fobos" 2>&1 || true
```

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
from shared.fwrapper import FobosSDR
sdr = FobosSDR()
sdr.open(0)
rates = sdr.get_samplerates()
print('sample rates:', rates)
assert 50e6 in rates, f'50 MSPS not supported: {rates}'
sdr.close()
print('OK')
"
```
