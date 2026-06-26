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

### USB port choice on OPI5 matters — not all USB 3.0 ports are equal

Validated 2026-06-18 on OPI5-1: two of the OPI5 Max's USB 3.0 SuperSpeed
ports (both nominally 5 Gbps) deliver materially different sustained
throughput for the Fobos workload. Same SoC, same Fobos device, same
`bench/profile_wrapper.py --mode async --rate 50e6 --duration 30`,
edge container paused:

| xhci controller | IRQ | bus# (at measurement time) | sustained eff @ 50 MSPS |
|---|---:|---:|---:|
| **xhci-hcd.12.auto** (fast) | **87** | Bus 005 (initial) / Bus 004 (post-reboot) | **24–28 MSPS (48–57%)** |
| xhci-hcd.11.auto (slow) | 86 | Bus 002 | ~6.6 MSPS (13%) |

The 2-4× difference isn't a wrapper or libfobos issue — both ports
enumerate as SuperSpeed at the link layer; the controllers differ in
practical DMA / DRAM / fabric throughput. Possible causes (not pinned
down):

- One xhci controller shares DRAM bandwidth with PCIe / eMMC / NPU
- Different DMA engine priorities at the SoC interconnect
- Different USB phy quality between the two physical ports

**Bus numbers can rotate across reboots.** Confirmed 2026-06-19:
after a power-cycle the Fobos moved from Bus 005 to Bus 004 without
the cable being touched — kernel re-enumerated the xhci controllers
in a different order. The PHYSICAL port and CONTROLLER didn't change;
only the bus number label. **Identify by xhci controller IRQ name,
not by bus number**, when validating which port a device is on.

How to identify which xhci controller a device is on (robust across
reboots):

```bash
# Find the bus the Fobos is on
lsusb | grep -i fobos
# → Bus 004 Device 002: ID 16d0:132e MCS Fobos SDR (this run)
# → may be Bus 005 or another after next reboot — bus number is not stable

# Look up which xhci controller serves that bus via /proc/interrupts.
# The "xhci-hcd:usbN" name in the IRQ description is the controller's
# USB 2.0 root-hub name, and is STABLE across reboots — the controller
# always advertises the same pair of {USB 2.0 hub, USB 3.0 hub}.
grep -i xhci /proc/interrupts
#  86: ... GICv3 252 Level  xhci-hcd:usb1   ← slower controller
#  87: ... GICv3 253 Level  xhci-hcd:usb3   ← faster controller (where Fobos should be)
```

For OPI5 Max specifically, **prefer the USB 3.0 port served by
`xhci-hcd:usb3` (IRQ 87, kernel name xhci-hcd.12.auto)** for any SDR
workload. The blue USB 3.0 port physically further from the HDMI
socket on the operator's unit was the faster one in our testing;
verify on yours before committing.

If the Fobos lands on the wrong controller after a reboot (lsusb
shows it on the bus paired with IRQ 86 = xhci-hcd:usb1), unplug-and-
replug into the OTHER physical USB 3.0 port. Bus number may differ
again; the IRQ-name check is the source of truth.

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

## Storage class for IQ captures

**Policy (operator-confirmed 2026-06-17):** the OPI5's SD card
(`/dev/mmcblk1p1`, Class 10 ext4) is for the OS and small writes only.
Bulk IQ captures must NOT land on it. Sustained-write benchmarks at
16 MSPS hit the SD card's sequential-write ceiling
(~30 MB/s effective) well below libfobos's 128 MB/s sustained data
rate, the writer queue fills, and the read pipeline blocks long enough
to trigger the libusb-9 cliff at ~63 s wall-clock (despite the
background-writer queue in `bench/observation_mode_research/capture_noise.py`
that decouples the libfobos read from the disk syscall).

### Two distinct storage roles

The OPI5 plays two different roles in this project and they have very
different storage needs:

| role | what it does | working storage need |
|---|---|---|
| **Deployed edge node** | runs `watchtower-edge` inference: reads IQ via `FobosIQSource`, runs WST → CFAR → ML pipeline, emits `DetectionReport` over MQTT. IQ is processed and discarded; only model weights, logs, detection metadata, and occasional triggered captures land on disk. | **~50 GB working space** — 128–256 GB drive is ample |
| **Research / dev bench** | runs bulk noise corpus captures, calibration sweeps, ML offline analysis. Sustained IQ-to-disk for minutes at a time. | **~250+ GB working space** — 1 TB+ drive recommended, sustained write critical |

The libusb-9 cliff problem is bench territory. Production edge nodes
don't sustain-write IQ to disk in the inference path, so the cliff
doesn't apply to them. The storage class table below reflects this:

| storage | sustained write | edge node | research bench |
|---|---|---|---|
| SD card (Class 10 / mmcblk1) | ~30 MB/s | ✗ OS+small writes only | ✗ |
| tmpfs (RAM-backed) | ≫1 GB/s | ✗ no persistence | ✓ short sessions, ≤RAM size |
| eMMC | ~100–200 MB/s | ✓ situational, board-dependent | ✗ insufficient sustained |
| M.2 NVMe (256 GB, Apacer-class) | ~400 MB/s sustained, ~1.3 GB/s burst | ✓ ample at any tier | ✓ adequate for 16 MSPS |
| M.2 NVMe (1 TB, Apacer-class) | ~800 MB/s sustained, ~1.5 GB/s burst | ✓ overspecced but fine | ✓ comfortable at 50 MSPS |

### Validated / target hardware

For the dev bench at the time of writing, the target drive is
**Apacer AS2280P4-1 256 GB (`AP256GAS2280P4-1`)** — industrial-grade
M.2 2280 NVMe, PCIe Gen 3 x4, DRAM cache, TLC NAND, -40 to +85 °C
operating range, optional PLP variant available.

Same drive ports to a field-deployed edge node without re-spec'ing —
the industrial temp range + PLP option future-proof the choice across
both roles.

#### Predicted speedup vs SD card (pending hardware validation)

Apacer datasheet specs for the 256 GB SKU + workload analysis:

| metric | SD card (measured 2026-06-17) | Apacer AS2280P4-1 256 GB (predicted) | speedup |
|---|---|---|---|
| Sequential write burst | n/a | ~1300 MB/s | — |
| Sustained write after SLC cache exhausts | 30 MB/s | ~400 MB/s (estimated, capacity-bound on 256 GB SKU) | **~13×** |
| 16 MSPS / 60 s capture | `ok: false`, 23.9% completion, cliff at 63 s | `ok: true`, ~100% completion, **no cliff expected** | qualitative — unlocks the workload |
| 50 MSPS / 60 s capture | cliff at 2.5–10 s (chunk-size-dependent, per embedded-agent `20260616T222139`) | 60 s clean **OR** cliff at 30–45 s | best case ~10–25× |
| 6-band E1.2/E1.3 noise-capture session (~46 GB total) | impossible | ~30–40 min straightforward session | qualitative — unlocks E1.2/E1.3 |

The 16 MSPS case is high confidence: 60 s × 128 MB/s = 7.7 GB total
write fits comfortably within the 256 GB drive's dynamic SLC cache
(~16–32 GB on Phison E12S / SMI 2263 class controllers). The drive
writes at near-burst speeds throughout, far above the 128 MB/s libfobos
produces.

The 50 MSPS case is medium confidence: 24 GB written over 60 s likely
exceeds the SLC cache mid-capture, dropping the drive to TLC native
write (~400 MB/s estimated for this capacity). At the device's 400 MB/s
sustained rate the headroom is tight — if TLC native lands below
400 MB/s, the cliff returns, just at 30–45 s instead of 10 s. If you
need rock-solid 50 MSPS sustained, bump to the 1 TB SKU; sustained
write scales roughly linearly with capacity on TLC drives because more
NAND channels run in parallel.

#### What needs validation post-install

When the drive is mounted on OPI5-1 (next hardware lease):

1. **Quick `fio` baseline** — sequential write at 4 KB and 1 MB block
   sizes, 30 s duration, confirms the OPI5's PCIe Gen 3 x4
   implementation reaches the datasheet specs (RK3588 PCIe lands at
   80–90% of theoretical typically).
2. **16 MSPS / 60 s `capture_noise.py` re-run** at 868 MHz / quiet_lab,
   same script as the SD-card baseline. Expected: `ok: true`,
   `queue_full_blocks: 0`, ~100% sample completion. Direct apples-to-
   apples vs `868MHz_16MSPS_quiet_lab_20260617T164623Z`.
3. **50 MSPS / 60 s stress test** — same band, same gain, same script.
   Maps the upper end of the sustained-write envelope. Either the cliff
   stays away (drive headroom validates) or it shifts to 30–45 s
   (TLC-native sustained measured; decide whether to upgrade SKU for
   50 MSPS work).
4. **One band of the actual E1.2/E1.3 capture per embedded-agent's
   spec** — converts validation into a deliverable. 60 s @ 16 MSPS at
   one of the spec bands (e.g. 868 MHz), full sidecar JSON, delivered
   to embedded-agent for the PD@PFA pipeline.

The four steps fit inside one ~15-minute hardware lease.

### Capture session recipes

**Dev / research session (NVMe target)** — mount the drive and run
captures directly to the mount point. No tmpfs rotation needed at
16 MSPS:

```bash
# One-time setup (operator side):
#   - Install M.2 NVMe (Apacer AS2280P4-1 or similar)
#   - mkfs.ext4 /dev/nvme0n1p1
#   - Add to /etc/fstab: /dev/nvme0n1p1 /mnt/nvme ext4 defaults,noatime 0 2
#   - sudo mount -a

uv run python bench/observation_mode_research/capture_noise.py \
    --freq 868e6 --rate 16e6 --duration 60 \
    --env quiet_lab --antenna "<descriptor>" \
    --out-dir /mnt/nvme/noise_captures
```

The `noatime` mount option matters — default `relatime` writes a 4 KB
timestamp metadata update per file open, which on a 500k-sample-per-file
workload adds up. `noatime` removes that overhead.

**Dev / research session (tmpfs fallback)** — if the NVMe isn't yet
installed, tmpfs + rsync rotation works for short windows:

```bash
sudo mkdir -p /mnt/capture-tmp
sudo mount -t tmpfs -o size=4G tmpfs /mnt/capture-tmp
uv run python bench/observation_mode_research/capture_noise.py \
    --freq 868e6 --rate 16e6 --duration 60 \
    --env quiet_lab --antenna "<descriptor>" \
    --out-dir /mnt/capture-tmp
rsync -a /mnt/capture-tmp/ \
    ~/watchtower-edge-pfobos/bench/observation_mode_research/noise_captures/
sudo umount /mnt/capture-tmp
```

Plan tmpfs size for ~7.7 GB per band at 16 MSPS / 60 s. Multi-band
sessions: rsync between bands, or scale tmpfs to the per-band budget
(OPI5 typically has 8–16 GB RAM total; don't starve the inference
pipeline).

**Production / operational** — provision the edge image with an M.2
NVMe SSD as the capture destination. The same `--out-dir` flag points
at the NVMe mount; the script doesn't care about the underlying
storage. Operator-confirmed direction; specific SSD model + mount
point are deployment-stage choices.

### Why the SD card can't be the bulk-capture target

At 16 MSPS / complex64 the device sustains 128 MB/s. The OPI5's
mmcblk1 SD card sustained-writes at ~30 MB/s on the
`868MHz_16MSPS_quiet_lab_20260617T164623Z` validation run:

```
total_samples:           240,058,368  (1.92 GB)
actual_duration_s:       62.91
effective_rate:          30.5 MB/s    ← SD bandwidth ceiling
queue_full_blocks:       202          ← writer fell behind 202×
sample_completion_pct:   23.9%
end_state:               Fobos error -9: libusb error
```

The writer-queue refactor (pfobos PR #12) decoupled the libfobos read
from the disk syscall and shifted the cliff timing 5.8× later vs the
pre-fix 10.79 s. But the cliff itself still fires — the bottleneck
moved from "syscall stalls inner loop" to "disk capacity ceiling".

### Beyond the SD card: tmpfs-as-core capture path

Operator-confirmed pattern on 2026-06-17: captures should write into
a tmpfs (RAM-backed filesystem, ≫1 GB/s) and a separate post-capture
step rsyncs to persistent storage (NVMe / NAS / S3). This is the
**core capture-path architecture, not a fallback**:

- RAM is faster than any persistent tier. NVMe at ~1 GB/s sustained
  is 10–50× slower than DDR4. Capture-to-RAM is the only writeable
  target that's never the disk-side bottleneck.
- Decouples capture wall-clock from persistence wall-clock. The SDR
  read pipeline runs at libfobos's USB delivery rate; persistence
  happens after, at whatever the storage backend supports.
- Storage-tier-agnostic. Same capture path works whether persistence
  is NVMe, SD, NAS, or S3.
- Simpler error model. If persistence fails (disk full, rsync hang,
  NAS unreachable), the IQ is in tmpfs until explicitly flushed.

`bench/observation_mode_research/capture_noise.py` is designed for this
pattern out of the box — `--out-dir` points at any directory; tmpfs
is the recommended target for capture sessions.

### The libusb-9 cliff is upstream of disk

**Validated 2026-06-17 on OPI5-1.** Running capture_noise.py against
a tmpfs target at three rates, with the hwtel module sampling thermal
zones and CPU frequency at 0.5 s intervals:

| run | rate | duration captured | total samples | eff rate | queue_full_blocks | end state |
|---|---:|---:|---:|---:|---:|---|
| tmpfs 16 MSPS | 16 | ~6 s eff | 108 M | 12.7 MSPS | **0** | libusb-9 |
| tmpfs 10 MSPS | 10 | 3.96 s | 19 M | 4.79 MSPS | **0** | libusb-9 |
| tmpfs 8 MSPS  | 8  | 21.86 s | 86 M | 3.92 MSPS | **0** | libusb-9 |

**`queue_full_blocks: 0` on every tmpfs run** — the writer queue
NEVER stalled. Tmpfs is bandwidth-unbounded relative to libfobos's
output rate. Yet libusb-9 still fires.

**The cliff is not primarily disk-bandwidth-bound.** Tmpfs proves
disk can be infinitely fast and the cliff still fires at all rates.

### Why thermal is NOT the cause

The `pfobos-noise-capture/2` schema added by pfobos PR #15 carries
thermal + CPU-freq + loadavg trajectories in the sidecar so any
performance finding can be checked against the hardware environment
that produced it. Telemetry from the 2026-06-17 8 MSPS / 21.86 s
clean session (RK3588 thermal zones, °C):

| zone | min | max | delta |
|---|---:|---:|---:|
| soc-thermal | 54.5 | 55.5 | 1.0 |
| bigcore0-thermal | 55.5 | 56.4 | 0.9 |
| bigcore1-thermal | 55.5 | 57.3 | 1.8 |
| littlecore-thermal | 55.5 | 56.4 | 0.9 |
| center-thermal | 53.6 | 55.5 | 1.9 |
| gpu-thermal | 53.6 | 54.5 | 0.9 |
| npu-thermal | 54.5 | 55.5 | 1.0 |

Max delta across all 7 zones during a libusb-9-firing capture:
**1.8 °C**. RK3588 throttles at 85 °C+. We're at less than two-thirds
of the throttle threshold and thermal barely moves under load.
**Thermal throttling is ruled out.**

### Cause confirmed: watchtower-edge container contention

Validated 2026-06-18 with two experiments back-to-back on OPI5-1.

| run | edge container | governor | end state | duration | samples |
|---|---|---|---|---:|---:|
| baseline (2026-06-17) | running | ondemand | libusb-9 | 21.86 s | 86 M |
| experiment B (perf gov) | running | **performance** | libusb-9 | **2.72 s** | 10 M |
| experiment A (edge paused) | **paused** | ondemand | **NO ERROR ✓** | **30.02 s** | **113.7 M** |

Pausing the watchtower-edge container eliminates the cliff entirely.
Capture ran to the requested 30 s deadline cleanly with
`queue_full_blocks: 0` and no `Fobos error -9`.

The contention mechanism is not yet precisely localized — could be
CPU scheduling, memory bandwidth, USB bus interference from
container-side activity, soft-IRQ context migration, or some
combination — but the **trigger** is unambiguously the container's
runtime presence on this OPI5+Fobos+libfobos+kernel combo.

### Operational pattern: docker pause for bulk captures

Recommended pattern for any bulk sustained-capture session on a host
running the watchtower-edge inference container:

```bash
docker pause watchtower-edge-edge-1
# ... bulk capture loop, all bands ...
docker unpause watchtower-edge-edge-1
```

The pause is fast (sub-second), reversible, and leaves the container's
process state intact (no restart, no reload of model weights, no
MQTT reconnect). For research / dev sessions on a node also running
inference this is the cleanest viable operational pattern.

**Production deployments are not affected.** Triggered captures in the
operational path are short bursts (≤2 s typically) that fit comfortably
within the cliff window even with the container running at full load.
Only sustained captures (≥5–20 s depending on rate) hit the cliff.

### Second-order finding: effective rate ~50% of nominal

The same experiments surfaced an independent issue: at the
**capture-paused** run that completed cleanly, the effective sample rate
was 113.7 M / 30.02 s = 3.79 MSPS for a requested 8 MSPS — **47%** of
nominal. The same shortfall (49%) appeared in the baseline-running run.

This is NOT cliff-related (same ratio whether edge runs or pauses) and
NOT thermal-related. It's an independent libfobos / Fobos / USB behavior
worth investigating later. Hypotheses to test (priority order):

1. **libfobos `actual_len` reporting** — does `fobos_rx_read_sync` set
   `actual_len[0]` to less than the requested buffer?
2. **USB transfer count per polling interval** — does
   `start_rx_sync(buf_length=N)` actually allocate the full N, or does
   libfobos halve it internally for some safety margin?
3. **Fobos firmware halving rate internally** — requires firmware-side
   instrumentation; lowest priority.

Tracked as a separate thread (embedded-agent bus `20260617T220626`),
not blocking the bulk noise-capture session.

### Hypotheses falsified along the way

Worth documenting so the next agent who hits a similar cliff knows
what's already been ruled out and doesn't repeat the experiments.

| hypothesis | experiment that killed it | evidence |
|---|---|---|
| Disk bandwidth ceiling is the primary cause | tmpfs capture (RAM target, ≫1 GB/s) | Cliff still fires at 8/10/16 MSPS on tmpfs; `queue_full_blocks: 0` on every tmpfs run |
| Thermal throttling on passive-cooled OPI5 | hwtel-instrumented capture | Max 1.8 °C delta across all 7 RK3588 zones during a cliff-firing capture; threshold is 85 °C+ |
| cpufreq scaling on xhci-handling cores | All-CPU performance governor + xhci IRQ already on cpu0 (A55 1800 MHz steady) | Performance governor made the cliff fire **8× faster** (2.72 s vs 21.86 s baseline); IRQ 87 was already landing 100% on cpu0 by default |

### Why thermal is NOT the cause (telemetry detail)

The `pfobos-noise-capture/2` schema added by pfobos PR #15 carries
thermal + CPU-freq + loadavg trajectories in the sidecar so any
performance finding can be checked against the hardware environment
that produced it. Telemetry from the 2026-06-17 8 MSPS / 21.86 s
cliff-firing run (RK3588 thermal zones, °C):

| zone | min | max | delta |
|---|---:|---:|---:|
| soc-thermal | 54.5 | 55.5 | 1.0 |
| bigcore0-thermal | 55.5 | 56.4 | 0.9 |
| bigcore1-thermal | 55.5 | 57.3 | 1.8 |
| littlecore-thermal | 55.5 | 56.4 | 0.9 |
| center-thermal | 53.6 | 55.5 | 1.9 |
| gpu-thermal | 53.6 | 54.5 | 0.9 |
| npu-thermal | 54.5 | 55.5 | 1.0 |

Max delta across all 7 zones during a libusb-9-firing capture:
**1.8 °C**. RK3588 throttles at 85 °C+. We're at less than two-thirds
of the throttle threshold and thermal barely moves under load.

The 2026-06-18 edge-paused capture confirmed independently: thermal
dropped 4 °C (50–53 °C vs 54–57 °C baseline) — the edge container is
the primary heat source on this passively-cooled OPI5, but even with
that heat, we're nowhere near the throttle threshold.

### State of the §5.3 investigation

| factor | status | evidence |
|---|---|---|
| Writer-side queue stalls disk I/O | ✓ confirmed contributing factor | SD card run had queue_full_blocks=202 |
| Disk bandwidth ceiling on SD | ✓ confirmed contributing factor on SD | 30 MB/s vs 128 MB/s demand |
| tmpfs lifts writer-side ceiling | ✓ confirmed | queue_full_blocks=0 on all rates |
| libusb-9 persists at any disk speed | ✓ confirmed | cliff fires at 8/10/16 MSPS on tmpfs |
| Thermal throttling | ✗ ruled out by telemetry | max delta 1.8 °C; threshold 85 °C+ |
| cpufreq scaling on xhci cores | ✗ ruled out by experiment | perf gov made cliff 8× WORSE, not better |
| **watchtower-edge container contention** | ✓ **CONFIRMED CAUSE** | docker pause → no cliff at 30 s |
| Effective rate ~50% of nominal | OPEN (second-order, not cliff-related) | 47–49% ratio with both edge running and paused |

## CaptureSession library API (Phase 3 §5.1 corpus capture)

`bench/observation_mode_research/capture_noise.py` exposes a library-first
API as of the 2026-06-26 refactor. Embedded callers (edge runtime
detection-metric writers, K-calibration harnesses, anything that needs
both the IQ stream AND a parallel side-effect) should use the API
directly instead of shelling out to the CLI.

### Quick usage

```python
from pathlib import Path
from bench.observation_mode_research.capture_noise import (
    CaptureConfig, CaptureChunk, run_capture,
)

def metric_writer(chunk: CaptureChunk) -> None:
    rms = float(np.abs(chunk.samples).mean())
    metrics_jsonl.write(f'{{"file_idx": {chunk.file_idx}, "rms": {rms}}}\n')

config = CaptureConfig(
    freq_hz=2.4e9,
    rate_hz=16e6,
    duration_s=30.0,
    env="field",
    antenna="2.4 GHz dipole",
    out_dir=Path("/mnt/nvme/corpus/run-001"),
    fan_out=[metric_writer],
)
result = run_capture(config)
assert result.ok, result.metadata.get("error")
```

### fan_out hook contract

- Each consumer is `Callable[[CaptureChunk], None]`
- Invoked in the **writer thread**, immediately AFTER the .iq file lands on disk
- Receives the same chunks in the same order as the .iq filenames
- Exceptions are caught, logged in `metadata['fan_out']['errors']`, and do not break the IQ-write path
- Consumer names are recorded in `metadata['fan_out']['consumers']` for the sidecar audit trail
- Heavy work in the consumer is fine — it's already off the read path. But be aware that a slow consumer increases writer-thread latency and can backpressure the bounded queue (`queue_max_chunks`, default 64)

### CaptureChunk fields

| field | type | meaning |
|---|---|---|
| `file_idx` | int | 1-based file index matching the `_partNNNN.iq` filename |
| `samples` | `np.ndarray[complex64]` | The chunk that was just written (default 500 000 IQ pairs) |
| `path` | `Path` | The .iq file that was just written (in case a consumer wants to re-open it via downstream tooling) |
| `band_label` | str | E.g. `2.4GHz`, `868MHz` |
| `base` | str | Filename base: `<band>_<rate>MSPS_<env>_<utc>` |

### CaptureResult fields

| field | meaning |
|---|---|
| `ok` | True iff the session completed without errors and was not interrupted |
| `metadata` | Full `pfobos-noise-capture/2` sidecar dict (also written to `metadata_path`) |
| `metadata_path` | Path to the `.json` sidecar |
| `iq_paths` | List of all `.iq` files written, in order |
| `total_samples` | Total IQ samples consumed from the SDR |
| `actual_duration_s` | Wall-clock duration of the read loop |
| `error` | Writer-thread exception if it died, else None |

### Storage target

Always set `out_dir` on the NVMe mount, e.g. `/mnt/nvme/corpus/<run-tag>/`.
- SD card writes (~30 MB/s sustained) are well below the 128 MB/s budget for 16 MSPS captures and will trip the libusb-9 cliff via queue backpressure.
- Apacer AS2280P4-1 256 GB at .136 was validated 2026-06-25 at 830 MB/s sequential — 60–90 s captures fit comfortably.

### Operational checklist before sustained captures

1. `docker pause watchtower-edge-edge-1` — eliminates edge-container contention (the confirmed root cause of the libusb-9 cliff at 8+ MSPS sustained)
2. Verify the Fobos is on the IRQ 87 USB port (`xhci-hcd:usb3`); see "USB port choice on OPI5 matters" above
3. For multi-phase sessions in the same process: ensure `sdr.get_board_info()` is called between `open()` and `set_samplerate()` (libfobos quirk); CaptureSession does this for you automatically
4. Confirm `out_dir` resolves under NVMe; never under `/`, `/home`, or any SD-backed path

## See also

- `setup/aarch64/Dockerfile.build-aarch64` — builder image that produces the artifact set referenced above
- `setup/aarch64/build-docker.sh` — script that runs the build and exports `lib/`, `include/`, `udev/`, `VERSIONS`
- `pfobos/fwrapper.py::FobosSDR._load_library` — the discovery code itself
- `bench/observation_mode_research/capture_noise.py` — CaptureSession library API source
- `tests/test_capture_session.py` — fan_out hook contract tests (stub-backed, hardware-free)
- OpenSpec change `pfobos-as-primary-sdr-backend` (in `watchtower-specs`) — §4.2 of `tasks.md` is this document
- Findings memo `bench/observation_mode_research/findings/findings-2026-06-16-session-1.md` — full context for the runtime gotchas above
