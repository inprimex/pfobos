"""
Async-path integration tests for Fobos SDR wrapper. Require real hardware.

Skipped automatically when no Fobos device is connected. Run with:

    uv run pytest tests/test_async_integration.py -v

These tests catch regressions in `FobosSDR.start_rx_async` / callback wrapper
that the stub tests can't surface — the async path's `buf_length` parameter
has empirical semantics that differ from what the libfobos source literally
states (see the NOTE in `FobosSDR.start_rx_async`). Hardware-side checks are
the only way to validate the wrapper's actual contract.

Surfaced from the 2026-06-18 OPI5-1 session: an attempted "fix" in pfobos
0.4.1 was empirically wrong (yielded 200% of nominal rate); reverted in
0.4.2. These tests would have caught that within seconds of attempting it.
"""
from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from pfobos import FobosSDR, FobosException


def _hardware_available() -> bool:
    try:
        sdr = FobosSDR()
        return sdr.get_device_count() > 0
    except Exception:
        return False


@unittest.skipUnless(_hardware_available(), "No Fobos SDR hardware detected")
class TestAsyncIntegration(unittest.TestCase):
    """Async-path hardware tests.

    All tests use a clean open/close lifecycle per test to keep them
    independent — async lifecycle state can leak across runs if a test
    fails mid-callback.
    """

    BUF_LENGTH = 32768
    BUF_COUNT = 16
    FREQ_HZ = 868_000_000
    LNA = 1
    VGA = 8

    def _open(self) -> FobosSDR:
        sdr = FobosSDR()
        sdr.open(0)
        return sdr

    def _close(self, sdr: FobosSDR) -> None:
        try:
            sdr.stop_rx_async()
        except Exception:
            pass
        try:
            sdr.close()
        except Exception:
            pass

    def _run_async(self, sdr: FobosSDR, rate_hz: float, duration_s: float):
        """Run async for duration_s, return (samples, callbacks, elapsed_s, first_iq)."""
        sdr.set_samplerate(rate_hz)
        sdr.set_frequency(self.FREQ_HZ)
        sdr.set_lna_gain(self.LNA)
        sdr.set_vga_gain(self.VGA)

        lock = threading.Lock()
        total_samples = [0]
        n_callbacks = [0]
        first_iq: list[np.ndarray] = []

        def cb(iq: np.ndarray) -> None:
            with lock:
                total_samples[0] += iq.size
                n_callbacks[0] += 1
                if not first_iq:
                    first_iq.append(iq.copy())

        t_thread = threading.Thread(
            target=lambda: sdr.start_rx_async(
                cb, buf_count=self.BUF_COUNT, buf_length=self.BUF_LENGTH
            ),
            daemon=True,
        )
        t0 = time.monotonic()
        t_thread.start()
        time.sleep(duration_s)
        sdr.stop_rx_async()
        t_thread.join(timeout=5)
        return total_samples[0], n_callbacks[0], time.monotonic() - t0, (first_iq[0] if first_iq else None)

    def test_async_callbacks_fire(self):
        """start_rx_async must invoke the callback at least once in 2 seconds at 8 MSPS."""
        sdr = self._open()
        try:
            samples, callbacks, _, _ = self._run_async(sdr, 8e6, 2.0)
        finally:
            self._close(sdr)
        self.assertGreater(callbacks, 0, "no async callbacks fired in 2 s")
        self.assertGreater(samples, 0, "no samples received from async callbacks")

    def test_async_first_callback_shape_and_dtype(self):
        """First callback delivers complex64 array of plausible size."""
        sdr = self._open()
        try:
            _, _, _, first = self._run_async(sdr, 8e6, 1.0)
        finally:
            self._close(sdr)
        self.assertIsNotNone(first, "no callback fired in 1 s")
        self.assertEqual(first.dtype, np.complex64)
        self.assertGreater(first.size, 0, "first callback delivered empty array")
        # Should be reasonable — order of magnitude check against BUF_LENGTH.
        # Don't assert exact value because semantics are empirically buf_length/2
        # per the wrapper NOTE; just guard against truly broken sizes.
        self.assertLessEqual(first.size, self.BUF_LENGTH * 2,
                             f"first callback delivered suspiciously large array: {first.size}")
        self.assertGreaterEqual(first.size, self.BUF_LENGTH // 4,
                                f"first callback delivered suspiciously small array: {first.size}")

    def test_async_first_callback_healthy_samples(self):
        """First callback contains real-valued IQ data (no NaN, not all-zero)."""
        sdr = self._open()
        try:
            _, _, _, first = self._run_async(sdr, 8e6, 1.0)
        finally:
            self._close(sdr)
        self.assertIsNotNone(first)
        self.assertFalse(np.any(np.isnan(first.real)), "NaN in real part of first callback")
        self.assertFalse(np.any(np.isnan(first.imag)), "NaN in imag part of first callback")
        self.assertFalse(np.all(first == 0), "first callback was all-zero")

    def test_async_effective_rate_at_8msps(self):
        """Effective sample rate at 8 MSPS should be within tolerance of nominal.

        Tolerance is wide (40-200%) to accommodate the empirical 75%-ish
        effective rate the async path delivers AND to defend against
        regressions like the 0.4.1 over-extraction (which would land at 200%).
        Tighter assertions belong in a separate parameterised sweep.
        """
        sdr = self._open()
        try:
            samples, _, elapsed, _ = self._run_async(sdr, 8e6, 3.0)
        finally:
            self._close(sdr)
        eff_pct = samples / (8e6 * elapsed) * 100
        self.assertGreater(eff_pct, 40.0,
                           f"async effective rate {eff_pct:.1f}% suspiciously low (broken?)")
        self.assertLess(eff_pct, 110.0,
                        f"async effective rate {eff_pct:.1f}% > 100% — likely the 0.4.1-class over-extraction bug")

    def test_async_effective_rate_at_16msps(self):
        """Same check at 16 MSPS — catches rate-dependent regressions."""
        sdr = self._open()
        try:
            samples, _, elapsed, _ = self._run_async(sdr, 16e6, 3.0)
        finally:
            self._close(sdr)
        eff_pct = samples / (16e6 * elapsed) * 100
        self.assertGreater(eff_pct, 40.0,
                           f"async @ 16 MSPS effective rate {eff_pct:.1f}% suspiciously low")
        self.assertLess(eff_pct, 110.0,
                        f"async @ 16 MSPS effective rate {eff_pct:.1f}% > 100% — over-extraction bug")

    def test_async_stop_clears_state(self):
        """stop_rx_async leaves the device in a state where re-opening works."""
        sdr = self._open()
        try:
            _, callbacks, _, _ = self._run_async(sdr, 8e6, 1.0)
        finally:
            self._close(sdr)
        self.assertGreater(callbacks, 0)

        # Re-open and do another short async cycle. If state leaked, this
        # would raise or hang.
        sdr2 = self._open()
        try:
            _, callbacks2, _, _ = self._run_async(sdr2, 8e6, 1.0)
        finally:
            self._close(sdr2)
        self.assertGreater(callbacks2, 0, "second async cycle failed — state leaked from first")

    def test_sync_vs_async_rms_within_order_of_magnitude(self):
        """Sync and async at the same band/gain should produce comparable RMS.

        Catches the case where the async wrapper exposes garbage data
        (e.g. the 0.4.1 over-extraction): RMS would be wildly different
        from sync because half the samples would be uninitialized memory.
        """
        sdr = self._open()
        try:
            sdr.set_samplerate(8e6)
            sdr.set_frequency(self.FREQ_HZ)
            sdr.set_lna_gain(self.LNA)
            sdr.set_vga_gain(self.VGA)
            sdr.start_rx_sync(self.BUF_LENGTH)
            sync_chunks = []
            t_end = time.monotonic() + 0.5
            while time.monotonic() < t_end:
                sync_chunks.append(sdr.read_rx_sync())
            sdr.stop_rx_sync()
        finally:
            self._close(sdr)
        sync_iq = np.concatenate(sync_chunks)
        sync_rms = float(np.sqrt(np.mean(np.abs(sync_iq) ** 2)))

        sdr2 = self._open()
        try:
            _, _, _, async_first = self._run_async(sdr2, 8e6, 1.0)
        finally:
            self._close(sdr2)
        self.assertIsNotNone(async_first)
        async_rms = float(np.sqrt(np.mean(np.abs(async_first) ** 2)))

        # Order-of-magnitude check. Sync and async on the same quiet band
        # should be within ~3x of each other; wider tolerance because each
        # path uses different DC-removal / scaling internally in libfobos.
        ratio = max(sync_rms, async_rms) / max(min(sync_rms, async_rms), 1e-12)
        self.assertLess(ratio, 10.0,
                        f"sync RMS {sync_rms:.3e} vs async RMS {async_rms:.3e} "
                        f"differ by {ratio:.1f}x — likely a wrapper bug exposing "
                        f"uninitialized memory")


if __name__ == "__main__":
    unittest.main()
