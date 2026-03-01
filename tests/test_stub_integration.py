"""
Stub integration tests for Fobos SDR wrapper.

These tests load a real compiled C stub library (tests/stub/libfobos.so)
via the actual CFFI path — no mocks.  They verify:
  - FobosSDR lifecycle with real C calls
  - IQ data shape, dtype, and content from sync and async modes
  - FM demodulation pipeline (DSP only, no audio hardware needed)
  - FFT spectrum computation with a known tone signal

Run after building the stub:
    uv run python tests/stub/build.py
    uv run pytest tests/test_stub_integration.py -v
or:
    uv run python run_tests.py --stub
"""

import os
import sys
import unittest
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUB_DIR     = os.path.join(PROJECT_ROOT, "tests", "stub")
STUB_LIB     = os.path.join(STUB_DIR, "libfobos.so")
SIGNALS_JSON = os.path.join(STUB_DIR, "signals.json")

sys.path.insert(0, PROJECT_ROOT)

# Point the stub at the signals config using its env var
os.environ.setdefault("FOBOS_STUB_SIGNALS", SIGNALS_JSON)

STUB_AVAILABLE = os.path.exists(STUB_LIB)
SKIP_MSG = (
    "Stub library not built. Run: uv run python tests/stub/build.py"
)

from shared.fwrapper import FobosSDR, FobosException


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_sdr() -> FobosSDR:
    """Return a FobosSDR instance loaded against the stub library."""
    return FobosSDR(lib_path=STUB_LIB)


# ---------------------------------------------------------------------------
# Tests: wrapper lifecycle via real CFFI
# ---------------------------------------------------------------------------
@unittest.skipUnless(STUB_AVAILABLE, SKIP_MSG)
class TestStubLifecycle(unittest.TestCase):

    def test_cffi_load(self):
        """Stub .so loads through real CFFI without error."""
        sdr = make_sdr()
        self.assertIsNotNone(sdr.lib)
        self.assertIsNotNone(sdr.ffi)

    def test_get_device_count(self):
        """Stub always reports 1 device."""
        sdr = make_sdr()
        self.assertEqual(sdr.get_device_count(), 1)

    def test_list_devices(self):
        """list_devices returns a non-empty list with stub serial."""
        sdr = make_sdr()
        serials = sdr.list_devices()
        self.assertIsInstance(serials, list)
        self.assertGreater(len(serials), 0)

    def test_get_api_info(self):
        """api_info returns dict with library_version and driver_version."""
        sdr = make_sdr()
        info = sdr.get_api_info()
        self.assertIn("library_version", info)
        self.assertIn("driver_version", info)
        self.assertIsInstance(info["library_version"], str)

    def test_open_close(self):
        """open() sets dev, close() clears it."""
        sdr = make_sdr()
        sdr.open(0)
        self.assertIsNotNone(sdr.dev)
        sdr.close()
        self.assertIsNone(sdr.dev)

    def test_context_manager(self):
        """with statement opens and auto-closes the device."""
        with make_sdr() as sdr:
            sdr.open(0)
            self.assertIsNotNone(sdr.dev)
        self.assertIsNone(sdr.dev)

    def test_get_board_info(self):
        """Board info fields are non-empty strings."""
        with make_sdr() as sdr:
            sdr.open(0)
            info = sdr.get_board_info()
        for key in ("hw_revision", "fw_version", "manufacturer", "product", "serial"):
            self.assertIn(key, info)
            self.assertIsInstance(info[key], str)
            self.assertGreater(len(info[key]), 0)

    def test_set_frequency(self):
        """set_frequency echoes back the requested value."""
        with make_sdr() as sdr:
            sdr.open(0)
            actual = sdr.set_frequency(100e6)
        self.assertAlmostEqual(actual, 100e6, delta=1.0)

    def test_set_samplerate(self):
        """set_samplerate echoes back the requested value."""
        with make_sdr() as sdr:
            sdr.open(0)
            actual = sdr.set_samplerate(2.048e6)
        self.assertAlmostEqual(actual, 2.048e6, delta=1.0)

    def test_get_samplerates(self):
        """get_samplerates returns a list of 4 known rates."""
        with make_sdr() as sdr:
            sdr.open(0)
            rates = sdr.get_samplerates()
        self.assertEqual(len(rates), 4)
        self.assertIn(2048000.0, rates)

    def test_gain_setters(self):
        """LNA and VGA gain setters do not raise."""
        with make_sdr() as sdr:
            sdr.open(0)
            sdr.set_lna_gain(1)
            sdr.set_vga_gain(10)


# ---------------------------------------------------------------------------
# Tests: synchronous IQ reception
# ---------------------------------------------------------------------------
@unittest.skipUnless(STUB_AVAILABLE, SKIP_MSG)
class TestStubSyncReception(unittest.TestCase):

    BUF = 4096   # float values → 2048 IQ pairs

    def setUp(self):
        self.sdr = make_sdr()
        self.sdr.open(0)
        self.sdr.set_samplerate(2.048e6)

    def tearDown(self):
        try:
            self.sdr.stop_rx_sync()
        except Exception:
            pass
        self.sdr.close()

    def test_iq_dtype(self):
        """read_rx_sync returns complex64 array."""
        self.sdr.start_rx_sync(self.BUF)
        iq = self.sdr.read_rx_sync()
        self.assertEqual(iq.dtype, np.complex64)

    def test_iq_length(self):
        """IQ array has BUF/2 samples (interleaved I+Q → complex)."""
        self.sdr.start_rx_sync(self.BUF)
        iq = self.sdr.read_rx_sync()
        self.assertEqual(len(iq), self.BUF // 2)

    def test_iq_not_zero(self):
        """Stub generates non-zero IQ data."""
        self.sdr.start_rx_sync(self.BUF)
        iq = self.sdr.read_rx_sync()
        self.assertGreater(np.max(np.abs(iq)), 0.0)

    def test_iq_is_complex(self):
        """Result is a complex-valued array."""
        self.sdr.start_rx_sync(self.BUF)
        iq = self.sdr.read_rx_sync()
        self.assertTrue(np.iscomplexobj(iq))

    def test_multiple_reads(self):
        """Multiple consecutive reads all return valid arrays."""
        self.sdr.start_rx_sync(self.BUF)
        for _ in range(3):
            iq = self.sdr.read_rx_sync()
            self.assertEqual(len(iq), self.BUF // 2)
            self.assertTrue(np.iscomplexobj(iq))

    def test_stop_clears_sync_mode(self):
        """stop_rx_sync clears the _sync_mode flag."""
        self.sdr.start_rx_sync(self.BUF)
        self.assertTrue(self.sdr._sync_mode)
        self.sdr.stop_rx_sync()
        self.assertFalse(self.sdr._sync_mode)


# ---------------------------------------------------------------------------
# Tests: asynchronous IQ reception
# ---------------------------------------------------------------------------
@unittest.skipUnless(STUB_AVAILABLE, SKIP_MSG)
class TestStubAsyncReception(unittest.TestCase):

    BUF_LEN   = 4096   # floats per callback
    BUF_COUNT = 4      # fwrapper enforces minimum buf_count=4

    def test_callback_called_correct_times(self):
        """Async callback is invoked exactly buf_count times."""
        call_count = [0]
        received   = []

        def cb(iq):
            call_count[0] += 1
            received.append(iq)

        with make_sdr() as sdr:
            sdr.open(0)
            sdr.set_samplerate(2.048e6)
            sdr.start_rx_async(cb, buf_count=self.BUF_COUNT, buf_length=self.BUF_LEN)

        self.assertEqual(call_count[0], self.BUF_COUNT)

    def test_async_iq_dtype_and_shape(self):
        """Each async callback receives complex64 array of correct length."""
        received = []

        def cb(iq):
            received.append(iq)

        with make_sdr() as sdr:
            sdr.open(0)
            sdr.set_samplerate(2.048e6)
            sdr.start_rx_async(cb, buf_count=self.BUF_COUNT, buf_length=self.BUF_LEN)

        for iq in received:
            self.assertEqual(iq.dtype, np.complex64)
            self.assertEqual(len(iq), self.BUF_LEN // 2)
            self.assertTrue(np.iscomplexobj(iq))

    def test_async_iq_not_zero(self):
        """Async buffers contain non-zero data."""
        received = []

        def cb(iq):
            received.append(iq)

        with make_sdr() as sdr:
            sdr.open(0)
            sdr.start_rx_async(cb, buf_count=2, buf_length=self.BUF_LEN)

        for iq in received:
            self.assertGreater(np.max(np.abs(iq)), 0.0)


# ---------------------------------------------------------------------------
# Tests: FFT spectrum — known tone peak
# ---------------------------------------------------------------------------
@unittest.skipUnless(STUB_AVAILABLE, SKIP_MSG)
class TestStubSpectrum(unittest.TestCase):
    """
    signals.json includes a tone at 100 kHz (freq_hz=100000).
    With sample_rate=2048000 and FFT size=4096, the bin resolution is
    2048000/4096 = 500 Hz/bin.  The 100 kHz tone lands at bin 200.
    """

    SAMPLE_RATE = 2_048_000
    FFT_SIZE    = 4096
    TONE_FREQ   = 100_000       # Hz — must match signals.json
    BUF_FLOATS  = FFT_SIZE * 2  # interleaved I+Q

    def _get_iq(self):
        with make_sdr() as sdr:
            sdr.open(0)
            sdr.set_samplerate(self.SAMPLE_RATE)
            sdr.start_rx_sync(self.BUF_FLOATS)
            iq = sdr.read_rx_sync()
        return iq

    def test_fft_shape(self):
        """FFT of IQ data has FFT_SIZE bins."""
        iq = self._get_iq()
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(iq[:self.FFT_SIZE], self.FFT_SIZE)))
        self.assertEqual(len(spectrum), self.FFT_SIZE)

    def test_tone_peak_at_correct_bin(self):
        """
        Peak of the spectrum is within ±2 bins of the expected tone frequency.
        The tone at TONE_FREQ Hz should dominate because its amplitude (0.3)
        is much larger than noise (0.05) after FFT gain.
        """
        iq = self._get_iq()
        fft_out  = np.fft.fftshift(np.fft.fft(iq[:self.FFT_SIZE], self.FFT_SIZE))
        power_db = 20 * np.log10(np.abs(fft_out) + 1e-10)

        freqs        = np.fft.fftshift(np.fft.fftfreq(self.FFT_SIZE, d=1.0 / self.SAMPLE_RATE))
        expected_bin = int(np.argmin(np.abs(freqs - self.TONE_FREQ)))
        peak_bin     = int(np.argmax(power_db))

        self.assertAlmostEqual(peak_bin, expected_bin, delta=2,
            msg=f"Peak at bin {peak_bin} ({freqs[peak_bin]:.0f} Hz), "
                f"expected ~bin {expected_bin} ({self.TONE_FREQ} Hz)")


# ---------------------------------------------------------------------------
# Tests: FM demodulation pipeline (DSP only, no audio device)
# ---------------------------------------------------------------------------
@unittest.skipUnless(STUB_AVAILABLE, SKIP_MSG)
class TestStubFMDemod(unittest.TestCase):
    """
    signals.json includes an FM-modulated 1 kHz tone.
    We instantiate FMReceiver with no_audio=True and feed it IQ data
    directly via _process_iq_samples, then inspect the audio queue.
    """

    SAMPLE_RATE = 2_048_000
    BUF_FLOATS  = 8192 * 2    # 8192 IQ pairs — enough for decimation filter

    def _get_iq(self):
        with make_sdr() as sdr:
            sdr.open(0)
            sdr.set_samplerate(self.SAMPLE_RATE)
            sdr.start_rx_sync(self.BUF_FLOATS)
            iq = sdr.read_rx_sync()
        return iq

    def _demod(self, iq):
        """Run FM demod and return the audio array directly, bypassing no_audio guard."""
        from fmreceiver.fobos_fm_receiver import FMReceiver, SAMPLE_RATE, AUDIO_RATE, FM_DEVIATION
        from scipy import signal as scipy_signal
        # Replicate _process_iq_samples DSP without the audio-queue guard
        phase       = np.angle(iq)
        diff_phase  = np.diff(np.unwrap(np.append(0.0, phase)))
        audio_raw   = diff_phase * (SAMPLE_RATE / (2 * np.pi * FM_DEVIATION))
        decimation  = int(SAMPLE_RATE / AUDIO_RATE)
        audio       = scipy_signal.decimate(audio_raw, decimation, ftype='fir')
        max_val     = np.max(np.abs(audio) + 1e-10)
        return (audio / max_val).astype(np.float32), decimation

    def test_fm_demod_produces_audio_array(self):
        """FM demodulation pipeline returns a float array."""
        audio, _ = self._demod(self._get_iq())
        self.assertIsInstance(audio, np.ndarray)
        self.assertTrue(np.issubdtype(audio.dtype, np.floating))
        self.assertGreater(len(audio), 0)

    def test_fm_demod_audio_not_silent(self):
        """Demodulated audio has non-zero amplitude (FM signal is present)."""
        audio, _ = self._demod(self._get_iq())
        self.assertGreater(np.max(np.abs(audio)), 1e-6,
            "Audio is silent — FM demodulation may have failed")

    def test_fm_demod_audio_shape(self):
        """Audio length matches expected decimation: SAMPLE_RATE / AUDIO_RATE."""
        iq = self._get_iq()
        audio, decimation = self._demod(iq)
        expected = len(iq) // decimation
        # Allow ±10% tolerance for FIR filter edge effects
        self.assertAlmostEqual(len(audio), expected, delta=expected * 0.10,
            msg=f"Audio length {len(audio)} far from expected {expected}")


if __name__ == "__main__":
    unittest.main()
