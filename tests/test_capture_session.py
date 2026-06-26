"""
Tests for bench.observation_mode_research.capture_noise.CaptureSession.

Uses the stub libfobos.so so these run without hardware. The stub
synthesises an IQ stream from tests/stub/signals.json so all the
metadata, file-layout, and fan_out behaviors are exercised end-to-end.
"""

import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STUB_DIR = PROJECT_ROOT / "tests" / "stub"
STUB_LIB = STUB_DIR / "libfobos.so"
SIGNALS_JSON = STUB_DIR / "signals.json"

sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("FOBOS_STUB_SIGNALS", str(SIGNALS_JSON))

STUB_AVAILABLE = STUB_LIB.exists()
SKIP_MSG = "Stub library not built. Run: uv run python tests/stub/build.py"

from bench.observation_mode_research.capture_noise import (  # noqa: E402
    CaptureChunk,
    CaptureConfig,
    CaptureResult,
    CaptureSession,
    run_capture,
)


def _short_config(tmp_dir: Path, **overrides) -> CaptureConfig:
    defaults = dict(
        freq_hz=868e6,
        rate_hz=8e6,
        duration_s=0.5,
        env="quiet_lab",
        antenna="stub-test-antenna",
        out_dir=tmp_dir,
        lna_gain=1,
        vga_gain=8,
        hwtel_interval_s=10.0,  # high — we don't need samples in 0.5 s
        samples_per_file=50_000,  # small so a 0.5 s capture produces several files
        sync_buf_floats=8192,
        queue_max_chunks=32,
        lib_path=str(STUB_LIB),
    )
    defaults.update(overrides)
    return CaptureConfig(**defaults)


@unittest.skipUnless(STUB_AVAILABLE, SKIP_MSG)
class TestCaptureSessionBasic(unittest.TestCase):
    """End-to-end: session.run() against the stub returns a populated result."""

    def setUp(self):
        self.tmp = PROJECT_ROOT / "tests" / "_tmp_capture_session"
        if self.tmp.exists():
            for p in self.tmp.iterdir():
                p.unlink()
        self.tmp.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.tmp.exists():
            for p in self.tmp.iterdir():
                p.unlink()
            self.tmp.rmdir()

    def test_run_returns_capture_result(self):
        config = _short_config(self.tmp)
        result = run_capture(config)

        self.assertIsInstance(result, CaptureResult)
        self.assertTrue(result.ok, f"capture not ok; metadata={result.metadata}")
        self.assertGreater(result.total_samples, 0)
        self.assertGreater(result.actual_duration_s, 0.0)
        self.assertIsNone(result.error)

    def test_metadata_sidecar_written(self):
        config = _short_config(self.tmp)
        result = run_capture(config)

        self.assertTrue(result.metadata_path.exists())
        loaded = json.loads(result.metadata_path.read_text())
        self.assertEqual(loaded["schema"], "pfobos-noise-capture/2")
        self.assertEqual(loaded["config"]["antenna"], "stub-test-antenna")
        self.assertEqual(loaded["config"]["environment_tag"], "quiet_lab")
        self.assertTrue(loaded["ok"])
        self.assertEqual(loaded["fan_out"]["consumers"], [])
        self.assertEqual(loaded["fan_out"]["errors"], [])

    def test_iq_files_match_metadata(self):
        config = _short_config(self.tmp)
        result = run_capture(config)

        self.assertGreater(len(result.iq_paths), 0)
        for path in result.iq_paths:
            self.assertTrue(path.exists())
            # complex64 = 8 bytes/sample
            self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(path.stat().st_size % 8, 0)

        files_record = result.metadata["files"]
        self.assertEqual(len(files_record), len(result.iq_paths))


@unittest.skipUnless(STUB_AVAILABLE, SKIP_MSG)
class TestFanOutHook(unittest.TestCase):
    """fan_out consumers receive chunks; exceptions are isolated."""

    def setUp(self):
        self.tmp = PROJECT_ROOT / "tests" / "_tmp_capture_fanout"
        if self.tmp.exists():
            for p in self.tmp.iterdir():
                p.unlink()
        self.tmp.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.tmp.exists():
            for p in self.tmp.iterdir():
                p.unlink()
            self.tmp.rmdir()

    def test_fan_out_receives_chunks(self):
        received: list[CaptureChunk] = []
        lock = threading.Lock()

        def collector(chunk: CaptureChunk) -> None:
            with lock:
                received.append(chunk)

        config = _short_config(self.tmp, fan_out=[collector])
        result = run_capture(config)

        self.assertTrue(result.ok)
        # Same number of chunks delivered as files written.
        self.assertEqual(len(received), len(result.iq_paths))
        # Chunks arrived in order.
        for expected_idx, chunk in enumerate(received, start=1):
            self.assertEqual(chunk.file_idx, expected_idx)
            self.assertEqual(chunk.samples.dtype, np.complex64)
            self.assertTrue(chunk.path.exists())
            self.assertEqual(chunk.band_label, "868MHz")

    def test_multiple_fan_out_consumers(self):
        received_a: list[int] = []
        received_b: list[int] = []

        def consumer_a(chunk: CaptureChunk) -> None:
            received_a.append(chunk.file_idx)

        def consumer_b(chunk: CaptureChunk) -> None:
            received_b.append(chunk.file_idx)

        config = _short_config(self.tmp, fan_out=[consumer_a, consumer_b])
        result = run_capture(config)

        self.assertTrue(result.ok)
        self.assertEqual(received_a, received_b)
        self.assertEqual(len(received_a), len(result.iq_paths))

    def test_fan_out_exception_does_not_break_writer(self):
        good_received: list[int] = []

        def bad_consumer(chunk: CaptureChunk) -> None:
            raise RuntimeError(f"intentional failure on file {chunk.file_idx}")

        def good_consumer(chunk: CaptureChunk) -> None:
            good_received.append(chunk.file_idx)

        config = _short_config(self.tmp, fan_out=[bad_consumer, good_consumer])
        result = run_capture(config)

        # Primary path completed despite the bad consumer.
        self.assertTrue(result.ok)
        self.assertGreater(len(result.iq_paths), 0)
        # Good consumer kept receiving even though bad_consumer kept raising.
        self.assertEqual(len(good_received), len(result.iq_paths))
        # Errors recorded in metadata for diagnostics.
        errors = result.metadata["fan_out"]["errors"]
        self.assertEqual(len(errors), len(result.iq_paths))
        for err in errors:
            self.assertEqual(err["consumer"], "bad_consumer")
            self.assertIn("RuntimeError", err["error"])

    def test_fan_out_consumer_names_recorded(self):
        def metric_writer(chunk: CaptureChunk) -> None:
            pass

        config = _short_config(self.tmp, fan_out=[metric_writer])
        result = run_capture(config)

        self.assertEqual(
            result.metadata["fan_out"]["consumers"], ["metric_writer"],
        )


if __name__ == "__main__":
    unittest.main()
