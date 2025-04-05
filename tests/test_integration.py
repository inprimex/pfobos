"""
Integration tests for Fobos SDR wrapper.
These tests require actual hardware to be connected.

To run these tests:
python -m unittest tests.test_integration

Skip these tests when no hardware is available:
python -m unittest tests.test_integration -k "not requires_hardware"
"""

import unittest
import numpy as np
import time
import sys
import os
import logging
from collections import deque

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path to import the wrapper
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fobos_wrapper import FobosSDR, FobosException


def requires_hardware(test_method):
    """Decorator to skip tests that require hardware when none is available."""
    def wrapper(*args, **kwargs):
        try:
            # Try to detect hardware
            sdr = FobosSDR()
            device_count = sdr.get_device_count()
            if device_count == 0:
                raise unittest.SkipTest("No Fobos SDR hardware detected")
            return test_method(*args, **kwargs)
        except Exception as e:
            if "Could not load Fobos SDR library" in str(e):
                raise unittest.SkipTest("Fobos SDR library not found")
            raise  # Re-raise any other exceptions
    return wrapper


class TestFobosSDRIntegration(unittest.TestCase):
    """Integration tests for Fobos SDR wrapper with actual hardware."""

    @classmethod
    def setUpClass(cls):
        """Set up test class. Will be skipped if no hardware is available."""
        try:
            cls.sdr = FobosSDR()
            cls.device_count = cls.sdr.get_device_count()
            if cls.device_count == 0:
                cls.skipTest(cls, "No Fobos SDR hardware detected")
            else:
                # Get connected device serials for info
                cls.device_serials = cls.sdr.list_devices()
                logger.info(f"Found {cls.device_count} devices: {cls.device_serials}")
        except Exception as e:
            if "Could not load Fobos SDR library" in str(e):
                cls.skipTest(cls, "Fobos SDR library not found")
            else:
                raise

    @requires_hardware
    def setUp(self):
        """Set up test case."""
        self.sdr = FobosSDR()
        self.sdr.open(0)  # Open first device
        
        # Get and log device info
        info = self.sdr.get_board_info()
        logger.info(f"Testing with device: {info['product']} (SN: {info['serial']})")

    def tearDown(self):
        """Clean up after test case."""
        if hasattr(self, 'sdr') and self.sdr.dev is not None:
            self.sdr.close()

    @requires_hardware
    def test_basic_device_info(self):
        """Test getting basic device information."""
        # Get API info
        api_info = self.sdr.get_api_info()
        self.assertIsNotNone(api_info["library_version"])
        self.assertIsNotNone(api_info["driver_version"])
        logger.info(f"API info: {api_info}")
        
        # Get board info
        board_info = self.sdr.get_board_info()
        self.assertIsNotNone(board_info["hw_revision"])
        self.assertIsNotNone(board_info["fw_version"])
        self.assertIsNotNone(board_info["manufacturer"])
        self.assertIsNotNone(board_info["product"])
        self.assertIsNotNone(board_info["serial"])
        logger.info(f"Board info: {board_info}")

    @requires_hardware
    def test_frequency_setting(self):
        """Test setting frequency."""
        # Test setting frequency to 100 MHz
        target_freq = 100e6
        actual_freq = self.sdr.set_frequency(target_freq)
        
        logger.info(f"Set frequency to {actual_freq/1e6:.3f} MHz (target: {target_freq/1e6:.3f} MHz)")
        
        # Check that frequency is reasonably close to target (within 1%)
        self.assertAlmostEqual(actual_freq, target_freq, delta=target_freq*0.01)

    @requires_hardware
    def test_sample_rate_setting(self):
        """Test setting sample rate."""
        # Get available sample rates
        rates = self.sdr.get_samplerates()
        self.assertGreater(len(rates), 0, "No sample rates available")
        logger.info(f"Available sample rates: {[r/1e6 for r in rates]} MHz")
        
        # Try to set to first available rate
        target_rate = rates[0]
        actual_rate = self.sdr.set_samplerate(target_rate)
        
        logger.info(f"Set sample rate to {actual_rate/1e6:.3f} MHz (target: {target_rate/1e6:.3f} MHz)")
        
        # Check that rate is reasonably close to target (within 1%)
        self.assertAlmostEqual(actual_rate, target_rate, delta=target_rate*0.01)

    @requires_hardware
    def test_gain_settings(self):
        """Test setting LNA and VGA gain."""
        # Test LNA gain settings (0-2)
        for gain in range(3):
            self.sdr.set_lna_gain(gain)
            logger.info(f"Set LNA gain to {gain}")
            # No assertion needed, just verify no exceptions are raised
        
        # Test VGA gain settings (0-15)
        for gain in range(0, 16, 5):  # Test 0, 5, 10, 15
            self.sdr.set_vga_gain(gain)
            logger.info(f"Set VGA gain to {gain}")
            # No assertion needed, just verify no exceptions are raised

    @requires_hardware
    def test_direct_sampling(self):
        """Test direct sampling mode."""
        # Enable direct sampling
        self.sdr.set_direct_sampling(True)
        logger.info("Enabled direct sampling")
        
        # Disable direct sampling
        self.sdr.set_direct_sampling(False)
        logger.info("Disabled direct sampling")
        
        # No assertions needed, just verify no exceptions are raised

    @requires_hardware
    def test_sync_reception(self):
        """Test synchronous reception mode."""
        # Configure SDR
        self.sdr.set_frequency(100e6)  # 100 MHz
        self.sdr.set_samplerate(2.048e6)  # 2.048 MHz
        self.sdr.set_lna_gain(1)
        self.sdr.set_vga_gain(10)
        
        # Start synchronous receiving
        buf_length = 1024
        logger.info(f"Starting synchronous reception (buffer size: {buf_length})")
        self.sdr.start_rx_sync(buf_length)
        
        # Read samples
        iq_data = self.sdr.read_rx_sync()
        
        # Stop synchronous receiving
        self.sdr.stop_rx_sync()
        
        # Verify we got the expected number of samples (buffer_size/2 for complex)
        expected_samples = buf_length // 2
        self.assertEqual(len(iq_data), expected_samples)
        
        # Check that data looks reasonable (should be complex values)
        self.assertTrue(np.iscomplexobj(iq_data))
        
        # Log summary stats
        logger.info(f"Received {len(iq_data)} IQ samples")
        logger.info(f"Sample mean: {np.mean(np.abs(iq_data)):.4f}")
        logger.info(f"Sample std: {np.std(np.abs(iq_data)):.4f}")    

    @requires_hardware
    def test_user_gpo(self):
        """Test setting user GPO bits."""
        # Test different bit patterns
        for bits in [0x00, 0x55, 0xAA, 0xFF]:
            self.sdr.set_user_gpo(bits)
            logger.info(f"Set GPO bits to 0x{bits:02X}")
            # Brief pause to allow hardware to respond
            time.sleep(0.1)
            
    
    @requires_hardware
    def test_context_manager(self):
        """Test using the SDR with a context manager."""
        # Close current device first
        self.sdr.close()
        
        # Use context manager
        with FobosSDR() as sdr:
            # Open device
            sdr.open(0)
            
            # Get info
            info = sdr.get_board_info()
            logger.info(f"Using device via context manager: {info['product']} (SN: {info['serial']})")
            
            # Set frequency
            freq = sdr.set_frequency(100e6)
            logger.info(f"Set frequency to {freq/1e6:.3f} MHz")
            
            # Device should be automatically closed when exiting the context manager
            
        # Verify device was closed (trying to use it should raise an exception)
        with self.assertRaises(FobosException):
            sdr.get_board_info()