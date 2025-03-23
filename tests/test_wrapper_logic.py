"""
Unit tests for Fobos SDR wrapper logic.
These tests focus on parameter validation, error handling, and other wrapper functionality.
"""

import unittest
from unittest.mock import patch, Mock, call
import numpy as np
import sys
import os

# Add parent directory to path to import the wrapper
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fobos_wrapper import FobosSDR, FobosException, FobosError


class TestWrapperLogic(unittest.TestCase):
    """Test logical behavior of the wrapper."""

    def setUp(self):
        """Set up test fixtures."""
        # Create patchers
        self.ffi_patcher = patch('fobos_wrapper.FFI')
        
        # Start patchers
        self.mock_ffi = self.ffi_patcher.start()
        
        # Configure mocks
        self.mock_ffi_instance = Mock()
        self.mock_ffi.return_value = self.mock_ffi_instance
        
        self.mock_lib = Mock()
        self.mock_ffi_instance.dlopen.return_value = self.mock_lib
        
        # Setup device mock
        self.mock_dev = Mock()
        self.mock_dev_ptr = Mock()
        self.mock_dev_ptr.__getitem__.return_value = self.mock_dev
        self.mock_ffi_instance.new.return_value = self.mock_dev_ptr

        # Default success return value
        self.mock_lib.fobos_rx_open.return_value = 0
        self.mock_lib.fobos_rx_close.return_value = 0
        
        # Create SDR instance
        self.sdr = FobosSDR()

    def tearDown(self):
        """Tear down test fixtures."""
        self.ffi_patcher.stop()

    def test_context_manager(self):
        """Test context manager functionality."""
        # Configure mocks
        self.mock_lib.fobos_rx_open.return_value = 0
        self.mock_lib.fobos_rx_close.return_value = 0
        
        # Use context manager
        with self.sdr as sdr:
            # Mock the open call that would happen in practice
            sdr.dev = self.mock_dev
            
            # Verify we got the right object
            self.assertEqual(sdr, self.sdr)
            
        # Verify close was called
        self.mock_lib.fobos_rx_close.assert_called_once_with(self.mock_dev)
        
    def test_check_device_open(self):
        """Test _check_device_open method."""
        # Test when device is not open
        with self.assertRaises(FobosException) as context:
            self.sdr._check_device_open()
            
        self.assertEqual(context.exception.code, FobosError.NOT_OPEN)
        
        # Test when device is open
        self.sdr.dev = self.mock_dev
        # Should not raise exception
        self.sdr._check_device_open()
        
    def test_set_lna_gain_validation(self):
        """Test validation in set_lna_gain method."""
        # Setup mock for successful call
        self.sdr.dev = self.mock_dev
        self.mock_lib.fobos_rx_set_lna_gain.return_value = 0
        
        # Test valid values
        for valid_value in [0, 1, 2]:
            self.sdr.set_lna_gain(valid_value)
            
        # Test invalid values (too low)
        with self.assertRaises(ValueError) as context:
            self.sdr.set_lna_gain(-1)
            
        # Test invalid values (too high)
        with self.assertRaises(ValueError) as context:
            self.sdr.set_lna_gain(3)
            
    def test_set_vga_gain_validation(self):
        """Test validation in set_vga_gain method."""
        # Setup mock for successful call
        self.sdr.dev = self.mock_dev
        self.mock_lib.fobos_rx_set_vga_gain.return_value = 0
        
        # Test valid values
        for valid_value in [0, 7, 15]:
            self.sdr.set_vga_gain(valid_value)
            
        # Test invalid values (too low)
        with self.assertRaises(ValueError) as context:
            self.sdr.set_vga_gain(-1)
            
        # Test invalid values (too high)
        with self.assertRaises(ValueError) as context:
            self.sdr.set_vga_gain(16)
    
    def test_set_max2830_frequency_validation(self):
        """Test validation in set_max2830_frequency method."""
        # Setup mock for successful call
        self.sdr.dev = self.mock_dev
        self.mock_lib.fobos_max2830_set_frequency.return_value = 0
        
        # Setup actual frequency
        def mock_getitem(idx):
            return 2.4e9
            
        self.mock_ffi_instance.new("double *").__getitem__.side_effect = mock_getitem
        
        # Test valid frequency
        result = self.sdr.set_max2830_frequency(2.4e9)
        self.assertEqual(result, 2.4e9)
        
        # Test frequency too low
        with self.assertRaises(ValueError) as context:
            self.sdr.set_max2830_frequency(2e9)
            
        # Test frequency too high
        with self.assertRaises(ValueError) as context:
            self.sdr.set_max2830_frequency(3e9)
            
    def test_set_rffc507x_lo_frequency_validation(self):
        """Test validation in set_rffc507x_lo_frequency method."""
        # Setup mock for successful call
        self.sdr.dev = self.mock_dev
        self.mock_lib.fobos_rffc507x_set_lo_frequency_hz.return_value = 0
        
        # Setup actual frequency
        def mock_getitem(idx):
            return 900e6
            
        self.mock_ffi_instance.new("uint64_t *").__getitem__.side_effect = mock_getitem
        
        # Test valid frequency
        result = self.sdr.set_rffc507x_lo_frequency(900e6)
        self.assertEqual(result, 900e6)
        
        # Test frequency too low
        with self.assertRaises(ValueError) as context:
            self.sdr.set_rffc507x_lo_frequency(20e6)
            
        # Test frequency too high
        with self.assertRaises(ValueError) as context:
            self.sdr.set_rffc507x_lo_frequency(6e9)
            
    def test_set_user_gpo_validation(self):
        """Test validation in set_user_gpo method."""
        # Setup mock for successful call
        self.sdr.dev = self.mock_dev
        self.mock_lib.fobos_rx_set_user_gpo.return_value = 0
        
        # Test valid values
        for valid_value in [0, 128, 255]:
            self.sdr.set_user_gpo(valid_value)
            
        # Test invalid values (too low)
        with self.assertRaises(ValueError) as context:
            self.sdr.set_user_gpo(-1)
            
        # Test invalid values (too high)
        with self.assertRaises(ValueError) as context:
            self.sdr.set_user_gpo(256)
            
    def test_read_rx_sync_validation(self):
        """Test validation in read_rx_sync method."""
        # Setup mock for successful call
        self.sdr.dev = self.mock_dev
        
        # Test with sync not started
        with self.assertRaises(RuntimeError) as context:
            self.sdr.read_rx_sync()
        
        self.assertIn("Synchronous mode not started", str(context.exception))
        
        # Setup mock buffer
        self.sdr._buffer_ptr = Mock()
        self.sdr._buffer_length = 1024
        
        # Configure mock for read_sync
        self.mock_lib.fobos_rx_read_sync.return_value = 0
        
        # Setup actual length
        mock_actual_len = Mock()
        mock_actual_len.__getitem__.return_value = 100
        self.mock_ffi_instance.new.return_value = mock_actual_len
        
        # Now read_sync should work without exception
        self.sdr.read_rx_sync()
        
        # Verify C function was called correctly
        self.mock_lib.fobos_rx_read_sync.assert_called_once()


if __name__ == '__main__':
    unittest.main()