"""
Unit tests for Fobos SDR wrapper logic.
These tests focus on parameter validation, error handling, and other wrapper functionality.
"""

import unittest
from unittest.mock import patch, Mock, MagicMock, call
import numpy as np
import sys
import os

# Import the wrapper from the shared module
from shared.fwrapper import FobosSDR, FobosException, FobosError


class TestWrapperLogic(unittest.TestCase):
    """Test logical behavior of the wrapper."""

    def setUp(self):
        """Set up test fixtures."""
        # Create patchers
        self.ffi_patcher = patch('shared.fwrapper.FFI')
        self.os_patcher = patch('shared.fwrapper.os')
        self.sys_patcher = patch('shared.fwrapper.sys')
        
        # Start patchers
        self.mock_ffi = self.ffi_patcher.start()
        self.mock_os = self.os_patcher.start()
        self.mock_sys = self.sys_patcher.start()
        
        # Configure platform detection
        self.mock_sys.platform = 'linux'
        
        # Configure mocks
        self.mock_ffi_instance = MagicMock()
        self.mock_ffi.return_value = self.mock_ffi_instance
        
        # Mock C library with proper integer returns
        self.mock_lib = MagicMock()
        self.mock_lib.fobos_rx_open.return_value = 0
        self.mock_lib.fobos_rx_close.return_value = 0
        self.mock_lib.fobos_rx_set_lna_gain.return_value = 0
        self.mock_lib.fobos_rx_set_vga_gain.return_value = 0
        self.mock_lib.fobos_rx_set_direct_sampling.return_value = 0
        self.mock_lib.fobos_rx_set_user_gpo.return_value = 0
        self.mock_lib.fobos_rx_read_sync.return_value = 0
        
        # IMPORTANT: Make sure these specific functions return integers
        self.mock_lib.fobos_max2830_set_frequency.return_value = 0
        self.mock_lib.fobos_rffc507x_set_lo_frequency_hz.return_value = 0
        
        self.mock_ffi_instance.dlopen.return_value = self.mock_lib
        
        # Set up mock error name function
        self.mock_lib.fobos_rx_error_name.return_value = b"MOCK_ERROR"
        
        # Mock device
        self.mock_dev = MagicMock()
        
        # Create proper pointer classes with __getitem__ methods
        
        # For device pointers
        class MockDevicePointer:
            def __init__(self, mock_dev):
                self.mock_dev = mock_dev
                
            def __getitem__(self, key):
                return self.mock_dev
        
        # For double pointers
        class MockDoublePointer:
            def __getitem__(self, key):
                return 100e6
        
        # For uint64 pointers
        class MockUInt64Pointer:
            def __getitem__(self, key):
                return 900e6
        
        # For uint32 pointers
        class MockUInt32Pointer:
            def __getitem__(self, key):
                return 1024
                
        # For float buffers
        class MockFloatBuffer:
            def __init__(self):
                self.buffer_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                
            def __getitem__(self, key):
                if isinstance(key, int) and key < len(self.buffer_data):
                    return self.buffer_data[key]
                return 0.0
        
        # Create a side effect for FFI's new function
        def mock_new_side_effect(arg_type, *args, **kwargs):
            if "fobos_dev_t **" in str(arg_type):
                return MockDevicePointer(self.mock_dev)
            elif "char[]" in str(arg_type) or "char *" in str(arg_type):
                # For strings, return the input if provided
                if args and args[0]:
                    return args[0]
                return b"test_string"
            elif "double *" in str(arg_type):
                return MockDoublePointer()
            elif "uint64_t *" in str(arg_type):
                return MockUInt64Pointer()
            elif "uint32_t *" in str(arg_type) or "unsigned int *" in str(arg_type):
                return MockUInt32Pointer()
            elif "float[" in str(arg_type) or "float *" in str(arg_type):
                return MockFloatBuffer()
            else:
                return MagicMock()
                
        self.mock_ffi_instance.new.side_effect = mock_new_side_effect
        
        # Mock string conversion
        self.mock_ffi_instance.string.return_value = b"test_string"
        
        # Create SDR instance
        self.sdr = FobosSDR()

    def tearDown(self):
        """Tear down test fixtures."""
        self.ffi_patcher.stop()
        self.os_patcher.stop()
        self.sys_patcher.stop()

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
        with self.assertRaises(ValueError):
            self.sdr.set_lna_gain(-1)
            
        # Test invalid values (too high)
        with self.assertRaises(ValueError):
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
        with self.assertRaises(ValueError):
            self.sdr.set_vga_gain(-1)
            
        # Test invalid values (too high)
        with self.assertRaises(ValueError):
            self.sdr.set_vga_gain(16)
    
    def test_set_max2830_frequency_validation(self):
        """Test validation in set_max2830_frequency method."""
        # Setup mock for successful call
        self.sdr.dev = self.mock_dev
        
        # Test valid frequency (just inside range)
        result = self.sdr.set_max2830_frequency(2.36e9)
        self.assertEqual(result, 100e6)  # mocked return value
        
        # Test frequency too low
        with self.assertRaises(ValueError):
            self.sdr.set_max2830_frequency(2e9)
            
        # Test frequency too high
        with self.assertRaises(ValueError):
            self.sdr.set_max2830_frequency(3e9)
            
    def test_set_rffc507x_lo_frequency_validation(self):
        """Test validation in set_rffc507x_lo_frequency method."""
        # Setup mock for successful call
        self.sdr.dev = self.mock_dev
        
        # Test valid frequency
        result = self.sdr.set_rffc507x_lo_frequency(900e6)
        self.assertEqual(result, 900e6)  # mocked return value
        
        # Test frequency too low
        with self.assertRaises(ValueError):
            self.sdr.set_rffc507x_lo_frequency(20e6)
            
        # Test frequency too high
        with self.assertRaises(ValueError):
            self.sdr.set_rffc507x_lo_frequency(6e9)
            
    def test_set_user_gpo_validation(self):
        """Test validation in set_user_gpo method."""
        # Setup mock for successful call
        self.sdr.dev = self.mock_dev
        
        # Test valid values
        for valid_value in [0, 128, 255]:
            self.sdr.set_user_gpo(valid_value)
            
        # Test invalid values (too low)
        with self.assertRaises(ValueError):
            self.sdr.set_user_gpo(-1)
            
        # Test invalid values (too high)
        with self.assertRaises(ValueError):
            self.sdr.set_user_gpo(256)
            
    def test_read_rx_sync_validation(self):
        """Test validation in read_rx_sync method."""
        # Setup mock for successful call
        self.sdr.dev = self.mock_dev
        
        # Test with sync not started
        with self.assertRaises(RuntimeError) as context:
            self.sdr.read_rx_sync()
        
        self.assertIn("Synchronous mode not started", str(context.exception))
        
        # Setup for sync mode test
        self.sdr._sync_mode = True
        self.sdr._buffer_ptr = MagicMock()
        self.sdr._buffer_length = 1024
        
        # Now read_sync should work without exception
        iq_data = self.sdr.read_rx_sync()
        
        # Verify C function was called correctly
        self.mock_lib.fobos_rx_read_sync.assert_called_once()
        
        # Verify we got complex data
        self.assertTrue(np.iscomplexobj(iq_data))


if __name__ == '__main__':
    unittest.main()