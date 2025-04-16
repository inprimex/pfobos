"""
Mock tests for Fobos SDR wrapper.
These tests don't require hardware to run and use a fully mocked device.
"""

import unittest
from unittest.mock import patch, Mock, MagicMock, call
import numpy as np
import sys
import os
import gc

# Import the wrapper from the shared module
from shared.fwrapper import FobosSDR, FobosException, FobosError

class TestMockFobos(unittest.TestCase):
    """Test Fobos SDR wrapper with a fully mocked device."""

    def setUp(self):
        """Set up test fixtures."""
        # Create comprehensive patchers
        self.ffi_patcher = patch('shared.fwrapper.FFI')
        self.os_patcher = patch('shared.fwrapper.os')
        self.sys_patcher = patch('shared.fwrapper.sys')
        
        # Start patchers
        self.mock_ffi = self.ffi_patcher.start()
        self.mock_os = self.os_patcher.start()
        self.mock_sys = self.sys_patcher.start()
        
        # Configure mocks for platform detection
        self.mock_sys.platform = 'linux'
        
        # Configure mock FFI instance
        self.mock_ffi_instance = MagicMock()
        self.mock_ffi.return_value = self.mock_ffi_instance
        
        # Mock C library - make it return integers for function calls
        self.mock_lib = MagicMock()
        self.mock_lib.fobos_rx_get_device_count.return_value = 1
        self.mock_lib.fobos_rx_list_devices.return_value = 0
        self.mock_lib.fobos_rx_open.return_value = 0
        self.mock_lib.fobos_rx_close.return_value = 0
        self.mock_lib.fobos_rx_get_api_info.return_value = 0
        self.mock_lib.fobos_rx_get_board_info.return_value = 0
        self.mock_lib.fobos_rx_get_samplerates.return_value = 0
        self.mock_lib.fobos_rx_set_samplerate.return_value = 0
        self.mock_lib.fobos_rx_set_frequency.return_value = 0
        self.mock_lib.fobos_rx_set_lna_gain.return_value = 0
        self.mock_lib.fobos_rx_set_vga_gain.return_value = 0
        self.mock_lib.fobos_rx_set_direct_sampling.return_value = 0
        self.mock_lib.fobos_rx_read_async.return_value = 0
        self.mock_lib.fobos_rx_cancel_async.return_value = 0
        self.mock_lib.fobos_rx_start_sync.return_value = 0
        self.mock_lib.fobos_rx_read_sync.return_value = 0
        self.mock_lib.fobos_rx_stop_sync.return_value = 0

        # Set up mock error name function
        self.mock_lib.fobos_rx_error_name.return_value = b"MOCK_ERROR"
        
        self.mock_ffi_instance.dlopen.return_value = self.mock_lib
        
        # Mock device pointer
        self.mock_dev = MagicMock()
        
        # Create class to properly handle __getitem__ for device pointers
        class MockDevicePointer:
            def __init__(self, mock_dev):
                self.mock_dev = mock_dev
                
            def __getitem__(self, key):
                return self.mock_dev
        
        # Classes for other pointer types
        class MockDoublePointer:
            def __getitem__(self, key):
                return 100e6
        
        class MockUInt32Pointer:
            def __getitem__(self, key):
                return 1024
                
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
            elif "uint32_t *" in str(arg_type) or "unsigned int *" in str(arg_type):
                return MockUInt32Pointer()
            elif "float[" in str(arg_type):
                return MockFloatBuffer()
            else:
                return MagicMock()
                
        self.mock_ffi_instance.new.side_effect = mock_new_side_effect
        
        # Mock string conversion
        self.mock_ffi_instance.string.return_value = b"test_string"
        
        # Create the SDR instance with mocks
        self.sdr = FobosSDR()

    def tearDown(self):
        """Tear down test fixtures."""
        self.ffi_patcher.stop()
        self.os_patcher.stop()
        self.sys_patcher.stop()

    def test_device_count(self):
        """Test getting device count."""
        count = self.sdr.get_device_count()
        self.assertEqual(count, 1)
        self.mock_lib.fobos_rx_get_device_count.assert_called_once()
        
    def test_list_devices(self):
        """Test listing device serials."""
        serials = self.sdr.list_devices()
        self.assertEqual(serials, ["test_string"])
        self.mock_lib.fobos_rx_list_devices.assert_called_once()
        
    def test_open_close(self):
        """Test opening and closing device."""
        # Open device
        self.sdr.open(0)
        self.assertEqual(self.sdr.dev, self.mock_dev)
        self.mock_lib.fobos_rx_open.assert_called_once()
        
        # Close device
        self.sdr.close()
        self.assertIsNone(self.sdr.dev)
        self.mock_lib.fobos_rx_close.assert_called_once_with(self.mock_dev)
    
    def test_get_board_info(self):
        """Test getting board info."""
        # Open device first
        self.sdr.open(0)
        
        # Get board info
        info = self.sdr.get_board_info()
        self.assertEqual(info["hw_revision"], "test_string")
        self.assertEqual(info["fw_version"], "test_string")
        self.assertEqual(info["manufacturer"], "test_string")
        self.assertEqual(info["product"], "test_string")
        self.assertEqual(info["serial"], "test_string")
        
        # Verify C function was called correctly
        self.mock_lib.fobos_rx_get_board_info.assert_called_once()
    
    def test_sync_mode(self):
        """Test synchronous reception mode."""
        # Open device first
        self.sdr.open(0)
        
        # Start sync mode
        self.sdr.start_rx_sync(1024)
        self.mock_lib.fobos_rx_start_sync.assert_called_once_with(self.mock_dev, 1024)
        self.assertTrue(self.sdr._sync_mode)
        
        # Read samples
        iq_data = self.sdr.read_rx_sync()
        self.mock_lib.fobos_rx_read_sync.assert_called_once()
        
        # Verify we got complex data
        self.assertTrue(np.iscomplexobj(iq_data))
        
        # Stop sync mode
        self.sdr.stop_rx_sync()
        self.mock_lib.fobos_rx_stop_sync.assert_called_once_with(self.mock_dev)
        self.assertFalse(self.sdr._sync_mode)
    
    def test_error_handling(self):
        """Test error handling."""
        # Make a function return an error
        self.mock_lib.fobos_rx_set_frequency.return_value = -1  # NO_DEV error
        
        # Open device first
        self.sdr.open(0)
        
        # This should raise an exception
        with self.assertRaises(FobosException):
            self.sdr.set_frequency(100e6)
    
    def test_async_callback_safety(self):
        """Test that async callback handling is safe."""
        # Open device first
        self.sdr.open(0)
        
        # Create a callback function
        callback_called = False
        
        def test_callback(samples):
            nonlocal callback_called
            callback_called = True
            self.assertTrue(len(samples) > 0)
        
        # Mock FFI callback function
        mock_callback = MagicMock()
        self.mock_ffi_instance.callback.return_value = mock_callback
        
        # Start async mode
        self.sdr.start_rx_async(test_callback)
        self.assertTrue(self.sdr._async_mode)
        
        # Call the callback directly through our mocked wrapper
        # We need to simulate this since we can't easily invoke the C callback
        buffer_data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        self.sdr._python_callback(buffer_data.view(np.complex64))
        
        # Verify our Python callback was called
        self.assertTrue(callback_called)
        
        # Stop async mode
        self.sdr.stop_rx_async()
        self.assertFalse(self.sdr._async_mode)
        
    def test_resource_cleanup(self):
        """Test that resources are properly cleaned up."""
        # Open device and set modes
        self.sdr.open(0)
        
        # Set up async mode
        mock_callback = MagicMock()
        self.mock_ffi_instance.callback.return_value = mock_callback
        
        def test_callback(samples):
            pass
            
        # Start async mode
        self.sdr.start_rx_async(test_callback)
        self.assertTrue(self.sdr._async_mode)
        
        # Close device, both modes should be stopped
        self.sdr.close()
        
        # Verify async mode was stopped
        self.assertFalse(self.sdr._async_mode)
        self.assertIsNone(self.sdr.dev)
        
        # Test with sync mode
        self.sdr.open(0)
        self.sdr.start_rx_sync(1024)
        self.assertTrue(self.sdr._sync_mode)
        
        # Close device
        self.sdr.close()
        
        # Verify sync mode was stopped
        self.assertFalse(self.sdr._sync_mode)
        self.assertIsNone(self.sdr.dev)

if __name__ == '__main__':
    unittest.main()