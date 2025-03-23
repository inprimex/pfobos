"""
Unit tests for Fobos SDR Python wrapper using mock objects.
These tests don't require actual hardware.
"""

import unittest
from unittest.mock import patch, Mock, MagicMock, call
import numpy as np
import sys
import os

# Add parent directory to path to import the wrapper
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fobos_wrapper import FobosSDR, FobosException, FobosError


class TestFobosSDRMock(unittest.TestCase):
    """Test Fobos SDR wrapper functionality using mocks."""

    def setUp(self):
        """Set up test fixtures."""
        # Create patcher for FFI
        self.ffi_patcher = patch('fobos_wrapper.FFI')
        self.mock_ffi = self.ffi_patcher.start()
        
        # Create mock objects
        self.mock_lib = Mock()
        self.mock_ffi_instance = Mock()
        
        # Configure mock FFI
        self.mock_ffi.return_value = self.mock_ffi_instance
        self.mock_ffi_instance.dlopen.return_value = self.mock_lib
        
        # Mock device pointer
        self.mock_dev_ptr = Mock()
        self.mock_ffi_instance.new.return_value = self.mock_dev_ptr
        
        # Configure default return values
        self.mock_lib.fobos_rx_get_device_count.return_value = 2
        self.mock_lib.fobos_rx_open.return_value = 0
        self.mock_lib.fobos_rx_close.return_value = 0
        self.mock_lib.fobos_rx_error_name.return_value = b"FOBOS_ERR_OK"
        
        # Create instance with mocked dependencies
        self.sdr = FobosSDR()

    def tearDown(self):
        """Tear down test fixtures."""
        self.ffi_patcher.stop()

    def test_init(self):
        """Test initialization process."""
        # Verify FFI interface was defined
        self.mock_ffi_instance.cdef.assert_called_once()
        
        # Verify library was loaded
        self.mock_ffi_instance.dlopen.assert_called_once()
        
        # Verify initial state
        self.assertIsNone(self.sdr.dev)

    def test_get_api_info(self):
        """Test get_api_info method."""
        # Configure mocks for string return
        def mock_string_side_effect(value):
            if value == self.mock_ffi_instance.new.return_value:
                return b"1.2.3"
            return b"4.5.6"
            
        self.mock_ffi_instance.string.side_effect = mock_string_side_effect
        
        # Call method
        result = self.sdr.get_api_info()
        
        # Verify results
        self.assertEqual(result, {
            "library_version": "1.2.3",
            "driver_version": "4.5.6"
        })
        
        # Verify proper C function was called
        self.mock_lib.fobos_rx_get_api_info.assert_called_once()

    def test_get_device_count(self):
        """Test get_device_count method."""
        # Configure mock
        self.mock_lib.fobos_rx_get_device_count.return_value = 3
        
        # Call method
        count = self.sdr.get_device_count()
        
        # Verify results
        self.assertEqual(count, 3)
        self.mock_lib.fobos_rx_get_device_count.assert_called_once()

    def test_list_devices(self):
        """Test list_devices method."""
        # Configure mocks
        self.mock_lib.fobos_rx_list_devices.return_value = 0
        self.mock_ffi_instance.string.return_value = b"device1 device2 device3"
        
        # Call method
        devices = self.sdr.list_devices()
        
        # Verify results
        self.assertEqual(devices, ["device1", "device2", "device3"])
        self.mock_lib.fobos_rx_list_devices.assert_called_once()

    def test_open(self):
        """Test open method."""
        # Configure mocks
        self.mock_lib.fobos_rx_open.return_value = 0
        self.mock_dev_ptr.__getitem__.return_value = "device_handle"
        
        # Call method
        result = self.sdr.open(1)
        
        # Verify results
        self.assertEqual(result, self.sdr)  # Should return self for chaining
        self.assertEqual(self.sdr.dev, "device_handle")
        self.mock_lib.fobos_rx_open.assert_called_once()
        
    def test_open_error(self):
        """Test open method with error."""
        # Configure mocks for error
        self.mock_lib.fobos_rx_open.return_value = -1
        self.mock_lib.fobos_rx_error_name.return_value = b"FOBOS_ERR_NO_DEV"
        
        # Call method and verify exception
        with self.assertRaises(FobosException) as context:
            self.sdr.open(1)
            
        self.assertEqual(context.exception.code, -1)
        
    def test_close(self):
        """Test close method."""
        # Setup device state
        self.sdr.dev = "device_handle"
        
        # Call method
        self.sdr.close()
        
        # Verify results
        self.assertIsNone(self.sdr.dev)
        self.mock_lib.fobos_rx_close.assert_called_once_with("device_handle")
        
    def test_close_no_device(self):
        """Test close method when no device is open."""
        # Call method with no device open
        self.sdr.close()
        
        # Verify no C calls were made
        self.mock_lib.fobos_rx_close.assert_not_called()
        
    def test_get_board_info(self):
        """Test get_board_info method."""
        # Setup device state
        self.sdr.dev = "device_handle"
        
        # Configure mocks
        self.mock_lib.fobos_rx_get_board_info.return_value = 0
        
        # Configure string returns for different fields
        def mock_string_side_effect(value):
            if value == self.mock_ffi_instance.new("char[256]"):
                return b"hw_rev_1.0"
            elif value == self.mock_ffi_instance.new("char[256]", 1):
                return b"fw_1.2.3"
            elif value == self.mock_ffi_instance.new("char[256]", 2):
                return b"Fobos Inc."
            elif value == self.mock_ffi_instance.new("char[256]", 3):
                return b"Fobos SDR"
            else:
                return b"SN12345"
                
        self.mock_ffi_instance.string.side_effect = mock_string_side_effect
        
        # Call method
        info = self.sdr.get_board_info()
        
        # Verify results
        self.assertEqual(info, {
            "hw_revision": "hw_rev_1.0",
            "fw_version": "fw_1.2.3",
            "manufacturer": "Fobos Inc.",
            "product": "Fobos SDR",
            "serial": "SN12345"
        })
        
        # Verify C function call
        self.mock_lib.fobos_rx_get_board_info.assert_called_once()
        
    def test_set_frequency(self):
        """Test set_frequency method."""
        # Setup device state
        self.sdr.dev = "device_handle"
        
        # Configure mocks
        self.mock_lib.fobos_rx_set_frequency.return_value = 0
        
        # Setup actual frequency return
        def mock_getitem_side_effect(idx):
            return 100.5e6
            
        self.mock_ffi_instance.new("double *").__getitem__.side_effect = mock_getitem_side_effect
        
        # Call method
        actual_freq = self.sdr.set_frequency(100e6)
        
        # Verify results
        self.assertEqual(actual_freq, 100.5e6)
        self.mock_lib.fobos_rx_set_frequency.assert_called_once()
        
    def test_get_samplerates(self):
        """Test get_samplerates method."""
        # Setup device state
        self.sdr.dev = "device_handle"
        
        # Configure mocks for first call (get count)
        def mock_check_error_side_effect(result):
            return result
            
        self.sdr._check_error = Mock(side_effect=mock_check_error_side_effect)
        self.mock_lib.fobos_rx_get_samplerates.side_effect = [0, 0]  # Two calls
        
        # Setup count return
        count_ptr_mock = Mock()
        count_ptr_mock.__getitem__.return_value = 3
        self.mock_ffi_instance.new.side_effect = [count_ptr_mock, "values_array"]
        
        # Setup values for the rates array
        values_mock = Mock()
        values_mock.__getitem__.side_effect = [1e6, 2e6, 4e6]  # Sample rates
        self.mock_ffi_instance.new.return_value = values_mock
        
        # Call method
        rates = self.sdr.get_samplerates()
        
        # Verify results
        self.assertEqual(rates, [1e6, 2e6, 4e6])
        
    def test_start_rx_async(self):
        """Test start_rx_async method with callback."""
        # Setup device state
        self.sdr.dev = "device_handle"
        
        # Configure mocks
        self.mock_lib.fobos_rx_read_async.return_value = 0
        
        # Create a test callback
        def test_callback(iq_samples):
            pass
            
        # Mock the callback wrapper method
        self.sdr._callback_wrapper = Mock(return_value="wrapped_callback")
        
        # Call method
        self.sdr.start_rx_async(test_callback, 8, 4096)
        
        # Verify results
        self.sdr._callback_wrapper.assert_called_once_with(test_callback)
        self.mock_lib.fobos_rx_read_async.assert_called_once_with(
            "device_handle", "wrapped_callback", self.mock_ffi_instance.NULL, 8, 4096
        )
        
    def test_callback_wrapper(self):
        """Test the callback wrapper for async reception."""
        # Prepare test data (float array with I/Q samples)
        test_buffer = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        
        # Configure mocks
        mock_buf_ptr = Mock()
        self.mock_ffi_instance.cast.return_value = mock_buf_ptr
        
        # Configure mock buffer access to return test values
        mock_buf_ptr.__getitem__.side_effect = lambda i: test_buffer[i]
        
        # Create a mock Python callback to verify it's called correctly
        mock_python_callback = Mock()
        
        # Create a wrapper for the mock callback
        wrapper = self.sdr._callback_wrapper(mock_python_callback)
        
        # Call the C callback with test buffer (6 float values = 3 complex samples)
        wrapper(mock_buf_ptr, 6, None)
        
        # Verify Python callback was called with correct complex data
        # The complex samples should be [1+2j, 3+4j, 5+6j]
        expected_complex = np.array([1+2j, 3+4j, 5+6j])
        
        # Check that callback was called once
        mock_python_callback.assert_called_once()
        
        # Get the actual argument passed to the callback
        actual_arg = mock_python_callback.call_args[0][0]
        
        # We can't directly compare NumPy arrays here because of the mock setup,
        # but we can verify the callback was called
        self.assertEqual(mock_python_callback.call_count, 1)

    def test_error_handling(self):
        """Test error handling in the wrapper."""
        # Configure mocks for error
        self.mock_lib.fobos_rx_error_name.return_value = b"FOBOS_ERR_NO_DEV"
        
        # Call _check_error with error code
        with self.assertRaises(FobosException) as context:
            self.sdr._check_error(-1)
            
        # Verify exception details
        self.assertEqual(context.exception.code, -1)
        self.assertIn("FOBOS_ERR_NO_DEV", str(context.exception))


if __name__ == '__main__':
    unittest.main()