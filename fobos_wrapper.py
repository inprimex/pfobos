"""
Python wrapper for Fobos SDR library.
This module provides a Pythonic interface to the Fobos SDR C library.
"""

import os
import sys
import ctypes
import numpy as np
from enum import IntEnum
from typing import Callable, List, Optional, Tuple, Union, Any
from cffi import FFI

# Error codes
class FobosError(IntEnum):
    OK = 0
    NO_DEV = -1
    NOT_OPEN = -2
    NO_MEM = -3
    CONTROL = -4
    ASYNC_IN_SYNC = -5
    SYNC_IN_ASYNC = -6
    SYNC_NOT_STARTED = -7
    UNSUPPORTED = -8
    LIBUSB = -9

class FobosException(Exception):
    """Exception raised for Fobos SDR errors."""
    def __init__(self, code: int, message: str = None):
        self.code = code
        self.message = message or f"Fobos error: {code}"
        super().__init__(self.message)


class FobosSDR:
    """Python wrapper for Fobos SDR library."""
    
    def __init__(self):
        self.ffi = FFI()
        self._define_ffi_interface()
        self._load_library()
        self.dev = None
        self._callback = None
        self._callback_ctx = None
        self._buffer_ptr = None
        self._keep_alive_ref = None
        self._async_mode = False
        self._sync_mode = False

    def _define_ffi_interface(self):
        self.ffi.cdef("""
        struct fobos_dev_t;
        typedef void(*fobos_rx_cb_t)(float *buf, uint32_t buf_length, void *ctx);
        
        int fobos_rx_get_api_info(char * lib_version, char * drv_version);
        int fobos_rx_get_device_count(void);
        int fobos_rx_list_devices(char * serials);
        int fobos_rx_open(struct fobos_dev_t ** out_dev, uint32_t index);
        int fobos_rx_close(struct fobos_dev_t * dev);
        int fobos_rx_reset(struct fobos_dev_t * dev);
        int fobos_rx_get_board_info(struct fobos_dev_t * dev, char * hw_revision, char * fw_version, char * manufacturer, char * product, char * serial);
        int fobos_rx_set_frequency(struct fobos_dev_t * dev, double value, double * actual);
        int fobos_rx_set_direct_sampling(struct fobos_dev_t * dev, unsigned int enabled);
        int fobos_rx_set_lna_gain(struct fobos_dev_t * dev, unsigned int value);
        int fobos_rx_set_vga_gain(struct fobos_dev_t * dev, unsigned int value);
        int fobos_rx_get_samplerates(struct fobos_dev_t * dev, double * values, unsigned int * count);
        int fobos_rx_set_samplerate(struct fobos_dev_t * dev, double value, double * actual);
        int fobos_rx_read_async(struct fobos_dev_t * dev, fobos_rx_cb_t cb, void *ctx, uint32_t buf_count, uint32_t buf_length);
        int fobos_rx_cancel_async(struct fobos_dev_t * dev);
        int fobos_rx_set_user_gpo(struct fobos_dev_t * dev, uint8_t value);
        int fobos_rx_set_clk_source(struct fobos_dev_t * dev, int value);
        int fobos_max2830_set_frequency(struct fobos_dev_t * dev, double value, double * actual);
        int fobos_rffc507x_set_lo_frequency_hz(struct fobos_dev_t * dev, uint64_t lo_freq, uint64_t * tune_freq_hz);
        int fobos_rx_start_sync(struct fobos_dev_t * dev, uint32_t buf_length);
        int fobos_rx_read_sync(struct fobos_dev_t * dev, float * buf, uint32_t * actual_buf_length);
        int fobos_rx_stop_sync(struct fobos_dev_t * dev);
        int fobos_rx_read_firmware(struct fobos_dev_t* dev, const char * file_name, int verbose);
        int fobos_rx_write_firmware(struct fobos_dev_t* dev, const char * file_name, int verbose);
        const char * fobos_rx_error_name(int error);
        """)

    def _load_library(self):
        """Load the Fobos SDR library based on the current platform."""
        if sys.platform.startswith('win'):
            lib_name = 'fobos.dll'
        elif sys.platform.startswith('linux'):
            lib_name = 'libfobos.so'
        elif sys.platform.startswith('darwin'):
            lib_name = 'libfobos.dylib'
        else:
            raise OSError(f"Unsupported platform: {sys.platform}")
        
        # Try to locate the library
        try:
            self.lib = self.ffi.dlopen(lib_name)
        except OSError:
            # Try current directory if not in path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            lib_path = os.path.join(current_dir, lib_name)
            try:
                self.lib = self.ffi.dlopen(lib_path)
            except OSError as e:
                raise OSError(f"Could not load Fobos SDR library: {e}")
    
    def _check_error(self, result: int):
        """Check if an API call resulted in an error and raise an exception if so."""
        if result < 0:
            error_str = self.ffi.string(self.lib.fobos_rx_error_name(result)).decode('utf-8')
            raise FobosException(result, f"Fobos error {result}: {error_str}")
        return result

    def get_api_info(self) -> dict:
        """Get API version information."""
        lib_version = self.ffi.new("char[256]")
        drv_version = self.ffi.new("char[256]")
        self._check_error(self.lib.fobos_rx_get_api_info(lib_version, drv_version))
        return {
            "library_version": self.ffi.string(lib_version).decode('utf-8'),
            "driver_version": self.ffi.string(drv_version).decode('utf-8')
        }

    def get_device_count(self) -> int:
        """Get the number of connected Fobos SDR devices."""
        return self._check_error(self.lib.fobos_rx_get_device_count())

    def list_devices(self) -> List[str]:
        """Get a list of connected device serial numbers."""
        serials = self.ffi.new("char[1024]")
        self._check_error(self.lib.fobos_rx_list_devices(serials))
        serial_str = self.ffi.string(serials).decode('utf-8')
        return serial_str.strip().split() if serial_str.strip() else []

    def open(self, index: int = 0):
        """Open a Fobos SDR device by index."""
        if self.dev is not None:
            self.close()
        
        dev_ptr = self.ffi.new("struct fobos_dev_t **")
        self._check_error(self.lib.fobos_rx_open(dev_ptr, index))
        self.dev = dev_ptr[0]
        return self

    def close(self):
        """Close the Fobos SDR device."""
        if self.dev is not None:
            # Make sure to stop any active streaming first
            if self._async_mode:
                self.stop_rx_async()
            if self._sync_mode:
                self.stop_rx_sync()
                
            self._check_error(self.lib.fobos_rx_close(self.dev))
            self.dev = None

    def reset(self):
        """Reset the Fobos SDR device."""
        if self.dev is not None:
            # Make sure to stop any active streaming first
            if self._async_mode:
                self.stop_rx_async()
            if self._sync_mode:
                self.stop_rx_sync()
                
            self._check_error(self.lib.fobos_rx_reset(self.dev))
            self.dev = None

    def get_board_info(self) -> dict:
        """Get information about the Fobos SDR board."""
        self._check_device_open()
        
        hw_revision = self.ffi.new("char[256]")
        fw_version = self.ffi.new("char[256]")
        manufacturer = self.ffi.new("char[256]")
        product = self.ffi.new("char[256]")
        serial = self.ffi.new("char[256]")
        
        self._check_error(self.lib.fobos_rx_get_board_info(
            self.dev, hw_revision, fw_version, manufacturer, product, serial
        ))
        
        return {
            "hw_revision": self.ffi.string(hw_revision).decode('utf-8'),
            "fw_version": self.ffi.string(fw_version).decode('utf-8'),
            "manufacturer": self.ffi.string(manufacturer).decode('utf-8'),
            "product": self.ffi.string(product).decode('utf-8'),
            "serial": self.ffi.string(serial).decode('utf-8')
        }

    def set_frequency(self, freq_hz: float) -> float:
        """Set the receiver frequency in Hz."""
        self._check_device_open()
        actual = self.ffi.new("double *")
        self._check_error(self.lib.fobos_rx_set_frequency(self.dev, freq_hz, actual))
        return actual[0]

    def set_direct_sampling(self, enabled: bool):
        """Enable or disable direct sampling mode."""
        self._check_device_open()
        self._check_error(self.lib.fobos_rx_set_direct_sampling(self.dev, 1 if enabled else 0))

    def set_lna_gain(self, value: int):
        """Set the LNA gain (0-2)."""
        self._check_device_open()
        if not 0 <= value <= 2:
            raise ValueError("LNA gain must be between 0 and 2")
        self._check_error(self.lib.fobos_rx_set_lna_gain(self.dev, value))

    def set_vga_gain(self, value: int):
        """Set the VGA gain (0-15)."""
        self._check_device_open()
        if not 0 <= value <= 15:
            raise ValueError("VGA gain must be between 0 and 15")
        self._check_error(self.lib.fobos_rx_set_vga_gain(self.dev, value))

    def get_samplerates(self) -> List[float]:
        """Get available sample rates."""
        self._check_device_open()
        
        # First get the count
        count_ptr = self.ffi.new("unsigned int *")
        self._check_error(self.lib.fobos_rx_get_samplerates(self.dev, self.ffi.NULL, count_ptr))
        count = count_ptr[0]
        
        # Then get the values
        values = self.ffi.new(f"double[{count}]")
        self._check_error(self.lib.fobos_rx_get_samplerates(self.dev, values, count_ptr))
        
        return [values[i] for i in range(count)]

    def set_samplerate(self, rate_hz: float) -> float:
        """Set the sample rate in Hz."""
        self._check_device_open()
        actual = self.ffi.new("double *")
        self._check_error(self.lib.fobos_rx_set_samplerate(self.dev, rate_hz, actual))
        return actual[0]

    def _callback_wrapper(self, python_callback):
        """Create a C callback wrapper for the Python callback function."""
        @self.ffi.callback("void(float *, uint32_t, void *)")
        def _c_callback(buf, buf_length, ctx):
            try:
                # Convert the buffer to a numpy array - use a safer approach with bounds checking
                buffer_size = buf_length
                if buffer_size <= 0:
                    return
                
                # Create a numpy array and copy data from the CFFI buffer
                buffer = np.zeros(buffer_size, dtype=np.float32)
                for i in range(buffer_size):
                    buffer[i] = self.ffi.cast("float *", buf)[i]
                
                # Make sure buffer length is even for complex conversion
                if len(buffer) % 2 != 0:
                    buffer = buffer[:-1]
                    
                # Reshape the buffer to complex IQ samples (I and Q are interleaved)
                iq_samples = buffer[0::2] + 1j * buffer[1::2]
                
                # Call the Python callback with the samples
                python_callback(iq_samples)
            except Exception as e:
                # Log any errors but don't let them propagate back to C
                print(f"Error in async callback: {e}")
            
        return _c_callback

    def start_rx_async(self, callback: Callable[[np.ndarray], None], buf_count: int = 16, buf_length: int = 32768):
        """Start asynchronous receiving of IQ data.
        
        Args:
            callback: Function to call with received IQ samples
            buf_count: Number of buffers to use
            buf_length: Buffer length in number of samples
                      
        Note: 
            Buffer length is in terms of float values, so for complex samples, 
            a buffer of 32768 will contain 16384 complex I/Q pairs.
        """
        self._check_device_open()
        
        # Make sure we're not already in another mode
        if self._sync_mode:
            self.stop_rx_sync()
        if self._async_mode:
            self.stop_rx_async()
        
        # Save the Python callback and create a C callback wrapper
        # The callback reference is kept to prevent garbage collection
        self._python_callback = callback
        self._callback = self._callback_wrapper(callback)
        
        # Ensure reasonable buffer values
        if buf_length < 1024:
            buf_length = 1024
        if buf_count < 4:
            buf_count = 4
        
        # Make buffer length even for I/Q pairs
        if buf_length % 2 != 0:
            buf_length += 1
        
        # Start async reading
        self._check_error(self.lib.fobos_rx_read_async(
            self.dev, self._callback, self.ffi.NULL, buf_count, buf_length
        ))
        
        self._async_mode = True

    def stop_rx_async(self):
        """Stop asynchronous receiving of IQ data."""
        if self.dev is not None and self._async_mode:
            try:
                self._check_error(self.lib.fobos_rx_cancel_async(self.dev))
            except Exception as e:
                # Just log error since we're cleaning up
                print(f"Warning: Error stopping async mode: {e}")
            finally:
                # Clean up references
                self._callback = None
                self._python_callback = None
                self._async_mode = False

    def start_rx_sync(self, buf_length: int = 32768):
        """Start synchronous receiving mode with a larger default buffer.
        
        Args:
            buf_length: Buffer length in number of samples (I+Q pairs)
                       Must be a multiple of 2, defaults to 32768
        """
        self._check_device_open()
        
        # Make sure we're not already in another mode
        if self._async_mode:
            self.stop_rx_async()
        if self._sync_mode:
            self.stop_rx_sync()
        
        # Ensure reasonable buffer size
        if buf_length < 1024:
            buf_length = 1024
            
        # Ensure buffer length is a multiple of 2 for I/Q pairs
        if buf_length % 2 != 0:
            buf_length += 1
        
        # Store the buffer length for later use
        self._buffer_length = buf_length
        
        # Start sync mode - this tells the device what buffer size to use
        self._check_error(self.lib.fobos_rx_start_sync(self.dev, buf_length))
        
        # Pre-allocate a buffer for read_sync - add some padding for safety
        # Multiply by 2 to ensure we have enough space
        self._buffer_ptr = self.ffi.new(f"float[{buf_length * 2}]")
        
        # Keep a reference to prevent garbage collection
        self._keep_alive_ref = self._buffer_ptr
        self._sync_mode = True

    def read_rx_sync(self) -> np.ndarray:
        """Read samples in synchronous mode and return complex IQ array.
        
        Returns:
            np.ndarray: Complex IQ samples
        
        Raises:
            RuntimeError: If synchronous mode not started
            FobosException: If an error occurs during reading
        """
        self._check_device_open()
        
        if not self._sync_mode or self._buffer_ptr is None:
            raise RuntimeError("Synchronous mode not started")
        
        # Create pointer for actual length
        actual_len_ptr = self.ffi.new("uint32_t *")
        
        try:
            # Read data into buffer
            ret = self.lib.fobos_rx_read_sync(self.dev, self._buffer_ptr, actual_len_ptr)
            self._check_error(ret)
            
            # Get actual number of samples read
            actual_len = actual_len_ptr[0]
            
            if actual_len == 0:
                # No data received
                return np.array([], dtype=np.complex64)
            
            if actual_len > self._buffer_length * 2:
                # Sanity check - shouldn't happen if C library is behaving
                actual_len = self._buffer_length * 2
            
            # Copy data to numpy array - safer than using fromiter
            buffer = np.zeros(actual_len, dtype=np.float32)
            for i in range(actual_len):
                buffer[i] = self._buffer_ptr[i]
            
            # Make sure length is even for complex conversion
            if len(buffer) % 2 != 0:
                buffer = buffer[:-1]
            
            # Create complex array from interleaved I/Q data
            iq_samples = buffer[0::2] + 1j * buffer[1::2]
            
            return iq_samples
            
        except Exception as e:
            # Make sure to stop sync mode if an error occurs to avoid device hanging
            self.stop_rx_sync()
            raise FobosException(-1, f"Error in read_rx_sync: {e}")

    def stop_rx_sync(self):
        """Stop synchronous receiving mode."""
        if self.dev is not None and self._sync_mode:
            try:
                self._check_error(self.lib.fobos_rx_stop_sync(self.dev))
            except Exception as e:
                # Just log this error since we're cleaning up
                print(f"Warning: Error stopping sync mode: {e}")
            finally:
                # Clean up references to allow garbage collection
                self._buffer_ptr = None
                self._keep_alive_ref = None
                self._sync_mode = False

    def set_user_gpo(self, value: int):
        """Set user general purpose output bits (0x00-0xFF)."""
        self._check_device_open()
        if not 0 <= value <= 255:
            raise ValueError("GPO value must be between 0 and 255")
        self._check_error(self.lib.fobos_rx_set_user_gpo(self.dev, value))

    def set_clk_source(self, external: bool):
        """Set clock source (0: internal, 1: external)."""
        self._check_device_open()
        self._check_error(self.lib.fobos_rx_set_clk_source(self.dev, 1 if external else 0))

    def set_max2830_frequency(self, freq_hz: float) -> float:
        """Explicitly set the MAX2830 frequency in Hz (2350-2550 MHz)."""
        self._check_device_open()
        if not 2.35e9 <= freq_hz <= 2.55e9:
            raise ValueError("MAX2830 frequency must be between 2350 MHz and 2550 MHz")
        actual = self.ffi.new("double *")
        self._check_error(self.lib.fobos_max2830_set_frequency(self.dev, freq_hz, actual))
        return actual[0]

    def set_rffc507x_lo_frequency(self, freq_hz: int) -> int:
        """Set RFFC507x LO frequency in Hz (25-5400 MHz)."""
        self._check_device_open()
        if not 25e6 <= freq_hz <= 5.4e9:
            raise ValueError("RFFC507x frequency must be between 25 MHz and 5400 MHz")
        actual = self.ffi.new("uint64_t *")
        self._check_error(self.lib.fobos_rffc507x_set_lo_frequency_hz(self.dev, freq_hz, actual))
        return actual[0]

    def read_firmware(self, filename: str, verbose: bool = False):
        """Read firmware from the device to a file."""
        self._check_device_open()
        self._check_error(self.lib.fobos_rx_read_firmware(
            self.dev, self.ffi.new("char[]", filename.encode('utf-8')), 1 if verbose else 0
        ))

    def write_firmware(self, filename: str, verbose: bool = False):
        """Write firmware to the device from a file."""
        self._check_device_open()
        self._check_error(self.lib.fobos_rx_write_firmware(
            self.dev, self.ffi.new("char[]", filename.encode('utf-8')), 1 if verbose else 0
        ))

    def _check_device_open(self):
        """Check if a device is open."""
        if self.dev is None:
            raise FobosException(FobosError.NOT_OPEN, "Device not open")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example usage
if __name__ == "__main__":
    try:
        # Create a Fobos SDR instance
        sdr = FobosSDR()
        
        # Get API info
        api_info = sdr.get_api_info()
        print(f"Library version: {api_info['library_version']}")
        print(f"Driver version: {api_info['driver_version']}")
        
        # Get connected devices
        device_count = sdr.get_device_count()
        print(f"Found {device_count} devices")
        
        if device_count > 0:
            # List device serials
            serials = sdr.list_devices()
            print(f"Device serials: {serials}")
            
            # Open first device
            sdr.open(0)
            
            # Get board info
            info = sdr.get_board_info()
            print(f"Hardware revision: {info['hw_revision']}")
            print(f"Firmware version: {info['fw_version']}")
            print(f"Manufacturer: {info['manufacturer']}")
            print(f"Product: {info['product']}")
            print(f"Serial: {info['serial']}")
            
            # Set frequency to 100 MHz
            actual_freq = sdr.set_frequency(100e6)
            print(f"Set frequency to {actual_freq/1e6:.3f} MHz")
            
            # Set sample rate to 2.048 MHz
            available_rates = sdr.get_samplerates()
            print(f"Available sample rates: {[rate/1e6 for rate in available_rates]} MHz")
            
            actual_rate = sdr.set_samplerate(2.048e6)
            print(f"Set sample rate to {actual_rate/1e6:.3f} MHz")
            
            # Example of synchronous receiving
            print("Starting synchronous reception...")
            sdr.start_rx_sync(32768)  # Use larger buffer
            
            # Read some samples
            iq_data = sdr.read_rx_sync()
            print(f"Received {len(iq_data)} IQ samples")
            if len(iq_data) > 0:
                print(f"First 5 samples: {iq_data[:5]}")
            
            # Stop synchronous receiving
            sdr.stop_rx_sync()
            
            # Close the device
            sdr.close()
    
    except FobosException as e:
        print(f"Fobos SDR error: {e}")
    except Exception as e:
        print(f"Error: {e}")
