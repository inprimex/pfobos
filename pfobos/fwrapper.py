"""
Python wrapper for Fobos SDR library.
This module provides a Pythonic interface to the Fobos SDR C library.
"""

import os
import sys
import ctypes
import numpy as np
import time
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
    """Python wrapper for Fobos SDR library.

    Synchronous reception pattern (pfobos 0.3.0+):

        sdr.start_rx_sync(buf_length)  # allocates buffer; sets _sync_mode = True
        try:
            while keep_going:
                try:
                    iq = sdr.read_rx_sync()  # complex64 ndarray
                except FobosException as e:
                    # Sync mode is still active. Decide what to do:
                    #   - transient USB error (e.code == FobosError.LIBUSB): often safe to retry
                    #   - persistent error: stop_rx_sync() + restart, or close
                    if e.code == FobosError.LIBUSB:
                        continue  # retry
                    raise
                process(iq)
        finally:
            sdr.stop_rx_sync()  # paired with start_rx_sync; caller-owned

    read_rx_sync does NOT auto-stop sync mode on error. The caller owns
    the start/stop lifecycle. See read_rx_sync docstring for the full
    error-handling contract.

    Asynchronous reception pattern:

        def cb(iq_samples): ...  # called from C thread
        sdr.start_rx_async(cb, buf_count=16, buf_length=32768)
        # blocks until fobos_rx_cancel_async is called
        sdr.stop_rx_async()
    """
    
    def __init__(self, lib_path: str = None):
        self.ffi = FFI()
        self._lib_path_override = lib_path
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
        # Explicit path override — used for testing with stub library
        if self._lib_path_override:
            try:
                self.lib = self.ffi.dlopen(self._lib_path_override)
                return
            except OSError as e:
                raise OSError(f"Could not load Fobos SDR library from '{self._lib_path_override}': {e}")

        if sys.platform.startswith('win'):
            lib_name = 'fobos.dll'
        elif sys.platform.startswith('linux'):
            lib_name = 'libfobos.so'
        elif sys.platform.startswith('darwin'):
            lib_name = 'libfobos.dylib'
        else:
            raise OSError(f"Unsupported platform: {sys.platform}")

        # Try system path first, then the pfobos/ package directory alongside this file
        try:
            self.lib = self.ffi.dlopen(lib_name)
        except OSError:
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
        
        # Initialize flag for safe stopping
        self._async_stop_flag = False
        
        # Save the Python callback and create a C callback wrapper
        # The callback reference is kept to prevent garbage collection
        self._python_callback = callback
        
        # Create a wrapper for the callback that checks the stop flag.
        # Per libfobos contract (fobos.c, transfer-completion callback site):
        #   complex_samples_count = transfer->actual_length / 4
        #   dev->rx_cb(dev->rx_buff, complex_samples_count, ctx)
        # So `buf_length` in this callback is IQ pair count (not float count).
        # Buffer holds 2 floats per pair (interleaved I, Q) = 8 bytes per pair.
        # Pre-0.4.1 used `buf_length * 4` which extracted only half the chunk —
        # the async version of the byte-count bug fixed for sync in 0.4.0.
        @self.ffi.callback("void(float *, uint32_t, void *)")
        def _c_callback(buf, buf_length, ctx):
            try:
                # Check stop flag to prevent race conditions during shutdown
                if hasattr(self, '_async_stop_flag') and self._async_stop_flag:
                    return

                if buf_length <= 0:
                    return

                # Extract 2 floats per IQ pair = 8 bytes per pair.
                buffer = np.frombuffer(
                    self.ffi.buffer(buf, buf_length * 8), dtype=np.float32
                ).copy()

                # Defensive: should always be even given the IQ-pair layout
                if len(buffer) % 2 != 0:
                    buffer = buffer[:-1]

                iq_samples = buffer[0::2] + 1j * buffer[1::2]
                
                # Call the Python callback with the samples
                # Use try/except to ensure any callback errors don't propagate back to C
                try:
                    self._python_callback(iq_samples)
                except Exception as e:
                    print(f"Error in user callback: {e}")
            except Exception as e:
                # Log any errors but don't let them propagate back to C
                print(f"Error in async callback wrapper: {e}")
        
        # Store the callback for later use and to prevent garbage collection
        self._callback = _c_callback
        
        # Ensure reasonable buffer values
        if buf_length < 1024:
            buf_length = 1024
        if buf_count < 4:
            buf_count = 4
        
        # Make buffer length even for I/Q pairs
        if buf_length % 2 != 0:
            buf_length += 1
        
        try:
            # Start async reading
            self._check_error(self.lib.fobos_rx_read_async(
                self.dev, self._callback, self.ffi.NULL, buf_count, buf_length
            ))
            
            # Mark as running only after successful start
            self._async_mode = True
            
            # Store parameters for debugging and cleanup
            self._async_buf_count = buf_count
            self._async_buf_length = buf_length
            
        except Exception as e:
            # Clean up references if starting fails
            self._callback = None
            self._python_callback = None
            self._async_stop_flag = False
            raise FobosException(-1, f"Error starting async mode: {e}")

    def stop_rx_async(self):
        """Stop asynchronous receiving of IQ data."""
        if self.dev is not None and self._async_mode:
            # Set stop flag first to prevent callback race conditions
            self._async_stop_flag = True
            
            # Give a small delay to ensure callbacks notice the flag
            time.sleep(0.1)
            
            # Use a timeout approach to ensure we don't get stuck
            start_time = time.time()
            max_wait = 5.0  # Maximum 5 seconds to wait for cancellation
            
            try:
                # Attempt to cancel the async operations
                cancel_result = self.lib.fobos_rx_cancel_async(self.dev)
                
                # Wait for cancellation to complete, but with timeout
                while self._async_mode and time.time() - start_time < max_wait:
                    time.sleep(0.1)
                    # Try to handle events to help process cancellation
                    try:
                        # Optional: If we have access to libusb event handling
                        # self.lib.libusb_handle_events_timeout_completed(self.dev.libusb_ctx, ...)
                        pass
                    except:
                        pass
                
                # Force cleanup if we hit timeout
                if self._async_mode and time.time() - start_time >= max_wait:
                    print("Warning: Async cancellation timed out - forcing cleanup")
                
            except Exception as e:
                # Log error but continue with cleanup
                print(f"Warning: Error stopping async mode: {e}")
            finally:
                # Always clean up references even if cancel fails
                self._callback = None
                self._python_callback = None
                self._async_mode = False
                self._async_stop_flag = False

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

        Contract (pfobos 0.3.0+): on FobosException, sync mode remains
        active. The caller decides whether to:

          - Retry the read (cheap; transient USB hiccups often pass)
          - Clean restart (stop_rx_sync() + start_rx_sync() explicitly)
          - Give up and close (stop_rx_sync() + close())
          - Escalate (propagate the exception up)

        The original FobosException carrying the libfobos error code is
        re-raised verbatim — callers can inspect `.code` to distinguish
        transient errors (e.g. FobosError.LIBUSB = -9) from terminal
        ones. The wrapper does NOT auto-stop sync mode on errors; that
        decision is consumer-owned and matches the documented start/stop
        lifecycle pattern in the class header.

        Returns:
            np.ndarray: Complex IQ samples (complex64).

        Raises:
            RuntimeError: If synchronous mode not started (call start_rx_sync first).
            FobosException: From libfobos errors (code preserved from C return).
        """
        self._check_device_open()

        if not self._sync_mode or self._buffer_ptr is None:
            raise RuntimeError("Synchronous mode not started")

        # Create pointer for actual length
        actual_len_ptr = self.ffi.new("uint32_t *")

        # Read data into buffer. _check_error raises FobosException with the
        # original code from libfobos on ret < 0; we let it propagate without
        # tearing down sync state — caller owns recovery (see contract above).
        ret = self.lib.fobos_rx_read_sync(self.dev, self._buffer_ptr, actual_len_ptr)
        self._check_error(ret)

        # Per libfobos contract: *actual_buf_length is in complex IQ pair count.
        # See rigexpert/libfobos fobos.c: `*actual_buf_length = actual / 4` where
        # `actual` is the byte count of the int16-pair USB transfer (4 bytes per
        # IQ pair). The float* buffer libfobos writes into holds 2 floats per
        # IQ pair (interleaved I, Q) = 8 bytes per IQ pair.
        actual_len = actual_len_ptr[0]

        if actual_len == 0:
            return np.array([], dtype=np.complex64)

        if actual_len > self._buffer_length:
            # libfobos shouldn't write more pairs than we requested.
            actual_len = self._buffer_length

        # Extract 2 floats per IQ pair = 8 bytes per pair. Pre-0.4.0 used
        # actual_len * 4 which silently discarded the second half of every
        # chunk libfobos delivered (the ~47% effective-rate bug). See
        # CHANGELOG 0.4.0.
        buffer = np.frombuffer(
            self.ffi.buffer(self._buffer_ptr, actual_len * 8), dtype=np.float32
        ).copy()

        # Defensive: should always be even given the IQ-pair layout
        if len(buffer) % 2 != 0:
            buffer = buffer[:-1]

        return buffer[0::2] + 1j * buffer[1::2]

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
