# Fobos SDR Python Wrapper Documentation

## Overview

The `fwrapper.py` module provides a comprehensive Python interface to the Fobos Software-Defined Radio (SDR) hardware. This wrapper allows Python developers to easily interact with the Fobos SDR C library using a Pythonic API and seamlessly integrate with NumPy for signal processing.

## Class: FobosError

An `IntEnum` that defines error codes returned by the Fobos SDR library.

```python
class FobosError(IntEnum):
    OK = 0                  # Operation completed successfully
    NO_DEV = -1             # No device found or device error
    NOT_OPEN = -2           # Device not open
    NO_MEM = -3             # Memory allocation error
    CONTROL = -4            # Control transfer error
    ASYNC_IN_SYNC = -5      # Async mode requested while in sync mode
    SYNC_IN_ASYNC = -6      # Sync mode requested while in async mode
    SYNC_NOT_STARTED = -7   # Sync mode not started
    UNSUPPORTED = -8        # Operation not supported by device
    LIBUSB = -9             # LibUSB error
```

## Class: FobosException

A custom exception class raised when the Fobos SDR library returns an error.

### Properties

- `code`: The error code from `FobosError`
- `message`: A human-readable error message

### Methods

#### `__init__(code, message=None)`

Initializes a new `FobosException`.

- **Parameters:**
  - `code` (int): Error code from `FobosError`
  - `message` (str, optional): Custom error message. If not provided, a default message is generated.

## Class: FobosSDR

The main class for interacting with Fobos SDR hardware.

### Properties

- `dev`: Internal device handle (shouldn't be accessed directly)
- `_callback`: Internal callback reference for async mode
- `_callback_ctx`: Internal callback context
- `_buffer_ptr`: Internal buffer pointer
- `_keep_alive_ref`: Reference to prevent garbage collection
- `_async_mode`: Flag indicating if async mode is active
- `_sync_mode`: Flag indicating if sync mode is active

### Constructor

#### `__init__()`

Initializes a new `FobosSDR` instance.

- **Raises:**
  - `OSError`: If the Fobos SDR library can't be loaded

### Basic Methods

#### `get_api_info()`

Gets version information about the Fobos SDR API.

- **Returns:**
  - `dict`: Dictionary containing:
    - `library_version`: Library version string
    - `driver_version`: Driver version string
- **Raises:**
  - `FobosException`: If an error occurs

#### `get_device_count()`

Gets the number of connected Fobos SDR devices.

- **Returns:**
  - `int`: Number of connected devices
- **Raises:**
  - `FobosException`: If an error occurs

#### `list_devices()`

Gets a list of serials for connected Fobos SDR devices.

- **Returns:**
  - `List[str]`: List of device serial numbers
- **Raises:**
  - `FobosException`: If an error occurs

#### `open(index=0)`

Opens a Fobos SDR device by index.

- **Parameters:**
  - `index` (int): Device index, starting from 0
- **Returns:**
  - `self`: For method chaining
- **Raises:**
  - `FobosException`: If the device can't be opened

#### `close()`

Closes the currently open Fobos SDR device.

- **Raises:**
  - `FobosException`: If an error occurs during closing

#### `reset()`

Resets the currently open Fobos SDR device.

- **Raises:**
  - `FobosException`: If an error occurs during reset

#### `get_board_info()`

Gets information about the Fobos SDR board.

- **Returns:**
  - `dict`: Dictionary containing:
    - `hw_revision`: Hardware revision
    - `fw_version`: Firmware version
    - `manufacturer`: Manufacturer name
    - `product`: Product name
    - `serial`: Device serial number
- **Raises:**
  - `FobosException`: If the device is not open or another error occurs

### Configuration Methods

#### `set_frequency(freq_hz)`

Sets the receiver frequency in Hz.

- **Parameters:**
  - `freq_hz` (float): Target frequency in Hz
- **Returns:**
  - `float`: Actual frequency set (may differ slightly from requested)
- **Raises:**
  - `FobosException`: If the device is not open or another error occurs

#### `set_direct_sampling(enabled)`

Enables or disables direct sampling mode.

- **Parameters:**
  - `enabled` (bool): True to enable direct sampling, False to disable
- **Raises:**
  - `FobosException`: If the device is not open or another error occurs

#### `set_lna_gain(value)`

Sets the LNA gain.

- **Parameters:**
  - `value` (int): LNA gain value (0-2)
- **Raises:**
  - `ValueError`: If gain value is out of range
  - `FobosException`: If the device is not open or another error occurs

#### `set_vga_gain(value)`

Sets the VGA gain.

- **Parameters:**
  - `value` (int): VGA gain value (0-15)
- **Raises:**
  - `ValueError`: If gain value is out of range
  - `FobosException`: If the device is not open or another error occurs

#### `get_samplerates()`

Gets available sample rates for the device.

- **Returns:**
  - `List[float]`: List of available sample rates in Hz
- **Raises:**
  - `FobosException`: If the device is not open or another error occurs

#### `set_samplerate(rate_hz)`

Sets the sample rate in Hz.

- **Parameters:**
  - `rate_hz` (float): Target sample rate in Hz
- **Returns:**
  - `float`: Actual sample rate set (may differ slightly from requested)
- **Raises:**
  - `FobosException`: If the device is not open or another error occurs

#### `set_user_gpo(value)`

Sets user general purpose output bits.

- **Parameters:**
  - `value` (int): 8-bit value (0x00-0xFF)
- **Raises:**
  - `ValueError`: If value is out of range
  - `FobosException`: If the device is not open or another error occurs

#### `set_clk_source(external)`

Sets the clock source.

- **Parameters:**
  - `external` (bool): True for external clock, False for internal
- **Raises:**
  - `FobosException`: If the device is not open or another error occurs

### Advanced Frequency Control

#### `set_max2830_frequency(freq_hz)`

Explicitly sets the MAX2830 frequency in Hz.

- **Parameters:**
  - `freq_hz` (float): Target frequency in Hz (2350-2550 MHz)
- **Returns:**
  - `float`: Actual frequency set
- **Raises:**
  - `ValueError`: If frequency is out of valid range
  - `FobosException`: If the device is not open or another error occurs

#### `set_rffc507x_lo_frequency(freq_hz)`

Sets RFFC507x LO frequency in Hz.

- **Parameters:**
  - `freq_hz` (int): Target frequency in Hz (25-5400 MHz)
- **Returns:**
  - `int`: Actual frequency set
- **Raises:**
  - `ValueError`: If frequency is out of valid range
  - `FobosException`: If the device is not open or another error occurs

### Synchronous Reception

#### `start_rx_sync(buf_length=32768)`

Starts synchronous receiving mode.

- **Parameters:**
  - `buf_length` (int): Buffer length in samples (will be adjusted to be a multiple of 2)
- **Raises:**
  - `FobosException`: If the device is not open or another error occurs

#### `read_rx_sync()`

Reads samples in synchronous mode.

- **Returns:**
  - `np.ndarray`: Complex IQ samples as NumPy array
- **Raises:**
  - `RuntimeError`: If synchronous mode not started
  - `FobosException`: If an error occurs during reading

#### `stop_rx_sync()`

Stops synchronous receiving mode.

- **Raises:**
  - `FobosException`: If an error occurs while stopping

### Asynchronous Reception

#### `start_rx_async(callback, buf_count=16, buf_length=32768)`

Starts asynchronous receiving of IQ data.

- **Parameters:**
  - `callback` (Callable[[np.ndarray], None]): Function to call with received IQ samples
  - `buf_count` (int): Number of buffers to use (min: 4)
  - `buf_length` (int): Buffer length in samples (min: 1024, will be adjusted to be a multiple of 2)
- **Raises:**
  - `FobosException`: If the device is not open or another error occurs

#### `stop_rx_async()`

Stops asynchronous receiving of IQ data.

- **Raises:**
  - `FobosException`: If an error occurs while stopping

### Firmware Operations

#### `read_firmware(filename, verbose=False)`

Reads firmware from the device to a file.

- **Parameters:**
  - `filename` (str): Path to save the firmware
  - `verbose` (bool): Enable verbose output
- **Raises:**
  - `FobosException`: If the device is not open or another error occurs

#### `write_firmware(filename, verbose=False)`

Writes firmware to the device from a file.

- **Parameters:**
  - `filename` (str): Path to the firmware file
  - `verbose` (bool): Enable verbose output
- **Raises:**
  - `FobosException`: If the device is not open or another error occurs

### Internal Methods

#### `_define_ffi_interface()`

Defines the CFFI interface for the Fobos SDR library.

#### `_load_library()`

Loads the Fobos SDR library based on the current platform.

- **Raises:**
  - `OSError`: If the library can't be found or loaded

#### `_check_error(result)`

Checks if an API call resulted in an error and raises an exception if so.

- **Parameters:**
  - `result` (int): Return code from the API call
- **Returns:**
  - `int`: The result code if no error
- **Raises:**
  - `FobosException`: If the result code indicates an error

#### `_check_device_open()`

Checks if a device is open.

- **Raises:**
  - `FobosException`: If no device is open

#### `_callback_wrapper(python_callback)`

Creates a C callback wrapper for the Python callback function.

- **Parameters:**
  - `python_callback` (Callable): Python function to call with received samples
- **Returns:**
  - CFFI callback that wraps the Python function

### Context Manager Support

The `FobosSDR` class supports the context manager protocol (with statement).

```python
with FobosSDR() as sdr:
    sdr.open(0)
    # Use the SDR...
    # Device will be automatically closed when exiting the with block
```

#### `__enter__()`

Called when entering a with statement.

- **Returns:**
  - `self`: The FobosSDR instance

#### `__exit__(exc_type, exc_val, exc_tb)`

Called when exiting a with statement. Ensures the device is closed.

## Example Usage

### Basic Usage

```python
# Create and open a device
sdr = FobosSDR()
device_count = sdr.get_device_count()
print(f"Found {device_count} devices")

if device_count > 0:
    # Open first device
    sdr.open(0)
    
    # Get board info
    info = sdr.get_board_info()
    print(f"Connected to {info['product']} (SN: {info['serial']})")
    
    # Set frequency to 100 MHz
    actual_freq = sdr.set_frequency(100e6)
    print(f"Tuned to {actual_freq/1e6:.3f} MHz")
    
    # Set sample rate to 2.048 MHz
    actual_rate = sdr.set_samplerate(2.048e6)
    print(f"Sample rate: {actual_rate/1e6:.3f} MHz")
    
    # Close when done
    sdr.close()
```

### Synchronous Reception

```python
# Get samples in synchronous mode
sdr = FobosSDR()
sdr.open(0)

try:
    sdr.set_frequency(100e6)
    sdr.set_samplerate(2.048e6)
    sdr.set_vga_gain(10)
    
    # Start sync mode with 32KB buffer
    sdr.start_rx_sync(32768)
    
    # Read samples
    iq_samples = sdr.read_rx_sync()
    print(f"Received {len(iq_samples)} IQ samples")
    
    # Calculate spectrum with NumPy
    spectrum = np.fft.fftshift(np.fft.fft(iq_samples))
    power_db = 20 * np.log10(np.abs(spectrum) + 1e-10)
    
finally:
    # Always stop and close
    sdr.stop_rx_sync()
    sdr.close()
```

### Asynchronous Reception

```python
# Define callback function
def sample_callback(iq_samples):
    print(f"Received {len(iq_samples)} samples")
    # Process samples here...

# Set up device
sdr = FobosSDR()
sdr.open(0)

try:
    sdr.set_frequency(100e6)
    sdr.set_samplerate(2.048e6)
    sdr.set_vga_gain(10)
    
    # Start async reception with callback
    sdr.start_rx_async(sample_callback, buf_count=8, buf_length=16384)
    
    # Let it run for a while
    time.sleep(5)
    
finally:
    # Always stop and close
    sdr.stop_rx_async()
    sdr.close()
```

### Using Context Manager

```python
with FobosSDR() as sdr:
    # Check for devices
    if sdr.get_device_count() > 0:
        # Open first device
        sdr.open(0)
        
        # Configure the device
        sdr.set_frequency(915e6)
        sdr.set_samplerate(2.048e6)
        
        # Get data
        sdr.start_rx_sync(4096)
        samples = sdr.read_rx_sync()
        sdr.stop_rx_sync()
        
        # Device is automatically closed when exiting the with block
```

## Error Handling

```python
try:
    sdr = FobosSDR()
    sdr.open(0)
    # Use the SDR...
except FobosException as e:
    print(f"Fobos SDR error ({e.code}): {e.message}")
except ValueError as e:
    print(f"Parameter error: {e}")
except OSError as e:
    print(f"OS error: {e}")
finally:
    if 'sdr' in locals() and hasattr(sdr, 'dev') and sdr.dev is not None:
        sdr.close()
```

## Implementation Details

### Internal Buffer Management

The wrapper carefully manages C buffers using CFFI to prevent memory leaks and ensure safe data transfer:

- Synchronous mode pre-allocates buffers of the requested size
- Asynchronous mode safely bridges between C callbacks and Python functions
- All buffers are properly freed when modes are stopped or device is closed

### Sample Format

- IQ samples are returned as NumPy arrays of complex64 values
- I and Q samples arrive interleaved (I0, Q0, I1, Q1, ...) and are converted to complex values
- For a buffer of length N, approximately N/2 complex samples will be returned

### Thread Safety

- The wrapper is not inherently thread-safe
- Avoid calling methods from multiple threads without synchronization
- All async callbacks are handled on a single thread created by the C library

### Resource Management

- Use context managers (with statement) when possible to ensure proper cleanup
- Always stop async or sync mode before closing the device
- Always close the device when done to free resources
