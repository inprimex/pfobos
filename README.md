# PFobos SDR Python Wrapper

<span style="color:red">The development in the starting point</span>

A comprehensive Python wrapper for the Fobos Software Defined Radio (SDR) C library. This project allows Python developers to easily interface with Fobos SDR hardware for various radio applications including spectrum analysis and FM demodulation.

![Fobos SDR](https://github.com/rigexpert/libfobos)

## Features

- Full Python interface to all Fobos SDR C library functions
- Easy-to-use object-oriented API
- Support for both synchronous and asynchronous I/Q sample collection
- NumPy integration for efficient signal processing
- Example applications for spectrum analysis and FM radio reception
- Comprehensive error handling and resource management
- Cross-platform compatibility (Windows, Linux, macOS)

## Installation

### Prerequisites

- Python 3.7 or higher
- NumPy
- SciPy
- CFFI
- Matplotlib (for spectrum analyzer example)
- SoundDevice (for FM receiver example)
- Fobos SDR hardware
- Fobos SDR C library for your platform

### Install Dependencies

```bash
pip install numpy scipy cffi matplotlib sounddevice
```

### Library Installation

1. Obtain the Fobos SDR library for your platform:
   - Windows: `fobos.dll`
   - Linux: `libfobos.so`
   - macOS: `libfobos.dylib`

2. Place the library in your system path or in the same directory as your Python code.

3. Clone this repository:
```bash
git clone https://github.com/yourusername/fobos-sdr-python.git
cd fobos-sdr-python
```

## Quick Start

```python
from fobos_wrapper import FobosSDR

# Create and open a device
with FobosSDR() as sdr:
    # Get connected devices
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
        
        # Get samples in synchronous mode
        sdr.start_rx_sync(1024)
        iq_samples = sdr.read_rx_sync()
        sdr.stop_rx_sync()
        
        print(f"Received {len(iq_samples)} IQ samples")
```

## Example Applications

### Spectrum Analyzer

Run the spectrum analyzer example to visualize the RF spectrum:

```bash
python fobos_spectrum_analyzer.py
```

This will display a real-time spectrum plot centered at 100 MHz with a 2.048 MHz bandwidth.

### FM Radio Receiver

Run the FM radio receiver example to demodulate and listen to FM broadcasts:

```bash
python fobos_fm_receiver.py -f 95.5
```

Optional arguments:
- `-f, --frequency`: FM station frequency in MHz (default: 95.5)
- `-g, --gain`: Receiver gain in dB (default: 12)
- `-d, --device`: Audio output device name or ID

## API Reference

### FobosSDR Class

The main class providing access to the SDR hardware.

#### Basic Methods

- `get_api_info()`: Get API version information
- `get_device_count()`: Get the number of connected devices
- `list_devices()`: Get a list of connected device serial numbers
- `open(index)`: Open a device by index
- `close()`: Close the device
- `reset()`: Reset the device
- `get_board_info()`: Get device information

#### Configuration Methods

- `set_frequency(freq_hz)`: Set receiver frequency in Hz
- `set_direct_sampling(enabled)`: Enable/disable direct sampling mode
- `set_lna_gain(value)`: Set LNA gain (0-2)
- `set_vga_gain(value)`: Set VGA gain (0-15)
- `get_samplerates()`: Get available sample rates
- `set_samplerate(rate_hz)`: Set sample rate in Hz
- `set_user_gpo(value)`: Set user GPO bits
- `set_clk_source(external)`: Set clock source (internal/external)

#### Advanced Frequency Control

- `set_max2830_frequency(freq_hz)`: Set MAX2830 frequency explicitly
- `set_rffc507x_lo_frequency(freq_hz)`: Set RFFC507x LO frequency

#### Synchronous Reception

- `start_rx_sync(buf_length)`: Start synchronous reception
- `read_rx_sync()`: Read samples synchronously (returns NumPy array)
- `stop_rx_sync()`: Stop synchronous reception

#### Asynchronous Reception

- `start_rx_async(callback, buf_count, buf_length)`: Start asynchronous reception
- `stop_rx_async()`: Stop asynchronous reception

#### Firmware Operations

- `read_firmware(filename, verbose)`: Read firmware from device to file
- `write_firmware(filename, verbose)`: Write firmware from file to device

## Error Handling

The wrapper converts C library error codes to Python exceptions. All errors are raised as `FobosException` with appropriate error messages.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Fobos SDR team for the hardware and C library
- Contributors to this project
- The open-source SDR community

## Troubleshooting

### Common Issues

1. **Library not found**: Ensure the Fobos SDR library is in your system path or the same directory as your Python code.

2. **No devices found**: Check that your Fobos SDR is properly connected and recognized by your system.

3. **Permission errors**: On Linux, you may need to add udev rules to access the SDR hardware without root privileges.

4. **Import errors**: Verify that all required Python dependencies are installed.

## Contact

If you have questions, suggestions, or need help with this project, please:

- Open an issue on GitHub
- Contact the project maintainer at: [https://sunflowers.online]
