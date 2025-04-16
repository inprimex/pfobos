# Fobos SDR Spectrum Analyzer

## Overview

This project provides an enhanced real-time spectrum analyzer for the Fobos Software-Defined Radio (SDR) device. The application offers a powerful, interactive interface for spectrum analysis with flexible configuration options.

## Features

- Real-time spectrum visualization
- Dynamic SDR parameter configuration
- Interactive user interface
- Configurable frequency, sample rate, and gain settings
- Automatic plot saving
- Error handling and live updates

## Prerequisites

### Hardware
- Fobos SDR Device

### Software Requirements
- Python 3.7+
- Linux, macOS, or Windows operating system

### Dependencies

Install the required Python packages:

```bash
pip install numpy scipy matplotlib cffi
```

#### Specific Package Versions
- NumPy: Data manipulation
- SciPy: Signal processing
- Matplotlib: Plotting and UI
- CFFI: C Foreign Function Interface for SDR library communication

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/fobos-sdr-analyzer.git
cd fobos-sdr-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Architecture

### Directory Structure

The project follows a modular architecture to support multiple proof-of-concept (POC) implementations:

```
project_root/
│
├── shared/               # Shared components used across all POCs
│   ├── __init__.py       # Package definition exposing FobosSDR and FobosException
│   └── fwrapper.py       # SDR wrapper implementing the hardware interface
│
├── rtanalyzer/           # Spectrum analyzer implementation
│   ├── __init__.py       # Package definition exposing EnhancedRealTimeAnalyzer
│   └── rtanalyzer.py     # Main analyzer implementation
│
├── doc/                  # Documentation
│   └── rtanalyzer.md     # This documentation file
│
├── run_rtanalyzer.py     # Entry point script for spectrum analyzer
└── requirements.txt      # Python dependencies
```

### Technical Design

#### Shared Module
The `shared` module uses Python's package system to provide reusable components:

- **fwrapper.py**: Implements `FobosSDR` class that wraps C library functions using CFFI
- **__init__.py**: Properly exports `FobosSDR` and `FobosException` classes

#### Analyzer Implementation
The spectrum analyzer uses a class-based approach with these key components:

- **UI Layer**: Built with Matplotlib for real-time plotting and interactive controls
- **SDR Communication**: Uses the shared FobosSDR wrapper for device interaction
- **Configuration Management**: Handles parameter validation and live updates

#### Import Strategy
The project uses relative imports and path manipulation to ensure components can find each other:

```python
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.fwrapper import FobosSDR, FobosException
```

This approach allows:
- Running scripts from any directory
- Adding new POC modules without changing imports
- Maintaining clean separation between components

## Configuration Parameters

The analyzer supports configuration of the following parameters:

| Parameter | Range | Description |
|-----------|-------|-------------|
| Center Frequency | 10 MHz - 2.5 GHz | SDR center frequency |
| Sample Rate | 1 MHz - 20 MHz | Data sampling rate |
| FFT Size | 256 - 4096 (Power of 2) | Spectral resolution |
| VGA Gain | 0 - 15 | Variable Gain Amplifier |
| LNA Gain | 0 - 2 | Low Noise Amplifier |
| Buffer Size | 1024 - 65536 | Sample buffer length |
| Update Interval | 100 - 2000 ms | Plot refresh rate |
| Save Interval | 1 - 30 seconds | Automatic plot save frequency |

## Running the Application

Use the entry point script from the project root:

```bash
python run_rtanalyzer.py
```

This script:
1. Sets up proper Python path
2. Imports the analyzer class
3. Handles exceptions gracefully

### User Interface

- **Left Panel**: Configuration inputs
- **Main Plot**: Real-time spectrum visualization
- **Bottom Buttons**:
  - `Start`: Begin SDR acquisition
  - `Stop`: Halt acquisition
  - `Save`: Capture current spectrum plot

## Implementation Details

### Signal Processing Flow
1. **Data Acquisition**: Reads complex I/Q samples from SDR
2. **Windowing**: Applies Hann window function to reduce spectral leakage
3. **FFT Processing**: Computes Fast Fourier Transform with FFT size parameter
4. **Power Calculation**: Converts to dB scale (20*log10)
5. **Smoothing**: Applies exponential smoothing filter (adjustable alpha)

### Parameter Validation
All user inputs are validated against minimum/maximum ranges and additional constraints:
- FFT Size must be a power of 2
- Buffer Size must be large enough for processing
- Sample Rate must be within device capabilities

### Update Strategy
The code implements two key update approaches:
1. **Timer-based Plot Updates**: Refresh visualization at configurable intervals
2. **Live Parameter Changes**: Some parameters can be changed during operation

## Troubleshooting

### Common Issues and Solutions

#### No Device Found
- **Symptoms**: "No SDR devices found" error at startup
- **Causes**: 
  - Device not connected
  - Missing driver installation
  - Inadequate permissions
- **Solutions**:
  - Check USB connection
  - Install required drivers (platform-specific)
  - On Linux: Add udev rules or run with sudo (temporarily)

#### Library Loading Failure
- **Symptoms**: "Could not load Fobos SDR library" error
- **Causes**:
  - Missing Fobos SDR shared library (.dll, .so, or .dylib)
  - Library in wrong location
- **Solutions**:
  - Ensure library is installed and in system path
  - Place library file in project root or specify path in environment

#### Import Errors
- **Symptoms**: ModuleNotFoundError or ImportError
- **Causes**: Python path issues after code reorganization
- **Solutions**:
  - Ensure `__init__.py` files exist in all package directories
  - Run from project root directory
  - Check that sys.path is properly configured in modules

#### No Signal Visible
- **Symptoms**: Flat line or only noise visible in spectrum
- **Causes**: Incorrect frequency, insufficient gain, or no signal present
- **Solutions**:
  - Increase VGA gain (8-15 range typically works well)
  - Verify center frequency setting
  - Check antenna connection

### Debugging Tips

1. **Check Console Output**: The application provides real-time feedback on:
   - Parameter changes
   - Device operations
   - Error conditions

2. **Inspect Saved Plots**: Review generated plots in `spectrum_plots/` for anomalies

3. **Device Reset**: If the SDR behaves erratically:
   - Stop the application
   - Disconnect and reconnect the device
   - Restart the application

4. **Version Compatibility**: Ensure you're using compatible versions of:
   - Python (3.7+)
   - NumPy and SciPy
   - Matplotlib (backend may need adjustment for your platform)
   - Fobos SDR firmware

## Development

### Extending the Project

#### Adding New POC Modules
1. Create a new directory in project root
2. Add `__init__.py` to make it a package
3. Import `FobosSDR` from the shared module
4. Create a runner script in root directory

#### Modifying the Analyzer
Key areas for enhancement:
- Additional signal processing in `update_plot()`
- New UI controls in `create_plot()`
- Support for different visualization modes
- Export capabilities for spectrum data

### Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Specify your project's license here]

## Contact

[Your contact information or project maintainer details]
