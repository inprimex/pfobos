# Fobos SDR Project Structure

This document outlines the key components and organization of the Fobos SDR Python wrapper project. It serves as a guide to help developers understand the project structure and navigate the codebase efficiently.

## Core Components

### 1. Wrapper Module (`shared/`)

The foundational wrapper for the Fobos SDR C library:

```
shared/
├── __init__.py          # Package exports (FobosSDR, FobosException)
└── fwrapper.py          # Core wrapper implementation with CFFI bindings
```

### 2. Real-Time Analyzer (`rtanalyzer/`)

Spectrum analyzer application built on the wrapper:

```
rtanalyzer/
├── __init__.py          # Package exports
└── rtanalyzer.py        # Spectrum analyzer implementation with matplotlib
```

### 3. FM Receiver (`fmreceiver/`)

FM radio applications:

```
fmreceiver/
├── fobos_fm_receiver.py         # Standard FM receiver
└── fobos_fm_receiver_paplay.py  # FM receiver with PulseAudio output
```

### 4. Tests and Benchmarks (`tests/`)

Comprehensive testing framework:

```
tests/
├── __init__.py           # Test package definition
├── __main__.py           # Test runner with custom result handling
├── benchmark.py          # Performance benchmarking tool
├── test_integration.py   # Hardware integration tests
├── test_mock_fobos.py    # Tests with mocked hardware
├── test_performance.py   # Performance measurement tests
└── test_wrapper_logic.py # Tests for wrapper logic and error handling
```

### 5. Setup and Configuration (`setup/`)

Hardware and environment setup:

```
setup/
└── setup-fobos-sdr.sh    # Udev rules and permissions setup for Linux
```

### 6. Scripts and Utilities (`scripts/`)

Helper scripts and examples:

```
scripts/
├── async_test.py         # Asynchronous reception test
├── check-fobos-sdr.sh    # Hardware detection script
├── fobos-debug.py        # Debugging utilities
├── simple_async_test.py  # Simplified async test
├── sync_test.py          # Synchronous reception test
└── test_matplotlib.py    # Matplotlib configuration test
```

## Entry Points

Primary executable scripts:

```
├── run_rtanalyzer.py     # Launch the real-time spectrum analyzer
├── run_setup.py          # Run setup verification
└── run_tests.py          # Execute test suite with options
```

## Documentation

Comprehensive documentation:

```
doc/
├── benchmark.md          # Benchmark tool documentation
├── fwrapper.md           # API reference for the wrapper
├── rtanalyzer.md         # Spectrum analyzer usage and features
├── setup-fobos-sdr.md    # Hardware setup instructions
├── test_performance.md   # Performance testing framework
└── tests.md              # Testing infrastructure overview
```

## Project Configuration

```
├── requirements.txt      # Python dependencies
├── README.md             # Project overview and quick start
└── LICENSE               # License information
```

## Output Directories

```
├── spectrum_plots/       # Directory for spectrum analyzer output
└── benchmark_results/    # (Created at runtime) Benchmark results storage
```

## Common Tasks

1. **First-time setup**:
   ```bash
   sudo ./setup/setup-fobos-sdr.sh
   python run_setup.py
   ```

2. **Running tests**:
   ```bash
   python run_tests.py                # Run basic tests
   python run_tests.py --integration  # Include hardware tests
   python run_tests.py --benchmark    # Run benchmarks
   ```

3. **Running applications**:
   ```bash
   python run_rtanalyzer.py           # Launch spectrum analyzer
   ```

## Development Workflow

1. Modify wrapper code in `shared/fwrapper.py`
2. Run tests with `python run_tests.py`
3. Update/extend applications in `rtanalyzer/` or `fmreceiver/`
4. Run benchmarks to verify performance with `python run_tests.py --benchmark`
5. Update documentation in `doc/` with any API changes
