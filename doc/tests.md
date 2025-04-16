# Fobos SDR Tests

This directory contains tests for the Fobos SDR Python wrapper. The tests are designed to verify the functionality of the wrapper without requiring actual hardware, except for the integration tests.

## Test Structure

The tests are organized into three main categories:

1. **Mock Tests** (`test_mock_fobos.py`): These tests use a fully mocked Fobos SDR library to verify the basic functionality of the wrapper without requiring hardware.

2. **Logic Tests** (`test_wrapper_logic.py`): These tests focus on parameter validation, error handling, and the logical components of the wrapper that can be tested without hardware.

3. **Integration Tests** (`test_integration.py`): These tests require actual Fobos SDR hardware to be connected and test the wrapper's ability to communicate with the hardware.

## Running Tests

### All Tests (Except Integration)

```bash
python run_tests.py
```

### Including Integration Tests (Requires Hardware)

```bash
python run_tests.py --integration
```

### Verbose Output

```bash
python run_tests.py --verbose
```

### Selective Test Categories

```bash
# Skip mock tests
python run_tests.py --no-mock

# Skip logic tests
python run_tests.py --no-logic

# Only integration tests
python run_tests.py --integration --no-mock --no-logic
```

## Using unittest Directly

You can also run tests using Python's unittest module:

```bash
# Run all tests
python -m unittest discover -s tests

# Run a specific test file
python -m unittest tests.test_mock_fobos

# Run a specific test class
python -m unittest tests.test_integration.TestFobosSDRIntegration

# Run a specific test method
python -m unittest tests.test_integration.TestFobosSDRIntegration.test_basic_device_info

# Skip integration tests (requires hardware)
python -m unittest discover -k "not requires_hardware"
```

## Async Testing Note

The asynchronous reception mode tests have been carefully designed to prevent deadlock or segmentation faults. We've observed issues with stopping async mode, so these tests:

1. Run for a limited time (max 2 seconds)
2. Use a small buffer size for quicker response
3. Have multiple safety checks to ensure cleanup even if the test fails
4. Set a flag before attempting to stop async mode to prevent callbacks from processing more data

If you encounter issues with the async tests, try increasing the waiting time between starting and stopping async mode.

## System Setup for Testing

Before running integration tests, ensure:

1. Fobos SDR device is properly connected
2. Proper permissions are set (run `sudo ./setup/setup-fobos-sdr.sh` if needed)
3. The Fobos SDR shared library is installed and accessible
