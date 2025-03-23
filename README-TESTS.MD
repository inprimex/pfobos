# Fobos SDR Wrapper Test Suite

A comprehensive testing solution for the Fobos SDR Python wrapper library. This test suite verifies the wrapper's functionality across multiple levels, from mock-based unit tests to hardware integration tests.

## Overview

This test suite consists of three main components:

1. **Mock Tests** - Test the wrapper's interaction with the C library without real hardware
2. **Wrapper Logic Tests** - Test the Python logic, input validation, and error handling
3. **Integration Tests** - Test the wrapper with actual Fobos SDR hardware

![Test Architecture](https://via.placeholder.com/800x400?text=Fobos+SDR+Test+Architecture)

## Requirements

- Python 3.7+
- Fobos SDR Python wrapper (fobos_wrapper.py)
- Python packages:
  - unittest (standard library)
  - numpy
  - logging (standard library)
- Fobos SDR hardware (for integration tests only)

## Installation

1. Clone the repository or copy the test files to your project:

```bash
git clone https://github.com/yourusername/fobos-sdr-python.git
cd fobos-sdr-python
```

2. Ensure your folder structure looks like this:

```
fobos-sdr-python/
├── fobos_wrapper.py
├── run_tests.py
└── tests/
    ├── __init__.py
    ├── __main__.py
    ├── test_mock_fobos.py
    ├── test_wrapper_logic.py
    └── test_integration.py
```

## Running the Tests

### Basic Usage

Run all tests (excluding hardware-dependent integration tests):

```bash
python -m tests
```

Or use the convenience script:

```bash
python run_tests.py
```

### Running Integration Tests

To include integration tests (requires Fobos SDR hardware):

```bash
python -m tests --integration
```

### Additional Options

```bash
# Run with verbose output
python -m tests --verbose

# Skip specific test types
python -m tests --no-mock         # Skip mock tests
python -m tests --no-logic        # Skip wrapper logic tests

# Combine options
python -m tests --integration --verbose --no-mock
```

## Test Categories

### Mock Tests (`test_mock_fobos.py`)

These tests use Python's `unittest.mock` to simulate the C library behavior. They verify:

- Proper initialization and resource handling
- Correct parameter passing to C functions
- Proper handling of return values
- Error handling and exception propagation

Example:
```python
# Mock the C library function and verify parameters
self.mock_lib.fobos_rx_set_frequency.assert_called_once_with(
    "device_handle", 100e6, self.mock_ffi_instance.new("double *")
)
```

### Wrapper Logic Tests (`test_wrapper_logic.py`)

These tests verify the Python logic of the wrapper:

- Input parameter validation
- Error conditions and boundary checks
- Context manager functionality
- Object lifecycle management

Example:
```python
# Test that invalid gain values are rejected
with self.assertRaises(ValueError):
    self.sdr.set_vga_gain(16)  # VGA gain must be 0-15
```

### Integration Tests (`test_integration.py`)

These tests require actual Fobos SDR hardware and verify:

- Device connectivity and information retrieval
- Frequency, sample rate, and gain settings
- Synchronous and asynchronous data reception
- GPO and clock source settings

Example:
```python
# Test setting frequency with actual hardware
actual_freq = self.sdr.set_frequency(100e6)
self.assertAlmostEqual(actual_freq, 100e6, delta=100e6*0.01)
```

## Test Execution Flow

1. **Test Discovery**: The test runner finds all test cases
2. **Test Setup**: Each test prepares its environment
3. **Test Execution**: Individual test methods are run
4. **Test Teardown**: Resources are cleaned up
5. **Reporting**: Results are summarized

## Hardware Detection

Integration tests automatically detect if hardware is available:

```python
@requires_hardware
def test_sample_rate_setting(self):
    """Test setting sample rate."""
    # This test will be skipped if no hardware is available
```

## Continuous Integration

Add these tests to your CI pipeline:

```yaml
# Example GitHub Actions workflow
name: Run Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install numpy
      - name: Run tests
        run: python -m tests
```

## Adding New Tests

To add new test cases:

1. Identify which test file to modify based on test category
2. Add a new test method with a descriptive name
3. Add detailed docstring explaining what is being tested
4. Use appropriate assertions to verify behavior

Example:
```python
def test_new_feature(self):
    """Test the new feature functionality."""
    # Setup
    self.sdr.setup_new_feature(param1, param2)
    
    # Execution
    result = self.sdr.use_new_feature()
    
    # Verification
    self.assertEqual(result, expected_value)
```

## Troubleshooting

### Common Issues

1. **Tests hang during execution**:
   - This may happen with integration tests if hardware is disconnected
   - Use timeout in integration tests to prevent hanging

2. **Mock tests fail unexpectedly**:
   - Verify that mock configurations match the actual C function signatures
   - Check if the C library interface has changed

3. **Integration tests are skipped**:
   - Ensure hardware is properly connected
   - Check that the Fobos SDR library is correctly installed

## Contributing

Contributions to the test suite are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improved-tests`)
3. Add or modify tests
4. Run the test suite to verify your changes
5. Commit your changes (`git commit -m 'Add tests for new feature'`)
6. Push to the branch (`git push origin feature/improved-tests`)
7. Create a Pull Request

## License

This test suite is licensed under the MIT License - see the LICENSE file for details.