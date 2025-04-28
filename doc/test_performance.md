# Fobos SDR Performance Testing

This document describes the unit-test based performance testing solution for the Fobos SDR Python wrapper.

## Overview

The performance testing module (`tests/test_performance.py`) provides a structured approach to measure and validate the performance of various Fobos SDR operations. It's integrated with the unittest framework to allow for automated performance verification.

## Purpose

The primary goals of the performance tests are:

1. Detect performance regressions during development
2. Verify that the SDR wrapper meets minimum performance requirements
3. Ensure consistent behavior across different environments
4. Provide baseline performance metrics for comparison

Unlike the comprehensive benchmark tool (`benchmark.py`), these tests focus on quick verification rather than detailed analysis.

## Requirements

- Python 3.7+
- Fobos SDR Python wrapper (`shared/fwrapper.py`)
- Python packages:
  - numpy
  - scipy
- Fobos SDR hardware

## Running the Tests

### Using run_tests.py

To run only the performance tests:

```bash
# Run only performance tests
python run_tests.py --performance-only

# With verbose output
python run_tests.py --performance-only --verbose
```

### Using unittest directly

```bash
# Run all performance tests
python -m unittest tests.test_performance

# Run a specific test
python -m unittest tests.test_performance.TestFobosSDRPerformance.test_sync_read_performance
```

## Test Categories

The performance tests include:

### Device Control Tests
- `test_open_close_performance`: Tests how quickly the device can be opened and closed
- `test_frequency_change_performance`: Tests frequency tuning time across multiple frequencies
- `test_samplerate_change_performance`: Tests sample rate adjustment performance

### Data Reception Tests
- `test_sync_read_performance`: Tests synchronous data reading performance with various buffer sizes
- `test_async_read_performance`: Tests asynchronous data reception performance and callback latency

### Signal Processing Tests
- `test_signal_processing_performance`: Tests performance of common SDR operations:
  - FFT (various sizes)
  - Filtering
  - FM demodulation
  - Decimation

## Interpreting Results

Each test outputs timing and performance metrics to the console. Key metrics to monitor:

1. **Device operations**: Should complete within tens of milliseconds
2. **Synchronous read**: Throughput should be consistent with the configured sample rate
3. **Asynchronous callbacks**: Should have consistent intervals
4. **Signal processing**: Should scale according to complexity (O(n log n) for FFT, etc.)

## Performance Thresholds

While the tests don't enforce strict pass/fail thresholds, these guidelines can help identify potential issues:

- **Device open/close**: Typically < 50ms
- **Frequency change**: Typically < 10ms
- **Sample rate change**: Typically < 20ms
- **Synchronous read**: Should achieve at least 80% of the theoretical maximum rate
- **FFT processing**: Should complete in < 5ms for 1024-point FFT
- **FM demodulation**: Should process data faster than real-time

## Troubleshooting

Common issues and solutions:

1. **"No Fobos SDR hardware detected"**
   - Verify the device is properly connected
   - Check USB connection and permissions
   - Run `lsusb` to confirm device visibility

2. **Tests running slowly**
   - Close other applications that might be using USB bandwidth
   - Check for background processes consuming CPU
   - Verify system is not throttling due to power management

3. **"Error in async callback"**
   - This is often caused by USB communication issues
   - Try reconnecting the device
   - Reduce the number of iterations to minimize the chance of USB errors

4. **Division by zero errors**
   - This can happen if timing operations complete too quickly
   - Try larger buffer sizes or more complex operations

## Extending the Tests

To add a new performance test:

1. Create a new test method in the `TestFobosSDRPerformance` class
2. Use the `@requires_hardware` decorator to skip when hardware is unavailable
3. Use `@time_execution` to automatically log execution time
4. Use `@profile` if you want detailed profiling information
5. Add metrics collection with `self.metrics.add_timing(name, value)`

Example:

```python
@requires_hardware
@time_execution
def test_my_new_function(self):
    """Test performance of my new function."""
    iterations = 5
    
    for i in range(iterations):
        start_time = time.time()
        # Perform the operation to be tested
        result = self.sdr.my_function()
        elapsed_time = time.time() - start_time
        
        self.metrics.add_timing('my_function', elapsed_time)
        logger.info(f"Operation completed in {elapsed_time:.6f} seconds")
    
    # Print summary statistics
    self.metrics.print_stats('my_function')
```

## Comparing with Benchmark Tool

The performance tests provide quick verification, while the benchmark tool (`benchmark.py`) offers:

- More detailed metrics and statistics
- Visual plots and reports
- Historical comparison
- JSON output for further analysis

If you need comprehensive analysis rather than quick verification, consider using the benchmark tool.

## See Also

- [Benchmark Tool Documentation](./benchmark.md)
- [Integration Tests Documentation](./tests.md)
- [Setup Guide](./setup-fobos-sdr.md)
