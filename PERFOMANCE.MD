# Fobos SDR Performance Testing

A comprehensive performance testing solution for the Fobos SDR Python wrapper. This package helps benchmark and analyze the performance of key SDR operations.

## Overview

This performance testing package consists of two main components:

1. **Unit Test-Based Performance Tests** (`tests/test_performance.py`) - Performance tests integrated with the unittest framework
2. **Standalone Benchmark Tool** (`benchmark.py`) - A script for running benchmarks and generating reports

![Performance Testing Architecture](https://via.placeholder.com/800x400?text=Fobos+SDR+Performance+Testing)

## Requirements

- Python 3.7+
- Fobos SDR Python wrapper (fobos_wrapper.py)
- Python packages:
  - numpy
  - matplotlib
  - scipy (for signal processing tests)
- Fobos SDR hardware

## Installation

Ensure that the `test_performance.py` file is in your `tests` directory and that `benchmark.py` is in your project root.

## Running the Tests

### Unit Test-Based Performance Tests

Run performance tests using the unittest framework:

```bash
# Run all performance tests
python -m unittest tests.test_performance

# Run a specific test
python -m unittest tests.test_performance.TestFobosSDRPerformance.test_sync_read_performance
```

### Standalone Benchmark Tool

Run the benchmark tool for more comprehensive testing and reporting:

```bash
# Run all benchmarks
python benchmark.py

# Specify device index (if multiple SDRs are connected)
python benchmark.py --device 1

# Specify number of iterations
python benchmark.py --iterations 5

# Specify output directory
python benchmark.py --output-dir ./my_benchmarks

# Only generate comparison plots from existing results
python benchmark.py --plot-only
```

## Test Categories

### Device Operations Tests

- Open/close performance
- Frequency tuning performance
- Sample rate setting performance
- User GPO setting performance
- Clock source switching performance

### Data Reception Tests

- Synchronous read performance (various buffer sizes)
- Asynchronous read performance (various buffer sizes)
- Callback latency measurement
- Data throughput measurement

### Signal Processing Tests

- FFT performance (various sizes)
- Filter performance (various tap counts)
- FM demodulation performance
- Decimation performance

## Benchmark Results

The benchmark tool generates detailed results in multiple formats:

### JSON Results

Each benchmark run generates a JSON file with detailed metrics and statistics:

```
benchmark_results/
└── benchmark_20250323_120000.json
```

The JSON file includes:
- Timestamp of the run
- Device information
- Detailed metrics for each test
- Statistical analyses (min, max, mean, median, std dev, percentiles)

### Performance Plots

Individual plots for each metric are generated in the plots directory:

```
benchmark_results/
└── plots/
    ├── sync_reception_16384_throughput_20250323_120000.png
    ├── frequency_tuning_time_20250323_120000.png
    └── ...
```

### Comparison Plots

When multiple benchmark runs exist, comparison plots are generated:

```
benchmark_results/
└── comparison_plots/
    ├── compare_sync_reception_16384_throughput_20250323_120500.png
    ├── compare_frequency_tuning_time_20250323_120500.png
    └── ...
```

## Interpreting Results

### Key Metrics to Monitor

1. **Device Control Latency**
   - Open/close times
   - Frequency tuning times
   - Sample rate setting times

2. **Data Throughput**
   - Samples processed per second
   - Maximum sustainable sample rate

3. **Processing Performance**
   - FFT performance vs size
   - Filtering performance vs complexity
   - Demodulation speed

4. **Callback Latency**
   - Time between async callbacks
   - Callback processing time

### Performance Comparison

Use the comparison plots to track performance changes:
- Across different versions of the wrapper
- After hardware/firmware updates
- Between different host systems
- With different configuration settings

## Troubleshooting

### Common Issues

1. **Benchmark hangs during async tests**
   - The tests include timeouts to prevent indefinite hangs
   - Check USB connectivity if timeouts occur frequently

2. **High variability in results**
   - Run with more iterations for more stable averages
   - Check for background processes consuming CPU resources
   - Try closing other applications using USB bandwidth

3. **Plots not showing expected data**
   - Ensure the benchmark completed successfully
   - Check console output for error messages
   - Verify that the output directory is writable

## Extending the Tests

### Adding New Performance Tests

To add new performance tests to the unittest-based framework:

1. Add a new test method to `TestFobosSDRPerformance` in `test_performance.py`
2. Use the `@requires_hardware`, `@time_execution`, and/or `@profile` decorators
3. Use the metrics collection methods to record and analyze results

### Adding New Benchmarks

To add new benchmarks to the benchmark tool:

1. Add a new benchmark method to the `FobosSDRBenchmark` class in `benchmark.py`
2. Use the `record_result` method to store measurements
3. Make sure to update the `run_all_benchmarks` method to include your new benchmark

## License

This performance testing package is licensed under the MIT License - see the LICENSE file for details.