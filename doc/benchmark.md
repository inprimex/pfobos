# Fobos SDR Benchmark Tool

This document describes the comprehensive benchmark tool for the Fobos SDR Python wrapper.

## Overview

The benchmark tool (`benchmark.py`) provides in-depth performance analysis of the Fobos SDR wrapper with advanced reporting capabilities. It's designed for detailed performance evaluation, optimization, and long-term tracking.

## Purpose

While the unit-test based performance tests (`test_performance.py`) focus on quick verification, the benchmark tool serves several additional purposes:

1. Generate detailed performance metrics with statistical analysis
2. Create visual representations of performance data
3. Enable historical comparison between benchmark runs
4. Provide comprehensive reports that can be saved and shared
5. Support optimization efforts with fine-grained performance insights

## Requirements

- Python 3.7+
- Fobos SDR Python wrapper (shared/fwrapper.py)
- Python packages:
  - numpy
  - matplotlib (for plot generation)
  - scipy
- Fobos SDR hardware

## Running the Benchmark

### Using run_tests.py

The easiest way to run the benchmark tool is through the `run_tests.py` script:

```bash
# Run the benchmark with default settings
python run_tests.py --benchmark

# Run with 5 iterations per test
python run_tests.py --benchmark --iterations 5

# Use a specific device index (if multiple SDRs are connected)
python run_tests.py --benchmark --device 1

# Save results to a custom directory
python run_tests.py --benchmark --output-dir ./my_benchmark_results

# Only generate comparison plots from existing results
python run_tests.py --benchmark --plot-only
```

### Running Directly

You can also run the benchmark script directly:

```bash
# Run with default settings
python benchmark.py

# Run with custom settings
python benchmark.py --device 1 --iterations 5 --output-dir ./my_benchmarks

# Only generate comparison plots
python benchmark.py --plot-only
```

## Benchmark Categories

The benchmark tool runs tests across several categories:

### Device Operations
- Open/close performance
- Frequency tuning performance
- Sample rate setting performance
- Gain setting performance
- User GPO setting performance

### Data Reception
- Synchronous reception with various buffer sizes
- Asynchronous reception with callback timing
- Throughput measurement

### Signal Processing
- FFT performance (various sizes)
- Filter performance (various tap counts)
- FM demodulation
- Decimation

## Output and Reports

The benchmark generates several types of output:

### Console Output
- Real-time results during benchmark execution
- Summary statistics for each test
- Success/failure status of the overall benchmark

### JSON Results
Each benchmark run creates a JSON file with detailed metrics:

```
benchmark_results/
└── benchmark_20250428_120000.json
```

This file contains:
- Timestamp of the run
- Device information
- Raw metrics for each test
- Statistical analysis (min, max, mean, median, std dev, percentiles)

### Performance Plots
Individual plots for each metric:

```
benchmark_results/
└── plots/
    ├── sync_reception_16384_throughput_20250428_120000.png
    ├── frequency_tuning_time_20250428_120000.png
    └── ...
```

### Comparison Plots
Plots comparing multiple benchmark runs:

```
benchmark_results/
└── comparison_plots/
    ├── compare_sync_reception_16384_throughput_20250428_120500.png
    ├── compare_frequency_tuning_time_20250428_120500.png
    └── ...
```

## Understanding Benchmark Results

The benchmark tool provides several key insights:

### 1. Performance Baseline
Establishes a baseline for expected performance on your hardware.

### 2. Statistical Distribution
Shows not just averages but the full distribution of performance:
- Minimum and maximum values identify outliers
- Standard deviation shows consistency
- Percentiles (95th, 99th) help identify worst-case performance

### 3. Visual Patterns
The plots can reveal:
- Performance scaling with buffer size
- Frequency-dependent performance
- Consistent bottlenecks
- Anomalies that might indicate hardware issues

### 4. Historical Trends
The comparison plots help track:
- Performance improvements after code changes
- Degradation over time
- Impact of environmental factors
- Differences between hardware setups

## When to Use the Benchmark Tool

The benchmark tool is particularly useful in these scenarios:

1. **Initial Deployment**: Establish baseline performance for your hardware setup
2. **After Code Changes**: Verify that optimizations have improved performance
3. **Hardware Evaluation**: Compare performance across different SDR units
4. **System Optimization**: Identify which parameters yield optimal performance
5. **Troubleshooting**: Diagnose performance-related issues

## Customizing the Benchmark

The benchmark tool can be extended to test additional aspects of the SDR:

1. Add new benchmark methods to the `FobosSDRBenchmark` class
2. Use the `record_result` method to store measurements
3. Update the `run_all_benchmarks` method to include your new tests
4. Optional: Add custom visualization in the `generate_plots` method

## Interpreting Specific Metrics

### Device Control Latency
- **Open/close times**: Measures USB enumeration and initialization overhead
- **Frequency tuning**: Reflects hardware PLL lock time and USB communication latency
- **Sample rate setting**: Shows how quickly the device adapts to new data rates

### Data Throughput
- **Synchronous read**: Indicates maximum sustainable data transfer rate
- **Asynchronous read**: Shows real-world performance with callback overhead
- **Buffer size impact**: Reveals the optimal buffer size for your application

### Processing Performance
- **FFT performance**: Scales with O(n log n) complexity
- **Filtering performance**: Scales linearly with filter length
- **Demodulation speed**: Critical for real-time applications

## Common Patterns and Solutions

### Buffer Size Trade-offs
- Larger buffers increase throughput but also increase latency
- Look for the "sweet spot" where throughput plateaus

### USB Transfer Efficiency
- USB bulk transfers work best with larger payloads
- If small buffer throughput is low, consider buffering in your application

### Processing vs. Transfer Bottlenecks
- Compare raw transfer rates with processing rates
- If processing is slower than transfer, optimize algorithms
- If transfer is slower, optimize buffer sizes or USB settings

## Troubleshooting

### Benchmark Hangs
- Default timeouts prevent infinite hangs
- Check USB connection or try reconnecting the device
- Look for signs of USB bandwidth saturation

### High Variability
- Run more iterations for more stable averages
- Check for background processes
- Try isolating the benchmark from other system activities

### Plot Generation Errors
- Ensure matplotlib is properly installed
- Check that output directories are writable
- Verify JSON results files are valid

## Comparison with Performance Tests

While the unit test framework provides quick verification, the benchmark tool offers:

- Much more detailed metrics
- Statistical analysis beyond simple timings
- Visualization capabilities
- Historical tracking and comparison
- Permanent result storage

For continuous integration or quick checks, use `test_performance.py`.
For optimization, detailed analysis, or reporting, use `benchmark.py`.

## See Also

- [Performance Tests Documentation](./test_performance.md)
- [Integration Tests Documentation](./tests.md)
- [Setup Guide](./setup-fobos-sdr.md)
