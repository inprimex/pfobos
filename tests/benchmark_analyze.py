#!/usr/bin/env python
"""
Benchmark analysis script for Fobos SDR.
Analyzes benchmark result files and provides insights.

Usage:
  python benchmark_analyze.py path/to/benchmark_file.json
  python benchmark_analyze.py --latest   # Analyze most recent benchmark
  python benchmark_analyze.py --compare  # Compare all benchmark results
  python benchmark_analyze.py --latest --dir my_benchmark_results  # Specify custom directory
"""

import argparse
import json
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from tabulate import tabulate
import time
from datetime import datetime

def load_benchmark_file(filename):
    """Load a benchmark result file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading benchmark file {filename}: {e}")
        return None

def find_latest_benchmark(directory="benchmark_results"):
    """Find the most recent benchmark result file."""
    pattern = os.path.join(directory, "benchmark_*.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No benchmark files found in {directory}")
        return None
        
    # Sort by modification time
    latest_file = max(files, key=os.path.getmtime)
    print(f"Found latest benchmark file: {latest_file}")
    return latest_file

def find_all_benchmarks(directory="benchmark_results"):
    """Find all benchmark result files."""
    pattern = os.path.join(directory, "benchmark_*.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No benchmark files found in {directory}")
        return []
        
    # Sort by modification time
    files.sort(key=os.path.getmtime)
    return files

def print_benchmark_summary(data):
    """Print a summary of the benchmark run."""
    print("\n=== BENCHMARK SUMMARY ===\n")
    
    # Print timestamp and device info
    timestamp = data.get('timestamp', 'Unknown')
    try:
        formatted_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except:
        formatted_time = timestamp
        
    print(f"Benchmark run: {formatted_time}")
    
    # Device info
    device = data.get('device', {})
    if device:
        print("\nDevice Information:")
        for key, value in device.items():
            print(f"  {key}: {value}")
    
    # Run info
    run_info = data.get('run_info', {})
    if run_info:
        print("\nRun Information:")
        for key, value in run_info.items():
            print(f"  {key}: {value}")
    
    # Print summary statistics for each test
    summary_stats = data.get('summary_stats', {})
    
    if summary_stats:
        print("\nPerformance Summary:")
        
        for test_name, metrics in summary_stats.items():
            print(f"\n  {test_name}:")
            
            for metric, stats in metrics.items():
                if stats:
                    print(f"    {metric}:")
                    print(f"      Mean: {stats['mean']:.6f}")
                    print(f"      Min: {stats['min']:.6f}")
                    print(f"      Max: {stats['max']:.6f}")
                    print(f"      Std Dev: {stats['std']:.6f}")

def analyze_detailed_results(data):
    """Analyze detailed benchmark results."""
    detailed_results = data.get('detailed_results', {})
    
    if not detailed_results:
        print("No detailed results found in benchmark file")
        return
    
    print("\n=== DETAILED ANALYSIS ===\n")
    
    for test_name, metrics in detailed_results.items():
        print(f"\n## {test_name}")
        
        for metric, metric_data in metrics.items():
            iterations = metric_data.get('iterations', [])
            
            if not iterations:
                continue
                
            print(f"\n### {metric} ({len(iterations)} iterations)")
            
            # Create a DataFrame for easier analysis
            df = pd.DataFrame(iterations)
            
            # Print basic statistics
            if 'value' in df.columns:
                print("\nValue Statistics:")
                stats = df['value'].describe()
                for stat_name, stat_value in stats.items():
                    print(f"  {stat_name}: {stat_value:.6f}")
            
            # Look for trends across iterations
            if 'value' in df.columns and 'iteration' in df.columns:
                try:
                    # Calculate trend (positive trend means values are increasing)
                    trend = np.polyfit(df['iteration'], df['value'], 1)[0]
                    trend_percent = trend / df['value'].mean() * 100
                    
                    trend_description = "increasing" if trend > 0 else "decreasing"
                    print(f"\nTrend: Values are {trend_description} by {abs(trend):.6f} per iteration ({abs(trend_percent):.2f}%)")
                    
                    # Check for consistency
                    consistency = df['value'].std() / df['value'].mean() * 100
                    print(f"Consistency: {consistency:.2f}% relative standard deviation")
                    
                    if consistency < 5:
                        print("Very consistent results (< 5% variation)")
                    elif consistency < 10:
                        print("Reasonably consistent results (5-10% variation)")
                    else:
                        print("High variation in results (> 10% variation)")
                except:
                    pass

            # Additional analysis based on test type
            if test_name.startswith("sync_reception") and metric == "throughput":
                print("\nThroughput Analysis:")
                
                # Get buffer size from test name
                try:
                    buffer_size = int(test_name.split("_")[-1])
                    if 'sample_rate_hz' in df.columns.values[0]:
                        sample_rate = df['sample_rate_hz'].iloc[0]
                        theoretical_max = sample_rate  # One complex sample per clock
                        efficiency = df['value'].mean() / theoretical_max * 100
                        print(f"  Buffer size: {buffer_size}")
                        print(f"  Sample rate: {sample_rate/1e6:.2f} MHz")
                        print(f"  Avg throughput: {df['value'].mean():.2f} samples/second")
                        print(f"  Theoretical max throughput: {theoretical_max:.2f} samples/second")
                        print(f"  Transfer efficiency: {efficiency:.2f}%")
                except:
                    print("  Could not calculate throughput efficiency")
                    
            elif test_name == "frequency_tuning":
                if metric == "accuracy":
                    if 'target_frequency_mhz' in df.columns.values[0]:
                        print("\nFrequency Accuracy by Target Frequency:")
                        # Group by target frequency
                        for freq, group in df.groupby('target_frequency_mhz'):
                            error_pct = group['value'].mean() * 100
                            print(f"  {freq} MHz: {error_pct:.6f}% average error")
                            
            elif test_name.startswith("async_reception") and metric == "callback_interval":
                if 'callback_intervals_ms' in df.columns.values[0]:
                    print("\nCallback Interval Analysis:")
                    # Calculate jitter (variability in intervals)
                    all_intervals = []
                    for intervals in df['callback_intervals_ms']:
                        all_intervals.extend(intervals)
                        
                    if all_intervals:
                        jitter = np.std(all_intervals)
                        mean_interval = np.mean(all_intervals)
                        print(f"  Average interval: {mean_interval:.2f} ms")
                        print(f"  Interval jitter: {jitter:.2f} ms")
                        print(f"  Relative jitter: {jitter/mean_interval*100:.2f}%")
                        
                        if jitter/mean_interval < 0.05:
                            print("  Very stable callback timing (< 5% jitter)")
                        elif jitter/mean_interval < 0.10:
                            print("  Reasonably stable callback timing (5-10% jitter)")
                        else:
                            print("  Unstable callback timing (> 10% jitter)")

def compare_benchmarks(files):
    """Compare multiple benchmark files."""
    if not files:
        print("No benchmark files to compare")
        return
        
    print(f"\n=== COMPARING {len(files)} BENCHMARK RUNS ===\n")
    
    # Load all benchmark data
    benchmark_data = []
    for filename in files:
        data = load_benchmark_file(filename)
        if data:
            # Extract timestamp
            timestamp = data.get('timestamp', 'Unknown')
            try:
                formatted_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_time = timestamp
                
            data['formatted_time'] = formatted_time
            data['filename'] = os.path.basename(filename)
            benchmark_data.append(data)
    
    if not benchmark_data:
        print("No valid benchmark data to compare")
        return
    
    # Compare key metrics across benchmarks
    metrics_to_compare = []
    
    # Collect all test metrics
    for data in benchmark_data:
        summary_stats = data.get('summary_stats', {})
        for test_name, metrics in summary_stats.items():
            for metric in metrics.keys():
                metric_key = f"{test_name}_{metric}"
                if metric_key not in metrics_to_compare:
                    metrics_to_compare.append(metric_key)
    
    # Create comparison table
    table_data = []
    headers = ["Metric"]
    for data in benchmark_data:
        headers.append(data['formatted_time'])
    
    # Sort metrics for better readability
    metrics_to_compare.sort()
    
    for metric_key in metrics_to_compare:
        test_name, metric = metric_key.rsplit('_', 1)
        row = [f"{test_name} - {metric}"]
        
        for data in benchmark_data:
            summary_stats = data.get('summary_stats', {})
            value = summary_stats.get(test_name, {}).get(metric, {}).get('mean', "N/A")
            if isinstance(value, (int, float)):
                row.append(f"{value:.6f}")
            else:
                row.append("N/A")
                
        table_data.append(row)
    
    # Print comparison table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    os.makedirs("benchmark_analysis", exist_ok=True)
    
    # Group metrics by test name for better visualization
    test_metrics = defaultdict(list)
    for metric_key in metrics_to_compare:
        test_name, metric = metric_key.rsplit('_', 1)
        test_metrics[test_name].append(metric)
    
    # Create plots for each test
    for test_name, metrics in test_metrics.items():
        plt.figure(figsize=(12, 8))
        plt.title(f"Performance Comparison - {test_name}")
        
        x = np.arange(len(benchmark_data))
        width = 1.0 / (len(metrics) + 1)
        
        for i, metric in enumerate(metrics):
            values = []
            for data in benchmark_data:
                summary_stats = data.get('summary_stats', {})
                value = summary_stats.get(test_name, {}).get(metric, {}).get('mean', np.nan)
                values.append(value)
            
            # Plot bar for this metric
            pos = x + (i - len(metrics)/2 + 0.5) * width
            plt.bar(pos, values, width=width, label=metric)
        
        # Add labels and legend
        plt.xticks(x, [data['timestamp'] for data in benchmark_data], rotation=45)
        plt.xlabel('Benchmark Run')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join("benchmark_analysis", f"compare_{test_name}.png")
        plt.savefig(plot_filename)
        plt.close()
        
        print(f"  Plot saved: {plot_filename}")

def analyze_iteration_trends(data):
    """Analyze trends across iterations within a single benchmark run."""
    detailed_results = data.get('detailed_results', {})
    
    if not detailed_results:
        return
    
    print("\n=== ITERATION TRENDS ===\n")
    print("Analyzing how performance changes across iterations...")
    
    # Create directory for iteration plots
    os.makedirs("benchmark_analysis/iterations", exist_ok=True)
    
    for test_name, metrics in detailed_results.items():
        for metric, metric_data in metrics.items():
            iterations = metric_data.get('iterations', [])
            
            if not iterations or len(iterations) < 2:
                continue
                
            # Extract iteration values
            values = [item.get('value', np.nan) for item in iterations]
            iterations_nums = [item.get('iteration', i+1) for i, item in enumerate(iterations)]
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(iterations_nums, values, 'o-', label=f"{test_name} - {metric}")
            plt.xlabel('Iteration')
            plt.ylabel(metric)
            plt.title(f"Iteration Trend - {test_name} - {metric}")
            plt.grid(True)
            
            # Add trend line
            if len(values) > 2:
                try:
                    z = np.polyfit(iterations_nums, values, 1)
                    p = np.poly1d(z)
                    plt.plot(iterations_nums, p(iterations_nums), "r--", label=f"Trend: {z[0]:.6f}x + {z[1]:.6f}")
                except:
                    pass
            
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plot_filename = os.path.join("benchmark_analysis/iterations", f"{test_name}_{metric}_trend.png")
            plt.savefig(plot_filename)
            plt.close()
            
            print(f"  Iteration trend plot saved: {plot_filename}")
            
            # Print trend analysis
            if len(values) > 2:
                try:
                    trend = np.polyfit(iterations_nums, values, 1)[0]
                    mean_value = np.mean(values)
                    percent_change = trend / mean_value * 100
                    
                    print(f"  {test_name} - {metric}:")
                    print(f"    Change per iteration: {trend:.6f} ({percent_change:.2f}% of mean)")
                    
                    if abs(percent_change) < 1:
                        print(f"    Stable across iterations (< 1% change)")
                    elif abs(percent_change) < 5:
                        print(f"    Slight {'increase' if trend > 0 else 'decrease'} across iterations (1-5% change)")
                    else:
                        print(f"    Significant {'increase' if trend > 0 else 'decrease'} across iterations (> 5% change)")
                except:
                    print(f"  {test_name} - {metric}: Could not calculate trend")

def analyze_buffer_size_impact(data):
    """Analyze the impact of buffer size on performance."""
    detailed_results = data.get('detailed_results', {})
    
    if not detailed_results:
        return
    
    # Look for sync_reception tests with different buffer sizes
    sync_tests = {}
    for test_name in detailed_results.keys():
        if test_name.startswith('sync_reception_'):
            # Extract buffer size
            try:
                buffer_size = int(test_name.split('_')[-1])
                sync_tests[buffer_size] = test_name
            except:
                continue
    
    if not sync_tests:
        return
    
    print("\n=== BUFFER SIZE IMPACT ANALYSIS ===\n")
    
    # Extract throughput for each buffer size
    buffer_sizes = []
    throughputs = []
    read_times = []
    
    for buffer_size, test_name in sorted(sync_tests.items()):
        buffer_sizes.append(buffer_size)
        
        # Get throughput
        throughput_data = detailed_results.get(test_name, {}).get('throughput', {}).get('iterations', [])
        if throughput_data:
            avg_throughput = np.mean([item.get('value', 0) for item in throughput_data])
            throughputs.append(avg_throughput)
        else:
            throughputs.append(np.nan)
        
        # Get read time - look for the specific read_time metric for this buffer size
        read_time_test = f"{test_name}_read"
        read_time_data = detailed_results.get(read_time_test, {}).get('time', {}).get('iterations', [])
        if read_time_data:
            avg_read_time = np.mean([item.get('value', 0) for item in read_time_data])
            read_times.append(avg_read_time)
        else:
            read_times.append(np.nan)
    
    if not buffer_sizes or len(buffer_sizes) < 2:
        return
    
    # Create buffer size impact plots
    os.makedirs("benchmark_analysis", exist_ok=True)
    
    # Create a table for throughput and read times
    print("Buffer size impact analysis:")
    print(f"  Buffer sizes tested: {buffer_sizes}")
    
    # Check if we have valid throughput data
    valid_throughputs = [t for t in throughputs if not np.isnan(t)]
    if valid_throughputs:
        # Find optimal buffer size for throughput
        max_throughput_idx = np.nanargmax(throughputs)
        max_throughput_buffer = buffer_sizes[max_throughput_idx]
        print(f"  Optimal buffer size for throughput: {max_throughput_buffer} ({throughputs[max_throughput_idx]:.2f} samples/second)")
        
        # Throughput vs Buffer Size plot
        plt.figure(figsize=(10, 6))
        plt.plot(buffer_sizes, throughputs, 'o-')
        plt.xlabel('Buffer Size')
        plt.ylabel('Throughput (samples/second)')
        plt.title('Impact of Buffer Size on Throughput')
        plt.grid(True)
        plt.xscale('log', base=2)
        
        # Add buffer sizes as text
        for i, (size, throughput) in enumerate(zip(buffer_sizes, throughputs)):
            if not np.isnan(throughput):
                plt.text(size, throughput, f"{size}", ha='left', va='bottom')
        
        plt.tight_layout()
        plt.savefig("benchmark_analysis/buffer_size_throughput.png")
        plt.close()
        
        print(f"  Throughput analysis plot saved to benchmark_analysis/buffer_size_throughput.png")
    
    # Check if we have valid read time data
    valid_read_times = [t for t in read_times if not np.isnan(t)]
    if valid_read_times:
        # Find optimal buffer size for latency
        min_read_time_idx = np.nanargmin(read_times)
        min_read_time_buffer = buffer_sizes[min_read_time_idx]
        print(f"  Optimal buffer size for latency: {min_read_time_buffer} ({read_times[min_read_time_idx]:.6f} seconds)")
        
        # Read Time vs Buffer Size plot
        plt.figure(figsize=(10, 6))
        plt.plot(buffer_sizes, read_times, 'o-')
        plt.xlabel('Buffer Size')
        plt.ylabel('Read Time (seconds)')
        plt.title('Impact of Buffer Size on Read Time')
        plt.grid(True)
        plt.xscale('log', base=2)
        
        # Add buffer sizes as text
        for i, (size, time_val) in enumerate(zip(buffer_sizes, read_times)):
            if not np.isnan(time_val):
                plt.text(size, time_val, f"{size}", ha='left', va='bottom')
        
        plt.tight_layout()
        plt.savefig("benchmark_analysis/buffer_size_read_time.png")
        plt.close()
        
        print(f"  Read time analysis plot saved to benchmark_analysis/buffer_size_read_time.png")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Analyze Fobos SDR benchmark results')
    parser.add_argument('file', nargs='?', help='Benchmark result file to analyze')
    parser.add_argument('--latest', action='store_true', help='Analyze most recent benchmark')
    parser.add_argument('--compare', action='store_true', help='Compare all benchmarks')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    parser.add_argument('--dir', help='Directory to search for benchmark files', default="benchmark_results")
    args = parser.parse_args()
    
    benchmark_file = None
    
    if args.file:
        benchmark_file = args.file
    elif args.latest:
        benchmark_file = find_latest_benchmark(args.dir)
    elif args.compare or args.all:
        benchmark_files = find_all_benchmarks(args.dir)
        if benchmark_files:
            if args.compare:
                compare_benchmarks(benchmark_files)
            if args.all and benchmark_files:
                # Analyze latest as well
                benchmark_file = benchmark_files[-1]
    else:
        # No specific mode, look for latest
        benchmark_file = find_latest_benchmark(args.dir)
    
    # Analyze single benchmark file if specified
    if benchmark_file:
        print(f"Analyzing benchmark file: {benchmark_file}")
        data = load_benchmark_file(benchmark_file)
        
        if data:
            print_benchmark_summary(data)
            analyze_detailed_results(data)
            analyze_iteration_trends(data)
            analyze_buffer_size_impact(data)
            
            print("\nAnalysis complete. Results saved to benchmark_analysis/ directory.")
        else:
            print("Failed to load benchmark data.")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
