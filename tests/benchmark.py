#!/usr/bin/env python
"""
Benchmark script for Fobos SDR wrapper.
This script runs performance tests and generates reports.

Usage:
  python benchmark.py              # Run all benchmarks
  python benchmark.py --plot-only  # Generate plots from existing results
  python benchmark.py --help       # Show help message
"""

import argparse
import sys
import os
import time
import datetime
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import platform
from collections import defaultdict
import logging

# Add parent directory to path to import the wrapper
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
# Import from shared module
from shared.fwrapper import FobosSDR, FobosException


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FobosSDRBenchmark:
    """Benchmark utilities for Fobos SDR wrapper."""
    
    def __init__(self, device_index=0, output_dir="benchmark_results"):
        """Initialize the benchmark."""
        self.device_index = device_index
        self.output_dir = output_dir
        self.sdr = None
        self.results = defaultdict(list)
        self.result_timestamps = defaultdict(list)  # Track timestamps for each measurement
        self.result_details = defaultdict(list)     # Track additional details for each measurement
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)        

    def setup(self):
        """Set up the benchmark environment."""
        try:
            self.sdr = FobosSDR()
            
            # Check for connected devices
            device_count = self.sdr.get_device_count()
            if device_count == 0:
                logger.error("No Fobos SDR devices found.")
                return False
                
            if self.device_index >= device_count:
                logger.error(f"Invalid device index {self.device_index}. Only {device_count} devices available.")
                return False
                
            # Open the device
            self.sdr.open(self.device_index)
            
            # Get and log device info
            info = self.sdr.get_board_info()
            logger.info(f"Testing with device: {info['product']} (SN: {info['serial']})")
            logger.info(f"Hardware revision: {info['hw_revision']}")
            logger.info(f"Firmware version: {info['fw_version']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up benchmark: {e}")
            return False
            
    def teardown(self):
        """Clean up after benchmarking."""
        if self.sdr and self.sdr.dev is not None:
            self.sdr.close()
            
    def record_result(self, test_name, metric, value, details=None):
        """Record a benchmark result with timestamp and optional details."""
        key = f"{test_name}_{metric}"
        self.results[key].append(value)
        self.result_timestamps[key].append(time.time())
        
        # Store additional details if provided
        if details is None:
            details = {}
        self.result_details[key].append(details)
        
    def get_stats(self, test_name, metric):
        """Get statistics for a specific test metric."""
        key = f"{test_name}_{metric}"
        if key not in self.results or not self.results[key]:
            return None
            
        values = np.array(self.results[key])
        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'std': float(np.std(values)),
            'p95': float(np.percentile(values, 95)),
            'p99': float(np.percentile(values, 99))
        }
        
    # Updated save_results method to include all the detailed logs
    def save_results(self):
        """Save benchmark results to file with detailed iteration logs."""
        if not self.results:
            logger.warning("No results to save.")
            return
            
        # Create a timestamp for the results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare the detailed results structure
        detailed_results = {}
        summary_stats = {}
        
        # Process each result key (test_name_metric)
        for key in self.results:
            test_name, metric = key.rsplit('_', 1)
            
            # Create test_name entry if it doesn't exist
            if test_name not in detailed_results:
                detailed_results[test_name] = {}
                summary_stats[test_name] = {}
                
            # Create metric entry if it doesn't exist
            if metric not in detailed_results[test_name]:
                detailed_results[test_name][metric] = {}
                
            # Add iteration data for this metric
            values = self.results[key]
            timestamps = self.result_timestamps[key]
            details = self.result_details[key]
            
            iterations_data = []
            
            for i, (value, ts, detail) in enumerate(zip(values, timestamps, details)):
                iteration_data = {
                    "iteration": i + 1,
                    "value": float(value),
                    "timestamp": ts
                }
                # Add all additional details
                iteration_data.update(detail)
                iterations_data.append(iteration_data)
                
            detailed_results[test_name][metric]["iterations"] = iterations_data
            
            # Add summary statistics
            if test_name not in summary_stats:
                summary_stats[test_name] = {}
            summary_stats[test_name][metric] = self.get_stats(test_name, metric)
        
        # Log run environment information
        run_info = {
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "platform": sys.platform,
            "machine": platform.machine() if hasattr(platform, "machine") else "unknown"
        }
            
        # Get device info for metadata
        device_info = {}
        if self.sdr and self.sdr.dev is not None:
            try:
                device_info = self.sdr.get_board_info()
                # Add API info if available
                try:
                    api_info = self.sdr.get_api_info()
                    device_info.update(api_info)
                except:
                    pass
            except:
                logger.warning("Could not get device board info for results metadata")
                
        # Create result data with both detailed logs and summary stats
        result_data = {
            'timestamp': timestamp,
            'device': device_info,
            'run_info': run_info,
            'detailed_results': detailed_results,
            'summary_stats': summary_stats,
            'run_config': {
                'device_index': self.device_index,
                'output_dir': self.output_dir
            }
        }
        
        # Save to JSON file
        json_filename = os.path.join(self.output_dir, f"benchmark_{timestamp}.json")
        with open(json_filename, 'w') as f:
            json.dump(result_data, f, indent=2)
            
        logger.info(f"Results saved to {json_filename}")
        
        # Generate plots
        self.generate_plots(timestamp)
        
        return json_filename
        
    def generate_plots(self, timestamp):
        """Generate plots from benchmark results."""
        if not self.results:
            return
            
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Group results by test
        tests = defaultdict(dict)
        for key in self.results:
            test_name, metric = key.rsplit('_', 1)
            tests[test_name][metric] = self.results[key]
            
        # Generate plots for each test
        for test_name, metrics in tests.items():
            self._generate_test_plots(test_name, metrics, plots_dir, timestamp)
            
    def _generate_test_plots(self, test_name, metrics, plots_dir, timestamp):
        """Generate plots for a specific test."""
        for metric, values in metrics.items():
            if not values:
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Generate appropriate plot based on data
            if len(values) > 1:
                # Time series
                plt.plot(values, marker='o')
                plt.title(f"{test_name} - {metric}")
                plt.xlabel("Iteration")
                plt.ylabel(metric)
                plt.grid(True)
                
                # Add mean line
                mean_value = np.mean(values)
                plt.axhline(y=mean_value, color='r', linestyle='--', 
                            label=f"Mean: {mean_value:.6f}")
                
                # Add legend
                plt.legend()
                
            else:
                # Single value - show as bar
                plt.bar(0, values[0])
                plt.title(f"{test_name} - {metric}")
                plt.ylabel(metric)
                plt.xticks([])
                plt.grid(axis='y')
                
                # Add value text
                plt.text(0, values[0] / 2, f"{values[0]:.6f}", 
                         ha='center', va='center')
                
            # Save plot
            plot_filename = os.path.join(plots_dir, f"{test_name}_{metric}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(plot_filename)
            plt.close()
            
    def generate_comparison_plots(self):
        """Generate comparison plots from all benchmark results."""
        logger.info("Generating comparison plots from all benchmark results")
        
        # Find all benchmark result files
        result_files = glob.glob(os.path.join(self.output_dir, "benchmark_*.json"))
        if not result_files:
            logger.warning("No benchmark result files found")
            return
            
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, "comparison_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Load all results
        all_results = []
        for filename in result_files:
            try:
                with open(filename, 'r') as f:
                    result = json.load(f)
                    # Add base filename as identifier
                    result['filename'] = os.path.basename(filename)
                    all_results.append(result)
            except Exception as e:
                logger.warning(f"Error loading {filename}: {e}")
                
        if not all_results:
            logger.warning("No valid benchmark result files could be loaded")
            return
            
        # Sort by timestamp
        all_results.sort(key=lambda x: x.get('timestamp', ''))
        
        # Generate comparison plots
        self._generate_comparison_plots(all_results, plots_dir)
        
    def _generate_comparison_plots(self, all_results, plots_dir):
        """Generate plots comparing multiple benchmark runs."""
        # Group metrics across runs
        metrics_by_test = defaultdict(lambda: defaultdict(list))
        labels = []
        
        for result in all_results:
            # Use timestamp as label
            label = result.get('timestamp', 'unknown')
            labels.append(label)
            
            # Extract stats
            stats = result.get('stats', {})
            for test_name, test_stats in stats.items():
                for metric, metric_stats in test_stats.items():
                    if metric_stats and 'mean' in metric_stats:
                        metrics_by_test[test_name][metric].append(metric_stats['mean'])
                        
        # Generate a comparison plot for each test metric
        for test_name, metrics in metrics_by_test.items():
            for metric, values in metrics.items():
                if len(values) < 2:
                    continue  # Need at least 2 values to compare
                    
                plt.figure(figsize=(12, 6))
                
                # Bar chart
                x = np.arange(len(values))
                bars = plt.bar(x, values)
                
                # Add labels
                plt.xlabel('Benchmark Run')
                plt.ylabel(metric)
                plt.title(f"{test_name} - {metric} Comparison")
                
                # Add value labels on top of bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{values[i]:.6f}',
                            ha='center', va='bottom', rotation=45)
                
                # Set x-axis labels with benchmark timestamps
                plt.xticks(x, [l.split('_')[0] for l in labels[:len(values)]], rotation=45)
                
                # Add grid and adjust layout
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Save plot
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_filename = os.path.join(plots_dir, f"compare_{test_name}_{metric}_{timestamp}.png")
                plt.savefig(plot_filename)
                plt.close()
                
        logger.info(f"Comparison plots saved to {plots_dir}")
            
    def run_all_benchmarks(self, iterations=3):
        """Run all benchmarks."""
        if not self.setup():
            return False
            
        try:
            # 1. Test device open/close performance
            self.benchmark_open_close(iterations)
            
            # 2. Test frequency tuning performance
            self.benchmark_frequency_tuning(iterations)
            
            # 3. Test sample rate setting performance
            self.benchmark_samplerate_setting()
            
            # 4. Test synchronous reception performance
            self.benchmark_sync_reception(iterations)
            
            # 5. Test asynchronous reception performance
            self.benchmark_async_reception(iterations)
            
            # 6. Test data processing performance
            self.benchmark_data_processing()
            
            # Save results
            self.save_results()
            
            return True
            
        except Exception as e:
            logger.error(f"Error during benchmarking: {e}")
            return False
            
        finally:
            self.teardown()
            
    # The benchmark_open_close method with improved logging
    def benchmark_open_close(self, iterations=3):
        """Benchmark device open/close performance."""
        logger.info("=== Benchmarking device open/close performance ===")
        
        # Close the device that was opened in setup
        if self.sdr.dev is not None:
            self.sdr.close()
            
        for i in range(iterations):
            # Measure open time
            iteration_start = time.time()
            start_time = time.time()
            self.sdr.open(self.device_index)
            open_time = time.time() - start_time
            
            # Get some device info for context
            try:
                info = self.sdr.get_board_info()
                device_info = {
                    "serial": info.get("serial", "unknown"),
                    "product": info.get("product", "unknown")
                }
            except:
                device_info = {"error": "Could not get device info"}
            
            details = {
                "iteration": i + 1,
                "timestamp": start_time,
                "device_info": device_info
            }
            
            self.record_result("open_close", "open_time", open_time, details)
            logger.info(f"Iteration {i+1}/{iterations}: Device opened in {open_time:.6f} seconds")
            
            # Measure close time
            start_time = time.time()
            self.sdr.close()
            close_time = time.time() - start_time
            
            close_details = {
                "iteration": i + 1,
                "timestamp": start_time,
                "iteration_duration": time.time() - iteration_start
            }
            
            self.record_result("open_close", "close_time", close_time, close_details)
            logger.info(f"Iteration {i+1}/{iterations}: Device closed in {close_time:.6f} seconds")
            
        # Print summary
        logger.info("Open/close benchmark completed")
        
        # Reopen the device for subsequent tests
        self.sdr.open(self.device_index)
        
    # The benchmark_frequency_tuning method with improved logging
    def benchmark_frequency_tuning(self, iterations=3):
        """Benchmark frequency tuning performance."""
        logger.info("=== Benchmarking frequency tuning performance ===")
        
        # Test frequencies (in MHz)
        frequencies = [100, 200, 433, 868, 915]
        
        for i in range(iterations):
            for freq_mhz in frequencies:
                freq = freq_mhz * 1e6
                
                # Measure tuning time
                start_time = time.time()
                actual_freq = self.sdr.set_frequency(freq)
                tuning_time = time.time() - start_time
                
                # Record accuracy and timing with detailed context
                tuning_details = {
                    "iteration": i + 1,
                    "target_frequency_mhz": freq_mhz,
                    "target_frequency_hz": freq,
                    "actual_frequency_hz": actual_freq,
                    "error_percent": abs(actual_freq - freq) / freq * 100,
                    "timestamp": start_time
                }
                
                self.record_result("frequency_tuning", "time", tuning_time, tuning_details)
                self.record_result("frequency_tuning", "accuracy", abs(actual_freq - freq) / freq, tuning_details)
                
                logger.info(f"Frequency {freq_mhz} MHz: Tuned to {actual_freq/1e6:.3f} MHz "
                            f"in {tuning_time:.6f} seconds "
                            f"(error: {abs(actual_freq - freq) / freq * 100:.4f}%)")
                
        # Print summary
        logger.info("Frequency tuning benchmark completed")

        
    def benchmark_samplerate_setting(self):
        """Benchmark sample rate setting performance."""
        logger.info("=== Benchmarking sample rate setting performance ===")
        
        # Get available sample rates
        sample_rates = self.sdr.get_samplerates()
        
        for rate in sample_rates:
            # Measure setting time
            start_time = time.time()
            actual_rate = self.sdr.set_samplerate(rate)
            setting_time = time.time() - start_time
            
            self.record_result("samplerate_setting", "time", setting_time)
            self.record_result("samplerate_setting", "accuracy", abs(actual_rate - rate) / rate)
            
            logger.info(f"Sample rate {rate/1e6:.3f} MHz: Set to {actual_rate/1e6:.3f} MHz "
                        f"in {setting_time:.6f} seconds "
                        f"(error: {abs(actual_rate - rate) / rate * 100:.4f}%)")
                        
        # Print summary
        logger.info("Sample rate setting benchmark completed")
        
    # The benchmark_sync_reception method with improved logging
    def benchmark_sync_reception(self, iterations=3):
        """Benchmark synchronous reception performance."""
        logger.info("=== Benchmarking synchronous reception performance ===")
        
        try:
            # Test buffer sizes
            buffer_sizes = [1024, 4096, 16384, 65536]
            
            # Set standard parameters
            current_freq = self.sdr.set_frequency(100e6)
            current_rate = self.sdr.set_samplerate(2.048e6)
            
            for buf_size in buffer_sizes:
                logger.info(f"Testing with buffer size {buf_size}")
                
                # Start synchronous reception
                self.sdr.start_rx_sync(buf_size)
                
                try:
                    for i in range(iterations):
                        # Measure read time
                        start_time = time.time()
                        iq_data = self.sdr.read_rx_sync()
                        read_time = time.time() - start_time
                        
                        # Calculate throughput
                        throughput = len(iq_data) / read_time if read_time > 0 else 0
                        
                        # Record detailed metrics
                        details = {
                            "iteration": i + 1,
                            "buffer_size": buf_size,
                            "timestamp": start_time,
                            "frequency_hz": current_freq,
                            "sample_rate_hz": current_rate,
                            "samples_received": len(iq_data),
                            "expected_samples": buf_size // 2,  # Complex samples are half of buffer size
                            "signal_stats": {
                                "mean_amplitude": float(np.mean(np.abs(iq_data))) if len(iq_data) > 0 else 0,
                                "max_amplitude": float(np.max(np.abs(iq_data))) if len(iq_data) > 0 else 0,
                                "std_amplitude": float(np.std(np.abs(iq_data))) if len(iq_data) > 0 else 0
                            }
                        }
                        
                        self.record_result(f"sync_reception_{buf_size}", "read_time", read_time, details)
                        self.record_result(f"sync_reception_{buf_size}", "throughput", throughput, details)
                        
                        logger.info(f"Iteration {i+1}/{iterations}: Read {len(iq_data)} samples "
                                    f"in {read_time:.6f} seconds "
                                    f"(throughput: {throughput:.2f} samples/second)")
                                    
                finally:
                    # Stop synchronous reception
                    self.sdr.stop_rx_sync()
                    
            # Print summary
            logger.info("Synchronous reception benchmark completed")
            
        except Exception as e:
            logger.error(f"Error in synchronous reception benchmark: {e}")
            
    # The benchmark_async_reception method with improved logging
    def benchmark_async_reception(self, iterations=3):
        """Benchmark asynchronous reception performance."""
        logger.info("=== Benchmarking asynchronous reception performance ===")
        
        try:
            # Test buffer sizes (all powers of 2 from 1K to 64K)
            buffer_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
            
            # Set standard parameters
            current_freq = self.sdr.set_frequency(100e6)
            current_rate = self.sdr.set_samplerate(2.048e6)
            
            for buf_size in buffer_sizes:
                logger.info(f"Testing with buffer size {buf_size}")
                
                for i in range(iterations):
                    # Setup for gathering metrics
                    received_samples = 0
                    callback_times = []
                    callback_samples = []
                    iteration_start = time.time()
                    
                    def sample_callback(iq_samples):
                        callback_time = time.time()
                        callback_times.append(callback_time)
                        callback_samples.append(len(iq_samples))
                        nonlocal received_samples
                        received_samples += len(iq_samples)
                    
                    # Start async reception
                    start_time = time.time()
                    self.sdr.start_rx_async(sample_callback, buf_count=4, buf_length=buf_size)
                    
                    # Let it run for a short time
                    time.sleep(1.0)
                    
                    # Stop async reception
                    logger.info("Stopping async reception")
                    self.sdr.stop_rx_async()
                    
                    # Allow time for cleanup
                    time.sleep(0.3)
                    
                    # Calculate metrics
                    total_time = 0
                    throughput = 0
                    
                    if callback_times:
                        total_time = max(callback_times) - start_time
                        throughput = received_samples / total_time if total_time > 0 else 0
                    
                    # Prepare detailed metrics
                    callback_details = []
                    for idx, (cb_time, samples) in enumerate(zip(callback_times, callback_samples)):
                        callback_details.append({
                            "callback_index": idx + 1,
                            "timestamp": cb_time,
                            "elapsed_since_start": cb_time - start_time,
                            "samples_received": samples
                        })
                    
                    # Record all intervals if we have at least 2 callbacks
                    if len(callback_times) >= 2:
                        intervals = np.diff(callback_times)
                        mean_interval = np.mean(intervals)
                        
                        interval_details = {
                            "iteration": i + 1,
                            "buffer_size": buf_size,
                            "timestamp": start_time,
                            "frequency_hz": current_freq,
                            "sample_rate_hz": current_rate,
                            "total_samples": received_samples,
                            "total_callbacks": len(callback_times),
                            "total_duration": total_time,
                            "callback_intervals_ms": [interval * 1000 for interval in intervals.tolist()],
                            "mean_interval_ms": mean_interval * 1000,
                            "std_interval_ms": float(np.std(intervals)) * 1000 if len(intervals) > 0 else 0,
                            "min_interval_ms": float(np.min(intervals)) * 1000 if len(intervals) > 0 else 0,
                            "max_interval_ms": float(np.max(intervals)) * 1000 if len(intervals) > 0 else 0,
                            "callback_details": callback_details
                        }
                        
                        self.record_result(f"async_reception_{buf_size}", "callback_interval", mean_interval, interval_details)
                        logger.info(f"Mean interval between callbacks: {mean_interval:.6f} seconds")
                    
                    # Record throughput
                    throughput_details = {
                        "iteration": i + 1,
                        "buffer_size": buf_size,
                        "timestamp": start_time,
                        "frequency_hz": current_freq,
                        "sample_rate_hz": current_rate, 
                        "total_samples": received_samples,
                        "total_callbacks": len(callback_times),
                        "total_duration": total_time,
                        "callback_details": callback_details
                    }
                    
                    self.record_result(f"async_reception_{buf_size}", "throughput", throughput, throughput_details)
                    
                    logger.info(f"Iteration {i+1}/{iterations}: Received {received_samples} samples "
                                f"in {total_time:.6f} seconds across {len(callback_times)} callbacks "
                                f"(throughput: {throughput:.2f} samples/second)")
                    
                    # Allow some time between iterations
                    time.sleep(0.5)
                    
            # Print summary
            logger.info("Asynchronous reception benchmark completed")
            
        except Exception as e:
            logger.error(f"Error in asynchronous reception benchmark: {e}")
            import traceback
            traceback.print_exc()
            

    def benchmark_async_reception_legacy(self, iterations=3):
        """Benchmark asynchronous reception performance."""
        logger.info("=== Benchmarking asynchronous reception performance ===")
        
        try:
            # Test buffer sizes
            buffer_sizes = [4096, 16384, 65536]
            
            # Set standard parameters
            current_freq = self.sdr.set_frequency(100e6)
            current_rate = self.sdr.set_samplerate(2.048e6)
            
            for buf_size in buffer_sizes:
                logger.info(f"Testing with buffer size {buf_size}")
                
                for i in range(iterations):
                    # Setup for gathering metrics
                    received_samples = 0
                    callback_times = []
                    callback_samples = []
                    stop_flag = False
                    iteration_start = time.time()
                    
                    def sample_callback(iq_samples):
                        nonlocal received_samples, callback_times, callback_samples, stop_flag
                        if stop_flag:
                            return
                        callback_time = time.time()
                        callback_times.append(callback_time)
                        callback_samples.append(len(iq_samples))
                        received_samples += len(iq_samples)
                    
                    # Start async reception
                    start_time = time.time()
                    self.sdr.start_rx_async(sample_callback, buf_count=4, buf_length=buf_size)
                    
                    # Let it run for a short time
                    time.sleep(1.0)
                    
                    # Set stop flag to prevent callback race conditions
                    stop_flag = True
                    time.sleep(0.1)  # Brief pause for flag to take effect
                    
                    # Stop async reception
                    self.sdr.stop_rx_async()
                    
                    # Calculate metrics
                    total_time = callback_times[-1] - start_time if callback_times else 0
                    throughput = received_samples / total_time if total_time > 0 else 0
                    
                    # Prepare detailed metrics
                    callback_details = []
                    for idx, (cb_time, samples) in enumerate(zip(callback_times, callback_samples)):
                        callback_details.append({
                            "callback_index": idx + 1,
                            "timestamp": cb_time,
                            "elapsed_since_start": cb_time - start_time,
                            "samples_received": samples
                        })
                    
                    # Record all intervals if we have at least 2 callbacks
                    if len(callback_times) >= 2:
                        intervals = np.diff(callback_times)
                        mean_interval = np.mean(intervals)
                        
                        interval_details = {
                            "iteration": i + 1,
                            "buffer_size": buf_size,
                            "timestamp": start_time,
                            "frequency_hz": current_freq,
                            "sample_rate_hz": current_rate,
                            "total_samples": received_samples,
                            "total_callbacks": len(callback_times),
                            "total_duration": total_time,
                            "callback_intervals_ms": [interval * 1000 for interval in intervals.tolist()],
                            "mean_interval_ms": mean_interval * 1000,
                            "std_interval_ms": float(np.std(intervals)) * 1000,
                            "min_interval_ms": float(np.min(intervals)) * 1000 if len(intervals) > 0 else 0,
                            "max_interval_ms": float(np.max(intervals)) * 1000 if len(intervals) > 0 else 0,
                            "callback_details": callback_details
                        }
                        
                        self.record_result(f"async_reception_{buf_size}", "callback_interval", mean_interval, interval_details)
                        logger.info(f"Mean interval between callbacks: {mean_interval:.6f} seconds")
                    
                    # Record throughput
                    throughput_details = {
                        "iteration": i + 1,
                        "buffer_size": buf_size,
                        "timestamp": start_time,
                        "frequency_hz": current_freq,
                        "sample_rate_hz": current_rate, 
                        "total_samples": received_samples,
                        "total_callbacks": len(callback_times),
                        "total_duration": total_time,
                        "callback_details": callback_details
                    }
                    
                    self.record_result(f"async_reception_{buf_size}", "throughput", throughput, throughput_details)
                    
                    logger.info(f"Iteration {i+1}/{iterations}: Received {received_samples} samples "
                                f"in {total_time:.6f} seconds across {len(callback_times)} callbacks "
                                f"(throughput: {throughput:.2f} samples/second)")
                    
                    # Allow some time between iterations
                    time.sleep(0.5)
                    
            # Print summary
            logger.info("Asynchronous reception benchmark completed")
            
        except Exception as e:
            logger.error(f"Error in asynchronous reception benchmark: {e}")

        
    def benchmark_data_processing(self):
        """Benchmark data processing performance."""
        logger.info("=== Benchmarking data processing performance ===")
        
        # Get a buffer of data to process
        buf_size = 16384
        self.sdr.set_frequency(100e6)
        self.sdr.set_samplerate(2.048e6)
        
        self.sdr.start_rx_sync(buf_size)
        try:
            iq_data = self.sdr.read_rx_sync()
            
            # Benchmark FFT performance
            fft_sizes = [1024, 4096, 8192]
            for fft_size in fft_sizes:
                if len(iq_data) < fft_size:
                    continue
                    
                # Measure FFT time
                start_time = time.time()
                for _ in range(10):  # Run multiple times for better measurement
                    spectrum = np.fft.fft(iq_data[:fft_size])
                    _ = 20 * np.log10(np.abs(spectrum) + 1e-10)  # Convert to dB
                fft_time = (time.time() - start_time) / 10
                
                self.record_result("data_processing", f"fft_{fft_size}_time", fft_time)
                logger.info(f"FFT size {fft_size}: {fft_time:.6f} seconds per FFT")
                
            # Benchmark FM demodulation
            start_time = time.time()
            # FM demodulation process
            phase = np.angle(iq_data)
            diff_phase = np.diff(np.unwrap(phase))
            _ = diff_phase * (2.048e6 / (2 * np.pi * 75e3))
            demod_time = time.time() - start_time
            
            self.record_result("data_processing", "fm_demod_time", demod_time)
            logger.info(f"FM demodulation: {demod_time:.6f} seconds for {len(iq_data)} samples")
            
            # Benchmark filtering
            filter_sizes = [31, 63, 127]
            for filter_size in filter_sizes:
                # Create filter coefficients
                h = np.sinc(np.linspace(-filter_size//2, filter_size//2, filter_size)) * np.hamming(filter_size)
                h = h / np.sum(h)
                
                # Measure filtering time
                start_time = time.time()
                _ = np.convolve(iq_data, h, mode='valid')
                filter_time = time.time() - start_time
                
                self.record_result("data_processing", f"filter_{filter_size}_time", filter_time)
                logger.info(f"Filter size {filter_size}: {filter_time:.6f} seconds")
                
            # Benchmark decimation
            decimation_factors = [2, 4, 8, 16]
            for factor in decimation_factors:
                # Measure decimation time
                start_time = time.time()
                _ = iq_data[::factor]  # Simple decimation
                decim_time = time.time() - start_time
                
                self.record_result("data_processing", f"decimate_{factor}_time", decim_time)
                logger.info(f"Decimation factor {factor}: {decim_time:.6f} seconds")
                
        finally:
            self.sdr.stop_rx_sync()
            
        # Print summary
        logger.info("Data processing benchmark completed")


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description='Benchmark Fobos SDR wrapper performance')
    parser.add_argument('--device', type=int, default=0,
                        help='Device index to use (default: 0)')
    parser.add_argument('--output-dir', type=str, default="benchmark_results",
                        help='Directory to save benchmark results (default: benchmark_results)')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of iterations for each test (default: 3)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only generate comparison plots from existing results')
    
    args = parser.parse_args()
    
    benchmark = FobosSDRBenchmark(args.device, args.output_dir)
    
    if args.plot_only:
        # Just generate comparison plots from existing results
        benchmark.generate_comparison_plots()
    else:
        # Run all benchmarks
        success = benchmark.run_all_benchmarks(args.iterations)
        
        if success:
            # Also generate comparison plots
            benchmark.generate_comparison_plots()
            logger.info("Benchmarking completed successfully")
        else:
            logger.error("Benchmarking failed")
            return 1
            
    return 0


if __name__ == "__main__":
    sys.exit(main())