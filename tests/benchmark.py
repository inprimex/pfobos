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
from collections import defaultdict
import logging

# Add parent directory to path to import the wrapper
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from fobos_wrapper import FobosSDR, FobosException

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
            
    def record_result(self, test_name, metric, value):
        """Record a benchmark result."""
        self.results[f"{test_name}_{metric}"].append(value)
        
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
        
    def save_results(self):
        """Save benchmark results to file."""
        if not self.results:
            logger.warning("No results to save.")
            return
            
        # Create a timestamp for the results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results as JSON
        stats = {}
        for key in self.results:
            test_name, metric = key.rsplit('_', 1)
            if test_name not in stats:
                stats[test_name] = {}
            stats[test_name][metric] = self.get_stats(test_name, metric)
            
        # Get device info for metadata
        device_info = {}
        if self.sdr and self.sdr.dev is not None:
            try:
                device_info = self.sdr.get_board_info()
            except:
                pass
                
        # Create result data
        result_data = {
            'timestamp': timestamp,
            'device': device_info,
            'stats': stats
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
            
    def benchmark_open_close(self, iterations=3):
        """Benchmark device open/close performance."""
        logger.info("=== Benchmarking device open/close performance ===")
        
        # Close the device that was opened in setup
        if self.sdr.dev is not None:
            self.sdr.close()
            
        for i in range(iterations):
            # Measure open time
            start_time = time.time()
            self.sdr.open(self.device_index)
            open_time = time.time() - start_time
            
            self.record_result("open_close", "open_time", open_time)
            logger.info(f"Iteration {i+1}/{iterations}: Device opened in {open_time:.6f} seconds")
            
            # Verify device is open by getting board info
            info = self.sdr.get_board_info()
            
            # Measure close time
            start_time = time.time()
            self.sdr.close()
            close_time = time.time() - start_time
            
            self.record_result("open_close", "close_time", close_time)
            logger.info(f"Iteration {i+1}/{iterations}: Device closed in {close_time:.6f} seconds")
            
        # Print summary
        logger.info("Open/close benchmark completed")
        
        # Reopen the device for subsequent tests
        self.sdr.open(self.device_index)
        
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
                
                self.record_result("frequency_tuning", "time", tuning_time)
                self.record_result("frequency_tuning", "accuracy", abs(actual_freq - freq) / freq)
                
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
        
    def benchmark_sync_reception(self, iterations=3):
        """Benchmark synchronous reception performance."""
        logger.info("=== Benchmarking synchronous reception performance ===")
        
        # Test buffer sizes
        buffer_sizes = [1024, 4096, 16384, 65536]
        
        # Set standard parameters
        self.sdr.set_frequency(100e6)
        self.sdr.set_samplerate(2.048e6)
        
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
                    
                    self.record_result(f"sync_reception_{buf_size}", "read_time", read_time)
                    self.record_result(f"sync_reception_{buf_size}", "throughput", throughput)
                    
                    logger.info(f"Iteration {i+1}/{iterations}: Read {len(iq_data)} samples "
                                f"in {read_time:.6f} seconds "
                                f"(throughput: {throughput:.2f} samples/second)")
                                
            finally:
                # Stop synchronous reception
                self.sdr.stop_rx_sync()
                
        # Print summary
        logger.info("Synchronous reception benchmark completed")
        
    def benchmark_async_reception(self, iterations=3):
        """Benchmark asynchronous reception performance."""
        logger.info("=== Benchmarking asynchronous reception performance ===")
        
        # Test buffer sizes
        buffer_sizes = [4096, 16384, 65536]
        
        # Set standard parameters
        self.sdr.set_frequency(100e6)
        self.sdr.set_samplerate(2.048e6)
        
        for buf_size in buffer_sizes:
            logger.info(f"Testing with buffer size {buf_size}")
            
            for i in range(iterations):
                # Setup for gathering metrics
                received_samples = 0
                callback_times = []
                
                def sample_callback(iq_samples):
                    nonlocal received_samples
                    callback_time = time.time()
                    callback_times.append(callback_time)
                    received_samples += len(iq_samples)
                
                # Start async reception
                start_time = time.time()
                self.sdr.start_rx_async(sample_callback, buf_count=4, buf_length=buf_size)
                
                # Let it run for a short time
                time.sleep(1.0)
                
                # Stop async reception
                self.sdr.stop_rx_async()
                
                # Calculate metrics
                total_time = callback_times[-1] - start_time if callback_times else 0
                throughput = received_samples / total_time if total_time > 0 else 0
                
                # Calculate latency if we have at least 2 callbacks
                if len(callback_times) >= 2:
                    intervals = np.diff(callback_times)
                    mean_interval = np.mean(intervals)
                    self.record_result(f"async_reception_{buf_size}", "callback_interval", mean_interval)
                    logger.info(f"Mean interval between callbacks: {mean_interval:.6f} seconds")
                
                self.record_result(f"async_reception_{buf_size}", "throughput", throughput)
                
                logger.info(f"Iteration {i+1}/{iterations}: Received {received_samples} samples "
                            f"in {total_time:.6f} seconds across {len(callback_times)} callbacks "
                            f"(throughput: {throughput:.2f} samples/second)")
                
                # Allow some time between iterations
                time.sleep(0.5)
                
        # Print summary
        logger.info("Asynchronous reception benchmark completed")
        
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