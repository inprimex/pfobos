"""
Performance tests for Fobos SDR wrapper.
These tests measure the performance of various operations and data processing capabilities.
"""

import unittest
import time
import numpy as np
import logging
import sys
import os
import cProfile
import pstats
import io
from functools import wraps
from collections import deque
import threading
import queue
from scipy import signal  # Added missing import for signal module

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path to import the wrapper properly
# This ensures imports work correctly regardless of the execution directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from shared module directly to maintain consistency with other tests
from shared.fwrapper import FobosSDR, FobosException, FobosError

# Import the hardware detection decorator from integration tests
# Using relative import which is more appropriate for a test package
from .test_integration import requires_hardware


def profile(func):
    """Decorator to profile a function using cProfile."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        try:
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            return result
        finally:
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Print top 20 functions by cumulative time
            logger.info(f"Profile for {func.__name__}:\n{s.getvalue()}")
    return wrapper


def time_execution(func):
    """Decorator to time the execution of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"{func.__name__} executed in {elapsed_time:.6f} seconds")
        return result
    return wrapper


class PerformanceMetrics:
    """Class to collect and analyze performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        
    def add_timing(self, name, value):
        """Add a timing measurement."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        
    def get_stats(self, name):
        """Get statistics for a specific metric."""
        if name not in self.metrics or not self.metrics[name]:
            return None
            
        values = np.array(self.metrics[name])
        return {
            'count': len(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values),
            'std': np.std(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }
        
    def print_stats(self, name=None):
        """Print statistics for metrics."""
        names = [name] if name else self.metrics.keys()
        
        for n in names:
            stats = self.get_stats(n)
            if stats:
                logger.info(f"Performance stats for {n}:")
                logger.info(f"  Count: {stats['count']}")
                logger.info(f"  Mean: {stats['mean']:.6f} seconds")
                logger.info(f"  Median: {stats['median']:.6f} seconds")
                logger.info(f"  Min: {stats['min']:.6f} seconds")
                logger.info(f"  Max: {stats['max']:.6f} seconds")
                logger.info(f"  Std Dev: {stats['std']:.6f} seconds")
                logger.info(f"  95th percentile: {stats['p95']:.6f} seconds")
                logger.info(f"  99th percentile: {stats['p99']:.6f} seconds")


class TestFobosSDRPerformance(unittest.TestCase):
    """Performance tests for Fobos SDR wrapper."""

    @classmethod
    def setUpClass(cls):
        """Set up the test class."""
        try:
            cls.sdr = FobosSDR()
            cls.device_count = cls.sdr.get_device_count()
            if cls.device_count == 0:
                cls.skipTest(cls, "No Fobos SDR hardware detected")
            else:
                # Log device information
                device_serials = cls.sdr.list_devices()
                logger.info(f"Found {cls.device_count} devices: {device_serials}")
                
                # Initialize performance metrics
                cls.metrics = PerformanceMetrics()
        except Exception as e:
            if "Could not load Fobos SDR library" in str(e):
                cls.skipTest(cls, "Fobos SDR library not found")
            else:
                raise

    @requires_hardware
    def setUp(self):
        """Set up each test."""
        self.sdr = FobosSDR()
        self.sdr.open(0)  # Open first device
        
        # Configure SDR with standard settings
        self.sdr.set_frequency(100e6)  # 100 MHz
        self.sdr.set_samplerate(2.048e6)  # 2.048 MHz
        self.sdr.set_lna_gain(1)
        self.sdr.set_vga_gain(10)

    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, 'sdr') and self.sdr.dev is not None:
            # Make sure to stop any active modes
            if hasattr(self.sdr, '_async_mode') and self.sdr._async_mode:
                try:
                    logger.info("Stopping async mode in tearDown")
                    self.sdr.stop_rx_async()
                    # Give a short pause for resources to be freed properly
                    time.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Error stopping async mode in tearDown: {e}")
                    
            if hasattr(self.sdr, '_sync_mode') and self.sdr._sync_mode:
                try:
                    logger.info("Stopping sync mode in tearDown")
                    self.sdr.stop_rx_sync()
                except Exception as e:
                    logger.warning(f"Error stopping sync mode in tearDown: {e}")
                    
            try:
                self.sdr.close()
            except Exception as e:
                logger.warning(f"Error closing device in tearDown: {e}")

    @requires_hardware
    @time_execution
    def test_open_close_performance(self):
        """Test the performance of opening and closing the device."""
        iterations = 5
        
        # Close the device that was opened in setUp
        self.sdr.close()
        
        for i in range(iterations):
            # Measure open time
            start_time = time.time()
            self.sdr.open(0)
            open_time = time.time() - start_time
            self.metrics.add_timing('device_open', open_time)
            logger.info(f"Iteration {i+1}/{iterations}: Device opened in {open_time:.6f} seconds")
            
            # Verify device is open by getting board info
            info = self.sdr.get_board_info()
            
            # Measure close time
            start_time = time.time()
            self.sdr.close()
            close_time = time.time() - start_time
            self.metrics.add_timing('device_close', close_time)
            logger.info(f"Iteration {i+1}/{iterations}: Device closed in {close_time:.6f} seconds")
        
        # Print summary statistics
        self.metrics.print_stats('device_open')
        self.metrics.print_stats('device_close')
        
        # Reopen device for tearDown (otherwise it will try to close an already closed device)
        self.sdr.open(0)

    @requires_hardware
    @time_execution
    def test_frequency_change_performance(self):
        """Test the performance of changing frequencies."""
        iterations = 10
        frequencies = [100e6, 200e6, 433e6, 868e6, 915e6]
        
        for i in range(iterations):
            for freq in frequencies:
                start_time = time.time()
                actual_freq = self.sdr.set_frequency(freq)
                elapsed_time = time.time() - start_time
                
                self.metrics.add_timing('frequency_change', elapsed_time)
                logger.info(f"Set frequency to {actual_freq/1e6:.3f} MHz in {elapsed_time:.6f} seconds")
        
        # Print summary statistics
        self.metrics.print_stats('frequency_change')

    @requires_hardware
    @time_execution
    def test_samplerate_change_performance(self):
        """Test the performance of changing sample rates."""
        # Get available sample rates
        sample_rates = self.sdr.get_samplerates()
        
        if not sample_rates:
            self.skipTest("No sample rates available")
            
        # Test changing between available rates
        for rate in sample_rates:
            start_time = time.time()
            actual_rate = self.sdr.set_samplerate(rate)
            elapsed_time = time.time() - start_time
            
            self.metrics.add_timing('samplerate_change', elapsed_time)
            logger.info(f"Set sample rate to {actual_rate/1e6:.3f} MHz in {elapsed_time:.6f} seconds")
        
        # Print summary statistics
        self.metrics.print_stats('samplerate_change')

    @requires_hardware
    @profile
    @time_execution
    def test_sync_read_performance(self):
        """Test synchronous read performance."""
        # Define buffer sizes to test
        buffer_sizes = [1024, 4096, 16384, 65536]
        iterations_per_size = 5
        
        for buf_size in buffer_sizes:
            logger.info(f"Testing sync read with buffer size {buf_size}")
            
            # Start synchronous reception
            self.sdr.start_rx_sync(buf_size)
            
            try:
                read_times = []
                throughputs = []
                
                for i in range(iterations_per_size):
                    # Measure read time
                    start_time = time.time()
                    iq_data = self.sdr.read_rx_sync()
                    elapsed_time = time.time() - start_time
                    
                    # Calculate throughput in samples per second
                    throughput = len(iq_data) / elapsed_time if elapsed_time > 0 else 0
                    
                    read_times.append(elapsed_time)
                    throughputs.append(throughput)
                    
                    self.metrics.add_timing(f'sync_read_{buf_size}', elapsed_time)
                    logger.info(f"Read {len(iq_data)} samples in {elapsed_time:.6f} seconds "
                                f"(throughput: {throughput:.2f} samples/second)")
                
                # Calculate statistics for this buffer size
                mean_time = np.mean(read_times)
                mean_throughput = np.mean(throughputs)
                logger.info(f"Buffer size {buf_size}: Mean read time {mean_time:.6f} seconds, "
                            f"Mean throughput {mean_throughput:.2f} samples/second")
                
            finally:
                # Stop synchronous reception
                self.sdr.stop_rx_sync()
        
        # Print summary statistics for each buffer size
        for buf_size in buffer_sizes:
            self.metrics.print_stats(f'sync_read_{buf_size}')

    @requires_hardware
    @profile
    @time_execution
    def test_async_read_performance(self):
        """Test asynchronous read performance."""
        # Define buffer sizes to test (all powers of 2 from 1K to 64K)
        buffer_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
        iterations_per_size = 3  # Reduced to prevent timeout issues
        
        for buf_size in buffer_sizes:
            logger.info(f"Testing async read with buffer size {buf_size}")
            
            for i in range(iterations_per_size):
                # Create a queue to store timing information
                timing_queue = queue.Queue()
                samples_count = queue.Queue()
                
                # Create a callback function
                def sample_callback(iq_samples):
                    end_time = time.time()
                    timing_queue.put(end_time)
                    samples_count.put(len(iq_samples))
                
                # Start async reception
                start_time = time.time()
                self.sdr.start_rx_async(sample_callback, buf_count=4, buf_length=buf_size)
                
                # Wait for a few callbacks to be processed (with timeout)
                max_wait = 2.0  # seconds
                wait_end = time.time() + max_wait
                
                # Sleep until we have enough callbacks or timeout
                while (timing_queue.qsize() < 3) and (time.time() < wait_end):
                    time.sleep(0.1)
                
                # Stop async reception
                try:
                    self.sdr.stop_rx_async()
                except Exception as e:
                    logger.warning(f"Error stopping async reception: {e}")
                
                # Allow a moment for stop to take effect
                time.sleep(0.3)
                
                # Process timing information
                callback_times = []
                samples_received = 0
                
                while not timing_queue.empty() and not samples_count.empty():
                    callback_time = timing_queue.get() - start_time
                    samples = samples_count.get()
                    samples_received += samples
                    callback_times.append(callback_time)
                
                if callback_times:
                    # Calculate mean time between callbacks
                    if len(callback_times) > 1:
                        intervals = np.diff(callback_times)
                        mean_interval = np.mean(intervals)
                        self.metrics.add_timing(f'async_interval_{buf_size}', mean_interval)
                        logger.info(f"Mean interval between callbacks: {mean_interval:.6f} seconds")
                    
                    # Calculate overall throughput
                    elapsed_time = callback_times[-1] - callback_times[0] if len(callback_times) > 1 else callback_times[0]
                    throughput = samples_received / elapsed_time if elapsed_time > 0 else 0
                    logger.info(f"Received {samples_received} samples in {elapsed_time:.6f} seconds "
                                f"(throughput: {throughput:.2f} samples/second)")
                    
                    # Add to metrics
                    self.metrics.add_timing(f'async_throughput_{buf_size}', throughput)
                else:
                    logger.warning(f"No callbacks received for buffer size {buf_size}")
                
                # Allow some time between iterations
                time.sleep(0.5)
        
        # Print summary statistics for each buffer size
        for buf_size in buffer_sizes:
            self.metrics.print_stats(f'async_interval_{buf_size}')
            self.metrics.print_stats(f'async_throughput_{buf_size}')

    @requires_hardware
    @profile
    @time_execution
    def test_async_read_performance_legacy(self):
        """Test asynchronous read performance."""
        # Define buffer sizes to test
        buffer_sizes = [1024, 4096, 16384, 65536]
        iterations_per_size = 3  # Reduced to prevent timeout issues
        
        for buf_size in buffer_sizes:
            logger.info(f"Testing async read with buffer size {buf_size}")
            
            for i in range(iterations_per_size):
                # Create a queue to store timing information
                timing_queue = queue.Queue()
                samples_count = queue.Queue()
                stop_flag = threading.Event()
                
                # Create a callback function with safety flag
                def sample_callback(iq_samples):
                    # Check if we should stop processing callbacks
                    if stop_flag.is_set():
                        return
                    end_time = time.time()
                    timing_queue.put(end_time)
                    samples_count.put(len(iq_samples))
                
                # Start async reception
                start_time = time.time()
                self.sdr.start_rx_async(sample_callback, buf_count=4, buf_length=buf_size)
                
                # Wait for a few callbacks to be processed (with timeout)
                max_wait = 2.0  # seconds
                wait_end = time.time() + max_wait
                
                # Sleep until we have enough callbacks or timeout
                while (timing_queue.qsize() < 3) and (time.time() < wait_end):
                    time.sleep(0.1)
                
                # Set stop flag before stopping async mode to prevent callback race conditions
                stop_flag.set()
                time.sleep(0.1)  # Small delay to let callbacks notice the flag
                
                # Stop async reception
                try:
                    self.sdr.stop_rx_async()
                except Exception as e:
                    logger.warning(f"Error stopping async reception: {e}")
                
                # Process timing information
                callback_times = []
                samples_received = 0
                
                while not timing_queue.empty() and not samples_count.empty():
                    callback_time = timing_queue.get() - start_time
                    samples = samples_count.get()
                    samples_received += samples
                    callback_times.append(callback_time)
                
                if callback_times:
                    # Calculate mean time between callbacks
                    if len(callback_times) > 1:
                        intervals = np.diff(callback_times)
                        mean_interval = np.mean(intervals)
                        self.metrics.add_timing(f'async_interval_{buf_size}', mean_interval)
                        logger.info(f"Mean interval between callbacks: {mean_interval:.6f} seconds")
                    
                    # Calculate overall throughput
                    elapsed_time = callback_times[-1] - callback_times[0] if len(callback_times) > 1 else callback_times[0]
                    throughput = samples_received / elapsed_time if elapsed_time > 0 else 0
                    logger.info(f"Received {samples_received} samples in {elapsed_time:.6f} seconds "
                                f"(throughput: {throughput:.2f} samples/second)")
                    
                    # Add to metrics
                    self.metrics.add_timing(f'async_throughput_{buf_size}', throughput)
                else:
                    logger.warning(f"No callbacks received for buffer size {buf_size}")
                
                # Allow some time between iterations
                time.sleep(0.5)
        
        # Print summary statistics for each buffer size
        for buf_size in buffer_sizes:
            self.metrics.print_stats(f'async_interval_{buf_size}')
            self.metrics.print_stats(f'async_throughput_{buf_size}')

    @requires_hardware
    @profile
    def test_signal_processing_performance(self):
        """Test performance of common signal processing operations on SDR data."""
        # Configure test parameters
        buf_size = 16384
        fft_sizes = [1024, 4096, 8192]
        
        # Start synchronous reception
        self.sdr.start_rx_sync(buf_size)
        
        try:
            # Read a buffer of samples
            iq_data = self.sdr.read_rx_sync()
            
            if len(iq_data) < max(fft_sizes):
                logger.warning(f"Not enough samples for FFT test: {len(iq_data)} < {max(fft_sizes)}")
                self.skipTest("Not enough samples for signal processing tests")
                return
            
            # 1. Test FFT performance
            for fft_size in fft_sizes:
                # Ensure we have enough samples
                if len(iq_data) < fft_size:
                    continue
                    
                # Time the FFT operation
                start_time = time.time()
                for _ in range(10):  # Do multiple FFTs to get better timing
                    spectrum = np.fft.fft(iq_data[:fft_size])
                    spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-10)
                elapsed_time = (time.time() - start_time) / 10
                
                self.metrics.add_timing(f'fft_{fft_size}', elapsed_time)
                logger.info(f"FFT size {fft_size}: {elapsed_time:.6f} seconds per FFT")
            
            # 2. Test filter performance
            filter_sizes = [31, 63, 127]
            for filter_size in filter_sizes:
                # Create filter coefficients (simple low-pass FIR)
                h = np.sinc(np.linspace(-filter_size//2, filter_size//2, filter_size)) * np.hamming(filter_size)
                h = h / np.sum(h)  # Normalize
                
                # Time the filtering operation
                start_time = time.time()
                filtered = np.convolve(iq_data, h, mode='valid')
                elapsed_time = time.time() - start_time
                
                self.metrics.add_timing(f'filter_{filter_size}', elapsed_time)
                logger.info(f"Filter size {filter_size}: {elapsed_time:.6f} seconds "
                            f"(throughput: {len(filtered)/elapsed_time:.2f} samples/second)")
            
            # 3. Test demodulation performance
            # FM demodulation
            start_time = time.time()
            # Extract phase of the IQ samples
            phase = np.angle(iq_data)
            # Compute phase difference
            diff_phase = np.diff(np.unwrap(phase))
            # Scale to get instantaneous frequency
            demodulated = diff_phase * (2.048e6 / (2 * np.pi * 75e3))
            elapsed_time = time.time() - start_time
            
            self.metrics.add_timing('fm_demod', elapsed_time)
            logger.info(f"FM demodulation: {elapsed_time:.6f} seconds "
                        f"(throughput: {len(demodulated)/elapsed_time:.2f} samples/second)")
            
            # 4. Test decimation performance
            decimation_factors = [2, 4, 8, 16]
            for factor in decimation_factors:
                # Time the decimation operation
                start_time = time.time()
                try:
                    decimated = signal.decimate(iq_data, factor)
                    elapsed_time = time.time() - start_time
                    
                    self.metrics.add_timing(f'decimate_{factor}', elapsed_time)
                    logger.info(f"Decimation factor {factor}: {elapsed_time:.6f} seconds "
                                f"(throughput: {len(iq_data)/elapsed_time:.2f} samples/second)")
                except Exception as e:
                    logger.error(f"Error in decimation with factor {factor}: {e}")
                
        finally:
            # Stop synchronous reception
            self.sdr.stop_rx_sync()
        
        # Print summary statistics
        for fft_size in fft_sizes:
            self.metrics.print_stats(f'fft_{fft_size}')
            
        for filter_size in filter_sizes:
            self.metrics.print_stats(f'filter_{filter_size}')
            
        self.metrics.print_stats('fm_demod')
        
        for factor in decimation_factors:
            self.metrics.print_stats(f'decimate_{factor}')

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Print overall performance summary
        if hasattr(cls, 'metrics'):
            logger.info("===== OVERALL PERFORMANCE SUMMARY =====")
            for metric_name in sorted(cls.metrics.metrics.keys()):
                cls.metrics.print_stats(metric_name)


if __name__ == '__main__':
    unittest.main()
