#!/usr/bin/env python3
# Simple test script for Fobos SDR asynchronous reception
# Using key press detection for termination instead of Ctrl+C

import sys
import os
import time
import logging
import numpy as np
import threading
import queue
import select
import termios
import tty

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Assuming script is in ./scripts subdirectory
sys.path.append(project_root)

# Import the wrapper from the shared module
from shared.fwrapper import FobosSDR, FobosException

class AsyncTester:
    """Class-based approach similar to rtanalyzer.py"""
    
    def __init__(self):
        self.sdr = None
        self.is_running = False
        self.total_samples = 0
        self.buffer_count = 0
        self.data_queue = queue.Queue(maxsize=10)  # Buffer for async data
        self.processing_thread = None
    
    def async_callback(self, samples):
        """Callback function for async data reception"""
        if not self.is_running:
            return
            
        # Add samples to queue, non-blocking
        try:
            self.data_queue.put_nowait(samples)
        except queue.Full:
            # Skip if queue is full
            pass
    
    def process_data(self):
        """Process data from the queue in a separate thread"""
        while self.is_running:
            try:
                # Get samples with timeout
                samples = self.data_queue.get(timeout=0.5)
                
                # Process samples
                self.buffer_count += 1
                self.total_samples += len(samples)
                
                # Log some information periodically
                if self.buffer_count % 10 == 0:
                    logger.info(f"Received buffer #{self.buffer_count} with {len(samples)} samples")
                    logger.info(f"Total samples: {self.total_samples}")
                    if len(samples) > 0:
                        logger.info(f"Sample mean magnitude: {np.mean(np.abs(samples)):.6f}")
                
                # Indicate task is done
                self.data_queue.task_done()
            
            except queue.Empty:
                # Just continue if no data
                continue
            except Exception as e:
                logger.error(f"Error processing data: {e}")
                if not self.is_running:
                    break
    
    def start(self):
        """Start the test"""
        if self.is_running:
            logger.info("Already running")
            return False
            
        # Reset counters
        self.total_samples = 0
        self.buffer_count = 0
        
        try:
            # Create SDR instance
            self.sdr = FobosSDR()
            
            # Check for devices
            device_count = self.sdr.get_device_count()
            logger.info(f"Found {device_count} devices")
            
            if device_count == 0:
                logger.error("No SDR devices found")
                return False
            
            # Open first device
            self.sdr.open(0)
            logger.info("Device opened successfully")
            
            # Get device info
            info = self.sdr.get_board_info()
            logger.info(f"Connected to {info['product']} (Serial: {info['serial']})")
            
            # Configure device
            actual_freq = self.sdr.set_frequency(100e6)
            logger.info(f"Set frequency to {actual_freq/1e6:.3f} MHz")
            
            # Set sample rate
            rates = self.sdr.get_samplerates()
            sample_rate = 10e6 if 10e6 in rates else rates[-1]
            actual_rate = self.sdr.set_samplerate(sample_rate)
            logger.info(f"Set sample rate to {actual_rate/1e6:.3f} MHz")
            
            # Set gain settings
            self.sdr.set_lna_gain(1)
            self.sdr.set_vga_gain(5)
            
            # Set running flag before starting
            self.is_running = True
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self.process_data)
            self.processing_thread.daemon = True  # Daemon thread will exit when main thread exits
            self.processing_thread.start()
            
            # Start async reception
            logger.info("Starting async reception")
            self.sdr.start_rx_async(self.async_callback, 16, 32768)
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting test: {e}")
            self.stop()
            return False
    
    def stop(self):
        """Stop the test and clean up resources"""
        logger.info("Stopping test...")
        
        # Set flag first to stop processing
        old_running = self.is_running
        self.is_running = False
        
        # Only do cleanup if we were previously running
        if old_running:
            # Wait for processing thread to finish
            if self.processing_thread and self.processing_thread.is_alive():
                logger.info("Waiting for processing thread to finish...")
                self.processing_thread.join(timeout=1.0)
            
            # Clean up SDR
            self.cleanup_sdr()
            
            # Calculate statistics if we have data
            if self.buffer_count > 0:
                logger.info(f"Received {self.total_samples} samples in {self.buffer_count} buffers")
    
    def cleanup_sdr(self):
        """Clean up SDR resources - similar to rtanalyzer approach"""
        if self.sdr is not None:
            try:
                # First stop async reception if running
                logger.info("Stopping async reception...")
                self.sdr.stop_rx_async()
                
                # Then close device
                logger.info("Closing device...")
                self.sdr.close()
                logger.info("SDR stopped and closed")
            except Exception as e:
                logger.error(f"Error during SDR cleanup: {e}")
            finally:
                # Set to None to allow garbage collection
                self.sdr = None

def is_key_pressed():
    """Check if a key has been pressed without blocking"""
    # Save the terminal settings
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        # Set terminal to raw mode
        tty.setraw(sys.stdin.fileno(), termios.TCSANOW)
        
        # Check if there's data ready to read from stdin
        if select.select([sys.stdin], [], [], 0)[0]:
            # Read one character
            key = sys.stdin.read(1)
            return True
        return False
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def main():
    """Main entry point"""
    # Create tester instance
    tester = AsyncTester()
    
    logger.info("Starting async test (will run for up to 10 seconds)")
    logger.info("Press any key to stop the test")
    
    try:
        # Start the test
        if tester.start():
            # Run for specified duration or until key press
            start_time = time.time()
            max_duration = 10
            
            while tester.is_running and time.time() - start_time < max_duration:
                # Check for key press
                if is_key_pressed():
                    logger.info("Key press detected, stopping test")
                    break
                
                # Sleep a short time
                time.sleep(0.1)
                
            if time.time() - start_time >= max_duration:
                logger.info("Test duration reached")
        else:
            logger.error("Failed to start test")
            return 1
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Always clean up properly
        logger.info("Cleaning up...")
        tester.stop()
    
    logger.info("Test completed")
    return 0

if __name__ == "__main__":
    # Make sure stdin is a tty
    if not sys.stdin.isatty():
        logger.error("This script requires a terminal with stdin access")
        sys.exit(1)
        
    exit_code = main()
    sys.exit(exit_code)
