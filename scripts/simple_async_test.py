#!/usr/bin/env python3
"""
Simplified test script for Fobos SDR asynchronous reception mode.
Runs for exactly 10 seconds and stops automatically.
"""

import sys
import os
import time
import logging
import numpy as np
import threading

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the wrapper - adjust path as needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fobos_wrapper import FobosSDR, FobosException

# Global variables for statistics
total_samples = 0
buffer_count = 0
stop_flag = False

def async_callback(samples):
    """Callback function for asynchronous reception."""
    global total_samples, buffer_count
    
    if stop_flag:  # Skip processing if we're stopping
        return
        
    buffer_count += 1
    total_samples += len(samples)
    
    # Log some information periodically
    if buffer_count % 10 == 0:
        logger.info(f"Received buffer #{buffer_count} with {len(samples)} samples")
        logger.info(f"Total samples: {total_samples}")
        if len(samples) > 0:
            logger.info(f"Sample mean magnitude: {np.mean(np.abs(samples)):.6f}")

def main():
    """Run the async test with a fixed duration."""
    global total_samples, buffer_count, stop_flag
    
    # Reset counters
    total_samples = 0
    buffer_count = 0
    stop_flag = False
    
    logger.info("Starting simplified async test (will run for 10 seconds)")
    
    try:
        # Create SDR instance
        sdr = FobosSDR()
        
        # Open first device
        sdr.open(0)
        logger.info("Device opened successfully")
        
        # Configure device
        actual_freq = sdr.set_frequency(100e6)
        logger.info(f"Set frequency to {actual_freq/1e6:.3f} MHz")
        
        rates = sdr.get_samplerates()
        # Use 10 MHz or the lowest available rate
        sample_rate = 10e6 if 10e6 in rates else rates[-1]
        actual_rate = sdr.set_samplerate(sample_rate)
        logger.info(f"Set sample rate to {actual_rate/1e6:.3f} MHz")
        
        # Set gain settings
        sdr.set_lna_gain(1)
        sdr.set_vga_gain(5)
        
        # Start async reception
        logger.info("Starting async reception")
        sdr.start_rx_async(async_callback, 16, 32768)
        
        # Run for exactly 10 seconds
        test_duration = 10
        logger.info(f"Test will run for {test_duration} seconds...")
        start_time = time.time()
        
        # Simple sleep-based approach
        time.sleep(test_duration)
        
        # Set stop flag to prevent callback from processing more data
        stop_flag = True
        
        # Calculate statistics
        elapsed = time.time() - start_time
        sample_rate_achieved = total_samples / elapsed if elapsed > 0 else 0
        
        logger.info(f"Test completed after {elapsed:.2f} seconds")
        logger.info(f"Received {total_samples} samples in {buffer_count} buffers")
        logger.info(f"Achieved sample rate: {sample_rate_achieved/1e6:.2f} MSps")
        
        # Stop reception and clean up
        logger.info("Stopping async reception...")
        sdr.stop_rx_async()
        logger.info("Closing device...")
        sdr.close()
        
        logger.info("Test completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in async test: {e}")
        try:
            # Attempt to clean up if there was an error
            if 'sdr' in locals() and sdr.dev is not None:
                sdr.stop_rx_async()
                sdr.close()
        except:
            pass
        return 1

if __name__ == "__main__":
    sys.exit(main())
