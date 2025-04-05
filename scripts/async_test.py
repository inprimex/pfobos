#!/usr/bin/env python3
# Simple test script for Fobos SDR asynchronous reception

import sys
import os
import time
import logging
import numpy as np
import signal
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the wrapper - adjust path as needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fobos_wrapper import FobosSDR, FobosException

# Global variables
total_samples = 0
buffer_count = 0
run_test = True

def signal_handler(sig, frame):
    global run_test
    logger.info("Stopping test on signal...")
    run_test = False

def async_callback(samples):
    global total_samples, buffer_count
    
    buffer_count += 1
    total_samples += len(samples)
    
    # Log some information periodically
    if buffer_count % 10 == 0:
        logger.info(f"Received buffer #{buffer_count} with {len(samples)} samples")
        logger.info(f"Total samples: {total_samples}")
        if len(samples) > 0:
            logger.info(f"Sample mean magnitude: {np.mean(np.abs(samples)):.6f}")

def main():
    global total_samples, buffer_count, run_test
    
    # Reset counters
    total_samples = 0
    buffer_count = 0
    run_test = True
    
    # Set up signal handler for cleaner termination
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting async test (will run for up to 10 seconds)")
    
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
        sample_rate = 10e6 if 10e6 in rates else rates[-1]
        actual_rate = sdr.set_samplerate(sample_rate)
        logger.info(f"Set sample rate to {actual_rate/1e6:.3f} MHz")
        
        # Set gain settings
        sdr.set_lna_gain(1)
        sdr.set_vga_gain(5)
        
        # Start async reception
        logger.info("Starting async reception")
        sdr.start_rx_async(async_callback, 16, 32768)
        
        # Run for 10 seconds or until interrupted
        test_duration = 10
        logger.info(f"Test will run for up to {test_duration} seconds...")
        start_time = time.time()
        
        # Main loop with timeout
        while run_test and (time.time() - start_time < test_duration):
            time.sleep(0.1)
        
        # Calculate statistics
        elapsed = time.time() - start_time
        sample_rate_achieved = total_samples / elapsed if elapsed > 0 else 0
        
        logger.info(f"Test completed after {elapsed:.2f} seconds")
        logger.info(f"Received {total_samples} samples in {buffer_count} buffers")
        logger.info(f"Achieved sample rate: {sample_rate_achieved/1e6:.2f} MSps")
        
    except Exception as e:
        logger.error(f"Error in async test: {e}")
        return 1
    finally:
        # Always clean up regardless of how we exit
        logger.info("Cleaning up...")
        try:
            if 'sdr' in locals() and hasattr(sdr, 'dev') and sdr.dev is not None:
                logger.info("Stopping async reception...")
                sdr.stop_rx_async()
                logger.info("Closing device...")
                sdr.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    logger.info("Test finished")
    return 0

if __name__ == "__main__":
    sys.exit(main())
