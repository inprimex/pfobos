#!/usr/bin/env python3
"""
Minimal debug script for Fobos SDR to isolate and fix segmentation fault issues.
"""

import sys
import os
import time
import logging
import traceback
import numpy as np
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the wrapper - adjust path as needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fobos_wrapper import FobosSDR, FobosException

def test_device_info():
    """Test basic device info to verify connectivity."""
    try:
        with FobosSDR() as sdr:
            sdr.open(0)  # Open first device
            
            # Get API info
            api_info = sdr.get_api_info()
            logger.info(f"API info: {api_info}")
            
            # Get board info
            board_info = sdr.get_board_info()
            logger.info(f"Board info: {board_info}")
            
            return True
    except Exception as e:
        logger.error(f"Error in device info test: {e}")
        logger.error(traceback.format_exc())
        return False

def safe_rx_test():
    """Test synchronous reception with minimal parameters and safer code."""
    logger.info("Starting safe RX test")
    
    try:
        # Create a new SDR instance for each test
        with FobosSDR() as sdr:
            sdr.open(0)  # Open first device
            logger.info("Device opened")
            
            # Configure SDR with safe parameters
            actual_freq = sdr.set_frequency(100e6)  # 100 MHz
            logger.info(f"Set frequency to {actual_freq/1e6:.3f} MHz")
            
            # Get available sample rates
            rates = sdr.get_samplerates()
            logger.info(f"Available sample rates: {[r/1e6 for r in rates]} MHz")
            
            # Use a larger sample rate (usually safer)
            sample_rate = 10e6 if 10e6 in rates else rates[0]
            actual_rate = sdr.set_samplerate(sample_rate)
            logger.info(f"Set sample rate to {actual_rate/1e6:.3f} MHz")
            
            # Set moderate gains
            sdr.set_lna_gain(1)
            logger.info("Set LNA gain to 1")
            
            sdr.set_vga_gain(5)
            logger.info("Set VGA gain to 5")
            
            # Use a large buffer
            buffer_size = 32768
            logger.info(f"Starting synchronous reception with buffer size {buffer_size}")
            
            # Starting sync reception
            sdr.start_rx_sync(buffer_size)
            logger.info("Sync mode started successfully")
            
            # Introduce a small delay to allow device to stabilize
            time.sleep(0.1)
            
            try:
                # Try to read samples
                logger.info("Attempting to read samples...")
                iq_data = sdr.read_rx_sync()
                
                # Process and log data
                logger.info(f"Successfully received {len(iq_data)} IQ samples")
                if len(iq_data) > 0:
                    logger.info(f"Sample mean magnitude: {np.mean(np.abs(iq_data)):.4f}")
                    logger.info(f"Sample std: {np.std(np.abs(iq_data)):.4f}")
                    logger.info(f"First 5 samples: {iq_data[:5]}")
            finally:
                # Always stop sync mode to avoid device hanging
                logger.info("Stopping synchronous reception...")
                sdr.stop_rx_sync()
                logger.info("Sync mode stopped")
            
            # Force garbage collection
            gc.collect()
            
            return True
    except Exception as e:
        logger.error(f"Error in safe RX test: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to run minimal tests."""
    
    # First verify device connectivity
    if not test_device_info():
        logger.error("Failed to connect to device. Aborting tests.")
        return
    
    # Try the safe RX test
    if safe_rx_test():
        logger.info("Safe RX test completed successfully!")
    else:
        logger.error("Safe RX test failed.")

if __name__ == "__main__":
    main()