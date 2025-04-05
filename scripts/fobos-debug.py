#!/usr/bin/env python3
"""
Simplified debug script for Fobos SDR to isolate the segmentation fault issue.
"""

import sys
import os
import time
import logging
import traceback
import numpy as np

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

def test_rx_sync_with_params(buffer_size, sample_rate, frequency, lna_gain, vga_gain):
    """Test synchronous reception with specific parameters."""
    logger.info(f"Testing RX sync with: buffer={buffer_size}, rate={sample_rate/1e6}MHz, "
                f"freq={frequency/1e6}MHz, LNA={lna_gain}, VGA={vga_gain}")
    
    try:
        with FobosSDR() as sdr:
            sdr.open(0)  # Open first device
            
            # Configure SDR
            actual_freq = sdr.set_frequency(frequency)
            logger.info(f"Set frequency to {actual_freq/1e6:.3f} MHz")
            
            actual_rate = sdr.set_samplerate(sample_rate)
            logger.info(f"Set sample rate to {actual_rate/1e6:.3f} MHz")
            
            sdr.set_lna_gain(lna_gain)
            logger.info(f"Set LNA gain to {lna_gain}")
            
            sdr.set_vga_gain(vga_gain)
            logger.info(f"Set VGA gain to {vga_gain}")
            
            # Start synchronous receiving - this is where it might crash
            logger.info(f"Starting synchronous reception...")
            sdr.start_rx_sync(buffer_size)
            
            # Read samples
            logger.info(f"Reading samples...")
            iq_data = sdr.read_rx_sync()
            
            # Process and log data
            logger.info(f"Successfully received {len(iq_data)} IQ samples")
            if len(iq_data) > 0:
                logger.info(f"Sample mean: {np.mean(np.abs(iq_data)):.4f}")
                logger.info(f"Sample std: {np.std(np.abs(iq_data)):.4f}")
            
            # Stop synchronous receiving
            logger.info(f"Stopping synchronous reception...")
            sdr.stop_rx_sync()
            
            return True
    except Exception as e:
        logger.error(f"Error in RX sync test: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to run tests with different parameters."""
    
    # First verify device connectivity
    if not test_device_info():
        logger.error("Failed to connect to device. Aborting tests.")
        return
    
    # Test different buffer sizes
    buffer_sizes = [256, 512, 1024, 2048, 4096]
    
    # Test different sample rates (in Hz)
    sample_rates = [2.048e6, 5e6, 8e6, 10e6]
    
    # Fixed parameters for testing
    frequency = 100e6  # 100 MHz
    lna_gain = 1
    vga_gain = 10
    
    # Test with different buffer sizes
    logger.info("=== Testing different buffer sizes ===")
    for buffer_size in buffer_sizes:
        success = test_rx_sync_with_params(
            buffer_size=buffer_size,
            sample_rate=2.048e6,  # Fixed sample rate
            frequency=frequency,
            lna_gain=lna_gain,
            vga_gain=vga_gain
        )
        
        if not success:
            logger.warning(f"Test failed with buffer size {buffer_size}")
        
        # Short delay between tests
        time.sleep(1)
    
    # Test with different sample rates
    logger.info("=== Testing different sample rates ===")
    for sample_rate in sample_rates:
        success = test_rx_sync_with_params(
            buffer_size=512,  # Fixed buffer size that hopefully works
            sample_rate=sample_rate,
            frequency=frequency,
            lna_gain=lna_gain,
            vga_gain=vga_gain
        )
        
        if not success:
            logger.warning(f"Test failed with sample rate {sample_rate}")
        
        # Short delay between tests
        time.sleep(1)

if __name__ == "__main__":
    main()
