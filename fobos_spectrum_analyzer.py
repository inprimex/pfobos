"""
Simple spectrum analyzer using Fobos SDR.
This example shows how to use the Fobos SDR Python wrapper for spectral analysis.
"""

# Set matplotlib backend before other imports
import matplotlib
matplotlib.use('TkAgg')  # Try TkAgg backend which typically works well

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
import sys
import os
import time
import signal as system_signal
from fobos_wrapper import FobosSDR, FobosException

# Configuration
CENTER_FREQ = 100e6  # 100 MHz
SAMPLE_RATE = 2.048e6  # 2.048 MHz
FFT_SIZE = 1024
GAIN = 10  # dB
SAVE_PLOTS = True  # Set to True to save plots to files
SAVE_DIRECTORY = "spectrum_plots"  # Directory to save plots
BUFFER_SIZE = 32768  # Increased buffer size for stability

class SpectrumAnalyzer:
    def __init__(self):
        self.sdr = FobosSDR()
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line, = self.ax.plot([], [])
        self.spectrum_data = np.zeros(FFT_SIZE)
        self.freq_axis = np.zeros(FFT_SIZE)
        self.running = False
        self.last_save_time = 0
        self.last_print_time = 0
        
        # Create directory for saving plots if needed
        if SAVE_PLOTS and not os.path.exists(SAVE_DIRECTORY):
            os.makedirs(SAVE_DIRECTORY, exist_ok=True)
        
        # Setup signal handlers for clean shutdown
        system_signal.signal(system_signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        """Handle Ctrl+C and other signals for clean shutdown."""
        print("\nShutting down gracefully...")
        self.stop()
        sys.exit(0)

    def setup(self):
        """Initialize the plot."""
        print("Setting up plot...")
        
        # Configure plot
        self.ax.set_xlim(CENTER_FREQ/1e6 - SAMPLE_RATE/2e6, CENTER_FREQ/1e6 + SAMPLE_RATE/2e6)
        self.ax.set_ylim(-120, 0)
        self.ax.set_xlabel('Frequency (MHz)')
        self.ax.set_ylabel('Power (dB)')
        self.ax.set_title(f'Fobos SDR Spectrum Analyzer - {CENTER_FREQ/1e6:.3f} MHz')
        self.ax.grid(True)

        # Calculate frequency axis
        self.freq_axis = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1/SAMPLE_RATE)) + CENTER_FREQ
        self.line.set_data(self.freq_axis/1e6, self.spectrum_data)

        print("Plot setup complete")
        return self.line,

    def start(self):
        """Start the spectrum analyzer."""
        try:
            # Check for devices
            print("Checking for Fobos SDR devices...")
            device_count = self.sdr.get_device_count()
            if device_count == 0:
                raise RuntimeError("No Fobos SDR devices found")
            print(f"Found {device_count} device(s)")

            # Open the first device
            print("Opening SDR device...")
            self.sdr.open(0)
            
            # Print device info
            info = self.sdr.get_board_info()
            print(f"Connected to {info['product']} (SN: {info['serial']})")
            
            # Get available sample rates
            rates = self.sdr.get_samplerates()
            print(f"Available sample rates: {[r/1e6 for r in rates]} MHz")
            
            # Configure SDR
            print(f"Setting frequency to {CENTER_FREQ/1e6} MHz...")
            actual_freq = self.sdr.set_frequency(CENTER_FREQ)
            print(f"Center frequency: {actual_freq/1e6:.3f} MHz")
            
            print(f"Setting sample rate to {SAMPLE_RATE/1e6} MHz...")
            actual_rate = self.sdr.set_samplerate(SAMPLE_RATE)
            print(f"Sample rate: {actual_rate/1e6:.3f} MHz")
            
            # Set VGA gain (0-15)
            # Convert dB to VGA value (approximate)
            vga_value = min(15, max(0, int(GAIN / 2)))
            print(f"Setting VGA gain to {vga_value}...")
            self.sdr.set_vga_gain(vga_value)
            print(f"VGA gain set to {vga_value} (approximately {vga_value*2} dB)")
            
            # Set some LNA gain as well
            lna_value = 1  # Medium LNA gain
            print(f"Setting LNA gain to {lna_value}...")
            self.sdr.set_lna_gain(lna_value)
            
            # Start SDR in asynchronous mode with larger buffer
            print(f"Starting SDR async reception with buffer size {BUFFER_SIZE}...")
            self.running = True
            self.sdr.start_rx_async(self.process_samples, buf_count=16, buf_length=BUFFER_SIZE)
            
            # Print instructions
            print("Spectrum analyzer running. Press Ctrl+C to exit.")
            
            # Start animation with explicit save_count to avoid warning
            print("Starting display animation...")
            self.ani = FuncAnimation(
                self.fig, self.update, init_func=self.setup,
                frames=None, interval=100, blit=True,
                cache_frame_data=False  # Disable frame caching to avoid memory issues
            )
            
            # If using X11, try forcing active plot window
            plt.figure(self.fig.number)  # Activate the figure
            plt.draw()
            
            # Use a try-except block for plt.show() to catch exceptions
            try:
                print("Displaying plot window...")
                plt.show(block=True)
            except KeyboardInterrupt:
                print("\nReceived keyboard interrupt...")
                self.stop()
            
        except Exception as e:
            print(f"Error: {e}")
            self.stop()

    def process_samples(self, iq_samples):
        """Process IQ samples and calculate spectrum."""
        if not self.running:
            return
        
        try:
            # Check if we received data
            if len(iq_samples) == 0:
                print("Warning: Received empty IQ sample buffer")
                return
                
            # Print data stats occasionally
            current_time = time.time()
            if current_time - self.last_print_time > 5:
                self.last_print_time = current_time
                print(f"Received buffer with {len(iq_samples)} IQ samples")
                
                # Print simple ASCII spectrum occasionally
                max_val = np.max(self.spectrum_data)
                min_val = np.min(self.spectrum_data)
                print(f"Spectrum data range: {min_val:.1f} to {max_val:.1f} dB")
                
                # Print simple ASCII spectrum (5 lines)
                width = 50
                lines = 5
                step = len(self.spectrum_data) // lines
                for i in range(lines):
                    idx = i * step
                    val = self.spectrum_data[idx]
                    normalized = (val - min_val) / (max_val - min_val) if max_val > min_val else 0
                    bars = int(normalized * width)
                    print(f"{self.freq_axis[idx]/1e6:7.2f} MHz: {'█' * bars}")
                
            # Use only first FFT_SIZE samples from the buffer
            # or pad with zeros if not enough samples
            if len(iq_samples) < FFT_SIZE:
                padded = np.pad(iq_samples, (0, FFT_SIZE - len(iq_samples)), 'constant')
                samples_to_process = padded
            else:
                samples_to_process = iq_samples[:FFT_SIZE]
                
            # Apply window function to reduce spectral leakage
            windowed = samples_to_process * signal.windows.hann(FFT_SIZE)
            
            # Calculate FFT and shift it
            fft = np.fft.fftshift(np.fft.fft(windowed, FFT_SIZE))
            
            # Convert to power in dB
            power_db = 20 * np.log10(np.abs(fft) + 1e-10)
            
            # Apply some smoothing using exponential moving average
            alpha = 0.5
            self.spectrum_data = alpha * power_db + (1 - alpha) * self.spectrum_data
            
            # Save plot to file periodically if enabled
            if SAVE_PLOTS and (current_time - self.last_save_time > 2):
                self.last_save_time = current_time
                self.save_plot()
        
        except Exception as e:
            if self.running:
                print(f"Error processing samples: {e}")

    def update(self, frame):
        """Update the plot with new data."""
        try:
            # Update the plot data
            self.line.set_data(self.freq_axis/1e6, self.spectrum_data)
            return self.line,
        except Exception as e:
            print(f"Error updating plot: {e}")
            return self.line,
            
    def save_plot(self):
        """Save the current spectrum plot to a file."""
        if not SAVE_PLOTS:
            return
            
        try:
            timestamp = int(time.time())
            filename = f"{SAVE_DIRECTORY}/spectrum_{timestamp}.png"
            
            # Update the title with timestamp
            self.ax.set_title(f'Fobos SDR Spectrum - {CENTER_FREQ/1e6:.3f} MHz - {time.strftime("%H:%M:%S")}')
            
            # Save the figure
            self.fig.savefig(filename)
            print(f"Saved spectrum to {filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")

    def stop(self):
        """Stop the spectrum analyzer safely."""
        # Only run this once
        if not self.running:
            return
            
        self.running = False
        print("Stopping spectrum analyzer...")
        
        # Stop the animation if it's running
        if hasattr(self, 'ani'):
            try:
                self.ani.event_source.stop()
                print("Animation stopped.")
            except Exception as e:
                print(f"Error stopping animation: {e}")
        
        # Close the SDR properly
        if hasattr(self, 'sdr'):
            try:
                print("Stopping SDR async reception...")
                self.sdr.stop_rx_async()
                print("Closing SDR device...")
                self.sdr.close()
                print("SDR stopped and closed.")
            except Exception as e:
                print(f"Error while closing SDR: {e}")
        
        # Save final plot if enabled
        if SAVE_PLOTS:
            self.save_plot()
            
        print("Spectrum analyzer shutdown complete.")

if __name__ == "__main__":
    # Print environment info
    print(f"Python version: {sys.version}")
    print(f"Matplotlib backend: {matplotlib.get_backend()}")
    print(f"Display environment: {os.environ.get('DISPLAY', 'Not set')}")
    
    analyzer = SpectrumAnalyzer()
    try:
        analyzer.start()
    except KeyboardInterrupt:
        print("\nShutting down from main...")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        analyzer.stop()
        plt.close('all')  # Make sure all matplotlib windows are closed
        print("Program terminated.")
