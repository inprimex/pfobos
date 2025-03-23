"""
Simple spectrum analyzer using Fobos SDR.
This example shows how to use the Fobos SDR Python wrapper for spectral analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
from fobos_wrapper import FobosSDR, FobosException

# Configuration
CENTER_FREQ = 100e6  # 100 MHz
SAMPLE_RATE = 2.048e6  # 2.048 MHz
FFT_SIZE = 1024
GAIN = 10  # dB

class SpectrumAnalyzer:
    def __init__(self):
        self.sdr = FobosSDR()
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line, = self.ax.plot([], [])
        self.spectrum_data = np.zeros(FFT_SIZE)
        self.freq_axis = np.zeros(FFT_SIZE)
        self.running = False

    def setup(self):
        # Configure plot
        self.ax.set_xlim(CENTER_FREQ - SAMPLE_RATE/2, CENTER_FREQ + SAMPLE_RATE/2)
        self.ax.set_ylim(-120, 0)
        self.ax.set_xlabel('Frequency (MHz)')
        self.ax.set_ylabel('Power (dB)')
        self.ax.set_title(f'Fobos SDR Spectrum Analyzer - {CENTER_FREQ/1e6:.3f} MHz')
        self.ax.grid(True)

        # Calculate frequency axis
        self.freq_axis = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1/SAMPLE_RATE)) + CENTER_FREQ
        self.line.set_data(self.freq_axis/1e6, self.spectrum_data)

        return self.line,

    def start(self):
        try:
            # Check for devices
            device_count = self.sdr.get_device_count()
            if device_count == 0:
                raise RuntimeError("No Fobos SDR devices found")

            # Open the first device
            self.sdr.open(0)
            
            # Print device info
            info = self.sdr.get_board_info()
            print(f"Connected to {info['product']} (SN: {info['serial']})")
            
            # Configure SDR
            actual_freq = self.sdr.set_frequency(CENTER_FREQ)
            print(f"Center frequency: {actual_freq/1e6:.3f} MHz")
            
            actual_rate = self.sdr.set_samplerate(SAMPLE_RATE)
            print(f"Sample rate: {actual_rate/1e6:.3f} MHz")
            
            # Set VGA gain (0-15)
            # Convert dB to VGA value (approximate)
            vga_value = min(15, max(0, int(GAIN / 2)))
            self.sdr.set_vga_gain(vga_value)
            print(f"VGA gain set to {vga_value} (approximately {vga_value*2} dB)")
            
            # Start animation
            self.ani = FuncAnimation(
                self.fig, self.update, init_func=self.setup,
                frames=None, interval=100, blit=True
            )
            
            # Start SDR in asynchronous mode
            self.running = True
            self.sdr.start_rx_async(self.process_samples)
            
            plt.show()
            
        except Exception as e:
            print(f"Error: {e}")
            self.stop()

    def process_samples(self, iq_samples):
        if not self.running:
            return
            
        # Apply window function to reduce spectral leakage
        windowed = iq_samples[:FFT_SIZE] * signal.windows.hann(min(len(iq_samples), FFT_SIZE))
        
        # Calculate FFT and shift it
        fft = np.fft.fftshift(np.fft.fft(windowed, FFT_SIZE))
        
        # Convert to power in dB
        power_db = 20 * np.log10(np.abs(fft) + 1e-10)
        
        # Apply some smoothing using exponential moving average
        alpha = 0.5
        self.spectrum_data = alpha * power_db + (1 - alpha) * self.spectrum_data

    def update(self, frame):
        # Update the plot data
        self.line.set_data(self.freq_axis/1e6, self.spectrum_data)
        return self.line,

    def stop(self):
        self.running = False
        if hasattr(self, 'sdr'):
            try:
                self.sdr.stop_rx_async()
                self.sdr.close()
            except Exception as e:
                print(f"Error while closing: {e}")

if __name__ == "__main__":
    analyzer = SpectrumAnalyzer()
    try:
        analyzer.start()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        analyzer.stop()