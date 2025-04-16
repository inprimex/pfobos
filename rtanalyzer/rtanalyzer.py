"""
Enhanced Real-Time Spectrum Analyzer for Fobos SDR
Adds configurable UI controls for SDR parameters
"""

import matplotlib
matplotlib.use('TkAgg')  # Try TkAgg backend for best compatibility

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, RadioButtons
import numpy as np
from scipy import signal
import time
import os
import sys
# from fobos_wrapper import FobosSDR, FobosException

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.fwrapper import FobosSDR, FobosException

class EnhancedRealTimeAnalyzer:
    def __init__(self):
        # Configuration defaults
        self.config = {
            'center_freq': 100e6,     # 100 MHz
            'sample_rate': 8e6,       # 8 MHz
            'fft_size': 1024,
            'gain_vga': 10,           # VGA gain (0-15)
            'gain_lna': 1,            # LNA gain (0-2)
            'buffer_size': 32768,     # Buffer size
            'update_interval': 500,   # Update interval in milliseconds
            'save_interval': 5        # Save a plot every X seconds
        }
        
        # Configuration limits
        self.config_limits = {
            'center_freq': (10e6, 2.5e9),      # 10 MHz to 2.5 GHz
            'sample_rate': (1e6, 20e6),        # 1 MHz to 20 MHz
            'fft_size': (256, 4096),           # Power of 2 sizes
            'gain_vga': (0, 15),               # VGA gain range
            'gain_lna': (0, 2),                # LNA gain range
            'buffer_size': (1024, 65536),      # Buffer size range
            'update_interval': (100, 2000),    # 100ms to 2s
            'save_interval': (1, 30)           # 1 to 30 seconds
        }
        
        # Create plot output directory
        if not os.path.exists("spectrum_plots"):
            os.makedirs("spectrum_plots", exist_ok=True)
        
        # Initialize SDR and variables
        self.sdr = None
        self.timer = None
        self.last_save_time = 0
        
        # Configuration tracking
        self.config_inputs = {}
        self.is_running = False
        
        # Initialize the plot
        self.create_plot()
        
        # Variables for spectrum data
        self.spectrum_data = np.ones(self.config['fft_size']) * -80  # Initialize with baseline
        self.freq_mhz = np.linspace(96, 104, self.config['fft_size'])  # Default freq axis
    
    def create_plot(self):
        """Create the main plot and controls."""
        # Create figure and axis with extra space for configuration
        self.fig = plt.figure(figsize=(16, 9))
        plt.subplots_adjust(bottom=0.15, left=0.25, right=0.95)
        
        # Spectrum subplot
        self.ax = self.fig.add_subplot(111)
        
        # Initial placeholder data
        x = np.linspace(96, 104, self.config['fft_size'])
        y = np.ones(self.config['fft_size']) * -80
        
        # Create the spectrum line
        self.line, = self.ax.plot(x, y, 'b-', linewidth=2)
        
        # Set labels and title
        self.ax.set_xlabel('Frequency (MHz)')
        self.ax.set_ylabel('Power (dB)')
        self.ax.set_title('Fobos SDR Spectrum Analyzer - 100 MHz')
        
        # Set axis limits
        self.ax.set_xlim(96, 104)
        self.ax.set_ylim(-80, 0)
        
        # Add grid
        self.ax.grid(True)
        
        # Create configuration inputs
        config_params = [
            ('center_freq', 'Center Freq (Hz)'), 
            ('sample_rate', 'Sample Rate (Hz)'), 
            ('fft_size', 'FFT Size'), 
            ('gain_vga', 'VGA Gain (0-15)'), 
            ('gain_lna', 'LNA Gain (0-2)'), 
            ('buffer_size', 'Buffer Size'), 
            ('update_interval', 'Update Interval (ms)'), 
            ('save_interval', 'Save Interval (s)')
        ]
        
        # Create configuration panel on the left side
        config_panel_left = 0.05
        config_panel_width = 0.15
        
        # Configuration panel background
        config_panel_ax = self.fig.add_axes([config_panel_left, 0.1, config_panel_width, 0.8])
        config_panel_ax.set_facecolor('#f0f0f0')
        config_panel_ax.set_zorder(-1)
        config_panel_ax.patch.set_alpha(0.5)
        config_panel_ax.axis('off')
        
        # Position for config inputs
        input_y_start = 0.9
        input_height = 0.05
        input_spacing = 0.08
        label_width = 0.12
        input_width = 0.08
        
        # Create text inputs for configuration
        for i, (param, label) in enumerate(config_params):
            # Label
            label_ax = self.fig.add_axes([config_panel_left, input_y_start - i*input_spacing, label_width, input_height])
            label_text = label_ax.text(0.5, 0.5, label, va='center', ha='center')
            label_ax.axis('off')
            
            # Input box
            input_ax = self.fig.add_axes([config_panel_left + label_width, input_y_start - i*input_spacing, input_width, input_height])
            input_box = TextBox(input_ax, '', initial=str(self.config[param]))
            input_box.on_submit(self.create_config_validator(param))
            self.config_inputs[param] = input_box
        
        # Add control buttons
        button_y = 0.02
        button_width = 0.15
        button_spacing = 0.2
        
        # Start button
        self.start_button_ax = self.fig.add_axes([0.1, button_y, button_width, 0.05])
        self.start_button = Button(self.start_button_ax, 'Start')
        self.start_button.on_clicked(self.on_start)
        
        # Stop button
        self.stop_button_ax = self.fig.add_axes([0.1 + button_spacing, button_y, button_width, 0.05])
        self.stop_button = Button(self.stop_button_ax, 'Stop')
        self.stop_button.on_clicked(self.on_stop)
        
        # Save button
        self.save_button_ax = self.fig.add_axes([0.1 + 2*button_spacing, button_y, button_width, 0.05])
        self.save_button = Button(self.save_button_ax, 'Save')
        self.save_button.on_clicked(self.on_save)
    
    def create_config_validator(self, param):
        """Create a configuration validator for a specific parameter."""
        def validate_config(value_str):
            try:
                # Convert to appropriate type
                if param in ['fft_size', 'buffer_size', 'update_interval', 'save_interval', 'gain_vga', 'gain_lna']:
                    value = int(float(value_str))
                else:
                    value = float(value_str)
                
                # Check against limits
                low, high = self.config_limits[param]
                if value < low or value > high:
                    raise ValueError(f"Value must be between {low} and {high}")
                
                # Special case for power of 2 for FFT size
                if param == 'fft_size':
                    if not (value & (value-1) == 0):  # Check if power of 2
                        raise ValueError("FFT size must be a power of 2")
                
                # Update config if validation passes
                self.config[param] = value
                
                # Provide visual feedback
                self.config_inputs[param].text_disp.set_color('black')
                
                # If running, attempt to update live
                if self.is_running:
                    self.update_sdr_config(param)
                
            except ValueError as e:
                # Visual feedback for invalid input
                self.config_inputs[param].text_disp.set_color('red')
                print(f"Invalid {param}: {e}")
        
        return validate_config
    
    def update_sdr_config(self, param):
        """Update SDR configuration for a specific parameter while running."""
        if self.sdr is None:
            return
        
        try:
            if param == 'center_freq':
                actual_freq = self.sdr.set_frequency(self.config[param])
                print(f"Updated frequency to {actual_freq/1e6:.3f} MHz")
            
            elif param == 'sample_rate':
                actual_rate = self.sdr.set_samplerate(self.config[param])
                print(f"Updated sample rate to {actual_rate/1e6:.3f} MHz")
                
                # Update frequency axis
                self.freq_mhz = np.linspace(
                    (actual_freq - actual_rate/2) / 1e6,
                    (actual_freq + actual_rate/2) / 1e6,
                    self.config['fft_size']
                )
                
                # Update plot axis
                self.ax.set_xlim(np.min(self.freq_mhz), np.max(self.freq_mhz))
            
            elif param == 'gain_vga':
                self.sdr.set_vga_gain(self.config[param])
                print(f"Updated VGA gain to {self.config[param]}")
            
            elif param == 'gain_lna':
                self.sdr.set_lna_gain(self.config[param])
                print(f"Updated LNA gain to {self.config[param]}")
            
            # For other parameters, restart is needed
            elif param in ['fft_size', 'buffer_size', 'update_interval', 'save_interval']:
                print(f"{param} change requires restart")
        
        except Exception as e:
            print(f"Error updating {param}: {e}")
    
    def on_start(self, event):
        """Start button handler."""
        print("Starting SDR and updates...")
        
        # Initialize SDR
        try:
            # Prevent multiple starts
            if self.is_running:
                print("Already running")
                return
            
            # Create SDR object if needed
            if self.sdr is None:
                self.sdr = FobosSDR()
            
            # Check for devices
            device_count = self.sdr.get_device_count()
            print(f"Found {device_count} devices")
            
            if device_count > 0:
                # Open first device
                print("Opening SDR device...")
                self.sdr.open(0)
                
                # Get board info
                info = self.sdr.get_board_info()
                print(f"Connected to {info['product']} (SN: {info['serial']})")
                
                # Set frequency
                print(f"Setting frequency to {self.config['center_freq']/1e6} MHz")
                actual_freq = self.sdr.set_frequency(self.config['center_freq'])
                print(f"Actual frequency: {actual_freq/1e6} MHz")
                
                # Set sample rate
                print(f"Setting sample rate to {self.config['sample_rate']/1e6} MHz")
                actual_rate = self.sdr.set_samplerate(self.config['sample_rate'])
                print(f"Actual sample rate: {actual_rate/1e6} MHz")
                
                # Set gains
                print(f"Setting VGA gain to {self.config['gain_vga']}")
                self.sdr.set_vga_gain(self.config['gain_vga'])
                
                print(f"Setting LNA gain to {self.config['gain_lna']}")
                self.sdr.set_lna_gain(self.config['gain_lna'])
                
                # Update frequency axis with actual values
                self.freq_mhz = np.linspace(
                    (actual_freq - actual_rate/2) / 1e6,
                    (actual_freq + actual_rate/2) / 1e6,
                    self.config['fft_size']
                )
                
                # Update title with actual frequency
                self.ax.set_title(f'Fobos SDR Spectrum Analyzer - {actual_freq/1e6:.1f} MHz')
                
                # Update axis limits
                self.ax.set_xlim(np.min(self.freq_mhz), np.max(self.freq_mhz))
                
                # Start SDR in synchronous mode
                print("Starting synchronous reception...")
                self.sdr.start_rx_sync(self.config['buffer_size'])
                
                # Mark as running
                self.is_running = True
                
                # Start update timer
                self.start_timer()
            else:
                print("No SDR devices found")
        
        except Exception as e:
            print(f"Error initializing SDR: {e}")
            import traceback
            traceback.print_exc()
    
    def on_stop(self, event):
        """Stop button handler."""
        print("Stopping updates and SDR...")
        self.stop_timer()
        self.cleanup_sdr()
    
    def on_save(self, event):
        """Save button handler."""
        print("Saving current spectrum...")
        self.save_plot()
    
    def start_timer(self):
        """Start the update timer."""
        # Stop existing timer if any
        self.stop_timer()
        
        # Create new timer
        self.timer = self.fig.canvas.new_timer(interval=self.config['update_interval'])
        self.timer.add_callback(self.update_plot)
        self.timer.start()
        print(f"Update timer started ({self.config['update_interval']}ms interval)")
    
    def stop_timer(self):
        """Stop the update timer."""
        if self.timer is not None:
            self.timer.stop()
            print("Update timer stopped")
            
        # Reset running flag
        self.is_running = False
    
    def update_plot(self):
        """Update the plot with new spectrum data."""
        if self.sdr is None:
            return
        
        try:
            # Get samples from SDR
            iq_samples = self.sdr.read_rx_sync()
            
            if len(iq_samples) < self.config['fft_size']:
                print(f"Warning: Not enough samples ({len(iq_samples)})")
                return
            
            # Calculate spectrum
            # Use a window function
            windowed = iq_samples[:self.config['fft_size']] * signal.windows.hann(self.config['fft_size'])
            
            # Calculate FFT
            fft_result = np.fft.fftshift(np.fft.fft(windowed, self.config['fft_size']))
            
            # Convert to power in dB
            power_db = 20 * np.log10(np.abs(fft_result) + 1e-10)
            
            # Apply some smoothing
            alpha = 0.7
            self.spectrum_data = alpha * power_db + (1 - alpha) * self.spectrum_data
            
            # Update the plot
            self.line.set_data(self.freq_mhz, self.spectrum_data)
            
            # Refresh the canvas
            self.fig.canvas.draw_idle()
            
            # Print stats occasionally
            current_time = time.time()
            if current_time - self.last_save_time > self.config['save_interval']:
                self.last_save_time = current_time
                print(f"Spectrum range: {np.min(self.spectrum_data):.1f} to {np.max(self.spectrum_data):.1f} dB")
        
        except Exception as e:
            print(f"Error updating plot: {e}")
            import traceback
            traceback.print_exc()
    
    def save_plot(self):
        """Save the current spectrum to a file."""
        try:
            # Generate filename with timestamp
            timestamp = int(time.time())
            filename = f"spectrum_plots/spectrum_{timestamp}.png"
            
            # Update title with timestamp
            current_time = time.strftime("%H:%M:%S")
            self.ax.set_title(f'Fobos SDR Spectrum - {self.config["center_freq"]/1e6:.1f} MHz - {current_time}')
            
            # Make sure the canvas is fully drawn
            self.fig.canvas.draw()
            
            # Save the figure
            self.fig.savefig(filename, dpi=100)
            print(f"Saved spectrum to {filename}")
        
        except Exception as e:
            print(f"Error saving plot: {e}")
    
    def cleanup_sdr(self):
        """Clean up SDR resources."""
        if self.sdr is not None:
            try:
                self.sdr.stop_rx_sync()
                self.sdr.close()
                print("SDR stopped and closed")
            except Exception as e:
                print(f"Error during SDR cleanup: {e}")
            
            self.sdr = None
            
            # Reset running flag
            self.is_running = False
    
    def run(self):
        """Run the main application."""
        try:
            # Show the plot window (blocks until closed)
            print("Showing plot window - close window to exit")
            plt.show()
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean up resources
            self.stop_timer()
            self.cleanup_sdr()
            print("Application terminated")

if __name__ == "__main__":
    print(f"Starting Enhanced Real-Time Spectrum Analyzer...")
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"Matplotlib backend: {matplotlib.get_backend()}")
    
    analyzer = EnhancedRealTimeAnalyzer()
    analyzer.run()