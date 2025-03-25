"""
Simple FM radio receiver using Fobos SDR.
This example shows how to implement FM demodulation with audio output.

Requirements:
- sounddevice (pip install sounddevice)
- numpy
- scipy
- fobos_wrapper.py (from the previous example)
"""

import numpy as np
import scipy.signal as signal
import sounddevice as sd
import threading
import time
import argparse
from queue import Queue
from fobos_wrapper import FobosSDR, FobosException

# Default configuration
DEFAULT_FREQ = 95.5e6  # Default FM station frequency (95.5 MHz)
SAMPLE_RATE = 2.048e6  # SDR sample rate
AUDIO_RATE = 48000     # Audio output sample rate
BUFFER_SIZE = 1024 * 256  # IQ buffer size
GAIN = 12              # Receiver gain setting

# FM demodulation parameters
FM_DEVIATION = 75e3    # FM deviation (75 kHz for broadcast FM)
STEREO_CARRIER = 19e3  # Stereo subcarrier frequency

class FMReceiver:
    def __init__(self, frequency=DEFAULT_FREQ, gain=GAIN, audio_device=None):
        self.frequency = frequency
        self.gain = gain
        self.audio_device = audio_device
        self.sdr = FobosSDR()
        
        # Signal processing state variables
        self.prev_phase = 0
        self.audio_queue = Queue(maxsize=10)
        
        # Control variables
        self.running = False
        self.volume = 0.5  # Initial volume (0.0-1.0)
        
    def stop(self):
        """Stop the FM receiver."""
        self.running = False
        
        # Wait for the audio thread to end
        if hasattr(self, 'audio_thread') and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
            
        # Stop the SDR
        try:
            self.sdr.stop_rx_async()
            self.sdr.close()
            print("FM receiver stopped.")
        except Exception as e:
            print(f"Error while stopping: {e}")
    
    def _process_iq_samples(self, iq_samples):
        """Process IQ samples from the SDR and perform FM demodulation."""
        if not self.running or iq_samples.size == 0:
            return
            
        # FM demodulation
        # 1. Extract phase of the complex IQ samples
        phase = np.angle(iq_samples)
        
        # 2. Compute phase difference (unwrapped)
        # Add 2π to phase differences < -π, subtract 2π from phase differences > π
        diff_phase = np.diff(np.unwrap(np.append(self.prev_phase, phase)))
        self.prev_phase = phase[-1]
        
        # 3. Scale to get instantaneous frequency
        # The frequency deviation is proportional to the audio signal
        audio = diff_phase * (SAMPLE_RATE / (2 * np.pi * FM_DEVIATION))
        
        # 4. Filter and decimate to audio rate
        # Design a low-pass filter with cutoff at 15 kHz (typical audio bandwidth)
        decimation_factor = int(SAMPLE_RATE / AUDIO_RATE)
        audio_filtered = signal.decimate(audio, decimation_factor, ftype='fir')
        
        # 5. Normalize and apply volume
        audio_normalized = audio_filtered / np.max(np.abs(audio_filtered) + 1e-10)
        audio_normalized = audio_normalized * self.volume
        
        # Add to audio queue
        try:
            self.audio_queue.put(audio_normalized, block=False)
        except:
            # If queue is full, just skip this buffer
            pass
            
    def _audio_worker(self):
        """Audio output worker thread."""
        # Initialize audio stream
        stream = sd.OutputStream(
            samplerate=AUDIO_RATE,
            channels=1,
            callback=self._audio_callback,
            blocksize=1024
        )
        
        with stream:
            # Keep the stream active while running
            while self.running:
                time.sleep(0.1)
                
    def _audio_callback(self, outdata, frames, time, status):
        """Audio callback function for sounddevice."""
        if not self.running:
            outdata.fill(0)
            return
            
        if self.audio_queue.empty():
            # No audio data available, output silence
            outdata.fill(0)
        else:
            # Get audio data from the queue
            audio_data = self.audio_queue.get()
            
            # Ensure we have enough samples
            if len(audio_data) < frames:
                # Pad with zeros if not enough samples
                audio_data = np.append(audio_data, np.zeros(frames - len(audio_data)))
            elif len(audio_data) > frames:
                # Truncate if too many samples
                audio_data = audio_data[:frames]
                
            # Fill output buffer
            outdata[:, 0] = audio_data
    
    def start(self):
        try:
            # Find and open the SDR device
            device_count = self.sdr.get_device_count()
            if device_count == 0:
                raise RuntimeError("No Fobos SDR devices found")
            
            self.sdr.open(0)
            
            # Print device info
            info = self.sdr.get_board_info()
            print(f"Connected to {info['product']} (SN: {info['serial']})")
            
            # Configure the SDR
            actual_freq = self.sdr.set_frequency(self.frequency)
            print(f"Tuned to {actual_freq/1e6:.2f} MHz")
            
            actual_rate = self.sdr.set_samplerate(SAMPLE_RATE)
            print(f"Sample rate: {actual_rate/1e6:.3f} MHz")
            
            # Set VGA gain (0-15)
            # Convert dB to VGA value (approximate)
            vga_value = min(15, max(0, int(self.gain / 2)))
            self.sdr.set_vga_gain(vga_value)
            print(f"Gain set to {vga_value} (approximately {vga_value*2} dB)")
            
            # Start the audio output thread
            self.running = True
            self.audio_thread = threading.Thread(target=self._audio_worker)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            # Start the SDR in asynchronous mode
            self.sdr.start_rx_async(self._process_iq_samples, buf_count=16, buf_length=BUFFER_SIZE)
            
            print("FM receiver started. Press Ctrl+C to stop.")
            
            # Simple command interface
            while self.running:
                cmd = input("> ")
                if cmd.lower() in ('q', 'quit', 'exit'):
                    self.stop()
                    break
                elif cmd.startswith('f '):
                    # Change frequency: f 100.1
                    try:
                        new_freq = float(cmd.split()[1]) * 1e6  # MHz to Hz
                        actual_freq = self.sdr.set_frequency(new_freq)
                        print(f"Tuned to {actual_freq/1e6:.2f} MHz")
                    except (ValueError, IndexError) as e:
                        print(f"Invalid frequency command: {e}")
                elif cmd.startswith('v '):
                    # Change volume: v 0.8
                    try:
                        self.volume = min(1.0, max(0.0, float(cmd.split()[1])))
                        print(f"Volume set to {self.volume:.1f}")
                    except (ValueError, IndexError) as e:
                        print(f"Invalid volume command: {e}")
                elif cmd.startswith('g '):
                    # Change gain: g 10
                    try:
                        self.gain = float(cmd.split()[1])
                        vga_value = min(15, max(0, int(self.gain / 2)))
                        self.sdr.set_vga_gain(vga_value)
                        print(f"Gain set to {vga_value} (approximately {vga_value*2} dB)")
                    except (ValueError, IndexError, FobosException) as e:
                        print(f"Invalid gain command: {e}")
                else:
                    print("Commands: f <MHz> (frequency), v <0.0-1.0> (volume), g <dB> (gain), q (quit)")
        except Exception as e:
            print(f"Error in FM receiver: {e}")
            self.stop()

def main():
    """Main entry point for the FM receiver application."""
    parser = argparse.ArgumentParser(description='Fobos SDR FM Receiver')
    parser.add_argument('-f', '--frequency', type=float, default=DEFAULT_FREQ/1e6,
                        help='FM station frequency in MHz (default: 95.5)')
    parser.add_argument('-g', '--gain', type=float, default=GAIN,
                        help='Receiver gain in dB (default: 12)')
    parser.add_argument('-d', '--device', type=str, default=None,
                        help='Audio output device name or ID')
    
    args = parser.parse_args()
    
    # Create and start the FM receiver
    receiver = FMReceiver(
        frequency=args.frequency * 1e6,
        gain=args.gain,
        audio_device=args.device
    )
    
    try:
        receiver.start()
    except KeyboardInterrupt:
        print("\nStopping FM receiver...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        receiver.stop()

if __name__ == "__main__":
    main()
