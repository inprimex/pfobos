"""
Simple FM radio receiver using Fobos SDR with PulseAudio support.
This example shows how to implement FM demodulation with audio output.

Requirements:
- sounddevice (pip install sounddevice)
- numpy
- scipy
- pulsectl (pip install pulsectl)
- fobos_wrapper.py (updated version)
"""

import numpy as np
from scipy import signal
import sounddevice as sd
import threading
import time
import argparse
import sys
import os
import signal as sig
from queue import Queue
import subprocess
from fobos_wrapper import FobosSDR, FobosException

# Try to import pulsectl (optional)
try:
    import pulsectl
    PULSECTL_AVAILABLE = True
except ImportError:
    PULSECTL_AVAILABLE = False
    print("pulsectl not available. Install with: pip install pulsectl")
    print("Some PulseAudio features will be limited.")

# Default configuration
DEFAULT_FREQ = 95.5e6  # Default FM station frequency (95.5 MHz)
SAMPLE_RATE = 2.048e6  # SDR sample rate
AUDIO_RATE = 48000     # Audio output sample rate
BUFFER_SIZE = 32768    # IQ buffer size (increased for stability)
GAIN = 12              # Receiver gain setting

# FM demodulation parameters
FM_DEVIATION = 75e3    # FM deviation (75 kHz for broadcast FM)
STEREO_CARRIER = 19e3  # Stereo subcarrier frequency

# Flag for controlling termination
terminate_flag = False

def list_pulse_sinks():
    """List all PulseAudio sinks using pulsectl."""
    if not PULSECTL_AVAILABLE:
        print("pulsectl not available. Cannot list PulseAudio sinks.")
        return []
        
    try:
        with pulsectl.Pulse('fm-receiver') as pulse:
            print("\nPULSEAUDIO SINKS:")
            print("----------------")
            sinks = pulse.sink_list()
            for i, sink in enumerate(sinks):
                print(f"  {i}: {sink.name} - {sink.description}")
            return sinks
    except Exception as e:
        print(f"Error listing PulseAudio sinks: {e}")
        return []

def list_all_audio_devices():
    """List all audio devices in the system."""
    try:
        devices = sd.query_devices()
        print("\nSOUNDDEVICE DEVICES:")
        print("------------------")
        for i, device in enumerate(devices):
            device_type = []
            if device['max_output_channels'] > 0:
                device_type.append("OUTPUT")
            if device['max_input_channels'] > 0:
                device_type.append("INPUT")
            
            type_str = ", ".join(device_type)
            print(f"  {i}: {device['name']} ({type_str}) - {device['hostapi']}")
            
        return devices
    except Exception as e:
        print(f"Error listing sounddevice devices: {e}")
        return []

def select_audio_device():
    """Let the user select an audio device interactively."""
    # List sounddevice devices
    sd_devices = list_all_audio_devices()
    
    # List PulseAudio sinks
    pulse_sinks = list_pulse_sinks()
    
    # Check if xrdp-sink exists in PulseAudio
    xrdp_sink_index = None
    if PULSECTL_AVAILABLE:
        for i, sink in enumerate(pulse_sinks):
            if 'xrdp' in sink.name.lower():
                print(f"\nFound xrdp sink: {sink.name}")
                confirm = input(f"Use this xrdp-sink? (Y/n): ").strip().lower()
                if confirm in ('', 'y', 'yes'):
                    xrdp_sink_index = i
                    return {'type': 'pulse', 'index': i, 'name': sink.name}
    
    # Let the user choose between PulseAudio and sounddevice
    print("\nSelect audio output method:")
    print("1. PulseAudio sink")
    print("2. Sounddevice device")
    print("3. No audio output")
    
    while True:
        try:
            method = input("Select option (1-3): ").strip()
            
            if not method or method == '3':
                print("Audio output disabled.")
                return None
                
            if method == '1':
                # PulseAudio sink
                if not PULSECTL_AVAILABLE:
                    print("pulsectl not available. Please install it or select another option.")
                    continue
                    
                if not pulse_sinks:
                    print("No PulseAudio sinks found. Please select another option.")
                    continue
                    
                print("Select PulseAudio sink by number:")
                sink_idx = int(input(f"Enter sink number (0-{len(pulse_sinks)-1}): ").strip())
                
                if 0 <= sink_idx < len(pulse_sinks):
                    return {'type': 'pulse', 'index': sink_idx, 'name': pulse_sinks[sink_idx].name}
                else:
                    print(f"Invalid sink number. Please select 0-{len(pulse_sinks)-1}")
                    
            elif method == '2':
                # Sounddevice device
                if not sd_devices:
                    print("No sounddevice devices found. Please select another option.")
                    continue
                    
                print("Select sounddevice device by number:")
                device_idx = int(input(f"Enter device number (0-{len(sd_devices)-1}): ").strip())
                
                if 0 <= device_idx < len(sd_devices):
                    return {'type': 'sounddevice', 'index': device_idx, 'name': sd_devices[device_idx]['name']}
                else:
                    print(f"Invalid device number. Please select 0-{len(sd_devices)-1}")
            else:
                print("Invalid option. Please select 1-3.")
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"Error: {e}")

class FMReceiver:
    def __init__(self, frequency=DEFAULT_FREQ, gain=GAIN, audio_device=None, 
                 buffer_size=BUFFER_SIZE, no_audio=False, use_xrdp=False):
        self.frequency = frequency
        self.gain = gain
        self.audio_device = audio_device
        self.buffer_size = buffer_size
        self.no_audio = no_audio
        self.use_xrdp = use_xrdp
        self.sdr = None  # Initialize later
        
        # Signal processing state variables
        self.prev_phase = 0
        self.audio_queue = Queue(maxsize=10)
        
        # Control variables
        self.running = False
        self.volume = 0.5  # Initial volume (0.0-1.0)
        self.stream = None
        self.pulse = None
        
    def stop(self):
        """Stop the FM receiver."""
        if not self.running:
            return
            
        print("Stopping FM receiver...")
        self.running = False
        
        time.sleep(0.2)  # Give threads time to notice the flag change
        
        # Clean up audio stream
        try:
            if self.stream is not None and hasattr(self.stream, 'active') and self.stream.active:
                print("Stopping audio stream...")
                self.stream.stop()
                self.stream.close()
                self.stream = None
        except Exception as e:
            print(f"Error stopping audio stream: {e}")
            
        # Close PulseAudio client if open
        try:
            if self.pulse is not None:
                print("Closing PulseAudio client...")
                self.pulse.close()
                self.pulse = None
        except Exception as e:
            print(f"Error closing PulseAudio client: {e}")
            
        # Wait for the audio thread to end
        if hasattr(self, 'audio_thread') and self.audio_thread.is_alive():
            try:
                print("Waiting for audio thread to terminate...")
                self.audio_thread.join(timeout=1.0)
            except Exception as e:
                print(f"Error joining audio thread: {e}")
            
        # Stop the SDR
        try:
            if self.sdr is not None and hasattr(self.sdr, 'dev') and self.sdr.dev is not None:
                print("Stopping SDR async reception...")
                self.sdr.stop_rx_async()
                print("Closing SDR device...")
                self.sdr.close()
                self.sdr = None
        except Exception as e:
            print(f"Error while closing SDR: {e}")
            
        print("FM receiver stopped.")
    
    def _process_iq_samples(self, iq_samples):
        """Process IQ samples from the SDR and perform FM demodulation."""
        global terminate_flag
        if terminate_flag or not self.running or iq_samples.size == 0:
            return
            
        try:
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
            max_val = np.max(np.abs(audio_filtered) + 1e-10)
            audio_normalized = audio_filtered / max_val if max_val > 0 else audio_filtered
            audio_normalized = audio_normalized * self.volume
            
            # Add to audio queue if still running
            if self.running and not terminate_flag and not self.no_audio:
                try:
                    self.audio_queue.put(audio_normalized, block=False)
                except:
                    # If queue is full, just skip this buffer
                    pass
        except Exception as e:
            if self.running and not terminate_flag:
                print(f"Error processing samples: {e}")
            
    def _audio_callback(self, outdata, frames, time, status):
        """Audio callback function for sounddevice."""
        if status and not terminate_flag:
            print(f"Audio callback status: {status}")
            
        if not self.running or terminate_flag or self.no_audio:
            outdata.fill(0)
            return
            
        if self.audio_queue.empty():
            # No audio data available, output silence
            outdata.fill(0)
        else:
            try:
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
            except Exception as e:
                if not terminate_flag:
                    print(f"Error in audio callback: {e}")
                outdata.fill(0)
    
    def _pulse_audio_worker(self, sink_name):
        """Audio worker thread for PulseAudio output."""
        try:
            print(f"Starting PulseAudio output to sink: {sink_name}")
            
            # Initialize PulseAudio client
            self.pulse = pulsectl.Pulse('fm-receiver')
            
            # Find the sink
            sink_found = False
            for sink in self.pulse.sink_list():
                if sink.name == sink_name:
                    sink_found = True
                    print(f"Found sink: {sink.name} - {sink.description}")
                    break
            
            if not sink_found:
                print(f"Could not find sink: {sink_name}")
                print("Audio playback disabled.")
                self.no_audio = True
                return
            
            # Create a stream
            sample_spec = {'format': 'float32le', 'rate': AUDIO_RATE, 'channels': 1}
            stream = self.pulse.stream_create(
                name="FM Receiver",
                sink_name=sink_name,
                sample_spec=sample_spec,
                buffering_attributes={'maxlength': -1, 'tlength': -1, 'minreq': 4*1024, 'prebuf': -1}
            )
            
            print(f"PulseAudio stream created")
            
            # Process audio data
            while self.running and not terminate_flag:
                if not self.audio_queue.empty():
                    try:
                        # Get audio data
                        audio_data = self.audio_queue.get()
                        
                        # Convert to bytes (float32)
                        audio_bytes = (audio_data.astype(np.float32)).tobytes()
                        
                        # Write to stream
                        self.pulse.stream_write(stream, audio_bytes)
                    except Exception as e:
                        if not terminate_flag:
                            print(f"Error writing to PulseAudio stream: {e}")
                else:
                    # Sleep briefly if no data
                    time.sleep(0.01)
            
            # Cleanup
            try:
                self.pulse.stream_delete(stream)
            except:
                pass
                
        except Exception as e:
            print(f"Error in PulseAudio worker: {e}")
            print("Audio playback disabled.")
            self.no_audio = True
        finally:
            # Clean up
            try:
                if self.pulse is not None:
                    self.pulse.close()
                    self.pulse = None
            except:
                pass
    
    def _audio_worker(self):
        """Audio output worker thread using sounddevice."""
        if self.no_audio:
            print("Audio playback disabled.")
            return
        
        # Check if using PulseAudio
        if isinstance(self.audio_device, dict) and self.audio_device['type'] == 'pulse':
            self._pulse_audio_worker(self.audio_device['name'])
            return
            
        # Otherwise use sounddevice
        try:
            # Get device index
            device_idx = None
            if isinstance(self.audio_device, dict) and self.audio_device['type'] == 'sounddevice':
                device_idx = self.audio_device['index']
                print(f"Using sounddevice device: {device_idx} ({self.audio_device['name']})")
            else:
                device_idx = None
                print("Using default sounddevice output device")
                
            # Initialize audio stream
            self.stream = sd.OutputStream(
                samplerate=AUDIO_RATE,
                channels=1,
                callback=self._audio_callback,
                blocksize=1024,
                device=device_idx
            )
            
            # Start the stream
            self.stream.start()
            print("Audio stream started successfully")
            
            # Keep the thread alive
            while self.running and not terminate_flag:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Error in audio worker: {e}")
            print("Audio playback disabled.")
            self.no_audio = True
        finally:
            # Clean up in case of thread exit
            try:
                if self.stream is not None and self.stream.active:
                    self.stream.stop()
                    self.stream.close()
            except:
                pass
    
    def start(self):
        """Start the FM receiver."""
        global terminate_flag
        terminate_flag = False
        
        try:
            # Check for xrdp-sink if requested
            if self.use_xrdp and PULSECTL_AVAILABLE:
                found = False
                try:
                    with pulsectl.Pulse('fm-receiver') as pulse:
                        for sink in pulse.sink_list():
                            if 'xrdp' in sink.name.lower():
                                print(f"Found xrdp sink: {sink.name}")
                                self.audio_device = {'type': 'pulse', 'index': 0, 'name': sink.name}
                                found = True
                                break
                        
                        if not found:
                            print("No xrdp sink found. Please select another output method.")
                except Exception as e:
                    print(f"Error looking for xrdp sink: {e}")
            
            # If audio device is not specified and audio is not disabled,
            # let the user select a device interactively
            if self.audio_device is None and not self.no_audio:
                self.audio_device = select_audio_device()
                if self.audio_device is None:
                    print("No audio device selected. Audio output disabled.")
                    self.no_audio = True
                
            # Create SDR instance
            self.sdr = FobosSDR()
            
            # Find and open the SDR device
            device_count = self.sdr.get_device_count()
            if device_count == 0:
                raise RuntimeError("No Fobos SDR devices found")
            
            print(f"Found {device_count} device(s)")
            self.sdr.open(0)
            
            # Print device info
            info = self.sdr.get_board_info()
            print(f"Connected to {info['product']} (SN: {info['serial']})")
            
            # Get available sample rates
            rates = self.sdr.get_samplerates()
            print(f"Available sample rates: {[r/1e6 for r in rates]} MHz")
            
            # Select appropriate sample rate
            # Find the closest available sample rate to our target
            if SAMPLE_RATE not in rates:
                closest_rate = min(rates, key=lambda x: abs(x - SAMPLE_RATE))
                print(f"Note: Requested {SAMPLE_RATE/1e6} MHz not available, using {closest_rate/1e6} MHz")
                use_rate = closest_rate
            else:
                use_rate = SAMPLE_RATE
            
            # Configure the SDR
            actual_freq = self.sdr.set_frequency(self.frequency)
            print(f"Tuned to {actual_freq/1e6:.2f} MHz")
            
            actual_rate = self.sdr.set_samplerate(use_rate)
            print(f"Sample rate: {actual_rate/1e6:.3f} MHz")
            
            # Set LNA gain (0-2)
            lna_value = 1  # Medium gain
            self.sdr.set_lna_gain(lna_value)
            print(f"LNA gain set to {lna_value}")
            
            # Set VGA gain (0-15)
            # Convert dB to VGA value (approximate)
            vga_value = min(15, max(0, int(self.gain / 2)))
            self.sdr.set_vga_gain(vga_value)
            print(f"VGA gain set to {vga_value} (approximately {vga_value*2} dB)")
            
            # Start the audio output thread
            self.running = True
            self.audio_thread = threading.Thread(target=self._audio_worker)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            # Brief pause to allow audio thread to start and report any errors
            time.sleep(0.5)
            
            # Start the SDR in asynchronous mode
            print(f"Starting SDR with buffer size {self.buffer_size}")
            self.sdr.start_rx_async(self._process_iq_samples, buf_count=16, buf_length=self.buffer_size)
            
            print("\nFM receiver started. Press Ctrl+C to stop.")
            print("Commands: f <MHz> (frequency), v <0.0-1.0> (volume), g <dB> (gain), q (quit)")
            
            # Simple command interface
            while self.running and not terminate_flag:
                try:
                    cmd = input("> ")
                    if cmd.lower() in ('q', 'quit', 'exit'):
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
                    elif cmd == 'devices':
                        # List all available devices
                        list_all_audio_devices()
                        list_pulse_sinks()
                    else:
                        print("Commands: f <MHz> (frequency), v <0.0-1.0> (volume), g <dB> (gain), devices (list devices), q (quit)")
                except EOFError:
                    # Handle Ctrl+D gracefully
                    print("\nReceived EOF. Stopping...")
                    break
                except KeyboardInterrupt:
                    # Handle Ctrl+C gracefully
                    print("\nReceived keyboard interrupt. Stopping...")
                    break
                
        except Exception as e:
            print(f"Error in FM receiver: {e}")
        finally:
            # Always stop in the finally block
            self.stop()

def signal_handler(sig, frame):
    """Handle Ctrl+C signal for clean shutdown."""
    global terminate_flag
    print("\nReceived termination signal. Shutting down gracefully...")
    terminate_flag = True

def main():
    """Main entry point for the FM receiver application."""
    # Register signal handler
    sig.signal(sig.SIGINT, signal_handler)
    sig.signal(sig.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description='Fobos SDR FM Receiver')
    parser.add_argument('-f', '--frequency', type=float, default=DEFAULT_FREQ/1e6,
                        help='FM station frequency in MHz (default: 95.5)')
    parser.add_argument('-g', '--gain', type=float, default=GAIN,
                        help='Receiver gain in dB (default: 12)')
    parser.add_argument('-b', '--buffer', type=int, default=BUFFER_SIZE,
                        help=f'Buffer size for SDR (default: {BUFFER_SIZE})')
    parser.add_argument('--no-audio', action='store_true',
                        help='Disable audio output')
    parser.add_argument('--list-devices', action='store_true',
                        help='List all audio devices and exit')
    parser.add_argument('--xrdp', action='store_true',
                        help='Use xrdp-sink automatically')
    
    args = parser.parse_args()
    
    # If requested, just list devices and exit
    if args.list_devices:
        list_all_audio_devices()
        list_pulse_sinks()
        return
    
    # Create and start the FM receiver
    receiver = FMReceiver(
        frequency=args.frequency * 1e6,
        gain=args.gain,
        buffer_size=args.buffer,
        no_audio=args.no_audio,
        use_xrdp=args.xrdp
    )
    
    try:
        receiver.start()
    except Exception as e:
        print(f"Fatal error: {e}")
    
    # Make sure we exit cleanly
    print("Program terminated.")
    
if __name__ == "__main__":
    main()
