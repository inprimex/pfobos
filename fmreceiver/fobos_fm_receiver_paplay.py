"""
Simple FM radio receiver using Fobos SDR with named pipe audio output.
This example shows how to implement FM demodulation with audio output.

Requirements:
- numpy
- scipy
- fobos_wrapper.py (updated version)
"""

import numpy as np
from scipy import signal
import threading
import time
import argparse
import sys
import os
import signal as sig
import subprocess
import tempfile
from queue import Queue
from fobos_wrapper import FobosSDR, FobosException

# Default configuration
DEFAULT_FREQ = 95.5e6  # Default FM station frequency (95.5 MHz)
SAMPLE_RATE = 2.048e6  # SDR sample rate
AUDIO_RATE = 44100     # Audio output sample rate
BUFFER_SIZE = 32768    # IQ buffer size (increased for stability)
GAIN = 12              # Receiver gain setting

# FM demodulation parameters
FM_DEVIATION = 75e3    # FM deviation (75 kHz for broadcast FM)
STEREO_CARRIER = 19e3  # Stereo subcarrier frequency

# Flag for controlling termination
terminate_flag = False

def find_xrdp_sink():
    """Find the xrdp sink using pactl."""
    try:
        result = subprocess.run(['pactl', 'list', 'sinks'], capture_output=True, text=True)
        if result.returncode == 0:
            sinks = result.stdout.split("Sink #")
            for sink in sinks[1:]:
                if 'xrdp' in sink.lower():
                    for line in sink.split('\n'):
                        if line.strip().startswith("Name:"):
                            return line.split(":", 1)[1].strip()
    except:
        pass
    
    return None

def list_pulse_sinks():
    """List all PulseAudio sinks using pactl."""
    try:
        print("\nPULSEAUDIO SINKS:")
        print("----------------")
        result = subprocess.run(['pactl', 'list', 'sinks'], capture_output=True, text=True)
        if result.returncode == 0:
            sinks = result.stdout.split("Sink #")
            for sink in sinks[1:]:  # Skip the first empty split
                sink_id = None
                name = None
                desc = None
                
                lines = sink.strip().split('\n')
                sink_id = lines[0].strip()
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("Name:"):
                        name = line.split(":", 1)[1].strip()
                    elif line.startswith("Description:"):
                        desc = line.split(":", 1)[1].strip()
                
                if name and desc:
                    print(f"  {sink_id}: {name} - {desc}")
        return None
    except Exception as e:
        print(f"Error listing PulseAudio sinks with pactl: {e}")
        return None

class FMReceiver:
    def __init__(self, frequency=DEFAULT_FREQ, gain=GAIN, buffer_size=BUFFER_SIZE, 
                 pulse_sink=None, no_audio=False):
        self.frequency = frequency
        self.gain = gain
        self.buffer_size = buffer_size
        self.pulse_sink = pulse_sink
        self.no_audio = no_audio
        self.sdr = None  # Initialize later
        
        # Signal processing state variables
        self.prev_phase = 0
        self.audio_queue = Queue(maxsize=10)
        
        # Control variables
        self.running = False
        self.volume = 0.5  # Initial volume (0.0-1.0)
        self.audio_process = None
        self.named_pipe = None
        
    def stop(self):
        """Stop the FM receiver."""
        if not self.running:
            return
            
        print("Stopping FM receiver...")
        self.running = False
        
        time.sleep(0.2)  # Give threads time to notice the flag change
        
        # Stop the audio process if running
        if self.audio_process is not None:
            try:
                print("Stopping audio process...")
                self.audio_process.terminate()
                self.audio_process.wait(timeout=1.0)
            except:
                try:
                    self.audio_process.kill()
                except:
                    pass
            self.audio_process = None
            
        # Remove named pipe
        if self.named_pipe and os.path.exists(self.named_pipe):
            try:
                os.unlink(self.named_pipe)
            except:
                pass
            self.named_pipe = None
            
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
    
    def _audio_worker(self):
        """Audio worker thread using a named pipe and parec."""
        if self.no_audio:
            print("Audio playback disabled.")
            return
        
        try:
            # Create a named pipe for audio data
            temp_dir = tempfile.gettempdir()
            self.named_pipe = os.path.join(temp_dir, f"fobos_fm_{os.getpid()}.fifo")
            
            # Remove existing pipe if it exists
            if os.path.exists(self.named_pipe):
                os.unlink(self.named_pipe)
                
            # Create the named pipe
            os.mkfifo(self.named_pipe)
            
            # Print audio output configuration
            print(f"Audio output: Using named pipe and PulseAudio")
            if self.pulse_sink:
                print(f"PulseAudio sink: {self.pulse_sink}")
            else:
                print("Using default PulseAudio sink")
                
            # Prepare parec command to play from the named pipe
            cmd = ['parec', '--format=s16le', '--rate=44100', '--channels=1']
            
            # Add sink if specified
            if self.pulse_sink:
                cmd.extend(['--device', self.pulse_sink])
                
            # Redirect to STDOUT
            cmd.append('-')
            
            # Start audio process with pipe
            print(f"Starting PulseAudio with command: {' '.join(cmd)}")
            
            # Open the named pipe for writing in non-blocking mode
            pipe_fd = os.open(self.named_pipe, os.O_WRONLY | os.O_NONBLOCK)
            
            # Use cat to read from pipe and parec to play
            self.audio_process = subprocess.Popen(
                ['cat', self.named_pipe], 
                stdout=subprocess.PIPE
            )
            
            play_process = subprocess.Popen(
                cmd,
                stdin=self.audio_process.stdout
            )
            
            print("Audio output started.")
            
            # Some initial silence
            try:
                silence = np.zeros(AUDIO_RATE//4, dtype=np.int16)
                os.write(pipe_fd, silence.tobytes())
            except:
                pass
            
            # Continuously write audio data to the named pipe
            while self.running and not terminate_flag:
                if not self.audio_queue.empty():
                    try:
                        # Get audio data from queue
                        audio_data = self.audio_queue.get()
                        
                        # Convert to 16-bit PCM
                        audio_int16 = (audio_data * 32767).astype(np.int16)
                        
                        # Write to the named pipe
                        os.write(pipe_fd, audio_int16.tobytes())
                    except Exception as e:
                        if not terminate_flag and not isinstance(e, BlockingIOError):
                            print(f"Error writing audio data: {e}")
                time.sleep(0.01)  # Short sleep when no data is available
                
            # Close the pipe
            try:
                os.close(pipe_fd)
            except:
                pass
                
        except Exception as e:
            print(f"Error in audio worker: {e}")
            print("Audio playback disabled.")
            self.no_audio = True
        finally:
            # Clean up
            if self.audio_process is not None:
                try:
                    self.audio_process.terminate()
                except:
                    pass
                self.audio_process = None
                
            if self.named_pipe and os.path.exists(self.named_pipe):
                try:
                    os.unlink(self.named_pipe)
                except:
                    pass
                self.named_pipe = None
    
    def start(self):
        """Start the FM receiver."""
        global terminate_flag
        terminate_flag = False
        
        try:
            # List all available sinks
            list_pulse_sinks()
            
            # Check for xrdp-sink if not specified
            if self.pulse_sink is None:
                xrdp_sink = find_xrdp_sink()
                if xrdp_sink:
                    print(f"Found xrdp sink: {xrdp_sink}")
                    self.pulse_sink = xrdp_sink
                
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
                    elif cmd == 'sinks':
                        # List all available sinks
                        list_pulse_sinks()
                    else:
                        print("Commands: f <MHz> (frequency), v <0.0-1.0> (volume), g <dB> (gain), sinks (list sinks), q (quit)")
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
    parser.add_argument('-s', '--sink', type=str, default=None,
                        help='PulseAudio sink name to use')
    parser.add_argument('--no-audio', action='store_true',
                        help='Disable audio output')
    parser.add_argument('--list-sinks', action='store_true',
                        help='List all PulseAudio sinks and exit')
    parser.add_argument('--xrdp', action='store_true',
                        help='Use xrdp-sink automatically')
    
    args = parser.parse_args()
    
    # If requested, just list devices and exit
    if args.list_sinks:
        list_pulse_sinks()
        return
    
    # Determine which sink to use
    pulse_sink = args.sink
    if args.xrdp and not pulse_sink:
        xrdp_sink = find_xrdp_sink()
        if xrdp_sink:
            pulse_sink = xrdp_sink
    
    # Create and start the FM receiver
    receiver = FMReceiver(
        frequency=args.frequency * 1e6,
        gain=args.gain,
        buffer_size=args.buffer,
        pulse_sink=pulse_sink,
        no_audio=args.no_audio
    )
    
    try:
        receiver.start()
    except Exception as e:
        print(f"Fatal error: {e}")
    
    # Make sure we exit cleanly
    print("Program terminated.")
    
if __name__ == "__main__":
    main()
