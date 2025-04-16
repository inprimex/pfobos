#!/bin/bash
# Script to test audio devices and setup for Fobos SDR audio processing

# Check if pactl is available
if ! command -v pactl &> /dev/null; then
    echo "Error: pactl command not found. Please install PulseAudio."
    echo "On Ubuntu/Debian: sudo apt-get install pulseaudio-utils"
    echo "On Fedora: sudo dnf install pulseaudio-utils"
    exit 1
fi

# Add a title banner
echo "====================================="
echo "Fobos SDR Audio Device Test Utility"
echo "====================================="
echo ""

# List all PulseAudio sinks (output devices)
echo "PulseAudio output devices (sinks):"
echo "-----------------------------------"
pactl list sinks | grep -E 'Sink|Name:|Description:'

# List all PulseAudio sources (input devices)
echo -e "\nPulseAudio input devices (sources):"
echo "-----------------------------------"
pactl list sources | grep -E 'Source|Name:|Description:'

# List sink inputs (applications using audio output)
echo -e "\nActive audio streams (sink-inputs):"
echo "-----------------------------------"
pactl list sink-inputs | grep -E 'Sink Input|application.name|media.name'

# For a shorter summary
echo -e "\nShort summary of devices:"
echo "-----------------------------------"
echo "Sinks (outputs):"
pactl list short sinks
echo -e "\nSources (inputs):"
pactl list short sources

# Check for default devices
echo -e "\nDefault audio devices:"
echo "-----------------------------------"
DEFAULT_SINK=$(pactl info | grep "Default Sink" | cut -d: -f2- | xargs)
DEFAULT_SOURCE=$(pactl info | grep "Default Source" | cut -d: -f2- | xargs)

echo "Default output device: $DEFAULT_SINK"
echo "Default input device: $DEFAULT_SOURCE"

# Test if we can get sample rates
echo -e "\nChecking device sample rates:"
echo "-----------------------------------"
for sink in $(pactl list short sinks | cut -f1); do
    RATE=$(pactl list sinks | grep -A20 "Sink #$sink" | grep "Sample Specification" | grep -oE '[0-9]+ Hz' | head -1)
    NAME=$(pactl list sinks | grep -A1 "Sink #$sink" | grep "Name:" | cut -d: -f2- | xargs)
    echo "Sink #$sink ($NAME): $RATE"
done

echo -e "\nTest completed successfully!"
echo "You can set default devices with:"
echo "  pactl set-default-sink [sink_name]"
echo "  pactl set-default-source [source_name]"
