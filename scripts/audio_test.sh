#!/bin/bash

# List all PulseAudio sinks (output devices)
echo "PulseAudio output devices (sinks):"
pactl list sinks

# List all PulseAudio sources (input devices)
echo -e "\nPulseAudio input devices (sources):"
pactl list sources

# List sink inputs (applications using audio output)
echo -e "\nActive audio streams (sink-inputs):"
pactl list sink-inputs

# For a shorter summary
echo -e "\nShort summary of devices:"
pactl list short sinks
pactl list short sources