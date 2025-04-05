#!/bin/bash
# run_spectrum_analyzer.sh - Script to run Fobos SDR spectrum analyzer with X11 backend

# Set environment variables for display
export QT_QPA_PLATFORM=xcb
export QT_QPA_PLATFORMTHEME=gtk2
export DISPLAY=:0
export MPLBACKEND=TkAgg

# Print header
echo "============================================="
echo "     Fobos SDR Spectrum Analyzer Runner      "
echo "============================================="
echo "Setting up environment..."
echo "Display: $DISPLAY"
echo "Qt platform: $QT_QPA_PLATFORM"
echo "Matplotlib backend: $MPLBACKEND"
echo

# Check for required packages
echo "Checking for required packages..."
if ! python3 -c "import matplotlib" 2>/dev/null; then
    echo "Warning: matplotlib not found. Installing..."
    pip install matplotlib
fi

if ! python3 -c "import numpy" 2>/dev/null; then
    echo "Warning: numpy not found. Installing..."
    pip install numpy
fi

if ! python3 -c "import scipy" 2>/dev/null; then
    echo "Warning: scipy not found. Installing..."
    pip install scipy
fi

# Try to fix common display issues
if [ -z "$DISPLAY" ]; then
    echo "DISPLAY environment variable not set. Setting to :0"
    export DISPLAY=:0
fi

# Create spectrum_plots directory if it doesn't exist
echo "Setting up plot directory..."
mkdir -p spectrum_plots

# Run the test script first to verify display works
echo "Running matplotlib test to verify display works..."
cat > test_matplotlib.py << 'EOF'
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# Simple test plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title("Test Plot")
plt.grid(True)

# Save to file
plt.savefig("test_plot.png")
print("Test plot saved to test_plot.png")

try:
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    print("Display test successful!")
except Exception as e:
    print(f"Display test failed: {e}")
    print("Will continue with file output only.")
EOF

python3 test_matplotlib.py

# Run the spectrum analyzer
echo
echo "Running Fobos SDR Spectrum Analyzer..."
echo "Plots will be saved to ./spectrum_plots/ directory"
echo "Press Ctrl+C to exit when done"
echo

# Run the spectrum analyzer
python3 fobos_spectrum_analyzer.py

# Exit message
echo
echo "Spectrum analyzer closed"
echo "You can find saved spectrum plots in ./spectrum_plots/ directory"
