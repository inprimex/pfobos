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
