import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

# Read the .spe file
im = imageio.imread('C:\\Users\\\Public\\Shared Confocal Files\\data\\temp\\N 2025 October 10 12_27_27 1.spe')

# Flatten the array if it's 2D with one row
spectrum = im.flatten()

# Create an x-axis (e.g., pixel indices)
x = np.arange(len(spectrum))

# Plot
plt.plot(x, spectrum)
plt.xlabel("Pixel")
plt.ylabel("Intensity")
plt.title("Spectrum from Test.spe")
plt.show()
