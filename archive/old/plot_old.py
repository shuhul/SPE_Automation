import numpy as np
import matplotlib.pyplot as plt

# Load data
filename = '20251103-PLSPC-PhENOM-Ch10-f014o018-150uw-1300msIntegration-test1.npy'

filename_fine='20251103-PLSPC-PhENOM-Ch10-f014o018-150uw-1300msIntegration-test1_fine.npy'

fine = True

if not fine:
    all_intensities = np.load(filename)
    xs = np.load('xs.npy')
    ys = np.load('ys.npy')
else:
    all_intensities = np.load(filename_fine)
    xs = np.load('xs_fine.npy')
    ys = np.load('ys_fine.npy')

# Sum over the last axis (collapse spectrum dimension)
summed = np.sum(all_intensities, axis=-1)

# --- Compute pixel edge coordinates from center coordinates ---
def edges_from_centers(coords):
    diffs = np.diff(coords) / 2
    edges = np.concatenate((
        [coords[0] - diffs[0]],
        coords[:-1] + diffs,
        [coords[-1] + diffs[-1]]
    ))
    return edges

x_edges = edges_from_centers(xs)
y_edges = edges_from_centers(ys)

# --- Plot heatmap ---
plt.figure(figsize=(8, 6))
plt.imshow(
    summed,
    extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]],  # flip y here
    origin='upper',  # ensures array[0,0] is top-left
    cmap='inferno',
    aspect='equal'
)
plt.colorbar(label='Summed Intensity')
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('Summed Spectrum Heatmap')
plt.show()
