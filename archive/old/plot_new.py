import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load data (replace with yours)
# -----------------------------
# filename = '20251103-PLSPC-PhENOM-Ch10-f014o018-150uw-1300msIntegration-test1.npy'

filename='20251124-PLSPC-f015o016-nwg-postetch-3.npy'
filename_fine = '20251103-PLSPC-PhENOM-Ch10-f014o018-150uw-1300msIntegration-test1_fine.npy'
fine = False

wl = np.load('wl.npy')

if not fine:
    all_intensities = np.load(filename)
    xs = np.load('xs.npy')
    ys = np.load('ys.npy')
    
else:
    all_intensities = np.load(filename_fine)
    xs = np.load('xs_fine.npy')
    ys = np.load('ys_fine.npy')

# Summed heatmap
summed = np.sum(all_intensities, axis=-1)

def edges_from_centers(coords):
    diffs = np.diff(coords) / 2
    edges = np.concatenate(([coords[0] - diffs[0]], coords[:-1] + diffs, [coords[-1] + diffs[-1]]))
    return edges

x_edges = edges_from_centers(xs)
y_edges = edges_from_centers(ys)

# -----------------------------
# Interactive plot setup
# -----------------------------
fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(12, 6))
im = ax_img.imshow(
    summed,
    extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]],
    origin='upper',
    cmap='inferno',
    aspect='equal'
)
ax_img.set_title('Summed Intensity Map')
ax_img.set_xlabel('x position')
ax_img.set_ylabel('y position')
fig.colorbar(im, ax=ax_img, label='Summed Intensity')

ax_spec.set_title('Hover over a pixel to see its spectrum')
ax_spec.set_xlabel('Wavelength')
ax_spec.set_ylabel('Intensity')

# For performance, keep a single line we update
(line,) = ax_spec.plot([], [], lw=2)
text = ax_spec.text(0.5, 0.9, '', transform=ax_spec.transAxes, ha='center', color='w')

# -----------------------------
# Hover callback
# -----------------------------
def on_move(event):
    if event.inaxes != ax_img:
        return
    # Convert mouse position (in data coords) to pixel indices
    xdata, ydata = event.xdata, event.ydata
    if xdata is None or ydata is None:
        return

    # Find nearest pixel index
    ix = np.argmin(np.abs(xs - xdata))
    iy = np.argmin(np.abs(ys - ydata))

    # Extract and plot spectrum
    spectrum = all_intensities[iy, ix, :]

    # line.set_data(np.arange(len(spectrum)), spectrum)
    line.set_data(wl, spectrum)
    ax_spec.relim()
    ax_spec.autoscale_view()

    text.set_text(f"x={xs[ix]:.3f}, y={ys[iy]:.3f}")

    fig.canvas.draw_idle()

# -----------------------------
# Connect and show
# -----------------------------
fig.canvas.mpl_connect("motion_notify_event", on_move)
plt.tight_layout()
plt.show()
