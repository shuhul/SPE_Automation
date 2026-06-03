import requests
import numpy as np
import matplotlib.pyplot as plt

TOKEN = "8463582982:AAG-izcwemLDy4l2A2ouEAXJDGzHL8xHD5A"

current_user = "kristina"

users = ["shuhul", "kristina"]
CHAT_IDS = ["8130896008", "7568051086"]

def send_telegram_message(message):
    for user, CHAT_ID in zip(users, CHAT_IDS):
        if user == current_user:
            url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
            payload = {
                "chat_id": CHAT_ID,
                "text": message
            }
            try:
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    print("Message sent!")
                else:
                    print(f"Failed: {response.text}")
            except Exception as e:
                print(f"Error: {e}")

send_telegram_message("Hi from python!")


# raise Exception


filename = '20251103-PLSPC-PhENOM-Ch10-f014o018-150uw-1300msIntegration-test1.npy'
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


laser_peak_cutoff_fraction = 0.5 # fraction of first intensity
first = True

peak_map = np.zeros((len(ys), len(xs)))
for iy, y in enumerate(ys):
    for ix, x in enumerate(xs):
        intensity = all_intensities[iy][ix]

        window = (wl > 529) & (wl < 535)
        peak_idx = np.argmax(intensity[window])
        peak_intensity = intensity[window][peak_idx]
        peak_wavelength = wl[window][peak_idx]

        if first:
            print(f'First peak intensity {peak_intensity} @ {np.round(peak_wavelength,0)} nm')
            cutoff = laser_peak_cutoff_fraction*peak_intensity
            print(f'Cutoff {cutoff}')
            first = False
        elif peak_intensity < cutoff:
            # Send notification
            # send_telegram_message("WARNING: Out of focus!")
            print(f'Out of focus at x={x}, y={y}')
            pass
        peak_map[iy, ix] = peak_intensity
        


# Summed heatmap
# summed = np.sum(all_intensities, axis=-1)

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
    peak_map,
    extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]],
    origin='upper',
    cmap='inferno',
    aspect='equal'
)
ax_img.set_title('Peak Intensity Map')
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
