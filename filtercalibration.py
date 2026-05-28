import time
import os
import numpy as np
from datetime import datetime

# Start the boot-up timer
t_start = time.time()

# Import your custom modules
import lf_spec
import filter

# --- 1. Configuration ---
exposure_time = 1  # seconds
angle_step = 0.5   # degrees
start_angle = 320
stop_angle = 20

# Edge finding parameters
peak_threshold = 10000
wl_min = 520
wl_max = 660

# --- 2. Create Save Directory ---
# Creates a folder named like "calibration/2026-04-07_12-30-00"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join("calibration", timestamp)
os.makedirs(save_dir, exist_ok=True)
print(f"Data will be saved to: {save_dir}")

# --- 3. Initialization ---
print("Initializing hardware...")
lf_spec.lf_connect()
lf_spec.lf_setup(exposure_s=exposure_time)
filter.filter_init()
filter.filter_on()
filter.flip_up()
filter.rotation_home()

# --- 4. Calculate Angles ---
total_sweep = (360 - start_angle) + stop_angle
continuous_angles = np.arange(start_angle, start_angle + total_sweep + angle_step, angle_step)

# Lists to hold our data
calibration_table = [] # Will hold [angle, center_wavelength]
calibration_data = []  # Will hold the raw intensity spectrum arrays

# --- 5. Execution Loop ---
t_scan_start = time.time()
boot_up_time = t_scan_start - t_start
print(f"\n[TIMING] Boot up complete in {boot_up_time:.2f} seconds.")
print("Starting Calibration Sweep...")

for current_angle in continuous_angles:
    # Wrap the angle using modulo
    target_angle = current_angle % 360 
    
    # Move Stage
    filter.rotation_move(target_angle)
    time.sleep(0.1) # Settling time
    
    # Acquire Spectrum
    intensity_raw, wl_raw = lf_spec.lf_acquire()
    intensity = np.array(intensity_raw).flatten()
    wl = np.array(wl_raw).flatten()
    
    # Store the raw spectrum for this angle
    calibration_data.append(intensity)
    
    # Process Spectrum: Use the edge-threshold method
    # Restrict the search to the specified wavelength window
    window_mask = (wl >= wl_min) & (wl <= wl_max)
    wl_window = wl[window_mask]
    spectrum_window = intensity[window_mask]
    
    # Find all indices where the signal is above the threshold
    above_thresh = np.where(spectrum_window > peak_threshold)[0]
    
    if len(above_thresh) > 0:
        # The first time it crosses the threshold
        rise_idx = above_thresh[0]
        # The last time it is above the threshold
        fall_idx = above_thresh[-1]
        
        rise_wl = wl_window[rise_idx]
        fall_wl = wl_window[fall_idx]
        
        # Calculate the mathematical center between the two edges
        center_wl = (rise_wl + fall_wl) / 2.0
    else:
        # If the peak never hits the threshold at this specific angle
        center_wl = np.nan

    print(f"Angle: {target_angle:>5.1f}° | Center Wavelength: {center_wl:.2f} nm")
    
    # Store the processed peak result
    calibration_table.append([target_angle, center_wl])

# End the scan timer
t_scan_end = time.time()
scan_time = t_scan_end - t_scan_start

# --- 6. Save Data and Shutdown ---
print(f"\n[TIMING] Sweep finished in {scan_time:.2f} seconds.")
print("Saving data...")

# Save the files into the timestamped directory
np.save(os.path.join(save_dir, 'calibration_table.npy'), np.array(calibration_table))
np.save(os.path.join(save_dir, 'calibration_data.npy'), np.array(calibration_data))
np.save(os.path.join(save_dir, 'wavelength.npy'), wl)

print(f"Saved files successfully to '{save_dir}'.")

filter.rotation_move(0)
# Safely turn off the hardware
filter.filter_off()
print("Hardware disconnected. Calibration complete!")