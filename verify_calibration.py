import time
import os
import numpy as np

# Import your custom modules
import pl_init
import filter
import pl_spec

# --- 1. Configuration ---
# Set these two strings to control the test
CALIBRATION_FOLDER_NAME = "2026-04-07_14-48-20" 
TEST_ANGLE = 350    # The specific angle you want to check

tolerance_nm = 2    # Acceptable difference in nm
exposure_time = 1     # seconds

# Edge finding parameters (Must match calibration script!)
peak_threshold = 10000
wl_min = 520
wl_max = 660

# --- 2. Load Calibration Data ---
base_dir = "calibration"
selected_dir = os.path.join(base_dir, CALIBRATION_FOLDER_NAME)

if not os.path.exists(selected_dir):
    print(f"Error: Calibration folder '{selected_dir}' not found.")
    exit()

table_path = os.path.join(selected_dir, 'calibration_table.npy')
calibration_table = np.load(table_path) # Matrix of [angle, wavelength]

# Find the expected wavelength for the chosen angle
# We look for the row where the angle matches TEST_ANGLE
idx = np.where(calibration_table[:, 0] == TEST_ANGLE)[0]

if len(idx) == 0:
    print(f"Error: Angle {TEST_ANGLE} not found in the calibration table.")
    print("Ensure the angle matches one of the steps used during calibration.")
    exit()

expected_wl = calibration_table[idx[0], 1]

if np.isnan(expected_wl):
    print(f"Error: The calibration entry for {TEST_ANGLE}° is NaN (no peak was found then).")
    exit()

# --- 3. Hardware Initialization ---
print(f"Initializing hardware to check {TEST_ANGLE}°...")
pl_init.pl_init()
filter.filter_init()
filter.filter_on()  
filter.flip_up()
filter.rotation_home()
pl_spec.connect_matlab()
pl_spec.pl_set_settings(exposure_time=exposure_time)

# --- 4. Execution ---
print(f"Moving to {TEST_ANGLE}°...")
filter.rotation_move(TEST_ANGLE)
time.sleep(0.5) # Extra settling time for a single check

# Acquire Spectrum
intensity, wl = pl_spec.pl_single_scan()

# Process Spectrum using the Threshold Edge Method
window_mask = (wl >= wl_min) & (wl <= wl_max)
wl_window = wl[window_mask]
spectrum_window = intensity[window_mask]

above_thresh = np.where(spectrum_window > peak_threshold)[0]

if len(above_thresh) > 0:
    rise_idx = above_thresh[0]
    fall_idx = above_thresh[-1]
    rise_wl = wl_window[rise_idx]
    fall_wl = wl_window[fall_idx]
    measured_wl = (rise_wl + fall_wl) / 2.0
else:
    measured_wl = np.nan

# --- 5. Comparison ---
print("\n--- Results ---")
if np.isnan(measured_wl):
    print(f"STATUS: [FAIL] Peak failed to cross {peak_threshold} counts at {TEST_ANGLE}°.")
else:
    diff = abs(measured_wl - expected_wl)
    passed = diff <= tolerance_nm
    status = "PASS" if passed else "FAIL"
    
    print(f"Target Angle: {TEST_ANGLE}°")
    print(f"Expected WL:  {expected_wl:.2f} nm")
    print(f"Measured WL:  {measured_wl:.2f} nm")
    print(f"Difference:   {diff:.2f} nm")
    print(f"Status:       [{status}] (Tolerance: {tolerance_nm} nm)")

# --- 6. Shutdown ---
filter.rotation_move(0)
filter.filter_off()
print("\nHardware disconnected.")