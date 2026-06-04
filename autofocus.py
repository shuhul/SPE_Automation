"""
Autofocus module for SPE automation — Z-axis focusing based on 532 nm laser peak intensity.
Uses Thorlabs Kinesis (BenchtopPiezo) to control piezo Z-stage voltage.
Integrates with LightField for spectral feedback.

Algorithm:
  Phase 1a: Symmetric scan ±10V @ 1V steps from current voltage
  Phase 1b: Directional extension (if peak at boundary) until peak found
  Phase 2:  Fine scan ±1V @ 0.1V steps around coarse peak

Integration time: 1 second per spectrum
"""

import os
import sys
import time
import clr
import numpy as np
from datetime import datetime
from scipy.signal import find_peaks
# Kinesis imports (BenchtopPiezo for piezo Z-stage)
clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.GenericPiezoCLI.dll")
clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\ThorLabs.MotionControl.Benchtop.PiezoCLI.dll")
clr.AddReference("System")

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from Thorlabs.MotionControl.GenericPiezoCLI import Piezo, DeviceUnits
from Thorlabs.MotionControl.Benchtop.PiezoCLI import BenchtopPiezo
from System import Decimal

import lf_spec
import pl_spec_python as psp

# ============================================================================
# CONFIGURATION
# ============================================================================

Z_STAGE_SERIAL = "71945320"  # Replace with your actual Z-stage serial number
Z_STAGE_CHANNEL = 1           # Channel number (e.g., 1, 2, 3 for multi-channel)
Z_MIN_VOLTAGE = 0.0           # Volts
Z_MAX_VOLTAGE = 150.0         # Volts

# Phase 1a: Symmetric scan
PHASE1A_OFFSET = 10.0         # ±10V from current position
PHASE1A_STEP = 1.0            # 1V steps

# Phase 1b: Directional extension
PHASE1B_STEP = 1.0            # Continue at 1V steps if peak at boundary

# Phase 2: Fine scan
PHASE2_OFFSET = 1.0           # ±1V around coarse peak
PHASE2_STEP = 0.1             # 0.1V steps

# Spectrum analysis
LASER_WAVELENGTH_MIN = 530    # nm
LASER_WAVELENGTH_MAX = 535    # nm
INTEGRATION_TIME_S = 2.0      # 1 second per spectrum

# Logging
DEBUG_LOG_FILE = 'autofocus_debug.log'
FOCUS_RESULTS_FILE = 'autofocus_results.csv'

_channel = None
_autofocus_enabled = False

# ============================================================================
# LOGGING
# ============================================================================

def _log_debug(message):
    """Print and log message with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    try:
        with open(DEBUG_LOG_FILE, 'a') as f:
            f.write(log_line + '\n')
    except Exception as e:
        print(f"[WARNING] Could not write to debug log: {e}")

def _log_results(emitter_pos, z_voltage, laser_intensity, phase):
    """Log autofocus results to CSV."""
    try:
        result_line = f"{datetime.now().isoformat()},{emitter_pos[0]:.4f},{emitter_pos[1]:.4f},{z_voltage:.2f},{laser_intensity:.1f},{phase}\n"

        # Write header if file doesn't exist or is empty
        if not os.path.exists(FOCUS_RESULTS_FILE) or os.path.getsize(FOCUS_RESULTS_FILE) == 0:
            with open(FOCUS_RESULTS_FILE, 'w') as f:
                f.write("timestamp,x_um,y_um,z_voltage,laser_intensity,phase\n")

        with open(FOCUS_RESULTS_FILE, 'a') as f:
            f.write(result_line)
    except Exception as e:
        _log_debug(f"WARNING: Could not log results: {e}")

# ============================================================================
# Z-STAGE CONTROL (KINESIS BENCHTOPPIEZO)
# ============================================================================

def autofocus_init(serial_no=None, channel_num=None):
    """Initialize Kinesis BenchtopPiezo Z-stage connection."""
    global _channel, _autofocus_enabled

    serial = serial_no or Z_STAGE_SERIAL
    channel = channel_num or Z_STAGE_CHANNEL

    _log_debug(f"Initializing Z-stage (Serial: {serial}, Channel: {channel})...")

    try:
        DeviceManagerCLI.BuildDeviceList()

        device = BenchtopPiezo.CreateBenchtopPiezo(serial)
        device.Connect(serial)
        _log_debug("Connected to BenchtopPiezo device.")

        _channel = device.GetChannel(channel)
        _log_debug(f"Retrieved channel {channel}.")

        if not _channel.IsSettingsInitialized():
            _log_debug("Waiting for settings initialization...")
            _channel.WaitForSettingsInitialized(10000)

        assert _channel.IsSettingsInitialized(), "Settings not initialized"
        _log_debug("Settings initialized.")

        _channel.StartPolling(250)
        time.sleep(0.25)
        _channel.EnableDevice()
        time.sleep(0.25)
        _log_debug("Device polling started and enabled.")

        device_info = _channel.GetDeviceInfo()
        _log_debug(f"Device: {device_info.Description}")

        _channel.GetPiezoConfiguration(_channel.DeviceID)
        time.sleep(0.25)

        max_volts = _channel.GetMaxOutputVoltage()
        _log_debug(f"Max output voltage: {max_volts}V")

        _autofocus_enabled = True
        _log_debug("Z-stage initialized successfully.")
        return True

    except Exception as e:
        _log_debug(f"ERROR: Failed to initialize Z-stage: {e}")
        _autofocus_enabled = False
        return False

def autofocus_shutdown():
    """Gracefully shut down Z-stage."""
    global _channel, _autofocus_enabled

    if _channel is not None:
        try:
            _channel.StopPolling()
            _log_debug("Z-stage polling stopped.")
        except Exception as e:
            _log_debug(f"WARNING: Error stopping polling: {e}")

    _autofocus_enabled = False
    _log_debug("Z-stage shut down.")

def set_z_voltage(voltage):
    """Set Z-stage output voltage."""
    global _channel

    if not _autofocus_enabled or _channel is None:
        _log_debug(f"WARNING: Z-stage not initialized; cannot set voltage to {voltage:.2f}V")
        return False

    voltage = np.clip(voltage, Z_MIN_VOLTAGE, Z_MAX_VOLTAGE)

    try:
        _channel.SetOutputVoltage(Decimal(str(voltage)))
        time.sleep(0.1)
        return True
    except Exception as e:
        _log_debug(f"ERROR: Failed to set Z voltage to {float(voltage):.2f}V: {e}")
        return False

def get_z_voltage():
    """Get current Z-stage output voltage."""
    global _channel

    if not _autofocus_enabled or _channel is None:
        return None

    try:
        voltage = _channel.GetOutputVoltage()
        return float(voltage)
    except Exception as e:
        _log_debug(f"WARNING: Could not read Z voltage: {e}")
        return None

# ============================================================================
# SPECTRUM ANALYSIS
# ============================================================================

def get_532nm_peak_intensity(spectrum, wl):
    """Extract 532 nm laser peak intensity from spectrum."""
    #from scipy.signal import find_peaks

    peaks, properties = find_peaks(spectrum, height=10, prominence=5, distance=5)

    if len(peaks) == 0:
        return -1.0

    peak_wls = wl[peaks]
    peak_heights = properties["peak_heights"]

    laser_mask = (peak_wls >= LASER_WAVELENGTH_MIN) & (peak_wls <= LASER_WAVELENGTH_MAX)
    laser_indices = np.where(laser_mask)[0]

    if len(laser_indices) == 0:
        return -1.0

    laser_idx = laser_indices[np.argmax(peak_heights[laser_indices])]
    return float(peak_heights[laser_idx])

# ============================================================================
# PHASE 1a: SYMMETRIC SCAN ± 10V @ 1V STEPS
# ============================================================================

def phase1a_symmetric_scan(center_x, center_y, grating, exposure_s, center_wl):
    """Phase 1a: Scan ±10V from current voltage at 1V steps."""
    _log_debug(f"\n[PHASE 1a] Symmetric scan ±{PHASE1A_OFFSET}V @ {PHASE1A_STEP}V steps")

    current_v = get_z_voltage()
    if current_v is None:
        _log_debug("ERROR: Could not read current voltage")
        return None
    #cv_f=str(current_v)

    _log_debug(f"Current voltage: {current_v:.2f}V")

    v_min = np.clip(current_v - PHASE1A_OFFSET, Z_MIN_VOLTAGE, Z_MAX_VOLTAGE)
    v_max = np.clip(current_v + PHASE1A_OFFSET, Z_MIN_VOLTAGE, Z_MAX_VOLTAGE)

    num_points = int(round((v_max - v_min) / PHASE1A_STEP)) + 1
    voltages = np.linspace(v_min, v_max, num_points)
    intensities = []

    _log_debug(f"Scanning voltage range: {v_min:.2f}V to {v_max:.2f}V ({num_points} points)")

    lf_spec.lf_setup(exposure_s=exposure_s, center_wavelength=center_wl, grating=grating)

    for i, voltage in enumerate(voltages):
        v = float(voltage)
        _log_debug(f"  [{i+1}/{len(voltages)}] Setting voltage to {v:.2f}V...")
        
        if not set_z_voltage(v):
            _log_debug(f"    ERROR: Could not set voltage; skipping.")
            intensities.append(-1.0)
            continue

        time.sleep(INTEGRATION_TIME_S)

        try:
            _log_debug(f"    Acquiring spectrum...")
            intensity_data, wl = lf_spec.lf_acquire()

            peak_intensity = get_532nm_peak_intensity(intensity_data, wl)

            _log_debug(f"    532nm intensity: {peak_intensity:.1f}")
            intensities.append(peak_intensity)

        except Exception as e:
            _log_debug(f"    ERROR: Acquisition failed: {e}")
            intensities.append(-1.0)

    valid_mask = np.array(intensities) >= 0
    if not np.any(valid_mask):
        _log_debug("ERROR: No valid spectra acquired in Phase 1a")
        return None

    valid_voltages = voltages[valid_mask]
    valid_intensities = np.array(intensities)[valid_mask]

    peak_idx = np.argmax(valid_intensities)
    peak_voltage = float(valid_voltages[peak_idx])
    peak_intensity = float(valid_intensities[peak_idx])

    peak_at_boundary = False
    boundary_direction = None

    if peak_idx == 0:
        peak_at_boundary = True
        boundary_direction = 'negative'
        _log_debug(f"Peak found at lower boundary ({peak_voltage:.2f}V) — will extend downward")
    elif peak_idx == len(valid_voltages) - 1:
        peak_at_boundary = True
        boundary_direction = 'positive'
        _log_debug(f"Peak found at upper boundary ({peak_voltage:.2f}V) — will extend upward")
    else:
        _log_debug(f"Peak found in interior at {peak_voltage:.2f}V — proceeding to Phase 2")

    return {
        'voltages': list(valid_voltages),
        'intensities': list(valid_intensities),
        'peak_voltage': peak_voltage,
        'peak_intensity': peak_intensity,
        'peak_at_boundary': peak_at_boundary,
        'boundary_direction': boundary_direction,
    }

# ============================================================================
# PHASE 1b: DIRECTIONAL EXTENSION (IF PEAK AT BOUNDARY)
# ============================================================================

def phase1b_directional_scan(center_x, center_y, grating, exposure_s, center_wl,
                              boundary_voltage, boundary_direction):
    """Phase 1b: Extend in direction of boundary until peak found."""
    _log_debug(f"\n[PHASE 1b] Directional scan from {boundary_voltage:.2f}V in '{boundary_direction}' direction")

    direction_sign = 1 if boundary_direction == 'positive' else -1

    scan_voltages = []
    scan_intensities = []
    previous_valid_intensity = None
    peak_found = False
    peak_voltage = boundary_voltage
    peak_intensity = -1.0

    step_count = 0
    max_steps = int(round((Z_MAX_VOLTAGE - Z_MIN_VOLTAGE) / PHASE1B_STEP)) + 1

    lf_spec.lf_setup(exposure_s=exposure_s, center_wavelength=center_wl, grating=grating)

    while step_count < max_steps and not peak_found:
        voltage = boundary_voltage + (step_count + 1) * PHASE1B_STEP * direction_sign

        if voltage < Z_MIN_VOLTAGE or voltage > Z_MAX_VOLTAGE:
            _log_debug(f"Reached voltage boundary ({voltage:.2f}V); stopping search")
            peak_found = True
            if scan_intensities:
                peak_voltage = float(scan_voltages[-1])
                peak_intensity = float(scan_intensities[-1])
            break

        _log_debug(f"  Step {step_count + 1}: Setting voltage to {voltage:.2f}V...")

        if not set_z_voltage(voltage):
            _log_debug(f"    ERROR: Could not set voltage; skipping")
            step_count += 1
            continue

        time.sleep(INTEGRATION_TIME_S)

        try:
            _log_debug(f"    Acquiring spectrum...")
            intensity_data, wl = lf_spec.lf_acquire()

            intensity = get_532nm_peak_intensity(intensity_data, wl)

            _log_debug(f"    532nm intensity: {intensity:.1f}")

            if intensity >= 0:
                if previous_valid_intensity is not None and intensity < previous_valid_intensity:
                    _log_debug(f"Peak detected: intensity declined from {previous_valid_intensity:.1f} to {intensity:.1f}")
                    peak_found = True
                    # Use previous voltage/intensity as peak (it was higher)
                    peak_voltage = float(scan_voltages[-1])
                    peak_intensity = float(scan_intensities[-1])
                else:
                    # Still ascending or first measurement, keep tracking
                    previous_valid_intensity = intensity
                    peak_voltage = float(voltage)
                    peak_intensity = float(intensity)

                scan_voltages.append(voltage)
                scan_intensities.append(intensity)

            step_count += 1

        except Exception as e:
            _log_debug(f"    ERROR: Acquisition failed: {e}")
            step_count += 1
            continue

    if len(scan_intensities) == 0:
        _log_debug("ERROR: No valid spectra acquired in Phase 1b")
        return None

    _log_debug(f"Coarse peak found at {peak_voltage:.2f}V (intensity: {peak_intensity:.1f})")

    return {
        'peak_voltage': peak_voltage,
        'peak_intensity': peak_intensity,
        'scan_voltages': scan_voltages,
        'scan_intensities': scan_intensities,
    }

# ============================================================================
# PHASE 2: FINE SCAN ± 1V @ 0.1V STEPS
# ============================================================================

def phase2_fine_scan(center_x, center_y, grating, exposure_s, center_wl, coarse_peak_voltage):
    """Phase 2: Fine scan ±1V around coarse peak at 0.1V steps."""
    
    coarse_peak_voltage = float(coarse_peak_voltage)

    _log_debug(f"\n[PHASE 2] Fine scan ±{PHASE2_OFFSET}V @ {PHASE2_STEP}V steps around {coarse_peak_voltage:.2f}V")

    v_min = np.clip(coarse_peak_voltage - PHASE2_OFFSET, Z_MIN_VOLTAGE, Z_MAX_VOLTAGE)
    v_max = np.clip(coarse_peak_voltage + PHASE2_OFFSET, Z_MIN_VOLTAGE, Z_MAX_VOLTAGE)

    num_points = int(round((v_max - v_min) / PHASE2_STEP)) + 1
    voltages = np.linspace(v_min, v_max, num_points)
    intensities = []

    _log_debug(f"Scanning voltage range: {v_min:.2f}V to {v_max:.2f}V ({num_points} points)")

    lf_spec.lf_setup(exposure_s=exposure_s, center_wavelength=center_wl, grating=grating)

    for i, voltage in enumerate(voltages):
        v = float(voltage)
        _log_debug(f"  [{i+1}/{len(voltages)}] Setting voltage to {v:.2f}V...")

       # voltage = Decimal(float(voltage))

        if not set_z_voltage(v):
            _log_debug(f"    ERROR: Could not set voltage; skipping")
            intensities.append(-1.0)
            continue

        time.sleep(INTEGRATION_TIME_S)

        try:
            _log_debug(f"    Acquiring spectrum...")
            intensity_data, wl = lf_spec.lf_acquire()

            peak_intensity = get_532nm_peak_intensity(intensity_data, wl)

            _log_debug(f"    532nm intensity: {peak_intensity:.1f}")
            intensities.append(peak_intensity)

        except Exception as e:
            _log_debug(f"    ERROR: Acquisition failed: {e}")
            intensities.append(-1.0)

    valid_mask = np.array(intensities) >= 0
    if not np.any(valid_mask):
        _log_debug("ERROR: No valid spectra acquired in Phase 2")
        return None

    valid_voltages = voltages[valid_mask]
    valid_intensities = np.array(intensities)[valid_mask]

    peak_idx = np.argmax(valid_intensities)
    peak_voltage = float(valid_voltages[peak_idx])
    peak_intensity = float(valid_intensities[peak_idx])

    _log_debug(f"Fine peak found at {peak_voltage:.2f}V (intensity: {peak_intensity:.1f})")

    return {
        'peak_voltage': peak_voltage,
        'peak_intensity': peak_intensity,
        'voltages': list(valid_voltages),
        'intensities': list(valid_intensities),
    }

# ============================================================================
# MAIN AUTOFOCUS ROUTINE
# ============================================================================

def autofocus_on_emitter(emitter_pos, grating, exposure_s, center_wl,
                         current_user=None, foldername=None):
    """Main autofocus routine: Phase 1a → Phase 1b (if needed) → Phase 2 → Lock focus."""
    if not _autofocus_enabled:
        _log_debug("WARNING: Autofocus not initialized")
        return None

    ex, ey = emitter_pos
    _log_debug(f"\n{'='*70}")
    _log_debug(f"AUTOFOCUS on emitter at ({ex:.2f}, {ey:.2f}) um")
    _log_debug(f"{'='*70}")

    phase1a_result = phase1a_symmetric_scan(ex, ey, grating, exposure_s, center_wl)

    if phase1a_result is None:
        error_msg = "Autofocus Phase 1a failed: No spectra acquired"
        _log_debug(f"ERROR: {error_msg}")
        if current_user:
            psp._send_telegram(current_user, f"⚠️ Focus Error: {error_msg}")
        return None

    coarse_peak_voltage = phase1a_result['peak_voltage']

    if phase1a_result['peak_at_boundary']:
        phase1b_result = phase1b_directional_scan(
            ex, ey, grating, exposure_s, center_wl,
            phase1a_result['peak_voltage'],
            phase1a_result['boundary_direction']
        )

        if phase1b_result is None:
            error_msg = "Autofocus Phase 1b failed: No spectra acquired during directional scan"
            _log_debug(f"ERROR: {error_msg}")
            if current_user:
                psp._send_telegram(current_user, f"⚠️ Focus Error: {error_msg}")
            return None

        coarse_peak_voltage = phase1b_result['peak_voltage']

    phase2_result = phase2_fine_scan(
        ex, ey, grating, exposure_s, center_wl,
        coarse_peak_voltage
    )

    if phase2_result is None:
        error_msg = "Autofocus Phase 2 failed: No spectra acquired during fine scan"
        _log_debug(f"ERROR: {error_msg}")
        if current_user:
            psp._send_telegram(current_user, f"⚠️ Focus Error: {error_msg}")
        return None

    final_voltage = phase2_result['peak_voltage']
    final_intensity = phase2_result['peak_intensity']

    _log_debug(f"\n[LOCK] Setting Z-voltage to {final_voltage:.2f}V...")

    if not set_z_voltage(final_voltage):
        error_msg = f"Failed to lock focus at {final_voltage:.2f}V"
        _log_debug(f"ERROR: {error_msg}")
        if current_user:
            psp._send_telegram(current_user, f"⚠️ Focus Error: {error_msg}")
        return None

    time.sleep(0.5)
    _log_debug(f"✓ Focus locked at {final_voltage:.2f}V (532nm intensity: {final_intensity:.1f})")

    _log_results(emitter_pos, final_voltage, final_intensity, 'locked')
    _log_debug(f"{'='*70}\n")

    return {
        'voltage': final_voltage,
        'intensity': final_intensity,
        'success': True,
    }

# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    """Quick test: initialize Z-stage and verify connection."""
    print("Testing autofocus module...")
    lf_spec.lf_connect()

    if autofocus_init():
        print("✓ Z-stage initialized successfully.")
        current_v = get_z_voltage()
        print(f"✓ Current voltage: {current_v} V")

        if set_z_voltage(current_v):
            print("✓ Successfully set voltage")
            time.sleep(0.5)
            new_v = get_z_voltage()
            print(f"✓ Verified voltage: {new_v} V")

            autofocus_on_emitter((0, 0), 150, INTEGRATION_TIME_S, 700,
                         current_user=None, foldername=None)

        autofocus_shutdown()
        print("✓ Z-stage shutdown complete.")
    else:
        print("✗ Failed to initialize Z-stage.")
