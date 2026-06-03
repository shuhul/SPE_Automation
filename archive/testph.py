import ctypes
import time

# 1. Setup Library and Constants
phlib = ctypes.windll.LoadLibrary("PHLib64.dll")
MODE_HIST = 0
MODE_T2 = 2
MODE_T3 = 3
dev_idx = 0  # device #0

# 2. Open the device
serial = ctypes.create_string_buffer(8)
if phlib.PH_OpenDevice(dev_idx, serial) != 0:
    raise RuntimeError("Could not open PicoHarp")

# 3. Basic Initialize
# This sets the mode (Histogramming)
if phlib.PH_Initialize(dev_idx, MODE_T2) < 0:
    raise RuntimeError("PH_Initialize failed")

# 4. Calibration (Essential for PH300 timing)
if phlib.PH_Calibrate(dev_idx) < 0:
    raise RuntimeError("Calibration failed")

# 5. Set Hardware Thresholds (CFD Settings)
# These match your MATLAB file's CFDLevel and CFDZeroX
phlib.PH_SetInputCFD(dev_idx, 0, 150, 10) # Channel 0
phlib.PH_SetInputCFD(dev_idx, 1, 150, 10) # Channel 1

# 6. Set Measurement Parameters
phlib.PH_SetSyncDiv(dev_idx, 1)
phlib.PH_SetBinning(dev_idx, 0)
phlib.PH_SetOffset(dev_idx, 0)

# 7. Get Resolution (Good practice to verify connection)
resolution = ctypes.c_double()
phlib.PH_GetResolution(dev_idx, ctypes.byref(resolution))
print(f"PicoHarp Ready. Resolution: {resolution.value} ps")


def get_pl(phlib, device_index):
    """
    Equivalent to GetPL(device) in MATLAB.
    Queries the current count rate for Channel 0 and Channel 1.
    """
    # Create C-compatible integer objects to hold the results
    count_rate_0 = ctypes.c_int32(0)
    count_rate_1 = ctypes.c_int32(0)
    
    # Call PH_GetCountRate for Channel 0
    # phlib.PH_GetCountRate(device, channel, pointer_to_result)
    ret0 = phlib.PH_GetCountRate(device_index, 0, ctypes.byref(count_rate_0))
    
    # Call PH_GetCountRate for Channel 1
    ret1 = phlib.PH_GetCountRate(device_index, 1, ctypes.byref(count_rate_1))
    
    # Check for errors (optional but recommended)
    if ret0 < 0 or ret1 < 0:
        print(f"Warning: Error reading count rates (Error codes: {ret0}, {ret1})")
    
    # Return the values as a tuple, just like MATLAB returns [Countrate0, Countrate1]
    return count_rate_0.value, count_rate_1.value

time.sleep(1)

print('Running PL')

rate0, rate1 = get_pl(phlib, 0)

print(rate0)
print(rate1)

print('Done')