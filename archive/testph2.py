import ctypes
import time
import sys
import os
from datetime import datetime
import numpy as np

# =============================================================================
# 1. Hardware Constants & Acquisition Settings
# =============================================================================

# Load the 64-bit PicoHarp DLL (Must be in the same folder or System PATH)
phlib = ctypes.windll.LoadLibrary("PHLib64.dll")

DEV_IDX = 0              # Device index (0 if you only have one PicoHarp plugged in)
MODE_T2 = 2              # T2 Mode: Records absolute arrival times of all photons on all channels independently
TTREADMAX = 131072       # Maximum number of records the hardware FIFO buffer can transfer to the PC in one read
FLAG_FIFOFULL = 0x0003   # Hardware flag indicating the FIFO buffer overflowed (data was lost)

# User-defined stopping conditions
ACQ_TIME_S = 10000       # Maximum measurement time in seconds

TARGET_RECORDS = 100000 # Measurement will stop exactly when this many records are collected

OUTPUT_DIR = "data_ph"  # Folder where .npz files are saved

# =============================================================================
# 2. Hardware Initialization & Configuration
# =============================================================================

# Open connection to the device
serial = ctypes.create_string_buffer(8)
if phlib.PH_OpenDevice(DEV_IDX, serial) != 0:
    raise RuntimeError("Could not open PicoHarp. Check USB connection and drivers.")

# Initialize the device specifically for T2 Mode
if phlib.PH_Initialize(DEV_IDX, MODE_T2) < 0:
    raise RuntimeError("PH_Initialize failed.")

# Calibrate the internal timing circuitry (Must be called after init and before measuring)
if phlib.PH_Calibrate(DEV_IDX) < 0:
    raise RuntimeError("Calibration failed.")

# --- Constant Fraction Discriminator (CFD) Settings ---
# The CFD isolates true photon pulses from electronic noise.
# Parameter 1: Device Index
# Parameter 2: Channel (0 is Sync/Ch0, 1 is Ch1)
# Parameter 3: CFD Level in millivolts (mV). The signal must drop below this threshold to be registered.
# Parameter 4: CFD ZeroCross in millivolts (mV). The precise timing point is taken when the signal crosses this value.
phlib.PH_SetInputCFD(DEV_IDX, 0, 150, 10) 
phlib.PH_SetInputCFD(DEV_IDX, 1, 150, 10) 

# --- Timing & Binning Settings ---
# Sync Divider: Divides the Sync channel frequency. Useful if the laser rate exceeds the 84 MHz hardware limit.
# 1 means no division (every pulse is counted).
phlib.PH_SetSyncDiv(DEV_IDX, 1)

# Binning: Defines the base timing resolution.
# 0 means base resolution (typically 4 ps for PicoHarp 300).
# Increasing this value doubles the bin width (e.g., 1 = 8 ps, 2 = 16 ps).
phlib.PH_SetBinning(DEV_IDX, 0)

# Offset: Adds a constant time shift (in picoseconds) to Channel 1 relative to Channel 0.
# Useful for compensating for different cable lengths.
phlib.PH_SetOffset(DEV_IDX, 0)

# =============================================================================
# 3. Helper Functions
# =============================================================================

def get_pl(phlib, device_index):
    """Queries the hardware for the current count rates (photons per second) on both channels."""
    rate0 = ctypes.c_int32(0)
    rate1 = ctypes.c_int32(0)
    phlib.PH_GetCountRate(device_index, 0, ctypes.byref(rate0))
    phlib.PH_GetCountRate(device_index, 1, ctypes.byref(rate1))
    return rate0.value, rate1.value

# =============================================================================
# 4. Acquisition Loop
# =============================================================================

print(f"Starting T2 Measurement. Max Time: {ACQ_TIME_S}s, Target Records: {TARGET_RECORDS}")

# C-compatible variables for reading hardware state
buffer = (ctypes.c_uint32 * TTREADMAX)()
flags = ctypes.c_int32(0)
ctc_status = ctypes.c_int32(0)

total_records = 0
record_chunks = []
start_time = time.time()

# Start the hardware timer and measurement (PH_StartMeas expects milliseconds)
phlib.PH_StartMeas(DEV_IDX, ACQ_TIME_S * 1000)

# Explicitly define the C function signature so the 64-bit pointer isn't truncated
phlib.PH_ReadFiFo.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
phlib.PH_ReadFiFo.restype = ctypes.c_int

nactual = ctypes.c_int(0)

try:
    while True:
        # Read the current contents of the hardware FIFO buffer into our Python 'buffer'
        phlib.PH_ReadFiFo(DEV_IDX, ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint32)), TTREADMAX, ctypes.byref(nactual))
        records_read = nactual.value
        
        if records_read > 0:
            # Convert the raw memory directly into a NumPy array of unsigned 32-bit integers
            # .copy() is required because the C-buffer will be overwritten on the next loop iteration
            chunk = np.frombuffer(buffer, dtype=np.uint32, count=records_read).copy()
            record_chunks.append(chunk)
            total_records += records_read
            
        # Update console status
        elapsed = time.time() - start_time
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        rate0, rate1 = get_pl(phlib, DEV_IDX)
        
        sys.stdout.write(f"\rElapsed Time: {elapsed_str} | Ch 0: {rate0:.2e} | Ch 1: {rate1:.2e} | Records: {total_records}")
        sys.stdout.flush()
        
        # Check if we hit the user-defined record limit
        if total_records >= TARGET_RECORDS:
            print("\n\nTarget record limit reached.")
            break
            
        # Check if the hardware timer finished (reached ACQ_TIME_S)
        phlib.PH_CTCStatus(DEV_IDX, ctypes.byref(ctc_status))
        if ctc_status.value != 0:
            print("\n\nAcquisition timer completed.")
            break
            
        # Check for data loss
        phlib.PH_GetFlags(DEV_IDX, ctypes.byref(flags))
        if (flags.value & FLAG_FIFOFULL) > 0:
            print("\n\nWarning: FIFO Full (Overrun). Data is arriving faster than the USB can transfer it.")
            break
            
        # Sleep briefly to prevent this loop from using 100% of a CPU core
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n\nMeasurement interrupted by user.")

# =============================================================================
# 5. Cleanup and File Saving
# =============================================================================

# Stop the hardware and close the connection
phlib.PH_StopMeas(DEV_IDX)
phlib.PH_CloseDevice(DEV_IDX)

# Combine all raw record chunks into one array
if record_chunks:
    raw = np.concatenate(record_chunks)

    # Trim to exact target if buffer chunking slightly overshot
    if len(raw) > TARGET_RECORDS:
        raw = raw[:TARGET_RECORDS]

    # --- Parse T2 records ---
    # Bit layout: bits 31-28 = channel, bits 27-0 = timetag (4 ps per unit)
    RESOLUTION_PS = 4
    T2_WRAPAROUND = 2**28

    channel_field = (raw >> 28) & 0xF
    timetag_field =  raw        & 0x0FFFFFFF

    overflow_mask = channel_field == 0xF
    overflow_increments = np.where(overflow_mask,
                                   np.where(timetag_field == 0, 1, timetag_field),
                                   0)
    cum_overflow = np.cumsum(overflow_increments) - overflow_increments
    abs_time_ps = (cum_overflow * T2_WRAPAROUND + timetag_field).astype(np.int64) * RESOLUTION_PS

    photon_mask = ~overflow_mask
    channels = channel_field[photon_mask]
    times_ps  = abs_time_ps[photon_mask]

    ch0_times = times_ps[channels == 0]
    ch1_times = times_ps[channels == 1]

    # --- Save to data_ph/<timestamp>.npz ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"{timestamp}.npz")

    np.savez(output_path, ch0=ch0_times, ch1=ch1_times)
    print(f"Hardware closed. Saved {len(ch0_times):,} ch0 + {len(ch1_times):,} ch1 photon times to '{output_path}'.")
else:
    print("Hardware closed. No records were captured.")