"""
One-off diagnostic: sweep the PicoHarp CFD discriminator level and print the
resulting count rates on both channels, so we can match the rates shown in
the PicoHarp software (without needing to read its CFD settings directly).

Close the PicoHarp software before running this (only one process can hold
the device), and don't touch the alignment between runs.
"""
import ctypes
import time

PHLIB = ctypes.windll.LoadLibrary("PHLib64.dll")
DEV_IDX = 0
MODE_T2 = 2

ZEROCROSS_MV = 10          # held fixed while sweeping level
LEVELS_MV    = range(50, 210, 10)   # 50, 60, ..., 200

serial = ctypes.create_string_buffer(8)
if PHLIB.PH_OpenDevice(DEV_IDX, serial) != 0:
    raise RuntimeError("Could not open PicoHarp. Check USB connection and drivers.")
if PHLIB.PH_Initialize(DEV_IDX, MODE_T2) < 0:
    raise RuntimeError("PH_Initialize failed.")
if PHLIB.PH_Calibrate(DEV_IDX) < 0:
    raise RuntimeError("PH_Calibrate failed.")

PHLIB.PH_SetSyncDiv(DEV_IDX, 1)
PHLIB.PH_SetBinning(DEV_IDX, 0)
PHLIB.PH_SetOffset(DEV_IDX, 0)

print(f"{'Level (mV)':>10} | {'Ch0 (cps)':>12} | {'Ch1 (cps)':>12} | {'ret0':>5} | {'ret1':>5}")
print("-" * 60)

r0 = ctypes.c_int32(0)
r1 = ctypes.c_int32(0)

for level in LEVELS_MV:
    ret0 = PHLIB.PH_SetInputCFD(DEV_IDX, 0, level, ZEROCROSS_MV)
    ret1 = PHLIB.PH_SetInputCFD(DEV_IDX, 1, level, ZEROCROSS_MV)

    time.sleep(0.5)  # let the count rate meter settle after changing CFD

    PHLIB.PH_GetCountRate(DEV_IDX, 0, ctypes.byref(r0))
    PHLIB.PH_GetCountRate(DEV_IDX, 1, ctypes.byref(r1))

    print(f"{level:>10} | {r0.value:>12.2e} | {r1.value:>12.2e} | {ret0:>5} | {ret1:>5}")

PHLIB.PH_CloseDevice(DEV_IDX)
print("\nDone. Pick the level whose Ch0/Ch1 rates are closest to "
      "the PicoHarp software's reading, then tell me which level to use.")
