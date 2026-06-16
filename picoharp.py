"""PicoHarp 300 T2 mode acquisition via ctypes DLL."""
import ctypes
import time
import sys
import os
import struct
import numpy as np
from datetime import datetime

PHLIB = None
DEV_IDX       = 0
MODE_T2       = 2
TTREADMAX     = 131072
FLAG_FIFOFULL = 0x0003
RESOLUTION_PS = 4
T2_WRAPAROUND = 2 ** 28

# Fixed CFD settings for this setup
_CFD_LEVEL_MV     = 150
_CFD_ZEROCROSS_MV = 10


def _write_minimal_ptu_header(f, n_records, resolution_s=4e-12):
    """
    Write a minimal but valid PicoQuant PTU header so the file can be
    opened by SymPhoTime, our Python parsers, and the MATLAB scripts.

    Only writes the tags that downstream parsers actually need.
    Everything else (e.g. hardware serial numbers) is left at defaults.
    """
    def write_tag(ident, idx, typ, value_bytes):
        padded = ident.encode('ascii')[:32].ljust(32, b'\x00')
        f.write(padded)
        f.write(struct.pack('<iI', idx, typ))
        f.write(value_bytes)

    TY_INT8      = 0x10000008
    TY_FLOAT8    = 0x20000008
    TY_ANSISTR   = 0x4001FFFF
    TY_EMPTY8    = 0xFFFF0008
    RT_PICOHARP_T2 = 0x00010203

    # Magic + version
    f.write(b'PQTTTR\x00\x00')
    f.write(b'1.0.00\x00\x00')

    # Required tags
    write_tag('TTResultFormat_TTTRRecType', -1, TY_INT8,
              struct.pack('<q', RT_PICOHARP_T2))
    write_tag('TTResult_NumberOfRecords', -1, TY_INT8,
              struct.pack('<q', n_records))
    write_tag('MeasDesc_Resolution', -1, TY_FLOAT8,
              struct.pack('<d', resolution_s))
    write_tag('MeasDesc_GlobalResolution', -1, TY_FLOAT8,
              struct.pack('<d', resolution_s))

    # Header end marker
    write_tag('Header_End', -1, TY_EMPTY8, struct.pack('<q', 0))

def ph_init():
    global PHLIB
    PHLIB = ctypes.windll.LoadLibrary("PHLib64.dll")

    serial = ctypes.create_string_buffer(8)
    if PHLIB.PH_OpenDevice(DEV_IDX, serial) != 0:
        raise RuntimeError("Could not open PicoHarp. Check USB connection and drivers.")
    if PHLIB.PH_Initialize(DEV_IDX, MODE_T2) < 0:
        raise RuntimeError("PH_Initialize failed.")
    if PHLIB.PH_Calibrate(DEV_IDX) < 0:
        raise RuntimeError("PH_Calibrate failed.")

    PHLIB.PH_SetInputCFD(DEV_IDX, 0, _CFD_LEVEL_MV, _CFD_ZEROCROSS_MV)
    PHLIB.PH_SetInputCFD(DEV_IDX, 1, _CFD_LEVEL_MV, _CFD_ZEROCROSS_MV)
    PHLIB.PH_SetSyncDiv(DEV_IDX, 1)
    PHLIB.PH_SetBinning(DEV_IDX, 0)
    PHLIB.PH_SetOffset(DEV_IDX, 0)

    PHLIB.PH_ReadFiFo.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int, ctypes.POINTER(ctypes.c_int)
    ]
    PHLIB.PH_ReadFiFo.restype = ctypes.c_int

    print("PicoHarp initialized.")


def ph_close():
    if PHLIB is not None:
        PHLIB.PH_CloseDevice(DEV_IDX)
        print("PicoHarp disconnected.")


def get_count_rates():
    """Return (ch0_cps, ch1_cps) count rates."""
    r0 = ctypes.c_int32(0)
    r1 = ctypes.c_int32(0)
    PHLIB.PH_GetCountRate(DEV_IDX, 0, ctypes.byref(r0))
    PHLIB.PH_GetCountRate(DEV_IDX, 1, ctypes.byref(r1))
    return r0.value, r1.value


def ph_acquire(target_records, acq_time_s=10000, out_folder='g2_data', progress_signal=None,
               stop_flag=None, save_ptu=True):
    """
    Run T2 acquisition until target_records collected or acq_time_s elapsed.
    Parses raw TTTR records and saves ch0/ch1 photon times to a timestamped .npz.

    Args:
        target_records  : stop when this many TTTR records have been collected
        acq_time_s      : hard time limit in seconds (safety cutoff)
        out_folder      : directory to save output .npz
        progress_signal : optional PyQt signal(int) for GUI progress updates
        stop_flag       : optional callable; if it returns True, stop early
                          and save whatever has been collected so far.

    Returns:
        Path to the saved .npz file, or None if no data was collected.
    """
    global PHLIB
    os.makedirs(out_folder, exist_ok=True)

    buffer     = (ctypes.c_uint32 * TTREADMAX)()
    flags      = ctypes.c_int32(0)
    ctc_status = ctypes.c_int32(0)
    nactual    = ctypes.c_int(0)

    total_records = 0
    record_chunks = []
    start_time    = time.time()

    PHLIB.PH_StartMeas(DEV_IDX, acq_time_s * 1000)
    print(f"Acquiring: target={target_records:,} records, limit={acq_time_s}s")

    try:
        while True:
            if stop_flag is not None and stop_flag():
                print("\nEmergency stop requested — stopping G2 acquisition early.")
                break

            PHLIB.PH_ReadFiFo(
                DEV_IDX,
                ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint32)),
                TTREADMAX,
                ctypes.byref(nactual)
            )
            n = nactual.value
            if n > 0:
                record_chunks.append(np.frombuffer(buffer, dtype=np.uint32, count=n).copy())
                total_records += n

            if progress_signal is not None:
                progress_signal.emit(int(min(total_records / target_records * 100, 99)))
            else:
                elapsed = time.time() - start_time
                r0, r1 = get_count_rates()
                sys.stdout.write(
                    f"\r  {elapsed:.0f}s | Ch0: {r0:.2e} cps | Ch1: {r1:.2e} cps"
                    f" | Records: {total_records:,}/{target_records:,}"
                )
                sys.stdout.flush()

            if total_records >= target_records:
                print("\nTarget record count reached.")
                break

            PHLIB.PH_CTCStatus(DEV_IDX, ctypes.byref(ctc_status))
            if ctc_status.value != 0:
                print("\nAcquisition time limit reached.")
                break

            PHLIB.PH_GetFlags(DEV_IDX, ctypes.byref(flags))
            if flags.value & FLAG_FIFOFULL:
                print("\nWarning: FIFO overrun — data loss. Stopping.")
                break

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nMeasurement interrupted by user.")

    PHLIB.PH_StopMeas(DEV_IDX)

    if not record_chunks:
        print("No records captured.")
        return None

    raw = np.concatenate(record_chunks)
    if len(raw) > target_records:
        raw = raw[:target_records]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#  Save raw PTU file
    if save_ptu:
       ptu_path = os.path.join(out_folder, f"{timestamp}.ptu")
       with open(ptu_path, 'wb') as f:
           _write_minimal_ptu_header(f, len(raw), resolution_s=RESOLUTION_PS * 1e-12)
           raw.tofile(f)
       print(f"Saved raw PTU: '{ptu_path}' ({len(raw):,} records)")

    # Parse T2 records into absolute photon times
    channel_field = (raw >> 28) & 0xF
    timetag_field =  raw        & 0x0FFFFFFF

    overflow_mask        = channel_field == 0xF
    overflow_increments  = np.where(overflow_mask,
                                    np.where(timetag_field == 0, 1, timetag_field),
                                    0)
    cum_overflow = np.cumsum(overflow_increments) - overflow_increments
    abs_time_ps  = (cum_overflow * T2_WRAPAROUND + timetag_field).astype(np.int64) * RESOLUTION_PS

    photon_mask = ~overflow_mask
    channels    = channel_field[photon_mask]
    times_ps    = abs_time_ps[photon_mask]

    ch0 = times_ps[channels == 0]
    ch1 = times_ps[channels == 1]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    npz_path  = os.path.join(out_folder, f"{timestamp}.npz")
    np.savez(npz_path, ch0=ch0, ch1=ch1)
    print(f"Saved {len(ch0):,} ch0 + {len(ch1):,} ch1 photons → '{npz_path}'")

    return npz_path
