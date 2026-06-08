"""
Full SPE automation: coarse scan -> fine scan -> long scan -> bandpass filter setup.
G2 measurement is not included yet.
Stop with Ctrl+C — the current acquisition finishes cleanly before exiting.
"""

# Set True to open a live heatmap window after each scan; 
# False to run silently (data and PNGs still saved).
# IF TRUE THEN MANUAL INTERACTION IS REQUIRED
MANUAL_PLOT_INTERACTION = True


import os
import signal
import threading
import time
import numpy as np
from datetime import datetime

import matplotlib
if not MANUAL_PLOT_INTERACTION:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import lf_spec
import sgd
import filter as fil
import plotter
import pl_spec_python as psp
import autofocus

# ============================================================================
# PARAMETERS — edit these before each session
# ============================================================================

FOLDERNAME   = datetime.now().strftime('%Y%m%d') + '-PLSPC-HT-Ch4-f1-500uW-1s-fullauto'
CURRENT_USER = 'kristina'
DATA_FOLDER  = 'data'
CAL_FOLDER   = '2026-05-28_18-08-36'   # bandpass calibration subfolder name

# Coarse scan — wide area to locate candidate emitters
COARSE_XDIM       = 20   # um
COARSE_YDIM       = 20   # um
COARSE_DX         = 0.5    # um step size
COARSE_DY         = 0.5
COARSE_CENTER     = (0, 0)
COARSE_GRATING    = 150
COARSE_EXPOSURE_S = 1.0
COARSE_CENTER_WL  = 700    # nm

# Fine scan — zoomed scan centred on each classified emitter
FINE_XDIM         = 3 # 3.0
FINE_YDIM         = 3 # 3.0
FINE_DX           = 0.25 # 0.25
FINE_DY           = 0.25 # 0.25
FINE_GRATING      = 600
FINE_EXPOSURE_S   = 1.0
FINE_CENTER_WL    = 595

# Long scan — single-point, high-exposure spectrum to measure ZPL precisely
LONG_GRATING      = 600
LONG_EXPOSURE_S   = 10.0


# ============================================================================
# STOP FLAG — set by Ctrl+C, checked between acquisitions
# ============================================================================

_stop = False

def _handle_stop(sig, frame):
    global _stop
    print('\nStop requested — finishing current acquisition and cleaning up...')
    _stop = True

signal.signal(signal.SIGINT,  _handle_stop)
signal.signal(signal.SIGTERM, _handle_stop)

# ============================================================================
# EMERGENCY STOP — Ctrl+X for immediate stop with partial data save
# ============================================================================

_stop_immediately = False
_keyboard_monitor_running = False
_monitor_thread = None
_filter_is_up = False

def _keyboard_monitor():
    """Monitor keyboard for Ctrl+X to trigger emergency stop."""
    global _stop_immediately
    try:
        import msvcrt
        print('[INFO] Keyboard monitor started (Ctrl+X = emergency stop, q = quit)')
        while _keyboard_monitor_running:
            try:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    # Try multiple key combos for emergency stop
                    # Ctrl+X = 0x18, Ctrl+Break = 0xE0 0x93, or also allow 'q'
                    if key == b'\x18':  # Ctrl+X
                        print('\n[CTRL+X] EMERGENCY STOP — stopping immediately with partial data save...')
                        _stop_immediately = True
                    elif key.lower() in [b'q']:  # Fallback: 'q' for quit
                        print(f'\n[Q] Emergency stop requested...')
                        _stop_immediately = True
            except Exception as e:
                print(f'[INFO] Keyboard monitor read error: {e}')
            time.sleep(0.05)
        print('[INFO] Keyboard monitor stopped')
    except ImportError:
        print('[INFO] msvcrt not available - keyboard monitor disabled')
    except Exception as e:
        print(f'[ERROR] Keyboard monitor failed: {e}')

# ============================================================================
# SPECTRUM HELPERS
# ============================================================================

def find_emission_fwhm_center(spectrum, wl, laser_cutoff_nm=560):
    """Return the FWHM centre of the brightest emission peak above the laser cutoff.
    Falls back to peak wavelength if FWHM crossings cannot be found."""
    mask = wl > laser_cutoff_nm
    if not mask.any():
        return None
    wl_m, sp_m  = wl[mask], spectrum[mask]
    peak_idx    = int(np.argmax(sp_m))
    half_max    = sp_m[peak_idx] / 2.0
    left_below  = np.where(sp_m[:peak_idx] < half_max)[0]
    right_below = np.where(sp_m[peak_idx:] < half_max)[0]

    if left_below.size == 0 or right_below.size == 0:
        return float(wl_m[peak_idx])

    li = left_below[-1]
    x0, x1 = wl_m[li], wl_m[li + 1]
    y0, y1 = sp_m[li], sp_m[li + 1]
    left_wl = x0 + (half_max - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else (x0 + x1) / 2

    ri = peak_idx + right_below[0]
    x0, x1 = wl_m[ri - 1], wl_m[ri]
    y0, y1 = sp_m[ri - 1], sp_m[ri]
    right_wl = x0 + (half_max - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else (x0 + x1) / 2

    return float((left_wl + right_wl) / 2.0)


def _angle_for_wavelength(target_wl):
    """Look up the rotation stage angle for target_wl from the calibration table."""
    table_path = os.path.join('calibration', CAL_FOLDER, 'calibration_table.npy')
    if not os.path.exists(table_path):
        return None
    table  = np.load(table_path)
    valid  = ~np.isnan(table[:, 1])
    if not valid.any():
        return None
    angles, wls = table[valid, 0], table[valid, 1]
    return float(angles[np.argmin(np.abs(wls - target_wl))])


# ============================================================================
# PLOTTING
# ============================================================================

def save_spectrum_plot(folder_path, title=''):
    """Save the spectrum from a single-point (long) scan as a PNG."""
    out = np.load(os.path.join(folder_path, 'out.npy'))
    wl  = np.load(os.path.join(folder_path, 'wl.npy'))
    spectrum = out[0, 0, :]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(wl, spectrum, lw=1.5)

    # Annotate FWHM centre if detectable
    centre = find_emission_fwhm_center(spectrum, wl)
    if centre is not None:
        ax.axvline(centre, color='r', ls='--', lw=1.5, label=f'ZPL: {centre:.1f} nm')
        ax.axhline(spectrum[np.argmin(np.abs(wl - centre))] / 2,
                   color='orange', ls=':', lw=1, label='half max')
        ax.legend()

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity (counts)')
    ax.set_title(title or os.path.basename(folder_path))
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'spectrum.png'), dpi=150)
    plt.close(fig)


# ============================================================================
# SCAN
# ============================================================================

def run_scan(scan_type, center, xdim, ydim, dx, dy, grating, exposure_s, center_wl):
    """
    Run a spectral scan via pl_spec_python and save results to
    DATA_FOLDER/FOLDERNAME/scan_type/. Single-point: xdim=ydim=dx=dy=0.
    Returns (folder_path, status) where status is 'complete' or 'stopped'.
    """
    global _stop_immediately
    folder_path = os.path.join(DATA_FOLDER, FOLDERNAME, scan_type)

    try:
        status = psp.pl_spec_lf(
            xdim=xdim, ydim=ydim, dx=dx, dy=dy,
            center=center,
            grating=grating,
            exposure_time=exposure_s,
            center_wavelength=center_wl,
            foldername=FOLDERNAME,
            current_user=CURRENT_USER,
            scan_type=scan_type,
            data_folder=DATA_FOLDER,
            stop_immediately_flag=lambda: _stop_immediately,
        )
        return folder_path, status
    except KeyboardInterrupt:
        global _stop
        _stop = True
        print('\nScan interrupted.')
        return folder_path, 'stopped'

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    global _keyboard_monitor_running, _monitor_thread, _filter_is_up
    _filter_is_up = False

    # Start keyboard monitor thread
    _keyboard_monitor_running = True
    _monitor_thread = threading.Thread(target=_keyboard_monitor, daemon=True)
    _monitor_thread.start()

    print('=== SPE Automation ===')
    print(f'Folder: {FOLDERNAME}')
    print(f'Cal:    {CAL_FOLDER}')
    print()

    # ── Hardware init ─────────────────────────────────────────────────────────
    print('Initializing hardware...')
    lf_spec.lf_connect()
    sgd.sgd_init()
    fil.filter_init()
    fil.filter_on()
    af_available = autofocus.autofocus_init()
    if af_available:
        print('Autofocus Z-stage initialised.')
    else:
        print('[WARNING] Autofocus Z-stage could not be initialised — '
              'fine scans will proceed without autofocus.')
    print()

    # ── STEP 1: COARSE SCAN ───────────────────────────────────────────────────
    # Scans the full area to find candidate emitter positions.
    print(f'[STEP 1] Coarse scan  ({COARSE_XDIM}x{COARSE_YDIM} um, {COARSE_DX} um step)...')
    _, status = run_scan(
        scan_type='coarse',
        center=COARSE_CENTER,
        xdim=COARSE_XDIM,     ydim=COARSE_YDIM,
        dx=COARSE_DX,         dy=COARSE_DY,
        grating=COARSE_GRATING,
        exposure_s=COARSE_EXPOSURE_S,
        center_wl=COARSE_CENTER_WL,
    )
    if _stop or _stop_immediately:
        print('Stopped.')
        return

    # Load coarse results — keep out/wl in memory for ZPL estimates later
    coarse_path = os.path.join(DATA_FOLDER, FOLDERNAME, 'coarse')
    classified  = np.load(os.path.join(coarse_path, 'classified.npy'))
    xs_c        = np.load(os.path.join(coarse_path, 'xs.npy'))
    ys_c        = np.load(os.path.join(coarse_path, 'ys.npy'))

    # Auto-select emitters; only pause for user if multiple candidates
    iys, ixs = np.where(classified == 1)
    auto_emitters = [(xs_c[ix], ys_c[iy]) for ix, iy in zip(ixs, iys)]

    if len(auto_emitters) == 0:
        print('No emitters found in coarse scan. Done.')
        return
    elif len(auto_emitters) == 1:
        print(f'Found 1 emitter at {auto_emitters[0]} — proceeding without pause.')
        emitters = auto_emitters
    else:
        # Multiple emitters: allow manual selection if requested
        if MANUAL_PLOT_INTERACTION:
            psp._send_telegram(CURRENT_USER, 'Coarse scan done. Select emitters and close the plot to continue.')
            emitters = plotter.select_emitters(FOLDERNAME, 'coarse', data_folder=DATA_FOLDER)
            if len(emitters) == 0:
                print('No emitters selected. Done.')
                return
        else:
            emitters = auto_emitters

    print(f'Running fine scans on {len(emitters)} emitter(s): {[(f"{x:.2f}", f"{y:.2f}") for x, y in emitters]}')

    # ── STEP 2: PER-EMITTER LOOP ──────────────────────────────────────────────
    results = []   # collect (emitter_xy, target_pos, zpl_wl, bandpass_angle) for summary

    for i, (ex, ey) in enumerate(emitters):
        if _stop or _stop_immediately:
            break

        print(f'\n=== Emitter {i+1}/{len(emitters)}  ({ex:.2f}, {ey:.2f}) ===')

        # ── STEP 2a: FINE SCAN ────────────────────────────────────────────────
        # Zoomed scan centred on the emitter to localise the brightest spot.
        if af_available:
            print(f'[AUTOFOCUS] Running autofocus at ({ex:.2f}, {ey:.2f})...')
            af_result = autofocus.autofocus_on_emitter(
                emitter_pos=(ex, ey),
                grating=FINE_GRATING,
                exposure_s=FINE_EXPOSURE_S,
                center_wl=FINE_CENTER_WL,
                current_user=CURRENT_USER,
                foldername=FOLDERNAME,
            )
            if af_result is not None:
                print(f'  Autofocus locked at {af_result["voltage"]:.2f} V '
                      f'(532 nm intensity: {af_result["intensity"]:.0f})')
            else:
                print('  [WARNING] Autofocus failed — continuing with current Z position.')

        if _stop or _stop_immediately:
            break
        fine_type = f'fine_x{ex:.1f}_y{ey:.1f}'
        print(f'[STEP 2a] Fine scan  ({FINE_XDIM}x{FINE_YDIM} um, {FINE_DX} um step)...')
        _, status = run_scan(
            scan_type=fine_type,
            center=(ex, ey),
            xdim=FINE_XDIM,   ydim=FINE_YDIM,
            dx=FINE_DX,       dy=FINE_DY,
            grating=FINE_GRATING,
            exposure_s=FINE_EXPOSURE_S,
            center_wl=FINE_CENTER_WL,
        )
        if _stop or _stop_immediately:
            break

        if MANUAL_PLOT_INTERACTION:
            plotter.open_heatmap(FOLDERNAME, fine_type, data_folder=DATA_FOLDER)

        # Pick the target position: brightest classified pixel, else brightest overall
        fine_path = os.path.join(DATA_FOLDER, FOLDERNAME, fine_type)
        fine_out  = np.load(os.path.join(fine_path, 'out.npy'))
        fine_wl   = np.load(os.path.join(fine_path, 'wl.npy'))
        fine_xs   = np.load(os.path.join(fine_path, 'xs.npy'))
        fine_ys   = np.load(os.path.join(fine_path, 'ys.npy'))

        emission_mask = fine_wl > 550
        peak_map      = fine_out[:, :, emission_mask].max(axis=-1)

        fine_cls_path = os.path.join(fine_path, 'classified.npy')
        had_classified_pixels = False
        if os.path.exists(fine_cls_path):
            fine_cls     = np.load(fine_cls_path)
            iys_f, ixs_f = np.where(fine_cls == 1)
            if len(ixs_f) > 0:
                had_classified_pixels = True
                # Among classified pixels, take the brightest one
                best    = np.argmax([peak_map[iy, ix] for iy, ix in zip(iys_f, ixs_f)])
                tx, ty  = fine_xs[ixs_f[best]], fine_ys[iys_f[best]]
            else:
                # No classified pixels in fine scan — skip to next emitter
                print(f'  Fine scan found no classified pixels for emitter '
                      f'({ex:.2f}, {ey:.2f}) — skipping to next emitter.')
                results.append((ex, ey, None, None, None, None, 'no fine classified'))
                continue
        else:
            # No classification file — skip to next emitter
            print(f'  Fine scan produced no classification file for emitter '
                  f'({ex:.2f}, {ey:.2f}) — skipping to next emitter.')
            results.append((ex, ey, None, None, None, None, 'no fine classified'))
            continue
        print(f'  Brightest spot: ({tx:.2f}, {ty:.2f})')

        # ── STEP 2b: LONG SCAN ────────────────────────────────────────────────
        # Single-point, high-exposure spectrum at the brightest spot.
        long_type = f'long_x{tx:.1f}_y{ty:.1f}'

        long_center_wl = 595
        print(f'[STEP 2b] Long scan  ({LONG_EXPOSURE_S}s, 600 g/mm)...')
        _, status = run_scan(
            scan_type=long_type,
            center=(tx, ty),
            xdim=0, ydim=0, dx=0, dy=0,
            grating=LONG_GRATING,
            exposure_s=LONG_EXPOSURE_S,
            center_wl=long_center_wl,
        )
        if _stop or _stop_immediately:
            break

        # ── STEP 2c: BANDPASS FILTER SETUP ───────────────────────────────────
        # Extract ZPL wavelength from the long scan, then align the bandpass
        # filter to that wavelength using the rotation stage calibration.
        long_path = os.path.join(DATA_FOLDER, FOLDERNAME, long_type)
        save_spectrum_plot(long_path, title=f'Long scan ({tx:.2f}, {ty:.2f})')
        long_out  = np.load(os.path.join(long_path, 'out.npy'))
        long_wl   = np.load(os.path.join(long_path, 'wl.npy'))
        target_wl = find_emission_fwhm_center(long_out[0, 0, :], long_wl)

        if target_wl is None:
            print('  No emission peak found in long scan — skipping this emitter.')
            results.append((ex, ey, tx, ty, None, None, 'no ZPL'))
            continue

        # if not had_classified_pixels:
        #     print('  Fine scan had no classified pixels — skipping bandpass filter.')
        #     results.append((ex, ey, tx, ty, target_wl, None, 'no classified'))
        #     continue

        angle = _angle_for_wavelength(target_wl)
        print(f'[STEP 2c] ZPL FWHM centre: {target_wl:.1f} nm — setting bandpass filter...')
        if angle is None:
            print(f'  No calibration data for {target_wl:.1f} nm — skipping filter.')
            results.append((ex, ey, tx, ty, target_wl, None, 'no cal'))
            continue
        fil.flip_up()
        _filter_is_up = True
        fil.rotation_move(angle)
        print(f'  Filter set to {angle:.1f} deg.')

        filter_long_type = f'long_filter_x{tx:.1f}_y{ty:.1f}'
        print(f'[STEP 2d] Long scan through filter ({LONG_EXPOSURE_S}s, 600 g/mm)...')
        _, status = run_scan(
            scan_type=filter_long_type,
            center=(tx, ty),
            xdim=0, ydim=0, dx=0, dy=0,
            grating=LONG_GRATING,
            exposure_s=LONG_EXPOSURE_S,
            center_wl=long_center_wl,
        )
        filter_long_path = os.path.join(DATA_FOLDER, FOLDERNAME, filter_long_type)
        save_spectrum_plot(filter_long_path, title=f'Filter scan ({tx:.2f}, {ty:.2f})')
        if MANUAL_PLOT_INTERACTION:
            plotter.open_heatmap(FOLDERNAME, filter_long_type, data_folder=DATA_FOLDER)
        if _stop or _stop_immediately:
            fil.flip_down()
            _filter_is_up = False
            break

        results.append((ex, ey, tx, ty, target_wl, angle, 'filter set'))
        # G2 measurement would run here
        print('  G2 measurement skipped (not implemented yet).')
        fil.flip_down()
        _filter_is_up = False

    fil.filter_off()

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────────
    print('\n=== Results Summary ===')
    print(f'{"#":<4} {"Coarse (x,y)":<18} {"Target (x,y)":<18} {"ZPL (nm)":<10} {"Angle (deg)":<12} {"Status"}')
    print('-' * 74)
    for i, (ex, ey, tx, ty, zpl, ang, status) in enumerate(results, 1):
        zpl_s = f'{zpl:.1f}' if zpl is not None else '—'
        ang_s = f'{ang:.1f}' if ang is not None else '—'
        tgt_s = f'({tx:.1f}, {ty:.1f})' if tx is not None else '—'
        print(f'{i:<4} ({ex:.1f}, {ey:.1f}){"":<8} {tgt_s:<18} {zpl_s:<10} {ang_s:<12} {status}')

    print('\n=== Automation complete ===')


if __name__ == '__main__':
    try:
        main()
    finally:
        _keyboard_monitor_running = False

        try:
            if _filter_is_up:
                fil.flip_down()
        except Exception as e:
            print(f'Error flipping down filter: {e}')
        try:
            autofocus.autofocus_shutdown()
        except Exception as e:
            print(f'Error shutting down autofocus Z-stage: {e}')

        try:
            sgd.sgd_off()
        except Exception as e:
            print(f'Error turning off SGD: {e}')

        try:
            fil.filter_off()
        except Exception as e:
            print(f'Error turning off filter: {e}')

        try:
            lf_spec.lf_shutdown()
        except Exception as e:
            print(f'Error shutting down LightField: {e}')
