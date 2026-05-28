"""
Full SPE automation: coarse scan -> fine scan -> long scan -> bandpass filter setup.
G2 measurement is not included yet.

Stop with Ctrl+C — the current acquisition finishes cleanly before exiting.
"""

# Set True to open a live heatmap window after each scan; False to run silently (data and PNGs still saved).
PLOT_INTERACTIVE = True

import os
import signal
import numpy as np
from datetime import datetime

import matplotlib
if not PLOT_INTERACTIVE:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import lf_spec
import sgd
import filter as fil
import plotter
import pl_spec_python as psp

# ============================================================================
# PARAMETERS — edit these before each session
# ============================================================================

FOLDERNAME   = datetime.now().strftime('%Y%m%d') + '-PLSPC-HT-plasma+anneal-Ch4-500uW-2s-fullauto-1'
CURRENT_USER = 'kristina'
DATA_FOLDER  = 'data'
CAL_FOLDER   = '2026-04-07_14-48-20'   # bandpass calibration subfolder name

# Coarse scan — wide area to locate candidate emitters
COARSE_XDIM       = 20.0   # um
COARSE_YDIM       = 20.0   # um
COARSE_DX         = 0.5    # um step size
COARSE_DY         = 0.5
COARSE_CENTER     = (0.0, 0.0)
COARSE_GRATING    = 150
COARSE_EXPOSURE_S = 2.0
COARSE_CENTER_WL  = 700    # nm

# Fine scan — zoomed scan centred on each classified emitter
FINE_XDIM         = 3.0
FINE_YDIM         = 3.0
FINE_DX           = 0.25
FINE_DY           = 0.25
FINE_GRATING      = 600
FINE_EXPOSURE_S   = 1.0
FINE_CENTER_WL    = 595

# Long scan — single-point, high-exposure spectrum to measure ZPL precisely
LONG_GRATING      = 600
LONG_EXPOSURE_S   = 10.0

# Bandpass filter alignment
BANDPASS_TOLERANCE_NM = 2.0
BANDPASS_MAX_ATTEMPTS = 3

# Replay mode — set to True to run the full pipeline on existing data without hardware.
# Point FOLDERNAME at a past dataset folder. Scans that already have out.npy are loaded
# directly; scans with no existing data get a single-point placeholder so the pipeline
# can still run. Hardware init and bandpass motor moves are skipped entirely.
REPLAY = False

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
    # NOTE if statement added to allow wrapping of angles
    if angles>360:
        angles-=360
    return float(angles[np.argmin(np.abs(wls - target_wl))])


def _bandpass_slope(target_wl, angle_window=30.0):
    """Estimate dangle/dwl (deg/nm) near target_wl from the calibration table.
    Filters to entries within angle_window degrees to exclude spurious points."""
    table_path = os.path.join('calibration', CAL_FOLDER, 'calibration_table.npy')
    if not os.path.exists(table_path):
        return None
    table = np.load(table_path)
    valid = ~np.isnan(table[:, 1])
    if valid.sum() < 2:
        return None
    angles, wls = table[valid, 0], table[valid, 1]

    expected_angle = float(angles[np.argmin(np.abs(wls - target_wl))])
    angle_diff     = np.abs(((angles - expected_angle) + 180) % 360 - 180)
    clean          = angle_diff <= angle_window
    if clean.sum() < 2:
        return None
    angles, wls = angles[clean], wls[clean]

    order        = np.argsort(wls)
    angles, wls  = angles[order], wls[order]
    idx          = int(np.argmin(np.abs(wls - target_wl)))
    i0, i1       = max(0, idx - 1), min(len(wls) - 1, idx + 1)
    dwl          = wls[i1] - wls[i0]
    if dwl == 0:
        return None
    dangle = (angles[i1] - angles[i0] + 180) % 360 - 180
    return float(dangle / dwl)

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
    In REPLAY mode: loads existing data if present, otherwise writes a
    placeholder to a tmp folder so the pipeline can continue without hardware.
    Returns the folder path where data was saved.
    """
    folder_path = os.path.join(DATA_FOLDER, FOLDERNAME, scan_type)

    # REPLAY: reuse existing data or write a placeholder — never touch hardware
    if REPLAY:
        if os.path.exists(os.path.join(folder_path, 'out.npy')):
            print(f'  [REPLAY] Using existing data in {folder_path}')
            return folder_path
        tmp_path = os.path.join(DATA_FOLDER, FOLDERNAME + '_replay_tmp', scan_type)
        os.makedirs(tmp_path, exist_ok=True)
        print(f'  [REPLAY] No existing data — placeholder at {center}')
        wl_fake = np.linspace(415, 980, 1340)
        np.save(os.path.join(tmp_path, 'out.npy'), np.zeros((1, 1, len(wl_fake))))
        np.save(os.path.join(tmp_path, 'xs.npy'),  np.array([center[0]]))
        np.save(os.path.join(tmp_path, 'ys.npy'),  np.array([center[1]]))
        np.save(os.path.join(tmp_path, 'wl.npy'),  wl_fake)
        return tmp_path

    # Live hardware — delegate entirely to pl_spec_python
    try:
        psp.pl_spec_lf(
            xdim=xdim, ydim=ydim, dx=dx, dy=dy,
            center=center,
            grating=grating,
            exposure_time=exposure_s,
            center_wavelength=center_wl,
            foldername=FOLDERNAME,
            current_user=CURRENT_USER,
            scan_type=scan_type,
            data_folder=DATA_FOLDER,
        )
    except KeyboardInterrupt:
        global _stop
        _stop = True
        print('\nScan interrupted.')

    return folder_path

# ============================================================================
# BANDPASS FILTER SETUP
# ============================================================================

def run_bandpass_setup(target_wl):
    """
    Insert the bandpass filter, rotate to target_wl, and verify with a single
    acquisition. Uses proportional angle correction if the measured peak is
    outside BANDPASS_TOLERANCE_NM. Flips filter back out on failure.
    Returns True if aligned, False otherwise.
    """
    #NOTE filter did not flip down on error
    angle = _angle_for_wavelength(target_wl)
    if angle is None:
        print(f'  No calibration data for {target_wl:.1f} nm — skipping filter.')
        return False

    slope = _bandpass_slope(target_wl)  # deg/nm, used for correction

    # REPLAY: skip all hardware, just verify the calibration lookup works
    if REPLAY:
        print(f'  [REPLAY] Would rotate to {angle:.2f} deg  slope: {slope}')
        print('  [REPLAY] Bandpass skipped — simulating success.')
        return True

    # Use long-scan settings so the ZPL is visible through the filter
    lf_spec.lf_setup(
        exposure_s=LONG_EXPOSURE_S,
        center_wavelength=int(target_wl),
        grating=LONG_GRATING,
    )

    fil.flip_up()  # insert filter into beam

    for attempt in range(BANDPASS_MAX_ATTEMPTS):
        print(f'  Attempt {attempt+1}/{BANDPASS_MAX_ATTEMPTS}: rotating to {angle:.2f} deg...')
        fil.rotation_move(angle)

        intensity, wl = lf_spec.lf_acquire()
        measured_wl   = find_emission_fwhm_center(
            np.array(intensity).flatten(),
            np.array(wl).flatten(),
        )

        if measured_wl is None:
            print('  No emission peak detected through filter.')
            break
        # NOTE error might be skewed by emission from PBS if wl is over estimated
        error = target_wl - measured_wl
        print(f'  Target: {target_wl:.1f} nm  Measured: {measured_wl:.1f} nm  Error: {error:+.1f} nm')

        if abs(error) <= BANDPASS_TOLERANCE_NM:
            print('  Bandpass aligned.')
            return True

        if slope is not None:
            correction = error * slope
            angle     += correction
            # NOTE if statement added to allow wrapping of angles
            if angle>360:
                angle-=360
            print(f'  Correction: {correction:+.2f} deg  new target: {angle:.2f} deg')
        else:
            print('  No slope data — cannot correct angle.')
            break

    # Alignment failed — remove filter so the next emitter is not obscured
    print(f'  Bandpass alignment failed after {BANDPASS_MAX_ATTEMPTS} attempts.')
    fil.flip_down()
    return False

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print('=== SPE Automation ===')
    print(f'Folder: {FOLDERNAME}')
    print(f'Cal:    {CAL_FOLDER}')
    print()

    # ── Hardware init ─────────────────────────────────────────────────────────
    if REPLAY:
        print('[REPLAY MODE] Hardware init skipped.\n')
    else:
        print('Initializing hardware...')
        lf_spec.lf_connect()
        sgd.sgd_init()
        fil.filter_init()
        fil.filter_on()
        print()

    # ── STEP 1: COARSE SCAN ───────────────────────────────────────────────────
    # Scans the full area to find candidate emitter positions.
    print(f'[STEP 1] Coarse scan  ({COARSE_XDIM}x{COARSE_YDIM} um, {COARSE_DX} um step)...')
    run_scan(
        scan_type='coarse',
        center=COARSE_CENTER,
        xdim=COARSE_XDIM,     ydim=COARSE_YDIM,
        dx=COARSE_DX,         dy=COARSE_DY,
        grating=COARSE_GRATING,
        exposure_s=COARSE_EXPOSURE_S,
        center_wl=COARSE_CENTER_WL,
    )
    if _stop:
        print('Stopped.')
        return

    if PLOT_INTERACTIVE:
        psp._send_telegram(CURRENT_USER, f"Interactive plot opened. Close to continue scan.")
        plotter.open_heatmap(FOLDERNAME, 'coarse', data_folder=DATA_FOLDER)

    # Load coarse results — keep out/wl in memory for ZPL estimates later
    coarse_path = os.path.join(DATA_FOLDER, FOLDERNAME, 'coarse')
    classified  = np.load(os.path.join(coarse_path, 'classified.npy'))
    coarse_out  = np.load(os.path.join(coarse_path, 'out.npy'))
    coarse_wl   = np.load(os.path.join(coarse_path, 'wl.npy'))
    xs_c        = np.load(os.path.join(coarse_path, 'xs.npy'))
    ys_c        = np.load(os.path.join(coarse_path, 'ys.npy'))

    iys, ixs = np.where(classified == 1)
    if len(ixs) == 0:
        print('No emitters found in coarse scan. Done.')
        return

    emitters = [(xs_c[ix], ys_c[iy]) for ix, iy in zip(ixs, iys)]
    print(f'Found {len(emitters)} emitter(s): {[(f"{x:.2f}", f"{y:.2f}") for x, y in emitters]}')

    # ── STEP 2: PER-EMITTER LOOP ──────────────────────────────────────────────
    results = []   # collect (emitter_xy, target_pos, zpl_wl, bandpass_angle) for summary

    for i, (ex, ey) in enumerate(emitters):
        if _stop:
            break

        print(f'\n=== Emitter {i+1}/{len(emitters)}  ({ex:.2f}, {ey:.2f}) ===')

        # ── STEP 2a: FINE SCAN ────────────────────────────────────────────────
        # Zoomed scan centred on the emitter to localise the brightest spot.
        fine_type = f'fine_x{ex:.1f}_y{ey:.1f}'
        print(f'[STEP 2a] Fine scan  ({FINE_XDIM}x{FINE_YDIM} um, {FINE_DX} um step)...')
        run_scan(
            scan_type=fine_type,
            center=(ex, ey),
            xdim=FINE_XDIM,   ydim=FINE_YDIM,
            dx=FINE_DX,       dy=FINE_DY,
            grating=FINE_GRATING,
            exposure_s=FINE_EXPOSURE_S,
            center_wl=FINE_CENTER_WL,
        )
        if _stop:
            break

        if PLOT_INTERACTIVE:
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
        if os.path.exists(fine_cls_path):
            fine_cls     = np.load(fine_cls_path)
            iys_f, ixs_f = np.where(fine_cls == 1)
            if len(ixs_f) > 0:
                # Among classified pixels, take the brightest one
                best    = np.argmax([peak_map[iy, ix] for iy, ix in zip(iys_f, ixs_f)])
                tx, ty  = fine_xs[ixs_f[best]], fine_ys[iys_f[best]]
            else:
                # No classified pixels — fall back to global peak
                iy_b, ix_b = np.unravel_index(np.argmax(peak_map), peak_map.shape)
                tx, ty     = fine_xs[ix_b], fine_ys[iy_b]
        else:
            iy_b, ix_b = np.unravel_index(np.argmax(peak_map), peak_map.shape)
            tx, ty     = fine_xs[ix_b], fine_ys[iy_b]

        print(f'  Brightest spot: ({tx:.2f}, {ty:.2f})')

        # ── STEP 2b: LONG SCAN ────────────────────────────────────────────────
        # Single-point, high-exposure spectrum at the brightest spot.
        # The center wavelength is derived from the coarse ZPL estimate so that
        # both the 532 nm laser line and the emission fit in the 600 g/mm window.
        long_type = f'long_x{tx:.1f}_y{ty:.1f}'

        ix_near    = int(np.argmin(np.abs(xs_c - ex)))
        iy_near    = int(np.argmin(np.abs(ys_c - ey)))
        coarse_zpl = find_emission_fwhm_center(coarse_out[iy_near, ix_near, :], coarse_wl)
        # NOTE: most likely redundant if statement
        if coarse_zpl is not None:
            long_center_wl = 595 #int((532 + coarse_zpl) / 2)
            print(f'  Coarse ZPL estimate: {coarse_zpl:.1f} nm  long center WL: {long_center_wl} nm')
        else:
            long_center_wl = 595
            print(f'  Could not estimate ZPL from coarse — using {long_center_wl} nm')

        print(f'[STEP 2b] Long scan  ({LONG_EXPOSURE_S}s, 600 g/mm)...')
        run_scan(
            scan_type=long_type,
            center=(tx, ty),
            xdim=0, ydim=0, dx=0, dy=0,
            grating=LONG_GRATING,
            exposure_s=LONG_EXPOSURE_S,
            center_wl=long_center_wl,
        )
        if _stop:
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

        angle = _angle_for_wavelength(target_wl)
        print(f'[STEP 2c] ZPL FWHM centre: {target_wl:.1f} nm — aligning bandpass filter...')
        aligned = run_bandpass_setup(target_wl)
        results.append((ex, ey, tx, ty, target_wl, angle, 'aligned' if aligned else 'failed'))

        if aligned:
            # G2 measurement would run here
            print('  Filter aligned. G2 measurement skipped (not implemented yet).')
            if not REPLAY:
                fil.flip_down()  # remove filter before moving to next emitter
        else:
            print('  Moving to next emitter.')

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────────
    print('\n=== Results Summary ===')
    print(f'{"#":<4} {"Coarse (x,y)":<18} {"Target (x,y)":<18} {"ZPL (nm)":<10} {"Angle (deg)":<12} {"Status"}')
    print('-' * 74)
    for i, (ex, ey, tx, ty, zpl, ang, status) in enumerate(results, 1):
        zpl_s = f'{zpl:.1f}' if zpl is not None else '—'
        ang_s = f'{ang:.1f}' if ang is not None else '—'
        print(f'{i:<4} ({ex:.1f}, {ey:.1f}){"":<8} ({tx:.1f}, {ty:.1f}){"":<8} {zpl_s:<10} {ang_s:<12} {status}')

    print('\n=== Automation complete ===')


if __name__ == '__main__':
    main()
