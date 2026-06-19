"""
Full SPE automation: coarse scan -> fine scan -> long scan -> bandpass filter setup.
Stop with Ctrl+C — the current acquisition finishes cleanly before exiting.
Ctrl+X — emergency stop immediately, saves partial data.
Ctrl+S — skip current emitter immediately, move to next one.
"""

# Set True to open a live heatmap window after each scan;
# False to run silently (data and PNGs still saved).
# IF TRUE THEN MANUAL INTERACTION IS REQUIRED
MANUAL_PLOT_INTERACTION = True


import os
import signal
import sys
import threading 
import time
import numpy as np
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lf_spec
import sgd
import filter as fil
import plotter
import pl_spec_python as psp
import autofocus
import picoharp
import g2 as g2mod

# ============================================================================
# PARAMETERS — edit these before each session
# ============================================================================

FOLDERNAME   = datetime.now().strftime('%Y%m%d') + '-PLSPC-HT-Ch4-f5-100uW-1s-fullauto-4'
CURRENT_USER = 'kristina'
DATA_FOLDER  = 'data'
CAL_FOLDER   = '2026-05-28_18-08-36'

COARSE_XDIM       = 2
COARSE_YDIM       = 2
COARSE_DX         = 0.5
COARSE_DY         = 0.5
COARSE_CENTER     = (13.50, 10.00)
COARSE_GRATING    = 150
COARSE_EXPOSURE_S = 1.0
COARSE_CENTER_WL  = 700

FINE_XDIM         = 2
FINE_YDIM         = 2
FINE_DX           = 0.25
FINE_DY           = 0.25
FINE_GRATING      = 150
FINE_EXPOSURE_S   = 1.0
FINE_CENTER_WL    = 700

LONG_GRATING      = 600
LONG_EXPOSURE_S   = 10.0

G2_TARGET_RECORDS = 50_000_000
G2_TIME_NS        = 100.0
G2_TIMEBIN_NS     = 0.25


# ============================================================================
# STOP / SKIP FLAGS
# ============================================================================

_stop             = False   # Ctrl+C  — finish current op then exit
_stop_immediately = False   # Ctrl+X  — stop everything immediately
_skip_emitter     = False   # Ctrl+S  — skip to next emitter immediately

def _handle_stop(sig, frame):
    global _stop
    print('\nStop requested — finishing current acquisition and cleaning up...')
    _stop = True

signal.signal(signal.SIGINT,  _handle_stop)
signal.signal(signal.SIGTERM, _handle_stop)

def _should_skip():
    """True if we should stop processing the current emitter."""
    return _skip_emitter or _stop_immediately or _stop

def _reset_skip():
    """Call at the top of each emitter iteration to clear the skip flag."""
    global _skip_emitter
    _skip_emitter = False


# ============================================================================
# KEYBOARD MONITOR
# ============================================================================

_keyboard_monitor_running = False
_monitor_thread           = None
_filter_is_up             = False
_monitor_paused           = threading.Event()


def _keyboard_monitor():
    """
    Background thread monitoring for:
      Ctrl+X (0x18) — emergency stop (existing behaviour)
      Ctrl+S (0x13) — skip current emitter immediately
      q             — same as Ctrl+X
    """
    global _stop_immediately, _skip_emitter
    try:
        import msvcrt
        print('[INFO] Keyboard monitor started  '
              '(Ctrl+X = emergency stop | Ctrl+S = skip emitter | q = quit)')
        while _keyboard_monitor_running:
            if _monitor_paused.is_set():
                time.sleep(0.05)
                continue
            try:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\x18':          # Ctrl+X
                        print('\n[CTRL+X] EMERGENCY STOP — stopping immediately...')
                        _stop_immediately = True
                    elif key == b'\x13':        # Ctrl+S  ← NEW
                        print('\n[CTRL+S] Skipping current emitter...')
                        _skip_emitter = True
                    elif key.lower() == b'q':
                        print('\n[Q] Emergency stop requested...')
                        _stop_immediately = True
            except Exception as e:
                print(f'[INFO] Keyboard monitor read error: {e}')
            time.sleep(0.05)
        print('[INFO] Keyboard monitor stopped')
    except ImportError:
        print('[INFO] msvcrt not available — keyboard monitor disabled')
    except Exception as e:
        print(f'[ERROR] Keyboard monitor failed: {e}')


def _flush_keys():
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        pass


def _paused_input(prompt):
    _monitor_paused.set()
    _flush_keys()
    try:
        return input(prompt)
    finally:
        _flush_keys()
        _monitor_paused.clear()


def _live_count_display():
    import msvcrt
    _monitor_paused.set()
    _flush_keys()
    print('  [G2] Live counts — adjust alignment, press any key to stop...')
    try:
        while True:
            r0, r1 = picoharp.get_count_rates()
            sys.stdout.write(f'\r         Ch0: {r0:.2e} cps   Ch1: {r1:.2e} cps   ')
            sys.stdout.flush()
            if msvcrt.kbhit():
                msvcrt.getch()
                break
            time.sleep(0.2)
    finally:
        print()
        _flush_keys()
        _monitor_paused.clear()


def _wait_start_or_align():
    import msvcrt
    _monitor_paused.set()
    _flush_keys()
    try:
        while True:
            print("  [G2] Press Enter to start acquisition, "
                  "or 'a' to view live counts for alignment...")
            while True:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key in (b'\r', b'\n'):
                        return
                    if key.lower() == b'a':
                        _live_count_display()
                        _monitor_paused.set()
                        _flush_keys()
                        break
                else:
                    time.sleep(0.05)
    finally:
        _flush_keys()
        _monitor_paused.clear()


# ============================================================================
# SPECTRUM HELPERS
# ============================================================================

def find_emission_fwhm_center(spectrum, wl, laser_cutoff_nm=560):
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
    out      = np.load(os.path.join(folder_path, 'out.npy'))
    wl       = np.load(os.path.join(folder_path, 'wl.npy'))
    spectrum = out[0, 0, :]
    fig, ax  = plt.subplots(figsize=(9, 4))
    ax.plot(wl, spectrum, lw=1.5)
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
            stop_immediately_flag=lambda: _stop_immediately or _skip_emitter,
        )
        return folder_path, status
    except KeyboardInterrupt:
        global _stop
        _stop = True
        print('\nScan interrupted.')
        return folder_path, 'stopped'


# ============================================================================
# PER-SPOT PIPELINE  (long scan + filter + G2 for one (tx, ty) position)
# ============================================================================

def _run_spot(i, n_emitters, ex, ey, tx, ty, spot_idx, n_spots,
              ph_available, results):
    """
    Run long scan → filter setup → G2 for a single confirmed spot (tx, ty).
    Respects _skip_emitter / _stop_immediately / _stop at every step.
    Appends one entry to results.
    Returns True if we should continue to the next spot, False to abort entirely.
    """
    spot_label = (f'Emitter {i+1}/{n_emitters}  spot {spot_idx+1}/{n_spots}'
                  f'  coarse=({ex:.2f},{ey:.2f})  target=({tx:.2f},{ty:.2f})')
    print(f'\n  --- {spot_label} ---')

    # ── LONG SCAN ─────────────────────────────────────────────────────────────
    if _should_skip():
        results.append((ex, ey, tx, ty, None, None, None, 'skipped before long scan'))
        return not (_stop or _stop_immediately)

    long_type      = f'long_x{tx:.2f}_y{ty:.2f}'
    long_center_wl = 595
    print(f'  [LONG] ({LONG_EXPOSURE_S}s, {LONG_GRATING} g/mm)...')
    _, status = run_scan(
        scan_type=long_type, center=(tx, ty),
        xdim=0, ydim=0, dx=0, dy=0,
        grating=LONG_GRATING, exposure_s=LONG_EXPOSURE_S,
        center_wl=long_center_wl,
    )

    if _should_skip():
        results.append((ex, ey, tx, ty, None, None, None, 'skipped after long scan'))
        return not (_stop or _stop_immediately)

    long_path = os.path.join(DATA_FOLDER, FOLDERNAME, long_type)
    save_spectrum_plot(long_path, title=spot_label)
    long_out  = np.load(os.path.join(long_path, 'out.npy'))
    long_wl   = np.load(os.path.join(long_path, 'wl.npy'))
    target_wl = find_emission_fwhm_center(long_out[0, 0, :], long_wl)

    if target_wl is None:
        print('  No emission peak found in long scan — skipping this spot.')
        results.append((ex, ey, tx, ty, None, None, None, 'no ZPL'))
        return True   # continue to next spot

    # ── BANDPASS FILTER ───────────────────────────────────────────────────────
    angle = _angle_for_wavelength(target_wl)
    print(f'  [FILTER] ZPL={target_wl:.1f} nm')
    if angle is None:
        print(f'  No calibration for {target_wl:.1f} nm — skipping filter.')
        results.append((ex, ey, tx, ty, target_wl, None, None, 'no cal'))
        return True

    fil.flip_up()
    global _filter_is_up
    _filter_is_up = True
    fil.rotation_move(angle)
    print(f'  Filter set to {angle:.1f} deg.')

    filter_long_type = f'long_filter_x{tx:.2f}_y{ty:.2f}'
    print(f'  [FILTER SCAN] ({LONG_EXPOSURE_S}s, {LONG_GRATING} g/mm)...')
    _, status = run_scan(
        scan_type=filter_long_type, center=(tx, ty),
        xdim=0, ydim=0, dx=0, dy=0,
        grating=LONG_GRATING, exposure_s=LONG_EXPOSURE_S,
        center_wl=long_center_wl,
    )
    filter_long_path = os.path.join(DATA_FOLDER, FOLDERNAME, filter_long_type)
    save_spectrum_plot(filter_long_path, title=f'Filter — {spot_label}')
    if MANUAL_PLOT_INTERACTION:
        plotter.open_heatmap(FOLDERNAME, filter_long_type, data_folder=DATA_FOLDER)

    if _should_skip():
        fil.flip_down()
        _filter_is_up = False
        results.append((ex, ey, tx, ty, target_wl, angle, None, 'skipped before G2'))
        return not (_stop or _stop_immediately)

    # ── G2 MEASUREMENT ────────────────────────────────────────────────────────
    g2_0      = None
    g2_status = 'g2 unavailable'

    if ph_available:
        sgd.goto(tx, ty)
        try:
            psp._send_telegram(CURRENT_USER,
                f'{spot_label}: ZPL={target_wl:.1f} nm, filter={angle:.1f} deg. '
                f'Flip mirror to APD path, then press Enter.')
            _paused_input('  [G2] Flip mirror to APD path, press Enter when ready...')

            if _should_skip():
                g2_status = 'skipped at G2 start'
            else:
                print('  [G2] Count rates:')
                for _ in range(4):
                    r0, r1 = picoharp.get_count_rates()
                    print(f'         Ch0: {r0:.2e} cps   Ch1: {r1:.2e} cps')
                    time.sleep(1.0)
                print(f'  [G2] Target: {G2_TARGET_RECORDS:,} records  '
                      f'(Ctrl+S saves partial and skips)')

                _wait_start_or_align()

                if _should_skip():
                    g2_status = 'skipped before acquire'
                else:
                    g2_folder = os.path.join(DATA_FOLDER, FOLDERNAME,
                                             f'g2_x{tx:.2f}_y{ty:.2f}')
                    # stop_flag checks BOTH _skip_emitter and _stop_immediately
                    # so Ctrl+S during acquisition stops ph_acquire immediately
                    # and whatever was collected so far gets saved
                    npz_path = picoharp.ph_acquire(
                        G2_TARGET_RECORDS,
                        out_folder=g2_folder,
                        stop_flag=lambda: _stop or _stop_immediately or _skip_emitter
                    )

                    if npz_path:
                        g2_result = g2mod.run(
                            npz_path, out_folder=g2_folder,
                            g2time_ns=G2_TIME_NS, timebin_ns=G2_TIMEBIN_NS
                        )
                        if g2_result['popt'] is not None:
                            g2_0 = g2_result['g2_0_norm']
                            print(f'  g²(0) = {g2_0:.3f}')
                            g2_status = 'g2 done' + (' (partial)' if _should_skip() else '')
                        else:
                            print('  g² fit did not converge.')
                            g2_status = 'g2 no fit'
                    else:
                        print('  G2 acquisition returned no data.')
                        g2_status = 'g2 no data'

            g2_0_str = f'{g2_0:.3f}' if g2_0 is not None else 'no fit'
            psp._send_telegram(CURRENT_USER,
                f'{spot_label} G2 done. g²(0)={g2_0_str}. '
                f'Flip mirror back to spectrometer path, then press Enter.')
            _paused_input('  [G2] Flip mirror back, press Enter to continue...')

        finally:
            sgd.sgd_off()

    fil.flip_down()
    _filter_is_up = False
    results.append((ex, ey, tx, ty, target_wl, angle, g2_0, g2_status))

    # Continue to next spot unless a hard stop was requested
    return not (_stop or _stop_immediately)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    global _keyboard_monitor_running, _monitor_thread, _filter_is_up
    _filter_is_up = False

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
    try:
        picoharp.ph_init()
        ph_available = True
    except Exception as e:
        print(f'[WARNING] PicoHarp init failed ({e}) — G2 will be skipped.')
        ph_available = False
    af_available = autofocus.autofocus_init()
    print('Autofocus Z-stage initialised.' if af_available else
          '[WARNING] Autofocus unavailable — fine scans proceed without it.')
    print()

    # ── STEP 1: COARSE SCAN ───────────────────────────────────────────────────
    if af_available:
        print(f'[AUTOFOCUS] Coarse centre {COARSE_CENTER}...')
        af_result = autofocus.autofocus_on_emitter(
            emitter_pos=COARSE_CENTER,
            grating=COARSE_GRATING, exposure_s=COARSE_EXPOSURE_S,
            center_wl=COARSE_CENTER_WL,
            current_user=CURRENT_USER, foldername=FOLDERNAME,
            stop_flag=lambda: _stop or _stop_immediately,
        )
        if af_result is not None:
            print(f'  Locked at {af_result["voltage"]:.2f} V  '
                  f'(532 nm: {af_result["intensity"]:.0f})')
        else:
            print('  [WARNING] Autofocus failed — using current Z.')

    print(f'[STEP 1] Coarse scan  ({COARSE_XDIM}x{COARSE_YDIM} um, {COARSE_DX} um step)...')
    _, status = run_scan(
        scan_type='coarse', center=COARSE_CENTER,
        xdim=COARSE_XDIM, ydim=COARSE_YDIM,
        dx=COARSE_DX, dy=COARSE_DY,
        grating=COARSE_GRATING, exposure_s=COARSE_EXPOSURE_S,
        center_wl=COARSE_CENTER_WL,
    )
    if _stop or _stop_immediately:
        print('Stopped.')
        return

    plotter.save_plot(FOLDERNAME, 'coarse', data_folder=DATA_FOLDER)

    coarse_path = os.path.join(DATA_FOLDER, FOLDERNAME, 'coarse')
    classified  = np.load(os.path.join(coarse_path, 'classified.npy'))
    xs_c        = np.load(os.path.join(coarse_path, 'xs.npy'))
    ys_c        = np.load(os.path.join(coarse_path, 'ys.npy'))

    iys, ixs       = np.where(classified == 1)
    auto_emitters  = [(xs_c[ix], ys_c[iy]) for ix, iy in zip(ixs, iys)]

    if len(auto_emitters) == 0:
        print('No emitters found in coarse scan. Done.')
        return
    elif len(auto_emitters) == 1:
        print(f'Found 1 emitter at {auto_emitters[0]} — proceeding.')
        emitters = auto_emitters
    else:
        if MANUAL_PLOT_INTERACTION:
            psp._send_telegram(CURRENT_USER,
                'Coarse scan done. Select emitters and close the plot to continue.')
            emitters = plotter.select_emitters(FOLDERNAME, 'coarse',
                                               data_folder=DATA_FOLDER)
            if len(emitters) == 0:
                print('No emitters selected. Done.')
                return
        else:
            emitters = auto_emitters

    print(f'Running fine scans on {len(emitters)} emitter(s): '
          f'{[(f"{x:.2f}", f"{y:.2f}") for x, y in emitters]}')

    # ── STEP 2: PER-EMITTER LOOP ──────────────────────────────────────────────
    results = []

    for i, (ex, ey) in enumerate(emitters):
        if _stop or _stop_immediately:
            break

        # Reset skip flag at the top of every emitter — a previous Ctrl+S
        # should not carry over to the next emitter.
        _reset_skip()

        print(f'\n=== Emitter {i+1}/{len(emitters)}  ({ex:.2f}, {ey:.2f}) ===')

        # ── STEP 2a: FINE SCAN ────────────────────────────────────────────────
        if af_available:
            print(f'[AUTOFOCUS] ({ex:.2f}, {ey:.2f})...')
            af_result = autofocus.autofocus_on_emitter(
                emitter_pos=(ex, ey),
                grating=FINE_GRATING, exposure_s=FINE_EXPOSURE_S,
                center_wl=FINE_CENTER_WL,
                current_user=CURRENT_USER, foldername=FOLDERNAME,
                stop_flag=lambda: _stop or _stop_immediately,
            )
            if af_result is not None:
                print(f'  Locked at {af_result["voltage"]:.2f} V  '
                      f'(532 nm: {af_result["intensity"]:.0f})')
            else:
                print('  [WARNING] Autofocus failed — using current Z.')

        if _should_skip():
            results.append((ex, ey, None, None, None, None, None,
                            'skipped before fine scan'))
            continue

        fine_type = f'fine_x{ex:.1f}_y{ey:.1f}'
        print(f'[STEP 2a] Fine scan  ({FINE_XDIM}x{FINE_YDIM} um, {FINE_DX} um step)...')
        _, status = run_scan(
            scan_type=fine_type, center=(ex, ey),
            xdim=FINE_XDIM, ydim=FINE_YDIM,
            dx=FINE_DX, dy=FINE_DY,
            grating=FINE_GRATING, exposure_s=FINE_EXPOSURE_S,
            center_wl=FINE_CENTER_WL,
        )

        if _should_skip():
            results.append((ex, ey, None, None, None, None, None,
                            'skipped after fine scan'))
            continue

        plotter.save_plot(FOLDERNAME, fine_type, data_folder=DATA_FOLDER)

        # ── STEP 2b: SELECT SPOTS FROM FINE SCAN ─────────────────────────────
        # Mirror exactly what the coarse scan does:
        # - 0 classified pixels  → skip this emitter
        # - 1 classified pixel   → use it automatically
        # - multiple pixels      → open interactive plot, user right-clicks to select
        fine_path     = os.path.join(DATA_FOLDER, FOLDERNAME, fine_type)
        fine_cls_path = os.path.join(fine_path, 'classified.npy')

        if not os.path.exists(fine_cls_path):
            print(f'  No classification file for fine scan — skipping emitter.')
            results.append((ex, ey, None, None, None, None, None,
                            'no fine classified'))
            continue

        fine_cls     = np.load(fine_cls_path)
        fine_xs      = np.load(os.path.join(fine_path, 'xs.npy'))
        fine_ys      = np.load(os.path.join(fine_path, 'ys.npy'))
        iys_f, ixs_f = np.where(fine_cls == 1)

        if len(ixs_f) == 0:
            print(f'  Fine scan found no classified pixels — skipping emitter.')
            results.append((ex, ey, None, None, None, None, None,
                            'no fine classified'))
            continue

        elif len(ixs_f) == 1:
            # Only one classified pixel — use it without pausing
            tx, ty      = fine_xs[ixs_f[0]], fine_ys[iys_f[0]]
            fine_spots  = [(tx, ty)]
            print(f'  1 spot found at ({tx:.2f}, {ty:.2f}) — proceeding.')

        else:
            # Multiple pixels — let user select, exactly like the coarse scan
            if MANUAL_PLOT_INTERACTION:
                psp._send_telegram(CURRENT_USER,
                    f'Fine scan done for emitter ({ex:.2f},{ey:.2f}). '
                    f'Select spots and close the plot to continue.')
                fine_spots = plotter.select_emitters(FOLDERNAME, fine_type,
                                                     data_folder=DATA_FOLDER)
                if len(fine_spots) == 0:
                    print('  No spots selected — skipping emitter.')
                    results.append((ex, ey, None, None, None, None, None,
                                    'no fine spots selected'))
                    continue
            else:
                # Non-interactive: use all classified pixels
                fine_spots = [(fine_xs[ix], fine_ys[iy])
                              for ix, iy in zip(ixs_f, iys_f)]

            print(f'  {len(fine_spots)} spot(s) selected: '
                  f'{[(f"{x:.2f}", f"{y:.2f}") for x, y in fine_spots]}')

        # ── STEP 2c–e: LONG SCAN + FILTER + G2 FOR EACH SELECTED SPOT ────────
        for spot_idx, (tx, ty) in enumerate(fine_spots):
            if _stop or _stop_immediately:
                break

            # Ctrl+S during a spot's processing stops that spot and moves to
            # the next one within the same emitter.  At the top of each spot
            # we reset the flag so it doesn't cascade to the next spot.
            _reset_skip()

            keep_going = _run_spot(
                i, len(emitters), ex, ey, tx, ty,
                spot_idx, len(fine_spots),
                ph_available, results
            )
            if not keep_going:
                break   # hard stop — exit spot loop and emitter loop

    fil.filter_off()

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────────
    print('\n=== Results Summary ===')
    print(f'{"#":<4} {"Coarse (x,y)":<18} {"Target (x,y)":<18} '
          f'{"ZPL (nm)":<10} {"Angle":<10} {"g²(0)":<8} Status')
    print('-' * 88)
    for i, (ex, ey, tx, ty, zpl, ang, g2_0, status) in enumerate(results, 1):
        zpl_s = f'{zpl:.1f}'  if zpl  is not None else '—'
        ang_s = f'{ang:.1f}'  if ang  is not None else '—'
        tgt_s = f'({tx:.2f},{ty:.2f})' if tx is not None else '—'
        g2_s  = f'{g2_0:.3f}' if g2_0 is not None else '—'
        print(f'{i:<4} ({ex:.1f},{ey:.1f}){"":<8} {tgt_s:<18} '
              f'{zpl_s:<10} {ang_s:<10} {g2_s:<8} {status}')

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
            picoharp.ph_close()
        except Exception as e:
            print(f'Error closing PicoHarp: {e}')
        try:
            autofocus.autofocus_shutdown()
        except Exception as e:
            print(f'Error shutting down autofocus: {e}')
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