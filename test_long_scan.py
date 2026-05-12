"""
Try long scans on all remaining classified emitters and run the classifier
on each result to find a good one for the test dataset.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matlab.engine
import _matlab_session
import sgd
import pl_spec
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from main import find_emission_fwhm_center
from classifier import classify_spectrum

FOLDERNAME = '20260512-PLSPC-PhENOM-Ch23-f011o015-f1a1-500uW-3s-1'

# All 20 classified emitters from the coarse map, sorted by brightness
# (center_x, center_y, coarse_fwhm_nm)
CANDIDATES = [
    (-11.5, -10.0, 621.0),
    (-10.0, -11.5, 586.1),
    (-13.0,  -7.5, 618.8),
    (-10.5, -10.5, 583.5),
    ( -8.5, -10.5, 583.4),
    ( -3.0, -12.5, 571.1),
    ( -4.5, -11.0, 570.6),
    ( -3.0, -12.0, 588.1),
    ( -9.5,  -7.5, 586.8),
    (-12.5,  -8.5, 608.6),
    (-10.5,  -5.0, 580.5),
    ( -4.0, -11.5, 581.8),
    ( -5.0, -10.0, 565.2),
    ( -3.5, -10.5, 612.5),
    ( -6.5, -10.0, 577.7),
    (-12.0,  -3.5, 576.8),
    ( -3.5, -14.0, 578.9),
    ( -4.0,  11.0, 597.2),
    (-12.5,  -2.0, 603.2),
    (-11.0,  -4.5, 587.8),
]

# ── 1. Connect to MATLAB ──────────────────────────────────────────────────────
print('Looking for MATLAB sessions...')
sessions = matlab.engine.find_matlab()
print(f'  Found: {sessions}')
if not sessions:
    print('ERROR: No MATLAB sessions found. Run: matlab.engine.shareEngine() in MATLAB.')
    sys.exit(1)

name = 'MySharedSession' if 'MySharedSession' in sessions else sessions[0]
print(f'  Connecting to: {name}')
try:
    eng = matlab.engine.connect_matlab(name)
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)

_matlab_session.name = name
eng.addpath(os.path.join(os.getcwd(), 'matlab'), nargout=0)
eng.pl_setup(nargout=0)
print('  MATLAB connected.')

# ── 2. Init SGD ───────────────────────────────────────────────────────────────
print('Initialising SGD...')
sgd.sgd_init()

# ── 3. Scan each candidate ────────────────────────────────────────────────────
results = []

for cx, cy, coarse_fwhm in CANDIDATES:
    cwl       = int((532 + coarse_fwhm) / 2)
    scan_type = f'long_x{cx:.1f}_y{cy:.1f}'

    print(f'\nScanning ({cx:+.1f}, {cy:+.1f})  coarse_fwhm={coarse_fwhm:.0f}nm  cwl={cwl}nm ...')
    pl_spec.pl_spec_manual(
        xdim=0, ydim=0, dx=0, dy=0,
        center=(cx, cy),
        grating=600,
        exposure_time=10,
        center_wavelength=cwl,
        foldername=FOLDERNAME,
        scan_type=scan_type,
        current_user='shuhul',
        eng=eng,
    )

    out      = np.load(f'data/{FOLDERNAME}/{scan_type}/out.npy')
    wl       = np.load(f'data/{FOLDERNAME}/{scan_type}/wl.npy')
    spectrum = out[0, 0, :]

    label, peak_h, peak_wl = classify_spectrum(spectrum, wl)
    fwhm_centre = find_emission_fwhm_center(spectrum, wl)
    status = 'PASS' if label == 1 else 'fail'
    print(f'  Classifier: {status}   peak_wl={peak_wl}   FWHM_centre={fwhm_centre:.1f}nm')
    results.append(dict(cx=cx, cy=cy, spectrum=spectrum, wl=wl,
                        label=label, peak_wl=peak_wl, fwhm_centre=fwhm_centre))

# ── 4. Plot all results ───────────────────────────────────────────────────────
n   = len(results)
cols = 4
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3.5*rows))
axes = axes.flatten()

for i, r in enumerate(results):
    ax = axes[i]
    ax.plot(r['wl'], r['spectrum'], lw=0.8)
    colour = 'green' if r['label'] == 1 else 'red'
    ax.axvline(r['fwhm_centre'], color=colour, ls='--', lw=1.2)
    status = 'PASS' if r['label'] == 1 else 'fail'
    ax.set_title(f"({r['cx']:.1f},{r['cy']:.1f})  {status}\n{r['fwhm_centre']:.1f}nm", fontsize=8)
    ax.tick_params(labelsize=7)

for ax in axes[n:]:
    ax.set_visible(False)

plt.suptitle('Long scan classifier results', fontsize=11)
plt.tight_layout()
plt.savefig('test_long_scan_output.png', dpi=130)
print('\nSaved: test_long_scan_output.png')

passing = [r for r in results if r['label'] == 1]
print(f'\n{len(passing)}/{len(results)} emitters passed the classifier:')
for r in passing:
    print(f"  ({r['cx']:+.1f}, {r['cy']:+.1f})  ZPL={r['peak_wl']:.1f}nm  FWHM_centre={r['fwhm_centre']:.1f}nm")
