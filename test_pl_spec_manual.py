"""
Test lf_spec-based scanning directly (bypasses pl_spec_manual / MATLAB).
This is the development version of the scan loop using lf_spec.
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

import lf_spec
import sgd
import classifier as classifer
from tqdm import tqdm

FOLDERNAME  = '20260519-PLSPC-example'
SCAN_TYPE   = 'coarse'
DATA_FOLDER = 'data'

XDIM, YDIM  = 1.0, 1.0
DX,   DY    = 0.5, 0.5
CENTER      = (0.0, 0.0)
GRATING     = 150
EXPOSURE_S  = 1.0
CENTER_WL   = 700

# ── Init ──────────────────────────────────────────────────────────────────────
lf_spec.lf_connect()
sgd.sgd_init()

# ── Setup spectrometer ────────────────────────────────────────────────────────
print('Setting up spectrometer...')
lf_spec.lf_setup(exposure_s=EXPOSURE_S, center_wavelength=CENTER_WL, grating=GRATING)

folder_path = os.path.join(DATA_FOLDER, FOLDERNAME, SCAN_TYPE)
os.makedirs(folder_path, exist_ok=True)

wl  = lf_spec.lf_get_wavelengths()
num = len(wl)
np.save(os.path.join(folder_path, 'wl.npy'), wl)
print(f'WL: {wl[0]:.1f}–{wl[-1]:.1f} nm  ({num} points)')

xs = np.arange(-XDIM/2 + CENTER[0], XDIM/2 + DX + CENTER[0], DX)
ys = np.arange(-YDIM/2 + CENTER[1], YDIM/2 + DY + CENTER[1], DY)
output = np.zeros((len(ys), len(xs), num))

# ── Scan ──────────────────────────────────────────────────────────────────────
print('Starting scan...')
sgd.sgd_on()

with tqdm(total=len(ys)*len(xs), desc='Scanning') as pbar:
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            sgd.set_position(x, y, silent=True)
            intensity, wl_acq = lf_spec.lf_acquire()
            intensity = np.array(intensity)
            if len(intensity) != num:
                intensity = np.resize(intensity, num)
            output[iy, ix, :] = intensity
            wl = wl_acq
            pbar.set_description(f'x={x:.2f}, y={y:.2f}')
            pbar.update(1)

sgd.sgd_off()

# ── Save ──────────────────────────────────────────────────────────────────────
np.save(os.path.join(folder_path, 'out.npy'), output)
np.save(os.path.join(folder_path, 'xs.npy'), xs)
np.save(os.path.join(folder_path, 'ys.npy'), ys)
np.save(os.path.join(folder_path, 'wl.npy'), wl)

print('Scan complete, classifying...')
classifer.classify_all(FOLDERNAME, SCAN_TYPE, data_folder=DATA_FOLDER)
print('Done.')
