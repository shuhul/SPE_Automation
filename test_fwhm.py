"""
Test find_emission_fwhm_center, _bandpass_slope, angle_for_wavelength
using real coarse scan data — no hardware required.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from main import find_emission_fwhm_center, _bandpass_slope, angle_for_wavelength

CAL  = '2026-04-07_14-48-20'
DATA = 'test_data/coarse'

out        = np.load(f'{DATA}/out.npy')
wl         = np.load(f'{DATA}/wl.npy')
xs         = np.load(f'{DATA}/xs.npy')
ys         = np.load(f'{DATA}/ys.npy')
classified = np.load(f'{DATA}/classified.npy')

emitter_positions = list(zip(*np.where(classified == 1)))  # (iy, ix) pairs
print(f'Testing on {len(emitter_positions)} classified emitters\n')

results = []
fig, axes = plt.subplots(4, 5, figsize=(18, 12))
axes = axes.flatten()

for i, (iy, ix) in enumerate(emitter_positions):
    spectrum  = out[iy, ix, :]
    x, y      = xs[ix], ys[iy]
    target_wl = find_emission_fwhm_center(spectrum, wl)

    angle = angle_for_wavelength(CAL, target_wl) if target_wl else None
    slope = _bandpass_slope(CAL, target_wl)       if target_wl else None

    status = 'OK' if target_wl else 'NO PEAK'
    print(f'  x={x:+6.1f} y={y:+6.1f}  FWHM: '
          f'{target_wl:.1f} nm  angle: {angle:.1f} deg  slope: {slope:.3f} deg/nm  [{status}]'
          if target_wl else
          f'  x={x:+6.1f} y={y:+6.1f}  [{status}]')

    results.append(dict(x=x, y=y, target_wl=target_wl, angle=angle, slope=slope))

    if i < len(axes):
        ax = axes[i]
        ax.plot(wl, spectrum, lw=0.8)
        if target_wl:
            peak_height = float(spectrum[wl > 560].max())
            ax.axvline(target_wl, color='r',      ls='--', lw=1.2)
            ax.axhline(peak_height / 2, color='orange', ls=':', lw=0.9)
        ax.set_title(f'x={x:.1f} y={y:.1f}\n{target_wl:.1f} nm' if target_wl else f'x={x:.1f} y={y:.1f}\nNO PEAK',
                     fontsize=7)
        ax.tick_params(labelsize=6)

for ax in axes[len(emitter_positions):]:
    ax.set_visible(False)

plt.suptitle('FWHM centre check — test_data/coarse', fontsize=10)
plt.tight_layout()
out_path = 'test_fwhm_output.png'
plt.savefig(out_path, dpi=120)
print(f'\nSaved plot: {out_path}')
