"""
Sweep filter angle past the calibration edge to find if/where the laser (532nm) passes through.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matlab.engine, _matlab_session
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pl_spec, filter as fil

ANGLES = np.arange(5, 45, 2)   # 5 to 44 degrees in 2-degree steps

# ── Connect MATLAB ────────────────────────────────────────────────────────────
print('Connecting...')
sessions = matlab.engine.find_matlab()
name = 'MySharedSession' if 'MySharedSession' in sessions else sessions[0]
eng  = matlab.engine.connect_matlab(name)
_matlab_session.name = name
eng.addpath(os.getcwd() + '/matlab', nargout=0)
eng.pl_setup(nargout=0)
pl_spec.engine = eng

# Set spectrometer: grating 600, centre at 540nm (puts laser + low-end emission in window)
pl_spec.pl_set_settings(exposure_time=1)
eng.eval("instance1.set(PrincetonInstruments.LightField.AddIns.SpectrometerSettings.GratingCenterWavelength, 540);", nargout=0)
eng.eval('instance1.set(PrincetonInstruments.LightField.AddIns.SpectrometerSettings.Grating, "[500nm,600][1][0]");', nargout=0)

# ── Init filter ───────────────────────────────────────────────────────────────
fil.filter_init()
fil.filter_on()
fil.flip_up()   # insert filter
time.sleep(0.5)

# ── Sweep ─────────────────────────────────────────────────────────────────────
laser_counts = []
print(f'\nSweeping {ANGLES[0]:.0f} to {ANGLES[-1]:.0f} degrees...')
print(f'{"Angle":>8}  {"Laser peak":>12}')

for angle in ANGLES:
    fil.rotation_move(float(angle))
    time.sleep(0.2)
    intensity, wl = pl_spec.pl_single_scan()
    intensity = np.array(intensity).flatten()
    wl        = np.array(wl).flatten()
    laser_mask = (wl > 529) & (wl < 535)
    peak = float(intensity[laser_mask].max()) if laser_mask.any() else 0.0
    laser_counts.append(peak)
    print(f'{angle:>8.1f}  {peak:>12.1f}')

# Also take a reference with filter OUT
fil.flip_down()
time.sleep(0.5)
i_ref, w_ref = pl_spec.pl_single_scan()
i_ref = np.array(i_ref).flatten()
w_ref = np.array(w_ref).flatten()
laser_ref = float(i_ref[(w_ref > 529) & (w_ref < 535)].max())
print(f'\nReference (no filter): {laser_ref:.1f} counts')

fil.filter_off()

# ── Plot 1: laser counts vs angle ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ax = axes[0]
ax.plot(ANGLES, laser_counts, 'o-', color='steelblue', ms=5)
ax.axhline(laser_ref, color='gray', ls='--', lw=1, label=f'No filter ({laser_ref:.0f} counts)')
ax.set_xlabel('Filter angle (deg)')
ax.set_ylabel('Laser peak counts (532nm)')
ax.set_title('Laser transmission vs filter angle')
ax.legend()

# ── Plot 2: best spectrum (highest laser) vs reference ────────────────────────
best_idx   = int(np.argmax(laser_counts))
best_angle = ANGLES[best_idx]
print(f'Best angle: {best_angle:.1f} deg  ({laser_counts[best_idx]:.1f} counts  =  {laser_counts[best_idx]/laser_ref*100:.1f}% of unfiltered)')

# re-acquire at best angle for a clean spectrum
fil.filter_init()
fil.filter_on()
fil.rotation_move(float(best_angle))
fil.flip_up()
time.sleep(0.5)
i_best, w_best = pl_spec.pl_single_scan()
i_best = np.array(i_best).flatten()
fil.flip_down()
fil.filter_off()

ax2 = axes[1]
ax2.plot(w_ref,  i_ref,   lw=1,   color='steelblue', label='No filter',               alpha=0.8)
ax2.plot(w_best, i_best,  lw=1.5, color='red',       label=f'Filter @ {best_angle:.0f} deg', alpha=0.9)
ax2.axvline(532, color='orange', ls='--', lw=1, label='532 nm laser')
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Intensity (counts)')
ax2.set_title(f'Spectra: no filter vs best angle ({best_angle:.0f} deg)')
ax2.legend()

plt.tight_layout()
plt.savefig('test_filter_laser.png', dpi=150)
print('Saved: test_filter_laser.png')
