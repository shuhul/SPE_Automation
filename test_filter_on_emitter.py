"""
Set bandpass filter to the emitter at (-3.5, -14.0), take a spectrum with and
without filter, and plot both to verify filtering is working.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matlab.engine, _matlab_session
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pl_spec, filter as fil, sgd
from main import find_emission_fwhm_center, angle_for_wavelength

CENTER    = (-3.5, -14.0)
CAL       = '2026-04-07_14-48-20'
SCAN_DIR  = 'data/20260512-PLSPC-PhENOM-Ch23-f011o015-f1a1-500uW-3s-1/long_x-3.5_y-14.0'

# ── 1. Connect MATLAB ─────────────────────────────────────────────────────────
print('Connecting to MATLAB...')
sessions = matlab.engine.find_matlab()
name = 'MySharedSession' if 'MySharedSession' in sessions else sessions[0]
eng  = matlab.engine.connect_matlab(name)
_matlab_session.name = name
eng.addpath(os.getcwd() + '/matlab', nargout=0)
eng.pl_setup(nargout=0)
pl_spec.engine = eng
print(f"  Connected to '{name}'.")

# ── 2. Init hardware ──────────────────────────────────────────────────────────
print('Initialising SGD + filter...')
sgd.sgd_init()
fil.filter_init()
fil.filter_on()

# ── 3. Get target wavelength from the saved long scan ─────────────────────────
spectrum_ref = np.load(f'{SCAN_DIR}/out.npy')[0, 0, :]
wl_ref       = np.load(f'{SCAN_DIR}/wl.npy')
target_wl    = find_emission_fwhm_center(spectrum_ref, wl_ref)
angle        = angle_for_wavelength(CAL, target_wl)
print(f'\nTarget wavelength: {target_wl:.2f} nm  ->  angle: {angle:.2f} deg')

# ── 4. Position SGD and set spectrometer to match long scan settings ──────────
sgd.sgd_on()
sgd.set_position(CENTER[0], CENTER[1])

# Use same settings as the long scan: grating 600, center_wavelength = midpoint
cwl = int((532 + target_wl) / 2)
pl_spec.pl_set_settings(exposure_time=3)
eng.eval(f"instance1.set(PrincetonInstruments.LightField.AddIns.SpectrometerSettings.GratingCenterWavelength, {cwl});", nargout=0)
eng.eval('instance1.set(PrincetonInstruments.LightField.AddIns.SpectrometerSettings.Grating, "[500nm,600][1][0]");', nargout=0)

# ── 5. Spectrum WITHOUT filter ────────────────────────────────────────────────
print('\nAcquiring spectrum WITHOUT filter...')
fil.flip_down()
time.sleep(0.5)
i_nofilter, w_nofilter = pl_spec.pl_single_scan()
i_nofilter = np.array(i_nofilter).flatten()
w_nofilter = np.array(w_nofilter).flatten()
print(f'  Peak: {i_nofilter.max():.0f} counts @ {w_nofilter[np.argmax(i_nofilter)]:.1f} nm')

# ── 6. Set filter angle + spectrum WITH filter ────────────────────────────────
print(f'\nRotating filter to {angle:.2f} deg and acquiring WITH filter...')
fil.rotation_move(angle)
fil.flip_up()
time.sleep(0.5)
i_filter, w_filter = pl_spec.pl_single_scan()
i_filter = np.array(i_filter).flatten()
w_filter = np.array(w_filter).flatten()
print(f'  Peak: {i_filter.max():.0f} counts @ {w_filter[np.argmax(i_filter)]:.1f} nm')

# ── 7. Remove filter + shut down ─────────────────────────────────────────────
fil.flip_down()
sgd.sgd_off()
fil.filter_off()

# ── 8. Plot both ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(w_nofilter, i_nofilter, lw=1,   color='steelblue', label='No filter',      alpha=0.9)
ax.plot(w_filter,   i_filter,   lw=1.5, color='red',       label='Filter in',      alpha=0.9)
ax.axvline(target_wl, color='orange', ls='--', lw=1.2, label=f'Target: {target_wl:.1f} nm')
ax.set_xlabel('Wavelength (nm)', fontsize=12)
ax.set_ylabel('Intensity (counts)', fontsize=12)
ax.set_title(f'Bandpass filter test — emitter ({CENTER[0]}, {CENTER[1]})  |  filter @ {angle:.1f} deg', fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('test_filter_on_emitter.png', dpi=150)
print('\nSaved: test_filter_on_emitter.png')
