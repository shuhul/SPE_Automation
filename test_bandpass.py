"""
Test run_bandpass_setup against real hardware using the test dataset target wavelength.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matlab.engine
import _matlab_session
import numpy as np
import pl_spec
import filter as fil
from main import find_emission_fwhm_center, run_bandpass_setup

CAL = '2026-04-07_14-48-20'

# ── 1. Connect MATLAB ─────────────────────────────────────────────────────────
print('Looking for MATLAB sessions...')
sessions = matlab.engine.find_matlab()
print(f'  Found: {sessions}')
if not sessions:
    print('ERROR: No MATLAB sessions found.')
    sys.exit(1)

name = 'MySharedSession' if 'MySharedSession' in sessions else sessions[0]
eng  = matlab.engine.connect_matlab(name)
_matlab_session.name = name
eng.addpath(os.path.join(os.getcwd(), 'matlab'), nargout=0)
eng.pl_setup(nargout=0)
pl_spec.engine = eng   # set directly so run_bandpass_setup doesn't reconnect
print(f"  Connected to '{name}'.")

# ── 2. Init filter hardware ───────────────────────────────────────────────────
print('Initialising filter...')
fil.filter_init()
fil.filter_on()
print('  Filter ready.')

# ── 3. Get target wavelength from test dataset ────────────────────────────────
spectrum = np.load('test_data/long_x-10.5_y-10.5/out.npy')[0, 0, :]
wl       = np.load('test_data/long_x-10.5_y-10.5/wl.npy')
target_wl = find_emission_fwhm_center(spectrum, wl)
print(f'\nTarget wavelength from test dataset: {target_wl:.2f} nm')

# ── 4. Run bandpass setup ─────────────────────────────────────────────────────
aligned = run_bandpass_setup(
    target_wl    = target_wl,
    cal_folder   = CAL,
    current_user = 'shuhul',
)

print(f'\nResult: {"ALIGNED" if aligned else "FAILED"}')

# ── 5. Flip filter back up when done ─────────────────────────────────────────
if aligned:
    fil.flip_down()
fil.filter_off()
