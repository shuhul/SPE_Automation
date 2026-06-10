"""Standalone repro/test for the OSError [WinError 10038] _may_clear_sock crash.

Mirrors automate.py's setup (custom SIGINT handler, Agg-by-default backend),
opens a heatmap window (close it manually to continue), then runs the same
sequence of input() prompts used around the G2 measurement. If the fix in
plotter.py works, no OSError should be raised by the input() calls after the
heatmap window is closed.
"""

import signal
import matplotlib
matplotlib.use('Agg')

import plotter

DATA_FOLDER = 'data'
FOLDERNAME = 'March252026-PhCh21-f007o008-a1-300uW-2s-1'
SCAN_TYPE = 'course'


def _handle_stop(sig, frame):
    print('\nStop requested...')


signal.signal(signal.SIGINT, _handle_stop)
signal.signal(signal.SIGTERM, _handle_stop)


print('Opening heatmap — close the window to continue...')
plotter.open_heatmap(FOLDERNAME, SCAN_TYPE, data_folder=DATA_FOLDER)

print('Heatmap closed. Now testing input() prompts (simulating G2 flow)...')
input('  [G2] Flip mirror to APD path, press Enter when ready...')
print('  [G2] Count rates: (simulated)')
input('  [G2] Press Enter to start acquisition...')
print('  [G2] Acquisition done (simulated)')
input('  [G2] Flip mirror back to spectrometer path, press Enter to continue...')

print('SUCCESS: no OSError raised during input() prompts.')
