"""
Direct Python interface to Princeton Instruments LightField via the Automation API.
Drop-in replacement for the MATLAB-based spectrometer calls in pl_spec.py.
LightField must already be open before using any function here.
"""
import sys
import threading
import numpy as np

import clr

_LF_PATH        = r"C:\Program Files\Princeton Instruments\LightField"
_LF_ADDINVIEWS  = _LF_PATH + r"\AddInViews"

sys.path.append(_LF_PATH)
sys.path.append(_LF_ADDINVIEWS)

clr.AddReference('PrincetonInstruments.LightFieldViewV5')
clr.AddReference('PrincetonInstruments.LightField.AutomationV5')
clr.AddReference('PrincetonInstruments.LightFieldAddInSupportServices')

from PrincetonInstruments.LightField.Automation import Automation
from PrincetonInstruments.LightField.AddIns import (
    SpectrometerSettings,
    CameraSettings,
)
from System.Collections.Generic import List
from System import String

_auto = None
_app  = None
_exp  = None

GRATING_150 = '[800nm,150][2][0]'
GRATING_600 = '[500nm,600][1][0]'


# ── Connection ────────────────────────────────────────────────────────────────

def lf_connect():
    """Connect to the running LightField instance."""
    global _auto, _app, _exp
    print('Connecting to LightField...')
    _auto = Automation(True, List[String]())
    _app  = _auto.LightFieldApplication
    _exp  = _app.Experiment
    print(f"  Connected. Experiment: '{_exp.Name}'")


def lf_disconnect():
    """Disconnect cleanly."""
    global _auto, _app, _exp
    if _auto is not None:
        _auto.Dispose()
        _auto = _app = _exp = None
    print('Disconnected from LightField.')


# ── Settings ──────────────────────────────────────────────────────────────────

def lf_set_exposure(exposure_s):
    """Set exposure time in seconds."""
    _exp.SetValue(CameraSettings.ShutterTimingExposureTime, float(exposure_s * 1000))


def lf_set_center_wavelength(wl_nm):
    """Set spectrometer center wavelength in nm."""
    _exp.SetValue(SpectrometerSettings.GratingCenterWavelength, float(wl_nm))


def lf_set_grating(grating):
    """Set grating. Pass 150 or 600 (g/mm), or the full string."""
    if grating == 150:
        grating = GRATING_150
    elif grating == 600:
        grating = GRATING_600
    _exp.SetValue(SpectrometerSettings.GratingSelected, grating)


def lf_get_wavelengths():
    """Return current wavelength calibration as a numpy array."""
    return np.array(list(_exp.SystemColumnCalibration))


# ── Acquire ───────────────────────────────────────────────────────────────────

def lf_acquire():
    """
    Trigger a single acquisition and return (intensity, wavelength) as numpy arrays.
    Blocks until the frame arrives (or 60s timeout).
    """
    result = {}
    event  = threading.Event()

    def _on_data(sender, args):
        try:
            frame          = args.ImageDataSet.GetFrame(0, 0)
            result['data'] = np.array(list(frame.GetData()), dtype=np.float64)
        except Exception as e:
            result['error'] = str(e)
        event.set()

    _exp.ImageDataSetReceived += _on_data
    _exp.Acquire()
    fired = event.wait(timeout=60)
    _exp.ImageDataSetReceived -= _on_data

    if not fired:
        raise TimeoutError('LightField acquisition timed out after 60s.')
    if 'error' in result:
        raise RuntimeError(f'LightField data extraction error: {result["error"]}')

    intensity = result['data']
    wl        = lf_get_wavelengths()
    return intensity, wl


# ── Convenience ───────────────────────────────────────────────────────────────

def lf_setup(exposure_s=1, center_wavelength=700, grating=150):
    """Set exposure, center wavelength, and grating in one call."""
    lf_set_exposure(exposure_s)
    lf_set_center_wavelength(center_wavelength)
    lf_set_grating(grating)
