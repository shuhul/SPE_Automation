import sys
import clr

LF_PATH = r"C:\Program Files\Princeton Instruments\LightField"
LF_ADDINVIEWS = LF_PATH + r"\AddInViews"

sys.path.append(LF_PATH)
sys.path.append(LF_ADDINVIEWS)

clr.AddReference('PrincetonInstruments.LightFieldViewV5')
clr.AddReference('PrincetonInstruments.LightField.AutomationV5')
clr.AddReference('PrincetonInstruments.LightFieldAddInSupportServices')

from PrincetonInstruments.LightField.Automation import Automation
from System.Collections.Generic import List
from System import String

# True = connect to existing visible LightField instance
auto = Automation(True, List[String]())
app  = auto.LightFieldApplication
exp  = app.Experiment

from PrincetonInstruments.LightField.AddIns import (
    SpectrometerSettings, CameraSettings, ExperimentSettings
)
import numpy as np
import threading

print(f'Connected. Experiment: {exp.Name}')

# 1. Wavelength calibration
wl = list(exp.SystemColumnCalibration)
print(f'Wavelength calibration: {len(wl)} points, {wl[0]:.2f} - {wl[-1]:.2f} nm')

# 2. Read current settings
cwl    = exp.GetValue(SpectrometerSettings.GratingCenterWavelength)
grat   = exp.GetValue(SpectrometerSettings.GratingSelected)
exp_ms = exp.GetValue(CameraSettings.ShutterTimingExposureTime)
print(f'Current: CWL={cwl}nm  Grating={grat}  Exposure={exp_ms}ms')

# 3. Set exposure to 500ms and read back
exp.SetValue(CameraSettings.ShutterTimingExposureTime, 500.0)
print(f'Set exposure to 500ms -> readback: {exp.GetValue(CameraSettings.ShutterTimingExposureTime)}ms')

# 4. Set grating center wavelength to 600nm and read back
exp.SetValue(SpectrometerSettings.GratingCenterWavelength, 600.0)
print(f'Set CWL to 600nm -> readback: {exp.GetValue(SpectrometerSettings.GratingCenterWavelength)}nm')

# 5. Set grating to 600 g/mm
exp.SetValue(SpectrometerSettings.GratingSelected, '[500nm,600][1][0]')
print(f'Set grating -> readback: {exp.GetValue(SpectrometerSettings.GratingSelected)}')

import time; time.sleep(2)  # let grating move

# 6. Read updated wavelength calibration
wl2 = list(exp.SystemColumnCalibration)
print(f'New wavelength cal: {len(wl2)} points, {wl2[0]:.2f} - {wl2[-1]:.2f} nm')

# 7. Restore original settings
exp.SetValue(CameraSettings.ShutterTimingExposureTime, exp_ms)
exp.SetValue(SpectrometerSettings.GratingCenterWavelength, cwl)
exp.SetValue(SpectrometerSettings.GratingSelected, grat)
print(f'Restored original settings.')

# 3. Try a synchronous acquire via event
result = {'data': None}
event  = threading.Event()

def on_data(sender, args):
    try:
        ds = args.ImageDataSet
        # Must extract data here before ds is disposed
        frame = ds.GetFrame(0, 0)
        buf   = frame.GetData()
        result['data'] = np.array(list(buf), dtype=np.float64)
    except Exception as e:
        # Try GetDataBuffer as fallback
        try:
            buf = args.ImageDataSet.GetDataBuffer(0)
            result['data'] = np.array(list(buf), dtype=np.float64)
        except Exception as e2:
            result['data'] = f'ERROR frame: {e} | buffer: {e2}'
    event.set()

exp.ImageDataSetReceived += on_data
exp.Acquire()
fired = event.wait(timeout=30)
exp.ImageDataSetReceived -= on_data

if not fired:
    print('Acquire timed out.')
elif isinstance(result['data'], str):
    print(result['data'])
else:
    d = result['data']
    print(f'\nAcquired {len(d)} points  min={d.min():.1f}  max={d.max():.1f}  mean={d.mean():.1f}')

auto.Dispose()
print('Done.')
