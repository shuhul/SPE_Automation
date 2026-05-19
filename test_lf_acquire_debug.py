"""Debug why ImageDataSetReceived doesn't fire in bridge context."""
import sys, os, threading, time
sys.path.insert(0, os.path.dirname(__file__))

_LF_PATH = r"C:\Program Files\Princeton Instruments\LightField"
sys.path.append(_LF_PATH)
sys.path.append(_LF_PATH + r"\AddInViews")

import clr
clr.AddReference('PrincetonInstruments.LightFieldViewV5')
clr.AddReference('PrincetonInstruments.LightField.AutomationV5')
clr.AddReference('PrincetonInstruments.LightFieldAddInSupportServices')

from PrincetonInstruments.LightField.Automation import Automation
from PrincetonInstruments.LightField.AddIns import CameraSettings, ExperimentSettings
from System.Collections.Generic import List
from System import String
import numpy as np

print('Connecting...')
auto = Automation(True, List[String]())
exp  = auto.LightFieldApplication.Experiment
print(f'Connected. Experiment: {exp.Name}')

# Check if experiment is ready
try:
    ready = exp.IsReadyToRun
    print(f'IsReadyToRun: {ready}')
except Exception as e:
    print(f'IsReadyToRun not available: {e}')

# Check exposure
try:
    exp_ms = exp.GetValue(CameraSettings.ShutterTimingExposureTime)
    print(f'Exposure: {exp_ms} ms')
except Exception as e:
    print(f'Exposure read error: {e}')

# Try acquire with detailed logging
result = {}
done   = threading.Event()

def on_data(sender, args):
    print(f'EVENT FIRED on thread {threading.current_thread().name}')
    try:
        frame = args.ImageDataSet.GetFrame(0, 0)
        data  = np.array(list(frame.GetData()), dtype=np.float64)
        result['data'] = data
        print(f'  Got {len(data)} points  max={data.max():.1f}')
    except Exception as e:
        result['error'] = str(e)
        print(f'  Data extraction error: {e}')
    done.set()

print('Subscribing to ImageDataSetReceived...')
exp.ImageDataSetReceived += on_data

print('Calling Acquire()...')
exp.Acquire()
print('Acquire() returned — waiting for event (30s timeout)...')

fired = done.wait(timeout=30)
exp.ImageDataSetReceived -= on_data

if fired:
    print('SUCCESS')
else:
    print('TIMEOUT — event never fired')
    # Check if acquisition is still running
    try:
        print(f'IsRunning: {exp.IsRunning}')
    except:
        pass

auto.Dispose()
print('Done.')
