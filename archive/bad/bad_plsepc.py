import clr
import sys
import os
from System.IO import *
from System import String
from System.Collections.Generic import List

print('Loading dll files')

sys.path.append(os.environ['LIGHTFIELD_ROOT'])
sys.path.append(os.environ['LIGHTFIELD_ROOT']+"\\AddInViews")
clr.AddReference('PrincetonInstruments.LightFieldViewV5')
clr.AddReference('PrincetonInstruments.LightField.AutomationV5')
clr.AddReference('PrincetonInstruments.LightFieldAddInSupportServices')

print('Done loading, starting LightField')

from time import sleep

from PrincetonInstruments.LightField.Automation import Automation
from PrincetonInstruments.LightField.AddIns import ExperimentSettings
from PrincetonInstruments.LightField.AddIns import DeviceType


def device_found():
    for device in experiment.ExperimentDevices:
        if (device.Type == DeviceType.Camera):
            return True
    print("Camera not found. Please add a camera and try again.")
    return False  


auto = Automation(True, List[String]())
experiment = auto.LightFieldApplication.Experiment

print('Light field started! (waiting 3 secs)')

sleep(3)


print('Loading Experiment PL')

experiment.Load("PL")

sleep(3)

print('Aquire')

if device_found():        
    experiment.Acquire()

    print(String.Format("{0} {1}",
                        "Image saved to",
                        experiment.GetValue(
                            ExperimentSettings.
                            FileNameGenerationDirectory)))          


print('Done')