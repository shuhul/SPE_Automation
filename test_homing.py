import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import clr
sys.path.append(r"C:\Program Files\Thorlabs\Kinesis")
clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
clr.AddReference("Thorlabs.MotionControl.KCube.DCServoCLI")
clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
clr.AddReference("System")

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from Thorlabs.MotionControl.KCube.DCServoCLI import KCubeDCServo
from System import Decimal

SERIAL = "27600279"

DeviceManagerCLI.BuildDeviceList()
rotation = KCubeDCServo.CreateKCubeDCServo(SERIAL)
rotation.Connect(SERIAL)
rotation.WaitForSettingsInitialized(5000)
rotation.LoadMotorConfiguration(SERIAL)
rotation.StartPolling(250)
time.sleep(0.5)
rotation.EnableDevice()
time.sleep(0.5)

params = rotation.GetHomingParams()
print(f'Before — Direction: {params.Direction}')

direction_type = type(params.Direction)
params.Direction = direction_type.Clockwise
rotation.SetHomingParams(params)

print(f'After  — Direction: {rotation.GetHomingParams().Direction}')
print('Homing now...')
rotation.Home(60000)
print('Done homing.')

rotation.StopPolling()
rotation.DisableDevice()
rotation.Disconnect(True)

print('\n>>> Did the stage rotate CLOCKWISE? Tell me yes or no.')
