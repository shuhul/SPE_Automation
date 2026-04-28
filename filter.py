import sys
import time
import clr

sys.path.append(r"C:\Program Files\Thorlabs\Kinesis")
clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
clr.AddReference("Thorlabs.MotionControl.FilterFlipperCLI") 
clr.AddReference("Thorlabs.MotionControl.KCube.DCServoCLI")
clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI") 
clr.AddReference("System")

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from Thorlabs.MotionControl.FilterFlipperCLI import FilterFlipper
from Thorlabs.MotionControl.KCube.DCServoCLI import KCubeDCServo
from System import Decimal
from Thorlabs.MotionControl.GenericMotorCLI.Settings import RotationSettings


flip_serial_num = "37010764" 

flipper = None


rotation_serial_num = "27600279"

rotation = None


wait4action = 0.5


def filter_init():
    global flipper, rotation

    print("Building device list for filter...")
    DeviceManagerCLI.BuildDeviceList()

    print(f"Connecting to Filter Flipper (Serial: {flip_serial_num})...")
    flipper = FilterFlipper.CreateFilterFlipper(flip_serial_num)
    flipper.Connect(flip_serial_num)

    print(f"Connecting to Filter Rotation Stage (Serial: {rotation_serial_num})...")
    rotation = KCubeDCServo.CreateKCubeDCServo(rotation_serial_num)
    rotation.Connect(rotation_serial_num)

    print("Initializing filter...")
    rotation.WaitForSettingsInitialized(5000)
    time.sleep(wait4action)
    flipper.WaitForSettingsInitialized(5000)
    time.sleep(wait4action)
    rotation.LoadMotorConfiguration(rotation_serial_num)
    rotation.StartPolling(250)
    time.sleep(wait4action)
    flipper.StartPolling(250)
    time.sleep(wait4action)
    print("Done initializing filter!")


def filter_on():
    global flipper, rotation
    print("Turning filter on...")
    flipper.EnableDevice()
    time.sleep(wait4action)
    rotation.EnableDevice()
    time.sleep(wait4action)
    print("Done turning on!")

def filter_off():
    global flipper, rotation
    print("Turning filter off...")
    flipper.StopPolling()
    flipper.DisableDevice()
    flipper.Disconnect(True)
    rotation.StopPolling()
    rotation.DisableDevice()
    rotation.Disconnect(True)
    time.sleep(wait4action)
    print("Done turning off!")

def flip_up():
    global flipper
    print("Flipping up...")
    flipper.SetPosition(2, 60000) 
    time.sleep(wait4action) 
    print("Done Flipping!")

def flip_down():
    global flipper
    print("Flipping down...")
    flipper.SetPosition(1, 60000) 
    time.sleep(wait4action) 
    print("Done Flipping!")

def rotation_home():
    global rotation
    print("Homing rotation stage...")
    rotation.Home(60000) 
    print("Done Homing!")

def rotation_move(target):
    global rotation
    print(f"Moving to {target} degrees...")
    rotation.MoveTo(Decimal(float(target)), 60000)
    print("Done Moving!")

def set_rotation_mode(mode="quickest"):
    """
    Sets the direction behavior for the rotation stage.
    
    Args:
        mode (str): The desired rotation mode. Valid options are:
            * 'quickest': Calculates the shortest path to the target angle.
            * 'forwards': Forces rotation in the positive direction only.
            * 'reverse': Forces rotation in the negative direction only.
    """
    global rotation
    
    # Clean up the input string to make it case-insensitive
    clean_mode = str(mode).strip().lower()
    
    # Map the friendly string to the Thorlabs C# enumeration
    if clean_mode == "quickest":
        enum_mode = RotationSettings.RotationDirections.Quickest
    elif clean_mode == "forwards":
        enum_mode = RotationSettings.RotationDirections.Forwards
    elif clean_mode == "reverse":
        enum_mode = RotationSettings.RotationDirections.Reverse
    else:
        raise ValueError(f"Invalid rotation mode '{mode}'. Please use 'quickest', 'forwards', or 'reverse'.")

    print(f"Setting rotation mode to {clean_mode}...")
    
    # Update the setting in memory
    rotation.MotorDeviceSettings.Rotation.RotationDirection = enum_mode
    
    # Push the setting to the hardware (False = don't save permanently to flash)
    rotation.SetSettings(rotation.MotorDeviceSettings, False, False)
    
    print("Done setting rotation mode!")
