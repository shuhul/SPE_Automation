import sys
import time
import clr

# --- 1. Configuration ---
serial_num = "27259021"  # Your KDC101 serial number
target_degree_1 = 45.0   # First position
target_degree_2 = 90.0   # Second position

# --- 2. Load Thorlabs Libraries ---
sys.path.append(r"C:\Program Files\Thorlabs\Kinesis")
clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
clr.AddReference("Thorlabs.MotionControl.KCube.DCServoCLI") # KDC101 library
clr.AddReference("System")

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from Thorlabs.MotionControl.KCube.DCServoCLI import KCubeDCServo
from System import Decimal

# --- 3. Connect to the Controller ---
print("Building device list...")
DeviceManagerCLI.BuildDeviceList()

print(f"Connecting to KDC101 (Serial: {serial_num})...")
device = KCubeDCServo.CreateKCubeDCServo(serial_num)
device.Connect(serial_num)

# --- 4. Initialize and Configure ---
print("Initializing...")
device.WaitForSettingsInitialized(5000)

# CRITICAL STEP: Load motor configuration so the controller knows what "degrees" are
device.LoadMotorConfiguration(serial_num)

device.StartPolling(250)
time.sleep(0.5)

device.EnableDevice()
time.sleep(0.5)

# (Optional but recommended) Home the device if it hasn't been homed since powering on
# print("Homing device...")
device.Home(60000) 

# --- 5. Move the Stage ---
print(f"Moving to {target_degree_1} degrees...")
# 60000 is the timeout in milliseconds (60 seconds) to wait for the move to finish
device.MoveTo(Decimal(float(target_degree_1)), 60000)
print(f"Reached {target_degree_1} degrees.")

time.sleep(1) # Pause so you can observe the position

print(f"Moving to {target_degree_2} degrees...")
device.MoveTo(Decimal(float(target_degree_2)), 60000)
print(f"Reached {target_degree_2} degrees.")

# --- 6. Safe Shutdown ---
print("Disconnecting...")
device.StopPolling()
device.DisableDevice()
device.Disconnect(True)

print("Done.")