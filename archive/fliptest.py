import sys
import time
import clr

# --- 1. Configuration ---
# MFF101 Serial numbers typically start with "37"
serial_num = "37000000"  # Replace with your actual MFF101 serial number

# --- 2. Load Thorlabs Libraries ---
sys.path.append(r"C:\Program Files\Thorlabs\Kinesis")
clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
clr.AddReference("Thorlabs.MotionControl.FilterFlipperCLI") # Note the Flipper DLL!
clr.AddReference("System")

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from Thorlabs.MotionControl.FilterFlipperCLI import FilterFlipper

# --- 3. Connect to the Flipper ---
print("Building device list...")
DeviceManagerCLI.BuildDeviceList()

print(f"Connecting to Filter Flipper (Serial: {serial_num})...")
# Create the specific Filter Flipper object
flipper = FilterFlipper.CreateFilterFlipper(serial_num)
flipper.Connect(serial_num)

# --- 4. Initialize ---
print("Initializing...")
flipper.WaitForSettingsInitialized(5000)
flipper.StartPolling(250)
time.sleep(0.5)

flipper.EnableDevice()
time.sleep(0.5)

# --- 5. Flip the Filter! ---
# The MFF101 only has two physical positions: 1 and 2

print("Moving to Position 1...")
# The SetPosition method takes the target position (1 or 2) 
# and a wait timeout in milliseconds (60000 ms = 60s)
flipper.SetPosition(1, 60000) 
time.sleep(1) # Wait a second so you can visually observe it

print("Moving to Position 2...")
flipper.SetPosition(2, 60000)
time.sleep(1) 

print("Moving back to Position 1...")
flipper.SetPosition(1, 60000)

# --- 6. Safe Shutdown ---
print("Disconnecting...")
flipper.StopPolling()
flipper.DisableDevice()
flipper.Disconnect(True)

print("Done.")