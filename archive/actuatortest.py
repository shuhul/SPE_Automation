import sys
import time
import clr 

# --- 1. Configuration ---
serial_num = "71000000"  # Replace with your actual BPC303 serial number
# Look at the Physical Device (Fastest)
# Check the back panel or the bottom of your BPC303 controller. 
# There will be a silver or white Thorlabs sticker with a barcode.
# The 8-digit number printed right below the barcode is your serial number.
channel_to_use = 1       # Channel your PE4 is plugged into (1, 2, or 3)
target_voltage = 15.0    # Voltage to apply to the actuator

# --- 2. Load Thorlabs Libraries ---
sys.path.append(r"C:\Program Files\Thorlabs\Kinesis")
clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
clr.AddReference("Thorlabs.MotionControl.Benchtop.PiezoCLI")
clr.AddReference("System") 

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from Thorlabs.MotionControl.Benchtop.PiezoCLI import BenchtopPiezo
from System import Decimal

# --- 3. Connect to the Controller ---
print("Building device list...")
DeviceManagerCLI.BuildDeviceList()

print(f"Connecting to BPC303 (Serial: {serial_num})...")
controller = BenchtopPiezo.CreateBenchtopPiezo(serial_num)
controller.Connect(serial_num)

# Access the specific channel 
channel = controller.GetChannel(channel_to_use)

# --- 4. Initialize and Enable ---
print(f"Initializing Channel {channel_to_use}...")
channel.WaitForSettingsInitialized(5000)
channel.StartPolling(250)
time.sleep(0.5)

channel.EnableDevice()
time.sleep(0.5)

# --- 5. Apply Voltage ---
print(f"Setting voltage to {target_voltage}V...")
channel.SetOutputVoltage(Decimal(target_voltage))

# Hold for 3 seconds so you can observe the movement
time.sleep(3)

# --- 6. Safe Shutdown ---
print("Resetting to 0V...")
channel.SetOutputVoltage(Decimal(0.0))
time.sleep(0.5) # Give it time to physically retract

print("Disconnecting...")
channel.StopPolling()
channel.DisableDevice()
controller.Disconnect(True)

print("Done. Disconnected safely.")