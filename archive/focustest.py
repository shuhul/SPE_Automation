import pl_init
import matlab.engine
import os
import datetime
import numpy as np
from tqdm import tqdm
import time
import clr 
import sys


pl_init.pl_init()


print('Connecting to matlab...')
eng = matlab.engine.connect_matlab('MySharedSession')

# data_folder = 'temp'
# foldername = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-Focus"

exposure_time = 1 # seconds
center_wavelength = 700 # nm

# print('Getting wavelengths and setting up...')
# folder_path = os.path.join(data_folder, foldername)
# os.makedirs(folder_path, exist_ok=True)

wl = np.array(eng.workspace['wl']).flatten()
# np.save(f'{data_folder}/{foldername}/wl', wl)
num = len(wl)

eng.eval(f"instance1.set_exposure({int(exposure_time*1000)});", nargout=0)
eng.eval(f"instance1.set(PrincetonInstruments.LightField.AddIns.SpectrometerSettings.GratingCenterWavelength, {int(center_wavelength)});", nargout=0)



serial_num = "71000000"  # Replace with your actual BPC303 serial number
channel_to_use = 1       # Channel your PE4 is plugged into
min_voltage = 0.0
max_voltage = 75.0       # Adjust based on your piezo's max range (usually 75V or 150V)
num_steps = 50           # Number of voltage points to test
settle_time = 0.1        # Seconds to wait for the piezo to stabilize before acquiring






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

channel = controller.GetChannel(channel_to_use)

print(f"Initializing Channel {channel_to_use}...")
channel.WaitForSettingsInitialized(5000)
channel.StartPolling(250)
time.sleep(0.5)

channel.EnableDevice()
time.sleep(0.5)




# --- 4. Autofocus Sweep ---
# Create an array of voltages to test
voltages = np.linspace(min_voltage, max_voltage, num_steps)
peak_intensities = np.zeros(num_steps)

print("Starting autofocus sweep...")

# Note: Assumes 'eng' (MATLAB engine) is already started in your environment
for i, v in enumerate(tqdm(voltages, desc="Sweeping Focus")):
    
    # Move Piezo to current voltage step
    # Note: We cast v to standard float before converting to Decimal to avoid type issues
    channel.SetOutputVoltage(Decimal(float(v)))
    time.sleep(settle_time) # Crucial: allow physical movement to finish
    
    # Acquire Data
    intensity, wavelength = eng.eval("instance1.acquire;", nargout=2)
    
    # Process Data
    intensity = np.array(intensity).flatten()
    wl = np.array(wavelength).flatten()
    
    # Evaluate Focus Metric (Max intensity in target window)
    window = (wl > 529) & (wl < 535)
    
    if np.any(window):
        peak_intensities[i] = np.max(intensity[window])
    else:
        peak_intensities[i] = 0

# --- 5. Determine and Set Optimal Focus ---
best_idx = np.argmax(peak_intensities)
best_voltage = voltages[best_idx]
best_intensity = peak_intensities[best_idx]

print(f"\nSweep complete!")
print(f"Optimal focus found at {best_voltage:.2f} V (Peak Intensity: {best_intensity:.2f})")

print("Moving piezo to optimal focus position...")
channel.SetOutputVoltage(Decimal(float(best_voltage)))
time.sleep(0.5)

# Optional: Stop polling but keep device enabled so it holds the voltage
channel.StopPolling()
# Do NOT call channel.DisableDevice() or set voltage to 0 if you want it to stay in focus!

print("Autofocus complete. Ready for 2D scan.")

# channel.DisableDevice()
# controller.Disconnect(True)

# print("Done. Disconnected safely.")