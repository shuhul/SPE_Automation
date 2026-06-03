import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import time
import pyvisa
import threading
import msvcrt
import requests

stop_flag = False

def key_listener():
    global stop_flag
    while True:
        if msvcrt.kbhit() and msvcrt.getch() == b'q':
            print("\n[KEYBOARD] Stop requested (q pressed).")
            stop_flag = True
            break

# Choose from kristina or shuhul
current_user = "kristina"

xdim = 2
ydim = 2
dx = dy = 0.1
fine = False
xdim_fine = 0.5
ydim_fine = 0.5
dx_fine=dy_fine=0.25

center = (0, 0)

filename='20251124-PLSPC-f015o016-nwg-postetch-3'

filename_fine='20251121-PLSPC-f015o016-nwg-postetch-2sIntegration_1_fine'


print('Scanning for intstruments')
resource_str = 'USB0::0xF4ED::0xEE3A::SDG10GAD1R1771::0::INSTR'
rm = pyvisa.ResourceManager()
try:
    obj1 = rm.open_resource(resource_str)
except pyvisa.VisaIOError:
    raise RuntimeError(f"Instrument {resource_str} not found.")

obj1.clear()
print("Getting ready!")
wait = 0.2
obj1.write('C1: BSWV WVTP, DC')
time.sleep(wait)
obj1.write('C1: BSWV OFST, 0')
time.sleep(wait)
obj1.write('C2: BSWV WVTP, DC')
time.sleep(wait)
obj1.write('C2: BSWV OFST, 0')
time.sleep(wait)
print('Ready for use!')

XCONV = -1.85 / 20
YCONV = 2.65 / 20
wait4action = 0.1
def sgd_on():
    print("Sgd on!")
    obj1.write('C1: OUTP ON')
    time.sleep(wait4action)
    obj1.write('C2: OUTP ON')
    time.sleep(wait4action)

def sgd_off():
    print("Sgd off!")
    obj1.write('C1: BSWV OFST, 0')
    time.sleep(wait4action)
    obj1.write('C2: BSWV OFST, 0')
    time.sleep(wait4action)
    obj1.write('C1: OUTP OFF')
    time.sleep(wait4action)
    obj1.write('C2: OUTP OFF')
    time.sleep(wait4action)

def set_position(x_um, y_um):
    print(f"Moving to ({x_um} um, {y_um} um)")
    x_volt = x_um * XCONV
    y_volt = -y_um * YCONV
    if abs(x_volt) > 10 or abs(y_volt) > 10:
        raise ValueError("Requested position is out of SDG range.")
    obj1.write(f'C1: BSWV OFST, {y_volt}')
    time.sleep(wait4action)
    obj1.write(f'C2: BSWV OFST, {x_volt}')
    time.sleep(wait4action)
    print("Done moving!")

def fix_length(arr, target_len):
    arr = np.array(arr)
    if len(arr) > target_len:
        return arr[:target_len]
    elif len(arr) < target_len:
        return np.pad(arr, (0, target_len - len(arr)), mode='constant')
    else:
        return arr

print('Connecting to matlab')
eng = matlab.engine.connect_matlab('MySharedSession')

print('Getting wavelengths')

wl = np.array(eng.workspace['wl']).flatten()
np.save(f'wl', wl)

num = len(wl)


print('Starting scan...')



sgd_on()



# Assume xdim, ydim, dx, dy, XCONV, YCONV are already defined

if abs(xdim / 2 * XCONV) > 10 or abs(ydim / 2 * YCONV) > 10:
    raise ValueError("Scan area too large for mirror")

if dx > xdim or dy > ydim:
    raise ValueError("Step size too large")

if not fine:
    xs = np.arange(-xdim/2, xdim/2+dx, dx)
    ys = np.arange(-ydim/2, ydim/2+dy, dy)
else:
    xs = np.arange(-xdim_fine/2 + center[0], xdim_fine/2+dx_fine + center[0], dx_fine)
    ys = np.arange(-ydim_fine/2 + center[1], ydim_fine/2+dy_fine + center[1], dy_fine)



listener = threading.Thread(target=key_listener, daemon=True)
listener.start()

output = np.zeros((len(ys), len(xs), num))

print('Press q to exit')

TOKEN = "8463582982:AAG-izcwemLDy4l2A2ouEAXJDGzHL8xHD5A"


users = ["shuhul", "kristina"]
CHAT_IDS = ["8130896008", "7568051086"]

def send_telegram_message(message):
    for user, CHAT_ID in zip(users, CHAT_IDS):
        if user == current_user:
            url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
            payload = {
                "chat_id": CHAT_ID,
                "text": message
            }
            try:
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    print("Message sent!")
                else:
                    print(f"Failed: {response.text}")
            except Exception as e:
                print(f"Error: {e}")

first = True
sent = False
cutoff = -1
laser_peak_cutoff_fraction = 0.5 # fraction of first intensity

# output = []
for iy, y in enumerate(ys):
    # output_x = []
    for ix, x in enumerate(xs):
        print(f'Aquire at x={x}, y={y}')
        set_position(x, y)
        intensity, wavelength = eng.eval("instance1.acquire;", nargout=2)
        intensity = np.array(intensity).flatten()
        intensity = fix_length(intensity, num)
        wl = np.array(wavelength).flatten()
        np.save(f'wl', wl)

        # print(wavelength)
    


        

        # if ix >= 1 :

            # if ix == 1:
            #     wl = np.array(eng.workspace['wl']).flatten()
            #     np.save(f'wl', wl)
    
        window = (wl > 529) & (wl < 535)
        peak_idx = np.argmax(intensity[window])
        peak_intensity = intensity[window][peak_idx]
        peak_wavelength = wl[window][peak_idx]

        if first:
            print(f'First peak intensity {peak_intensity} @ {np.round(peak_wavelength,0)} nm')
            cutoff = laser_peak_cutoff_fraction*peak_intensity
            print(f'Cutoff {cutoff}')
            first = False
        elif peak_intensity < cutoff and not sent:
            # Send notification
            send_telegram_message("WARNING: Out of focus!")
            sent = True
            print(f'Out of focus at x={x}, y={y}')
            pass


        # output_x.append(intensity)
        output[iy, ix, :] = intensity

        # sleep(1)

        # Get the initial intensity of the laser (532 peak)
        # Check every 5 pixels
        # Above cutoff to be in focus
        
        # If goes out of focus 
        # send some sort of notification, but continue, text 
        
        # Figure out a way to cancel, populate rest with 0 and still save

        # Put data into the classifier, places that are correct highlights

        if stop_flag:
            break
    
    
    # output.append(output_x)

    if stop_flag:
        break



if not fine:
    np.save(filename, np.array(output))
    np.save('xs', xs)
    np.save('ys', ys)
else:
    np.save(filename_fine, np.array(output))
    np.save('xs_fine', xs)
    np.save('ys_fine', ys)


sgd_off()
