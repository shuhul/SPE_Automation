import matlab.engine
import numpy as np
import threading
import requests
import sgd
import lf_spec
import classifier as classifer
from tqdm import tqdm
import os
import time
import _matlab_session

# --- EXIT CODES ---
EXIT_SUCCESS = 0
EXIT_STOPPED = 1
EXIT_FOCUS_WARNING = 2


simulate_out_of_focus = False


# --- Helper Functions ---
def fix_length(arr, target_len):
    arr = np.array(arr)
    if len(arr) > target_len:
        return arr[:target_len]
    elif len(arr) < target_len:
        return np.pad(arr, (0, target_len - len(arr)), mode='constant')
    else:
        return arr


def send_telegram_message(current_user, message):
    # Hardcoded credentials (as provided)
    TOKEN = "8463582982:AAG-izcwemLDy4l2A2ouEAXJDGzHL8xHD5A"
    users = ["shuhul", "kristina", "holland"]
    CHAT_IDS = ["8130896008", "7568051086", "8743893517"]

    if current_user in users:
        try:
            chat_id = CHAT_IDS[users.index(current_user)]
            url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
            payload = {"chat_id": chat_id, "text": message}
            requests.post(url, json=payload)
        except Exception as e:
            print(f"Telegram Error: {e}")

def pl_spec(xdim, ydim, dx, dy, foldername, current_user, center=(0,0), 
            data_folder='data', scan_type='course', exposure_time=1, center_wavelength=700, stop_event=None, progress_signal=None):
    global simulate_out_of_focus
    """
    Executes the PL Spectrum scan.

    Args:
        stop_event (threading.Event, optional): A thread-safe flag to trigger a stop.
        progress_signal (pyqtSignal, optional): A signal to emit progress (int 0-100).
    
    Returns:
        int: EXIT_SUCCESS (0), EXIT_STOPPED (1), or EXIT_FOCUS_WARNING (2)
    """

    # State flags
    out_of_focus_detected = False
    aborted = False

    # --- Hardware Initialization ---
    print('Connecting to matlab...')
    eng = matlab.engine.connect_matlab(_matlab_session.name)

    print('Getting wavelengths and setting up...')
    folder_path = os.path.join(data_folder, foldername)
    os.makedirs(folder_path, exist_ok=True)


    os.makedirs(f'{data_folder}/{foldername}/{scan_type}', exist_ok=True)

    wl = np.array(eng.workspace['wl']).flatten()
    np.save(f'{data_folder}/{foldername}/{scan_type}/wl', wl)
    num = len(wl)

    # # instance1.set(PrincetonInstruments.LightField.AddIns.SpectrometerSettings.Grating, "[800nm,150][2][0]");
    # instance1.set(PrincetonInstruments.LightField.AddIns.SpectrometerSettings.GratingCenterWavelength, 660)
    # [500nm,600][1][0]

    # Set Exposure Time
    eng.eval(f"instance1.set_exposure({int(exposure_time*1000)});", nargout=0)
    eng.eval(f"instance1.set(PrincetonInstruments.LightField.AddIns.SpectrometerSettings.GratingCenterWavelength, {int(center_wavelength)});", nargout=0)
    if 'long' in scan_type:
        eng.eval('instance1.set(PrincetonInstruments.LightField.AddIns.SpectrometerSettings.Grating, "[500nm,600][1][0]");', nargout=0)
    else:
        eng.eval('instance1.set(PrincetonInstruments.LightField.AddIns.SpectrometerSettings.Grating, "[800nm,150][2][0]");', nargout=0)

    

    print('Starting scan...')
    sgd.sgd_on()

    # Safety Checks
    if abs(xdim / 2 * sgd.XCONV) > 10 or abs(ydim / 2 * sgd.YCONV) > 10:
        raise ValueError("Scan area too large for mirror")
    if dx > xdim or dy > ydim:
        raise ValueError("Step size too large")
    
    if 'long' in scan_type:
        xs = np.array([center[0]])
        ys = np.array([center[1]])
    else:
        xs = np.arange(-xdim/2 + center[0], xdim/2+dx + center[0], dx)
        ys = np.arange(-ydim/2 + center[1], ydim/2+dy + center[1], dy)

    output = np.zeros((len(ys), len(xs), num))

    # --- Scan Logic ---
    first = True
    sent_warning = False
    cutoff = -1
    laser_peak_cutoff_fraction = 0.5 
 
    total_points = len(ys) * len(xs)
    current_point = 0

    # We use tqdm if running as a script, but suppress it if running in GUI (implied by progress_signal)
    disable_tqdm = progress_signal is not None
    
    with tqdm(total=total_points, desc="Scanning", disable=disable_tqdm) as pbar:
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                
                # 1. CHECK FOR STOP REQUEST
                if stop_event is not None and stop_event.is_set():
                    print("\n[STOP] Scan aborted by user.")
                    aborted = True
                    break

                # 2. MOVE AND ACQUIRE
                sgd.set_position(x, y)
                intensity, wavelength = eng.eval("instance1.acquire;", nargout=2)
                
                # Process Data
                intensity = fix_length(np.array(intensity).flatten(), num)
                # Note: We save wl every time in your original code, keeping that logic
                wl = np.array(wavelength).flatten() 
                np.save(f'{data_folder}/{foldername}/{scan_type}/wl', wl)

                # 3. FOCUS CHECK LOGIC
                window = (wl > 529) & (wl < 535)
                # Handle edge case where window is empty or invalid
                if np.any(window):
                    peak_idx = np.argmax(intensity[window])
                    peak_intensity = intensity[window][peak_idx]
                    
                    if first:
                        cutoff = laser_peak_cutoff_fraction * peak_intensity
                        first = False
                        if not disable_tqdm: pbar.write(f'Ref Intensity: {peak_intensity}, Cutoff: {cutoff}')
                    else:
                        if simulate_out_of_focus and ix == 1 and iy == 0 and not out_of_focus_detected:
                            peak_intensity = cutoff-10
                        if peak_intensity < cutoff:
                            # OUT OF FOCUS DETECTED
                            if not sent_warning:
                                send_telegram_message(current_user, f"WARNING: Out of focus at {x:.2f}, {y:.2f}!")
                                sent_warning = True
                            
                            out_of_focus_detected = True
                            # We do NOT break here, we continue the map as requested

                # Store Data
                output[iy, ix, :] = intensity

                # 4. UPDATE PROGRESS
                current_point += 1
                if not disable_tqdm:
                    pbar.set_description(f"Acquiring x={x:.2f}, y={y:.2f}")
                    pbar.update(1)
                
                if progress_signal:
                    percent = int((current_point / total_points) * 100)
                    progress_signal.emit(percent)

            if aborted:
                break

    # --- Finalization ---
    # Save whatever we have (even if aborted, saving partial data is usually good)

    np.save(f'{data_folder}/{foldername}/{scan_type}/out', np.array(output))
    np.save(f'{data_folder}/{foldername}/{scan_type}/xs', xs)
    np.save(f'{data_folder}/{foldername}/{scan_type}/ys', ys)

    sgd.sgd_off()
       
    if aborted:
        return EXIT_STOPPED

    # If we finished, run classifier
    print('Scan complete, classifying data...')
    try:
        classifer.classify_all(foldername, scan_type, data_folder=data_folder)
        print('Classification complete!')
    except Exception as e:
        print(f"Classifier failed: {e}")

    # Determine final exit code
    if out_of_focus_detected:
        print("[WARNING] Scan completed, but focus was lost during acquisition.")
        return EXIT_FOCUS_WARNING
    
    return EXIT_SUCCESS






def pl_spec_manual(xdim, ydim, dx, dy, foldername, current_user, center=(0,0),
                   grating=150, exposure_time=1, center_wavelength=700, classification_threshold=1.05,
                   scan_type='coarse', data_folder='data'):
    """
    Executes the PL Spectrum scan and saves the acquired data to a structured directory.

    Args:
        xdim (float): Total width of the scan area. Set to 0 for a single point scan.
        ydim (float): Total height of the scan area. Set to 0 for a single point scan.
        dx (float): Step size in the x direction.
        dy (float): Step size in the y direction.
        foldername (str): Base name of the folder where data will be saved.
        current_user (str): Telegram identifier for out of focus alerts. Must match a registered user.
        center (tuple): (x, y) coordinates for the center of the scan. Default is (0, 0).
        grating (int): Spectrometer grating option. Allowed values are 150 or 600. Default is 150.
        exposure_time (float): Spectrometer exposure time in seconds. Default is 1.
        center_wavelength (int): Spectrometer center wavelength in nanometers. Default is 700.
        classification_threshold (float): Minimum fraction of the laser peak that emitter peak should be
        scan_type (str): Subfolder category for the scan (e.g., 'coarse', 'fine'). Default is 'coarse'.
        data_folder (str): The root directory for all saved data. Default is 'data'.
    """

    out_of_focus_detected = False

    # *** Hardware Initialization ***
    print('Connecting to matlab...')
    eng = matlab.engine.connect_matlab(_matlab_session.name)

    print('Getting wavelengths and setting up...')
    folder_path = os.path.join(data_folder, foldername, scan_type)
    os.makedirs(folder_path, exist_ok=True)

    wl = np.array(eng.workspace['wl']).flatten()
    np.save(os.path.join(folder_path, 'wl.npy'), wl)
    num = len(wl)

    if grating == 150:
        grating_str = "[800nm,150][2][0]"
    elif grating == 600:
        grating_str = "[500nm,600][1][0]"
    else:
        raise ValueError("Invalid grating selected. Options are 150 or 600.")

    eng.eval(f"instance1.set_exposure({int(exposure_time*1000)});", nargout=0)
    eng.eval(f"instance1.set(PrincetonInstruments.LightField.AddIns.SpectrometerSettings.GratingCenterWavelength, {int(center_wavelength)});", nargout=0)
    eng.eval(f'instance1.set(PrincetonInstruments.LightField.AddIns.SpectrometerSettings.Grating, "{grating_str}");', nargout=0)

    print('Starting scan...')
    sgd.sgd_on()

    # Safety Checks
    if abs(xdim / 2 * sgd.XCONV) > 10 or abs(ydim / 2 * sgd.YCONV) > 10:
        raise ValueError("Scan area too large for mirror")
    if xdim > 0 and dx > xdim:
        raise ValueError("X step size too large")
    if ydim > 0 and dy > ydim:
        raise ValueError("Y step size too large")
    
    # Coordinate generation (handles single point if xdim/ydim are 0)
    xs = np.array([center[0]]) if xdim == 0 else np.arange(-xdim/2 + center[0], xdim/2 + dx + center[0], dx)
    ys = np.array([center[1]]) if ydim == 0 else np.arange(-ydim/2 + center[1], ydim/2 + dy + center[1], dy)

    output = np.zeros((len(ys), len(xs), num))

    # *** Scan Logic ***
    first = True
    sent_warning = False
    cutoff = -1
    laser_peak_cutoff_fraction = 0.5 

    total_points = len(ys) * len(xs)

    with tqdm(total=total_points, desc="Scanning") as pbar:
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                
                # Move and Acquire
                sgd.set_position(x, y, silent=True)
                intensity, wavelength = eng.eval("instance1.acquire;", nargout=2)

                # Process Data
                intensity = fix_length(np.array(intensity).flatten(), num)
                wl = np.array(wavelength).flatten()
                np.save(os.path.join(folder_path, 'wl.npy'), wl)

                # Focus Check Logic
                window = (wl > 529) & (wl < 535)
                if np.any(window):
                    peak_idx = np.argmax(intensity[window])
                    peak_intensity = intensity[window][peak_idx]
                    
                    if first:
                        cutoff = laser_peak_cutoff_fraction * peak_intensity
                        first = False
                        # pbar.write(f'Ref Intensity: {peak_intensity}, Cutoff: {cutoff}')
                    else:
                        if peak_intensity < cutoff:
                            if not sent_warning:
                                send_telegram_message(current_user, f"WARNING: Out of focus at {x:.2f}, {y:.2f}!")
                                sent_warning = True
                            out_of_focus_detected = True

                # Store Data
                output[iy, ix, :] = intensity

                # Update Progress
                pbar.set_description(f"Acquiring x={x:.2f}, y={y:.2f}")
                pbar.update(1)

    # *** Finalization ***
    np.save(os.path.join(folder_path, 'out.npy'), output)
    np.save(os.path.join(folder_path, 'xs.npy'), xs)
    np.save(os.path.join(folder_path, 'ys.npy'), ys)

    sgd.sgd_off()
    
    print('Scan complete, classifying data...')
    try:
        classifer.classify_all(foldername, scan_type, threshold=classification_threshold, data_folder=data_folder)
        print('Classification complete!')
    except Exception as e:
        print(f"Classifier failed: {e}")

    # Determine final exit code
    if out_of_focus_detected:
        print("[WARNING] Scan completed, but focus was lost during acquisition.")























engine = None

def connect_matlab():
    global engine
    print('Connecting to matlab...')
    engine = matlab.engine.connect_matlab(_matlab_session.name)
    print('Done connecting to matlab!')


def pl_set_settings(exposure_time=1, center_wavelength=700, grating=500):
    engine.eval(f"instance1.set_exposure({int(exposure_time*1000)});", nargout=0)
    # engine.eval(f"instance1.set(PrincetonInstruments.LightField.AddIns.SpectrometerSettings.GratingCenterWavelength, {int(center_wavelength)});", nargout=0)
    # if grating == 500:
    #     engine.eval('instance1.set(PrincetonInstruments.LightField.AddIns.SpectrometerSettings.Grating, "[500nm,600][1][0]");', nargout=0)
    # else:
    #     engine.eval('instance1.set(PrincetonInstruments.LightField.AddIns.SpectrometerSettings.Grating, "[800nm,150][2][0]");', nargout=0)

def pl_single_scan():
    intensity, wavelength = engine.eval("instance1.acquire;", nargout=2)
    intensity = np.array(intensity).flatten()
    wavelength = np.array(wavelength).flatten()
    return intensity, wavelength


# import matlab.engine
# import numpy as np
# from time import sleep
# import threading
# import msvcrt
# import requests
# import sgd
# import classifier as classifer
# from tqdm import tqdm
# import os



# def pl_spec(xdim, ydim, dx, dy, foldername, current_user, center=(0,0), data_folder='data'):

#     exit_code = 'stop requested'

#     stop_flag = False

#     def key_listener():
#         nonlocal stop_flag
#         while True:
#             if msvcrt.kbhit() and msvcrt.getch() == b'q':
#                 print("\n[KEYBOARD] Stop requested (q pressed).")
#                 stop_flag = True
#                 break


#     def fix_length(arr, target_len):
#         arr = np.array(arr)
#         if len(arr) > target_len:
#             return arr[:target_len]
#         elif len(arr) < target_len:
#             return np.pad(arr, (0, target_len - len(arr)), mode='constant')
#         else:
#             return arr

#     print('Connecting to matlab...')
#     eng = matlab.engine.connect_matlab(_matlab_session.name)

#     print('Getting wavelengths...')

#     folder_path = os.path.join(data_folder, foldername)
#     os.makedirs(folder_path, exist_ok=True)

#     wl = np.array(eng.workspace['wl']).flatten()
#     np.save(f'{data_folder}/{foldername}/wl', wl)

#     num = len(wl)

#     print('Starting scan...')

#     sgd.sgd_on()

#     if abs(xdim / 2 * sgd.XCONV) > 10 or abs(ydim / 2 * sgd.YCONV) > 10:
#         raise ValueError("Scan area too large for mirror")

#     if dx > xdim or dy > ydim:
#         raise ValueError("Step size too large")

#     xs = np.arange(-xdim/2 + center[0], xdim/2+dx + center[0], dx)
#     ys = np.arange(-ydim/2 + center[1], ydim/2+dy + center[1], dy)



#     listener = threading.Thread(target=key_listener, daemon=True)
#     listener.start()

#     output = np.zeros((len(ys), len(xs), num))

#     print('Press q to exit')

#     TOKEN = "8463582982:AAG-izcwemLDy4l2A2ouEAXJDGzHL8xHD5A"

#     users = ["shuhul", "kristina"]
#     CHAT_IDS = ["8130896008", "7568051086"]

#     def send_telegram_message(message):
#         for user, CHAT_ID in zip(users, CHAT_IDS):
#             if user == current_user:
#                 url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
#                 payload = {
#                     "chat_id": CHAT_ID,
#                     "text": message
#                 }
#                 try:
#                     response = requests.post(url, json=payload)
#                     if response.status_code == 200:
#                         print("Message sent!")
#                     else:
#                         print(f"Failed: {response.text}")
#                 except Exception as e:
#                     print(f"Error: {e}")

#     first = True
#     sent = False
#     cutoff = -1
#     laser_peak_cutoff_fraction = 0.5 # fraction of first intensity
 
#     total_points = len(ys) * len(xs)

#     with tqdm(total=total_points, desc="Starting Scan") as pbar:
#         for iy, y in enumerate(ys):
#             for ix, x in enumerate(xs):
                
#                 # Update the progress bar description instead of printing a new line every time
#                 pbar.set_description(f"Acquiring x={x:.2f}, y={y:.2f}")
                
#                 sgd.set_position(x, y)
#                 intensity, wavelength = eng.eval("instance1.acquire;", nargout=2)
#                 intensity = np.array(intensity).flatten()
#                 intensity = fix_length(intensity, num)
#                 wl = np.array(wavelength).flatten()
#                 np.save(f'{data_folder}/{foldername}/wl', wl)

#                 window = (wl > 529) & (wl < 535)
#                 peak_idx = np.argmax(intensity[window])
#                 peak_intensity = intensity[window][peak_idx]
#                 peak_wavelength = wl[window][peak_idx]

#                 if first:
#                     # Use pbar.write() instead of print() to avoid breaking the progress bar visual
#                     pbar.write(f'First peak intensity {peak_intensity} @ {np.round(peak_wavelength,0)} nm')
#                     cutoff = laser_peak_cutoff_fraction*peak_intensity
#                     pbar.write(f'Cutoff {cutoff}')
#                     first = False
#                 elif peak_intensity < cutoff and not sent:
#                     send_telegram_message("WARNING: Out of focus!")
#                     sent = True
#                     pbar.write(f'Out of focus at x={x}, y={y}')
#                     pass

#                 output[iy, ix, :] = intensity
#                 pbar.update(1)

#                 if stop_flag:
#                     break

#             if stop_flag:
#                 break

#     np.save(f'{data_folder}/{foldername}/out', np.array(output))
#     np.save(f'{data_folder}/{foldername}/xs', xs)
#     np.save(f'{data_folder}/{foldername}/ys', ys)

#     sgd.sgd_off()

#     print('Scan complete, classifying data...')

#     classifer.classify_all(foldername, data_folder=data_folder)

#     print('Classification complete!')


#     return exit_code