"""
Python-only spectrometer interface using the LightField bridge (lf_spec).
No MATLAB required — import this instead of pl_spec when not using MATLAB.
"""
import os
import numpy as np
import requests
from tqdm import tqdm

import lf_spec
import sgd
import classifier as classifer


def _fix_length(arr, target_len):
    arr = np.array(arr)
    if len(arr) > target_len:
        return arr[:target_len]
    elif len(arr) < target_len:
        return np.pad(arr, (0, target_len - len(arr)), mode='constant')
    return arr


def _send_telegram(current_user, message):
    TOKEN    = "8463582982:AAG-izcwemLDy4l2A2ouEAXJDGzHL8xHD5A"
    users    = ["shuhul", "kristina", "holland"]
    chat_ids = ["8130896008", "7568051086", "8743893517"]
    if current_user in users:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                json={"chat_id": chat_ids[users.index(current_user)], "text": message},
            )
        except Exception as e:
            print(f"Telegram error: {e}")


def pl_spec_lf(xdim, ydim, dx, dy, foldername, current_user, center=(0, 0),
               grating=150, exposure_time=1, center_wavelength=700,
               classification_threshold=1.05, scan_type='coarse', data_folder='data'):
    """
    Full PL spectral scan using the LightField Python backend.

    Args:
        xdim (float): Scan width in um. Set to 0 for single-point.
        ydim (float): Scan height in um. Set to 0 for single-point.
        dx (float): Step size in x (um).
        dy (float): Step size in y (um).
        foldername (str): Folder name for saved data.
        current_user (str): Telegram alert recipient. Options: 'shuhul', 'kristina', 'holland'.
        center (tuple): (x, y) scan centre in um.
        grating (int): 150 (150 g/mm) or 600 (600 g/mm).
        exposure_time (float): Exposure per point in seconds.
        center_wavelength (int): Spectrometer centre wavelength in nm.
        classification_threshold (float): Min fraction of laser peak for emitter classification.
        scan_type (str): Subfolder name, e.g. 'coarse' or 'fine'.
        data_folder (str): Root data directory.
    """
    out_of_focus_detected = False

    print('Connecting to LightField...')
    lf_spec.lf_connect()

    print('Getting wavelengths and setting up...')
    folder_path = os.path.join(data_folder, foldername, scan_type)
    os.makedirs(folder_path, exist_ok=True)

    lf_spec.lf_setup(exposure_s=exposure_time, center_wavelength=center_wavelength, grating=grating)

    wl  = lf_spec.lf_get_wavelengths()
    num = len(wl)
    np.save(os.path.join(folder_path, 'wl.npy'), wl)

    if abs(xdim / 2 * sgd.XCONV) > 10 or abs(ydim / 2 * sgd.YCONV) > 10:
        raise ValueError("Scan area too large for mirror")
    if xdim > 0 and dx > xdim:
        raise ValueError("X step size too large")
    if ydim > 0 and dy > ydim:
        raise ValueError("Y step size too large")

    xs = np.array([center[0]]) if xdim == 0 else np.arange(-xdim/2 + center[0], xdim/2 + dx + center[0], dx)
    ys = np.array([center[1]]) if ydim == 0 else np.arange(-ydim/2 + center[1], ydim/2 + dy + center[1], dy)

    output = np.zeros((len(ys), len(xs), num))

    print('Starting scan...')
    sgd.sgd_on()

    first        = True
    sent_warning = False
    cutoff       = -1

    with tqdm(total=len(ys) * len(xs), desc='Scanning') as pbar:
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                sgd.set_position(x, y, silent=True)
                intensity, wl_acq = lf_spec.lf_acquire()
                intensity = _fix_length(np.array(intensity).flatten(), num)
                wl        = np.array(wl_acq).flatten()
                np.save(os.path.join(folder_path, 'wl.npy'), wl)

                # Focus check via laser peak
                window = (wl > 529) & (wl < 535)
                if np.any(window):
                    peak_intensity = intensity[window].max()
                    if first:
                        cutoff = 0.5 * peak_intensity
                        first  = False
                    elif peak_intensity < cutoff:
                        if not sent_warning:
                            _send_telegram(current_user, f"WARNING: Out of focus at {x:.2f}, {y:.2f}!")
                            sent_warning = True
                        out_of_focus_detected = True

                output[iy, ix, :] = intensity
                pbar.set_description(f'Acquiring x={x:.2f}, y={y:.2f}')
                pbar.update(1)

    sgd.sgd_off()

    np.save(os.path.join(folder_path, 'out.npy'), output)
    np.save(os.path.join(folder_path, 'xs.npy'),  xs)
    np.save(os.path.join(folder_path, 'ys.npy'),  ys)

    print('Scan complete, classifying data...')
    try:
        classifer.classify_all(foldername, scan_type,
                               threshold=classification_threshold, data_folder=data_folder)
        print('Classification complete!')
    except Exception as e:
        print(f"Classifier failed: {e}")

    if out_of_focus_detected:
        print('[WARNING] Scan completed, but focus was lost during acquisition.')
