import numpy as np
from scipy.signal import find_peaks, peak_widths

def classify_spectrum(spectrum, wl, ratio_threshold=1.05):
    peaks, properties = find_peaks(spectrum, height=40, prominence=20, distance=10)
    widths_samples = peak_widths(spectrum, peaks, rel_height=0.5)[0]
    peak_wls = wl[peaks]
    peak_heights = properties["peak_heights"]

    if len(peak_wls) == 0:
        return 0, None, None

    # 1. Find the Laser Peak (530 - 535 nm)
    laser_mask = (peak_wls > 530) & (peak_wls < 535)
    valid_laser_indices = np.where(laser_mask)[0]

    if len(valid_laser_indices) == 0:
        return 0, None, None

    laser_peak_idx = valid_laser_indices[0]
    laser_peak_height = peak_heights[laser_peak_idx]

    # 2. Find the Emission Peak (560 - 630 nm)
    emission_mask = (peak_wls > 560) & (peak_wls < 630)
    valid_emission_indices = np.where(emission_mask)[0]

    if len(valid_emission_indices) == 0:
        return 0, None, None

    # Find the highest peak strictly within the emission range
    emission_heights = peak_heights[valid_emission_indices]
    local_max_idx = valid_emission_indices[np.argmax(emission_heights)]
    
    emission_peak_wl = peak_wls[local_max_idx]
    emission_peak_height = peak_heights[local_max_idx]

    # 3. Check Threshold
    if not (emission_peak_height > ratio_threshold * laser_peak_height):
        return 0, None, None

    # 4. Check FWHM
    emission_peak_fwhm = widths_samples[local_max_idx] * (wl[1] - wl[0])  # convert to nm

    if not (5 < emission_peak_fwhm < 35):
        return 0, None, None

    return 1, emission_peak_height, emission_peak_wl


def get_peak_annotation(spectrum, wl):
    """Return (peak_wl, peak_height, left_wl, right_wl, fwhm_nm) for the emission peak
    using the exact same scipy logic as classify_spectrum. Returns None if no emission
    peak is found in the 560-630 nm window."""
    peaks, properties = find_peaks(spectrum, height=40, prominence=20, distance=10)
    if len(peaks) == 0:
        return None
    widths, _, left_ips, right_ips = peak_widths(spectrum, peaks, rel_height=0.5)
    peak_wls    = wl[peaks]
    peak_heights = properties["peak_heights"]

    emission_mask = (peak_wls > 560) & (peak_wls < 630)
    valid = np.where(emission_mask)[0]
    if len(valid) == 0:
        return None

    idx        = valid[np.argmax(peak_heights[valid])]
    peak_wl    = float(peak_wls[idx])
    peak_height = float(peak_heights[idx])
    fwhm_nm    = float(widths[idx] * (wl[1] - wl[0]))
    left_wl    = float(np.interp(left_ips[idx],  np.arange(len(wl)), wl))
    right_wl   = float(np.interp(right_ips[idx], np.arange(len(wl)), wl))

    return peak_wl, peak_height, left_wl, right_wl, fwhm_nm


def classify_all(foldername, scan_type, data_folder='data', threshold=1.05, 
                 spatial_radius_um=0.8, wl_diff_threshold=6.0):
    
    intensities = np.load(f'{data_folder}/{foldername}/{scan_type}/out.npy')
    wl = np.load(f'{data_folder}/{foldername}/{scan_type}/wl.npy')
    xs = np.load(f'{data_folder}/{foldername}/{scan_type}/xs.npy')
    ys = np.load(f'{data_folder}/{foldername}/{scan_type}/ys.npy')
    
    # 1. Collect all valid candidates
    candidates = []
    for iy in range(intensities.shape[0]):
        for ix in range(intensities.shape[1]):
            spectrum = intensities[iy, ix, :]
            label, p_h, p_wl = classify_spectrum(spectrum, wl, ratio_threshold=threshold)  
            
            if label == 1:
                candidates.append({
                    'ix': ix, 
                    'iy': iy, 
                    'x_pos': xs[ix],   # Store the physical X coordinate
                    'y_pos': ys[iy],   # Store the physical Y coordinate
                    'height': p_h, 
                    'wl': p_wl
                })
                
    # 2. Sort candidates by peak height (brightest first)
    candidates.sort(key=lambda c: c['height'], reverse=True)
    
    classified = np.zeros((intensities.shape[0], intensities.shape[1]), dtype=int)
    accepted_emitters = []
    
    # 3. Non-Maximum Suppression (using physical micrometers)
    for cand in candidates:
        is_duplicate = False
        
        for acc in accepted_emitters:
            # Calculate physical distance in micrometers
            dist = np.sqrt((cand['x_pos'] - acc['x_pos'])**2 + (cand['y_pos'] - acc['y_pos'])**2)
            
            # Calculate wavelength difference
            wl_diff = abs(cand['wl'] - acc['wl'])
            
            # If it is physically close AND spectrally similar, it is the same emitter
            if dist <= spatial_radius_um and wl_diff <= wl_diff_threshold:
                is_duplicate = True
                break
                
        if not is_duplicate:
            accepted_emitters.append(cand)
            classified[cand['iy'], cand['ix']] = 1
            
    np.save(f'{data_folder}/{foldername}/{scan_type}/classified', classified)
    print(f"Found {len(accepted_emitters)} distinct emitters.")

# import numpy as np

# from scipy.signal import find_peaks, peak_widths


# def classify_spectrum(spectrum, wl):

#     peaks, properties = find_peaks(spectrum, height=40, prominence=20, distance=10)

#     widths_samples = peak_widths(spectrum, peaks, rel_height=0.5)[0]

#     peak_wls = wl[peaks]


#     if len(peak_wls) == 0:

#         return 0, None

#     range_mask = (peak_wls > 530) & (peak_wls < 535)

#     valid_peak_indices = np.where(range_mask)[0]



#     if len(valid_peak_indices) == 0:

#         return 0, None

   

#     peak_idx = valid_peak_indices[0]

#     first_peak_wavelength = peak_wls[peak_idx]

#     print("First peak wavelength (nm):", first_peak_wavelength)



#     peak_heights = properties["peak_heights"]



#     first_peak_height = peak_heights[peak_idx]

#     print("First peak height:", first_peak_height)



#     max_peak_height = np.max(peak_heights)



#     print("Max peak height:", max_peak_height)

#     if not (max_peak_height > 1.05*first_peak_height):

#         return 0, None

   

#     max_peak_wl = peak_wls[np.argmax(peak_heights)]

#     print("Max peak wavelength (nm):", max_peak_wl)

#     if not (560 < max_peak_wl < 630):

#         return 0, None





#     max_peak_fwhm = widths_samples[np.argmax(peak_heights)] * (wl[1] - wl[0])  # convert to nm

#     print("Max peak FWHM (nm):", max_peak_fwhm)

#     if not (5 < max_peak_fwhm < 35):

#         return 0, None

   

#     return 1, max_peak_height



# def classify_all(foldername, scan_type, data_folder='data'):

#     intensities = np.load(f'{data_folder}/{foldername}/{scan_type}/out.npy')

#     wl = np.load(f'{data_folder}/{foldername}/{scan_type}/wl.npy')

#     classified = np.zeros((intensities.shape[0], intensities.shape[1]), dtype=int)

#     for iy in range(intensities.shape[0]):

#         for ix in range(intensities.shape[1]):

#             spectrum = intensities[iy, ix, :]

#             label, _ = classify_spectrum(spectrum, wl)  

#             classified[iy, ix] = label

   

#     np.save(f'{data_folder}/{foldername}/{scan_type}/classified', classified)
