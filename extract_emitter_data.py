import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

DATA_DIR = 'data'
OUT_DIR  = 'analysis_output'

# ── Data extraction ───────────────────────────────────────────────────────────

_RUN_RE   = re.compile(r'(?P<date>\d{8})-PLSPC-HT-Ch(?P<chip>\w+)-f(?P<field>\d+)-')
_COORD_RE = re.compile(r'_x(?P<x>-?[\d.]+)_y(?P<y>-?[\d.]+)')


def _parse_run_meta(run_name):
    m = _RUN_RE.search(run_name)
    if not m:
        return {'chip': '?', 'field': '?', 'date': run_name[:8]}
    return {'chip': m.group('chip'), 'field': m.group('field'), 'date': m.group('date')}


def extract_emitters(data_dir, verbose=False):

    import pandas as pd

    """Yield one dict per emitter that has raw g2 data (.npz) and a matching spectrum.
    
    Recalculates g2 with 50 ns window from raw data, excluding processed files.

    T1_ns is only populated for CONFIRMED single emitters (g2_0 < SPE_THRESHOLD)
    whose fitted T1 also passes MAX_T1_NS; otherwise it's NaN, so a non-SPE
    emitter or an unphysically slow "antibunching" fit never contributes a
    T1 value downstream.
    """
    
    data_list = []

    index=0

    run_pattern = re.compile(r'.*HT.*fullauto.*', re.IGNORECASE)

    for run_name in sorted(os.listdir(data_dir)):
        if not run_pattern.match(run_name):
            continue
        run_path = os.path.join(data_dir, run_name)
        if not os.path.isdir(run_path):
            continue

        subfolders = set(os.listdir(run_path))
        meta = _parse_run_meta(run_name)

        for subdir in sorted(subfolders):
            if not subdir.startswith('g2_'):
                continue

            coord_str = subdir[2:]            # '_x10.25_y-6.75'
            lf_dir    = 'long' + coord_str
            lf_dir_fine = 'fine' + coord_str
            if lf_dir not in subfolders:
                continue

            # ── Find raw g2 data (.npz file, not _processed) ──
            g2_path = os.path.join(run_path, subdir)
            raw_files = sorted(f for f in os.listdir(g2_path) 
                              if f.endswith('.npz') and '_processed' not in f)
            if not raw_files:
                continue

            # Load raw data and recalculate g2
            try:
                npz = np.load(os.path.join(g2_path, raw_files[-1]))
                ch0 = npz['ch0'].astype(np.int64)
                ch1 = npz['ch1'].astype(np.int64)
            except Exception as e:
                if verbose:
                    print(f"Skipped {run_name}/{coord_str}: {e}")
                continue

            # ── filtered spectrum → ZPL + FWHM (Gaussian fit, 560-630 nm) ──
            lf_path = os.path.join(run_path, lf_dir)
            try:
                wl600       = np.load(os.path.join(lf_path, 'wl.npy'))
                out      = np.load(os.path.join(lf_path, 'out.npy'))
                spectrum600 = out[0, 0, :]
            except (FileNotFoundError, IndexError):
                continue
            lf_path = os.path.join(run_path, lf_dir_fine)
            try:
                wl150       = np.load(os.path.join(lf_path, 'wl.npy'))
                out      = np.load(os.path.join(lf_path, 'out.npy'))
                spectrum150 = out[0, 0, :]
            except (FileNotFoundError, IndexError):
                wl150=[]
                spectrum150=[]

                     

            label = f'{run_name}/{coord_str}'

            cm = _COORD_RE.search(coord_str)
            x  = float(cm.group('x')) if cm else None
            y  = float(cm.group('y')) if cm else None

            row_dict = {
                'index':index,
                'name':label,
                'ch0':ch0,
                'ch1':ch1,
                'spec150':spectrum150,
                'spec600':spectrum600,
                'wl150':wl150,
                'wl600':wl600,
            }

            data_list.append(row_dict)


            index+=1
    full_emitter_df = pd.DataFrame(data_list)
    full_emitter_df.to_pickle('full_emitter_df.pkl')

    return full_emitter_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Analyse HT fullauto emitter data.')
    ap.add_argument('--data-dir', default=DATA_DIR, help='Path to data/ folder')
    ap.add_argument('--out-dir',  default=OUT_DIR,  help='Output folder for CSV and plots')
    ap.add_argument('--verbose', action='store_true',
                     help='Print the reason each PSB (DWF) fit was skipped')
    args = ap.parse_args()

    print('Scanning data folders...')
    full_emitter_df = extract_emitters(args.data_dir, verbose=False)
    print(full_emitter_df)

    


if __name__ == '__main__':
    main()