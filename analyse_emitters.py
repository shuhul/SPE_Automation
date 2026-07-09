"""
analyse_emitters.py
Extracts ZPL, FWHM, g²(0) and T1 from all HT fullauto runs that have g2 data,
builds a summary DataFrame, saves emitter_summary.csv, and generates scatter
plots + histograms in analysis_output/.

Usage:
    python analyse_emitters.py
    python analyse_emitters.py --data-dir path/to/data --out-dir results
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

DATA_DIR = 'data'
OUT_DIR  = 'analysis_output'

# ── Spectrum helpers — Gaussian fitting ───────────────────────────────────────

_FWHM_K = 2.0 * np.sqrt(2.0 * np.log(2.0))   # 2.3548


def _gaussian1d(x, A, mu, sigma, bkg):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + bkg


def _fit_zpl_gaussian(spectrum, wl, laser_cutoff_nm=560,
                       window_nm=15.0, min_snr=3.0, max_fwhm_nm=45.0):
    """Fit a Gaussian to the ZPL peak.

    Returns (zpl_nm, fwhm_nm).  fwhm_nm is None if the fit fails or the
    spectrum has no clear ZPL (flat / edge-clipped peak with low SNR).
    """
    mask = wl > laser_cutoff_nm
    if not mask.any():
        return None, None
    wl_m, sp_m  = wl[mask], spectrum[mask]
    pk          = int(np.argmax(sp_m))
    peak_wl     = float(wl_m[pk])
    peak_val    = float(sp_m[pk])

    # Baseline and noise from the wings outside the fit window
    wing   = np.abs(wl_m - peak_wl) > window_nm
    bkg0   = float(np.median(sp_m[wing])) if wing.any() else float(np.percentile(sp_m, 10))
    noise  = float(np.std(sp_m[wing]))    if wing.sum() > 5 else 1.0

    if (peak_val - bkg0) < min_snr * max(noise, 1.0):
        return None, None   # flat spectrum / no clear ZPL

    win = np.abs(wl_m - peak_wl) <= window_nm
    if win.sum() < 5:
        return None, None

    x, y = wl_m[win], sp_m[win]
    A0   = peak_val - bkg0
    p0   = [A0, peak_wl, 1.5, bkg0]
    try:
        popt, _ = curve_fit(
            _gaussian1d, x, y, p0=p0,
            bounds=([0,      peak_wl - window_nm,   0.05,            -np.inf],
                    [A0 * 3, peak_wl + window_nm,   max_fwhm_nm / _FWHM_K, peak_val]),
            maxfev=3000,
        )
        A, mu, sigma, _ = popt
        if A <= 0:
            return peak_wl, None
        return float(mu), float(_FWHM_K * sigma)
    except Exception:
        return peak_wl, None


def _fit_psb_gaussian(spectrum, wl, zpl_nm,
                       psb_min_nm=10.0, psb_max_nm=100.0,
                       window_nm=20.0, min_snr=3.0):
    """Fit a Gaussian to the phonon sideband (red of ZPL).

    Returns (psb_nm, psb_fwhm_nm) or (None, None) if no PSB found.
    """
    mask = (wl > zpl_nm + psb_min_nm) & (wl < zpl_nm + psb_max_nm)
    if not mask.any() or mask.sum() < 10:
        return None, None
    wl_m, sp_m = wl[mask], spectrum[mask]
    pk      = int(np.argmax(sp_m))
    pk_wl   = float(wl_m[pk])
    pk_val  = float(sp_m[pk])

    wing   = np.abs(wl_m - pk_wl) > window_nm
    bkg0   = float(np.median(sp_m[wing])) if wing.any() else float(np.percentile(sp_m, 10))
    noise  = float(np.std(sp_m[wing]))    if wing.sum() > 5 else 1.0

    if (pk_val - bkg0) < min_snr * max(noise, 1.0):
        return None, None

    win = np.abs(wl_m - pk_wl) <= window_nm
    if win.sum() < 5:
        return None, None

    x, y = wl_m[win], sp_m[win]
    A0   = pk_val - bkg0
    p0   = [A0, pk_wl, 5.0, bkg0]
    try:
        popt, _ = curve_fit(
            _gaussian1d, x, y, p0=p0,
            bounds=([0,      pk_wl - 15,  1.0,  -np.inf],
                    [A0 * 3, pk_wl + 15,  30.0, pk_val]),
            maxfev=3000,
        )
        A, mu, sigma, _ = popt
        if A <= 0 or mu < zpl_nm + psb_min_nm or mu > zpl_nm + psb_max_nm:
            return None, None
        return float(mu), float(_FWHM_K * sigma)
    except Exception:
        return None, None


# ── Data extraction ───────────────────────────────────────────────────────────

_RUN_RE   = re.compile(r'(?P<date>\d{8})-PLSPC-HT-Ch(?P<chip>\w+)-f(?P<field>\d+)-')
_COORD_RE = re.compile(r'_x(?P<x>-?[\d.]+)_y(?P<y>-?[\d.]+)')


def _parse_run_meta(run_name):
    m = _RUN_RE.search(run_name)
    if not m:
        return {'chip': '?', 'field': '?', 'date': run_name[:8]}
    return {'chip': m.group('chip'), 'field': m.group('field'), 'date': m.group('date')}


def iter_emitters(data_dir):
    """Yield one dict per emitter that has a g2 processed file and a matching spectrum."""
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
            lf_dir    = 'long_filter' + coord_str
            if lf_dir not in subfolders:
                continue

            # ── g2 fit results ──
            g2_path    = os.path.join(run_path, subdir)
            proc_files = sorted(f for f in os.listdir(g2_path) if f.endswith('_processed.npz'))
            if not proc_files:
                continue

            g2d  = np.load(os.path.join(g2_path, proc_files[-1]), allow_pickle=True)
            popt = g2d['popt']
            if popt.ndim == 0 or popt.size < 5:
                continue  # fit failed

            _a, _b, T1, _T2, _g0 = popt.astype(float)
            g2_0_norm = float(g2d['g2_0_norm'])

            # ── filtered spectrum → ZPL + FWHM ──
            lf_path = os.path.join(run_path, lf_dir)
            try:
                wl       = np.load(os.path.join(lf_path, 'wl.npy'))
                out      = np.load(os.path.join(lf_path, 'out.npy'))
                spectrum = out[0, 0, :]
            except (FileNotFoundError, IndexError):
                continue

            zpl, fwhm = _fit_zpl_gaussian(spectrum, wl)
            if zpl is None:
                continue

            psb_nm, psb_fwhm_nm = _fit_psb_gaussian(spectrum, wl, zpl)
            psb_shift_nm = float(psb_nm - zpl) if psb_nm is not None else None

            cm = _COORD_RE.search(coord_str)
            x  = float(cm.group('x')) if cm else None
            y  = float(cm.group('y')) if cm else None

            yield {
                'run':         run_name,
                **meta,
                'x':           x,
                'y':           y,
                'ZPL_nm':      zpl,
                'FWHM_nm':     fwhm,
                'PSB_nm':      psb_nm,
                'PSB_FWHM_nm': psb_fwhm_nm,
                'PSB_shift_nm': psb_shift_nm,
                'g2_0':        g2_0_norm,
                'T1_ns':       float(T1),
            }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _chip_palette(chips):
    """Return {chip_label: color} with a stable, distinct color per chip."""
    unique  = sorted(set(chips))
    colors  = plt.cm.tab10.colors
    return {ch: colors[i % len(colors)] for i, ch in enumerate(unique)}


def _scatter(ax, df, xcol, ycol, xlabel, ylabel, cmap):
    """Scatter plot coloured by chip; stars mark g²(0) < 0.5 emitters."""
    chips  = df['chip'].fillna('?').astype(str)
    colors = chips.map(cmap)
    sub    = df[[xcol, ycol]].copy()
    valid  = sub.notna().all(axis=1)
    sub, colors = sub[valid], colors[valid]

    ax.scatter(sub[xcol], sub[ycol],
               c=colors, s=25, edgecolors='k', linewidths=0.4, zorder=3)

    se = df.loc[valid, 'g2_0'] < 0.5
    if se.any():
        ax.scatter(sub.loc[se, xcol], sub.loc[se, ycol],
                   marker='*', s=80, c=colors[se],
                   edgecolors='k', linewidths=0.4, zorder=4,
                   label='single emitter (g²(0) < 0.5)')

    if ycol == 'g2_0':
        ax.axhline(0.5, ls='--', color='#e74c3c', lw=1.2, zorder=2,
                   label='g²(0) = 0.5')

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')


def make_scatter_plots(df, out_dir):
    chips = df['chip'].fillna('?').astype(str)
    cmap  = _chip_palette(chips.unique())

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Emitter correlation analysis', fontsize=14, fontweight='bold')

    _scatter(axes[0, 0], df, 'ZPL_nm',  'g2_0',        'ZPL (nm)',           'g²(0)',          cmap)
    _scatter(axes[0, 1], df, 'FWHM_nm', 'g2_0',        'ZPL FWHM (nm)',      'g²(0)',          cmap)
    _scatter(axes[0, 2], df, 'ZPL_nm',  'FWHM_nm',     'ZPL (nm)',           'ZPL FWHM (nm)',  cmap)
    _scatter(axes[1, 0], df, 'T1_ns',   'ZPL_nm',      'T₁ (ns)',            'ZPL (nm)',       cmap)
    _scatter(axes[1, 1], df, 'T1_ns',   'FWHM_nm',     'T₁ (ns)',            'ZPL FWHM (nm)', cmap)
    _scatter(axes[1, 2], df, 'ZPL_nm',  'PSB_shift_nm', 'ZPL (nm)',          'PSB shift (nm)', cmap)

    # Shared chip legend at the bottom
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=c, markeredgecolor='k',
                   markersize=9, label=f'Ch{ch}')
        for ch, c in cmap.items()
    ]
    fig.legend(handles=handles, title='Chip', loc='lower center',
               ncol=len(cmap), fontsize=9, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    out_path = os.path.join(out_dir, 'emitter_correlations.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {out_path}')


def make_histogram_plots(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle('Emitter property distributions', fontsize=13)

    axes[0].hist(df['ZPL_nm'].dropna(),  bins=20, edgecolor='k', color='steelblue')
    axes[0].set_xlabel('ZPL (nm)');  axes[0].set_ylabel('Count')
    axes[0].set_title('ZPL')

    axes[1].hist(df['FWHM_nm'].dropna(), bins=20, edgecolor='k', color='seagreen')
    axes[1].set_xlabel('FWHM (nm)'); axes[1].set_ylabel('Count')
    axes[1].set_title('FWHM')

    axes[2].hist(df['g2_0'].dropna(),    bins=20, edgecolor='k', color='salmon')
    axes[2].axvline(0.5, ls='--', color='red', lw=1.2, label='g²(0) = 0.5')
    axes[2].set_xlabel('g²(0)');     axes[2].set_ylabel('Count')
    axes[2].set_title('g²(0)')
    axes[2].legend()

    fig.tight_layout()
    out_path = os.path.join(out_dir, 'emitter_histograms.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Analyse HT fullauto emitter data.')
    ap.add_argument('--data-dir', default=DATA_DIR, help='Path to data/ folder')
    ap.add_argument('--out-dir',  default=OUT_DIR,  help='Output folder for CSV and plots')
    args = ap.parse_args()

    print('Scanning data folders...')
    rows = list(iter_emitters(args.data_dir))

    if not rows:
        print('No emitters found with both g2 and long_filter spectrum data.')
        return

    df = pd.DataFrame(rows)

    # ── Flag and drop g²(0) < 0 (unphysical fit) ─────────────────────────────
    bad = df[df['g2_0'] < 0]
    if not bad.empty:
        print('\nIgnored (g²(0) < 0 — unphysical fit):')
        for _, row in bad.iterrows():
            print(f'  {row["run"]}  /  g2_{row["x"]}_{row["y"]}  '
                  f'g²(0) = {row["g2_0"]:.3f}')
        df = df[df['g2_0'] >= 0].reset_index(drop=True)

    n_single = int((df['g2_0'] < 0.5).sum())
    print(f'\nFound {len(df)} valid emitters across {df["run"].nunique()} runs '
          f'({n_single} single emitters with g²(0) < 0.5).\n')

    cols = ['chip', 'field', 'x', 'y', 'ZPL_nm', 'FWHM_nm', 'PSB_shift_nm', 'g2_0', 'T1_ns']
    print(df[cols].to_string(index=False, float_format=lambda v: f'{v:.3f}' if pd.notna(v) else 'None'))

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'emitter_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}')

    make_scatter_plots(df, args.out_dir)
    make_histogram_plots(df, args.out_dir)


if __name__ == '__main__':
    main()
