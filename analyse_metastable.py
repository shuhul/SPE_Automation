"""
analyse_metastable.py
Reads multi-scale g²(τ) fit results (*_multiscale_results.npz) produced by
g2_multiscale.py and correlates metastable-state parameters with the ZPL of
each emitter.

Metastable parameters extracted from popt = [a, b, T1, T2, c, T3]:
  bunching_height  = a * b   (how far above 1 the bunching shoulder reaches)
  T2_ns            = fast metastable lifetime  (max(T1, T2), the slower one)
  c                = slow-dynamics amplitude   (g²(∞) = 1 - c)
  T3_ns            = slow-dynamics lifetime

Usage:
    python analyse_metastable.py
    python analyse_metastable.py --data-dir path/to/data --out-dir results
    python analyse_metastable.py --from-date 20260617
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

DATA_DIR = 'data'
OUT_DIR  = 'analysis_output'

_RUN_RE   = re.compile(r'.*HT.*fullauto.*', re.IGNORECASE)
_DATE_RE  = re.compile(r'(?P<date>\d{8})-PLSPC-HT-Ch(?P<chip>\w+)-f(?P<field>\d+)-')
_COORD_RE = re.compile(r'_x(?P<x>-?[\d.]+)_y(?P<y>-?[\d.]+)')

SPE_THRESHOLD = 0.5
_FWHM_K = 2.0 * np.sqrt(2.0 * np.log(2.0))


# ── Spectrum helpers — Gaussian fitting ───────────────────────────────────────

def _gaussian1d(x, A, mu, sigma, bkg):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + bkg


def _fit_zpl_gaussian(spectrum, wl, laser_cutoff_nm=560,
                       window_nm=15.0, min_snr=3.0, max_fwhm_nm=45.0):
    mask = wl > laser_cutoff_nm
    if not mask.any():
        return None, None
    wl_m, sp_m = wl[mask], spectrum[mask]
    pk       = int(np.argmax(sp_m))
    peak_wl  = float(wl_m[pk])
    peak_val = float(sp_m[pk])
    wing     = np.abs(wl_m - peak_wl) > window_nm
    bkg0     = float(np.median(sp_m[wing])) if wing.any() else float(np.percentile(sp_m, 10))
    noise    = float(np.std(sp_m[wing]))    if wing.sum() > 5 else 1.0
    if (peak_val - bkg0) < min_snr * max(noise, 1.0):
        return None, None
    win = np.abs(wl_m - peak_wl) <= window_nm
    if win.sum() < 5:
        return None, None
    x, y = wl_m[win], sp_m[win]
    A0   = peak_val - bkg0
    p0   = [A0, peak_wl, 1.5, bkg0]
    try:
        popt, _ = curve_fit(
            _gaussian1d, x, y, p0=p0,
            bounds=([0, peak_wl - window_nm, 0.05, -np.inf],
                    [A0 * 3, peak_wl + window_nm, max_fwhm_nm / _FWHM_K, peak_val]),
            maxfev=3000,
        )
        A, mu, sigma, _ = popt
        if A <= 0:
            return peak_wl, None
        return float(mu), float(_FWHM_K * sigma)
    except Exception:
        return peak_wl, None


# ── Data extraction ───────────────────────────────────────────────────────────

def _parse_run_meta(run_name):
    m = _DATE_RE.search(run_name)
    if not m:
        return {'chip': '?', 'field': '?', 'date': run_name[:8]}
    return {'chip': m.group('chip'), 'field': m.group('field'), 'date': m.group('date')}


def iter_metastable(data_dir, from_date=None, to_date=None):
    """Yield one dict per emitter that has a multiscale results file and a
    matching long_filter spectrum."""
    for run_name in sorted(os.listdir(data_dir)):
        if not _RUN_RE.match(run_name):
            continue
        run_path = os.path.join(data_dir, run_name)
        if not os.path.isdir(run_path):
            continue

        date_m   = re.match(r'^(\d{8})', run_name)
        run_date = date_m.group(1) if date_m else None
        if run_date:
            if from_date and run_date < from_date:
                continue
            if to_date   and run_date > to_date:
                continue

        subfolders = set(os.listdir(run_path))
        meta = _parse_run_meta(run_name)

        for subdir in sorted(subfolders):
            if not subdir.startswith('g2_'):
                continue

            g2_path = os.path.join(run_path, subdir)
            res_files = sorted(f for f in os.listdir(g2_path)
                               if f.endswith('_multiscale_results.npz'))
            if not res_files:
                continue

            res = np.load(os.path.join(g2_path, res_files[-1]), allow_pickle=True)

            if not bool(res['fit_converged'][0]):
                continue

            popt = res['popt'].astype(float)
            if popt.size < 6:
                continue

            a, b, T1, T2, c, T3 = popt
            T1_ns = float(min(T1, T2))   # antibunching (shorter)
            T2_ns = float(max(T1, T2))   # metastable shelving (longer)
            bunching_height = float(a * b)

            coord_str = subdir[2:]        # '_x10.25_y-6.75'
            lf_dir    = 'long_filter' + coord_str
            if lf_dir not in subfolders:
                continue

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

            cm = _COORD_RE.search(coord_str)
            x  = float(cm.group('x')) if cm else None
            y  = float(cm.group('y')) if cm else None

            yield {
                'run':             run_name,
                **meta,
                'x':               x,
                'y':               y,
                'ZPL_nm':          zpl,
                'FWHM_nm':         fwhm,
                'g2_0':            float(res['g2_0'][0]),
                'g2_inf':          float(res['g2_inf'][0]),
                'a':               float(a),
                'b':               float(b),
                'T1_ns':           T1_ns,
                'T2_ns':           T2_ns,
                'c':               float(c),
                'T3_ns':           float(T3),
                'bunching_height': bunching_height,
                'wing_reliable':   bool(res['wing_reliable'][0]),
                'result_reliable': bool(res['result_reliable'][0]),
                'T2_pinned':       bool(res['T2_pinned'][0]),
                'T3_pinned':       bool(res['T3_pinned'][0]),
            }


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _chip_palette(chips):
    unique = sorted(set(chips))
    colors = plt.cm.tab10.colors
    return {ch: colors[i % len(colors)] for i, ch in enumerate(unique)}


def _scatter_meta(ax, df, xcol, ycol, xlabel, ylabel, cmap,
                  yscale='linear', pinned_col=None, hline=None):
    """Scatter coloured by chip; stars = single emitters; hatched = pinned fits."""
    chips  = df['chip'].fillna('?').astype(str)
    colors = chips.map(cmap)
    sub    = df[[xcol, ycol]].copy()
    valid  = sub.notna().all(axis=1)

    df_v    = df[valid].reset_index(drop=True)
    col_v   = colors[valid].reset_index(drop=True)
    x_v     = df_v[xcol].values
    y_v     = df_v[ycol].values
    is_spe  = df_v['g2_0'] < SPE_THRESHOLD
    is_pin  = df_v[pinned_col].astype(bool) if pinned_col and pinned_col in df_v else \
              pd.Series(False, index=df_v.index)
    is_rel  = df_v['result_reliable'].astype(bool)

    # reliable, not-pinned: solid circles / stars
    mask_ok = ~is_pin & is_rel
    if mask_ok.any():
        ax.scatter(x_v[mask_ok & ~is_spe], y_v[mask_ok & ~is_spe],
                   c=col_v[mask_ok & ~is_spe], s=30,
                   edgecolors='k', linewidths=0.4, zorder=3)
        if (mask_ok & is_spe).any():
            ax.scatter(x_v[mask_ok & is_spe], y_v[mask_ok & is_spe],
                       marker='*', s=100, c=col_v[mask_ok & is_spe],
                       edgecolors='k', linewidths=0.5, zorder=4,
                       label='single emitter (g²(0) < 0.5)')

    # pinned or low-reliability: open markers
    mask_q = is_pin | ~is_rel
    if mask_q.any():
        ax.scatter(x_v[mask_q], y_v[mask_q],
                   c=col_v[mask_q], s=30,
                   edgecolors=col_v[mask_q], linewidths=1.0,
                   facecolors='none', zorder=3, alpha=0.6,
                   label='low confidence / pinned')

    if hline is not None:
        ax.axhline(hline, ls='--', color='#e74c3c', lw=1.2, zorder=2)

    if yscale == 'log':
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(labelOnlyBase=False))

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, loc='best')


# ── Main plots ────────────────────────────────────────────────────────────────

def make_metastable_plots(df, out_dir):
    chips = df['chip'].fillna('?').astype(str)
    cmap  = _chip_palette(chips.unique())

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle('Metastable state structure vs ZPL', fontsize=14, fontweight='bold')

    _scatter_meta(axes[0, 0], df, 'ZPL_nm', 'bunching_height',
                  'ZPL (nm)', 'Bunching amplitude  a·b', cmap,
                  pinned_col='T2_pinned')

    _scatter_meta(axes[0, 1], df, 'ZPL_nm', 'T2_ns',
                  'ZPL (nm)', 'T₂  fast metastable lifetime (ns)', cmap,
                  yscale='log', pinned_col='T2_pinned')

    _scatter_meta(axes[0, 2], df, 'ZPL_nm', 'T3_ns',
                  'ZPL (nm)', 'T₃  slow dynamics lifetime (ns)', cmap,
                  yscale='log', pinned_col='T3_pinned')

    _scatter_meta(axes[1, 0], df, 'ZPL_nm', 'c',
                  'ZPL (nm)', 'Slow amplitude  c  [g²(∞) = 1−c]', cmap,
                  pinned_col='T3_pinned')

    _scatter_meta(axes[1, 1], df, 'ZPL_nm', 'g2_0',
                  'ZPL (nm)', 'g²(0)', cmap,
                  hline=SPE_THRESHOLD)

    _scatter_meta(axes[1, 2], df, 'bunching_height', 'g2_0',
                  'Bunching amplitude  a·b', 'g²(0)', cmap,
                  hline=SPE_THRESHOLD)

    # Shared chip legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=c, markeredgecolor='k',
                   markersize=9, label=f'Ch{ch}')
        for ch, c in cmap.items()
    ]
    handles += [
        plt.Line2D([0], [0], marker='*', color='w',
                   markerfacecolor='grey', markeredgecolor='k',
                   markersize=11, label='single emitter'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='none', markeredgecolor='grey',
                   markersize=9, label='low confidence / pinned'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=len(cmap) + 2,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(out_dir, 'metastable_vs_zpl.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {out_path}')


def make_lifetime_histograms(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Metastable lifetime distributions', fontsize=13)

    t2 = df['T2_ns'].dropna()
    axes[0].hist(t2, bins=20, edgecolor='k', color='steelblue', log=False)
    axes[0].set_xlabel('T₂ fast metastable (ns)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Fast metastable lifetime T₂')

    t3_ms = df['T3_ns'].dropna() / 1e6
    axes[1].hist(t3_ms, bins=20, edgecolor='k', color='darkorange')
    axes[1].set_xlabel('T₃ slow dynamics (ms)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Slow dynamics lifetime T₃')

    bh = df['bunching_height'].dropna()
    axes[2].hist(bh, bins=20, edgecolor='k', color='seagreen')
    axes[2].set_xlabel('Bunching amplitude  a·b')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Bunching shoulder height')

    fig.tight_layout()
    out_path = os.path.join(out_dir, 'metastable_histograms.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Analyse metastable state structure vs ZPL.')
    ap.add_argument('--data-dir',   default=DATA_DIR, help='Path to data/ folder')
    ap.add_argument('--out-dir',    default=OUT_DIR,  help='Output folder for plots')
    ap.add_argument('--from-date',  default=None,     metavar='YYYYMMDD')
    ap.add_argument('--to-date',    default=None,     metavar='YYYYMMDD')
    args = ap.parse_args()

    print('Scanning for multiscale g2 results...')
    rows = list(iter_metastable(args.data_dir,
                                from_date=args.from_date,
                                to_date=args.to_date))

    if not rows:
        print('No *_multiscale_results.npz files found with matching long_filter spectra.')
        print('Run g2_multiscale.py first to generate the results files.')
        return

    df = pd.DataFrame(rows)

    # Drop unphysical fits
    bad = df[df['g2_0'] < 0]
    if not bad.empty:
        print(f'\nDropped {len(bad)} emitter(s) with g²(0) < 0 (unphysical fit).')
        df = df[df['g2_0'] >= 0].reset_index(drop=True)

    n_spe     = int((df['g2_0'] < SPE_THRESHOLD).sum())
    n_pinned  = int(df['T2_pinned'].sum())
    n_t3pin   = int(df['T3_pinned'].sum())
    print(f'\nFound {len(df)} emitters with fit results'
          f' ({n_spe} single emitters, {n_pinned} T2-pinned, {n_t3pin} T3-pinned).\n')

    cols = ['chip', 'field', 'x', 'y', 'ZPL_nm', 'g2_0',
            'bunching_height', 'T2_ns', 'T3_ns', 'c', 'T2_pinned', 'T3_pinned']
    print(df[cols].to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    os.makedirs(args.out_dir, exist_ok=True)

    csv_path = os.path.join(args.out_dir, 'metastable_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}')

    make_metastable_plots(df, args.out_dir)
    make_lifetime_histograms(df, args.out_dir)


if __name__ == '__main__':
    main()
