import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

DATA_DIR = 'data'
OUT_DIR  = 'analysis_output'

# ── 50 ns g2 calculation config ───────────────────────────────────────────────
G2TIME_NS          = 50.0     # correlation half-window (ns)
TIMEBIN_NS         = 0.25     # bin width (ns)
AFTERFLASH_LOW_NS  = 15.0
AFTERFLASH_HIGH_NS = 35.0
WING_FRAC_LOW      = 0.90
WING_FRAC_HIGH     = 0.95
G0_FIXED           = 1.0
SPE_THRESHOLD      = 0.5

# ── Spectrum helpers — Gaussian fitting ───────────────────────────────────────

_FWHM_K = 2.0 * np.sqrt(2.0 * np.log(2.0))   # 2.3548
_AREA_K = np.sqrt(2.0 * np.pi)               # Gaussian area = A * sigma * sqrt(2*pi)


def _gaussian1d(x, A, mu, sigma, bkg):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + bkg


def _fit_zpl_gaussian(spectrum, wl, wl_min_nm=560.0, wl_max_nm=630.0,
                       window_nm=15.0, min_snr=3.0, max_fwhm_nm=45.0):
    """Fit a Gaussian to the ZPL peak, restricted to wl in [wl_min_nm, wl_max_nm].

    Returns (zpl_nm, fwhm_nm, area). All None if the fit fails or the
    spectrum has no clear ZPL (flat / edge-clipped peak with low SNR).
    Area is the integrated Gaussian intensity (A * sigma * sqrt(2*pi)),
    used downstream for the Debye-Waller factor.
    """
    mask = (wl > wl_min_nm) & (wl < wl_max_nm)
    if not mask.any():
        return None, None, None
    wl_m, sp_m  = wl[mask], spectrum[mask]
    pk          = int(np.argmax(sp_m))
    peak_wl     = float(wl_m[pk])
    peak_val    = float(sp_m[pk])

    # Baseline and noise from the wings outside the fit window
    wing   = np.abs(wl_m - peak_wl) > window_nm
    bkg0   = float(np.median(sp_m[wing])) if wing.any() else float(np.percentile(sp_m, 10))
    noise  = float(np.std(sp_m[wing]))    if wing.sum() > 5 else 1.0

    if (peak_val - bkg0) < min_snr * max(noise, 1.0):
        return None, None, None   # flat spectrum / no clear ZPL

    win = np.abs(wl_m - peak_wl) <= window_nm
    if win.sum() < 5:
        return None, None, None

    x, y = wl_m[win], sp_m[win]
    A0   = peak_val - bkg0
    p0   = [A0, peak_wl, 1.5, bkg0]
    try:
        popt, _ = curve_fit(
            _gaussian1d, x, y, p0=p0,
            bounds=([0,      max(peak_wl - window_nm, wl_min_nm), 0.05,            -np.inf],
                    [A0 * 3, min(peak_wl + window_nm, wl_max_nm), max_fwhm_nm / _FWHM_K, peak_val]),
            maxfev=3000,
        )
        A, mu, sigma, _ = popt
        if A <= 0:
            return peak_wl, None, None
        area = float(A * sigma * _AREA_K)
        return float(mu), float(_FWHM_K * sigma), area
    except Exception:
        return peak_wl, None, None


def _fit_psb_gaussian(spectrum, wl, zpl_nm,
                       psb_min_nm=10.0, psb_max_nm=100.0,
                       window_nm=20.0, min_snr=3.0,
                       label=None, verbose=False):
    """Fit a Gaussian to the phonon sideband (red of ZPL).

    Returns (psb_nm, psb_fwhm_nm, area) or (None, None, None) if no PSB found.
    Area is the integrated Gaussian intensity, used for the Debye-Waller factor.

    If verbose=True, prints the reason for a failed fit (prefixed with `label`
    if given) so failure modes can be diagnosed per-emitter.
    """
    tag = f'[{label}] ' if label else ''
    mask = (wl > zpl_nm + psb_min_nm) & (wl < zpl_nm + psb_max_nm)
    if not mask.any() or mask.sum() < 10:
        if verbose:
            wl_max = float(wl.max()) if wl.size else float('nan')
            print(f'{tag}PSB skip: only {int(mask.sum())} pts in '
                  f'[{zpl_nm + psb_min_nm:.1f}, {zpl_nm + psb_max_nm:.1f}] nm '
                  f'(wl data only goes to {wl_max:.1f} nm)')
        return None, None, None
    wl_m, sp_m = wl[mask], spectrum[mask]
    pk      = int(np.argmax(sp_m))
    pk_wl   = float(wl_m[pk])
    pk_val  = float(sp_m[pk])

    wing   = np.abs(wl_m - pk_wl) > window_nm
    bkg0   = float(np.median(sp_m[wing])) if wing.any() else float(np.percentile(sp_m, 10))
    noise  = float(np.std(sp_m[wing]))    if wing.sum() > 5 else 1.0

    snr = (pk_val - bkg0) / max(noise, 1.0)
    if snr < min_snr:
        if verbose:
            print(f'{tag}PSB skip: SNR={snr:.2f} < {min_snr} '
                  f'(peak={pk_val:.1f} @ {pk_wl:.1f} nm, bkg={bkg0:.1f}, noise={noise:.1f})')
        return None, None, None

    win = np.abs(wl_m - pk_wl) <= window_nm
    if win.sum() < 5:
        if verbose:
            print(f'{tag}PSB skip: only {int(win.sum())} pts within {window_nm} nm of peak')
        return None, None, None

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
            if verbose:
                print(f'{tag}PSB skip: fit converged but out of range '
                      f'(A={A:.1f}, mu={mu:.1f})')
            return None, None, None
        area = float(A * sigma * _AREA_K)
        return float(mu), float(_FWHM_K * sigma), area
    except Exception as exc:
        if verbose:
            print(f'{tag}PSB skip: curve_fit failed ({exc})')
        return None, None, None


def _debye_waller(zpl_area, psb_area):
    """DWF = I_ZPL / (I_ZPL + I_PSB), using integrated Gaussian areas."""
    if zpl_area is None or psb_area is None:
        return None
    total = zpl_area + psb_area
    if total <= 0:
        return None
    return float(zpl_area / total)


# ── 50 ns g2 calculation (from g2_standalone_50ns.py) ─────────────────────────

def _model_g2(x, a, b, T1, T2):
    """Model with g0 fixed at G0_FIXED."""
    return G0_FIXED - b * ((1 + a) * np.exp(-np.abs(x) / T1)
                           - a * np.exp(-np.abs(x) / T2))


def _cross_correlation_hist(ch0, ch1, g2time_ps, timebin_ps, chunk=200_000):
    """
    Right-closed bins: bin k iff edges[k] < dt <= edges[k+1]
    (np.searchsorted(..., side='left') - 1 gives this convention).
    """
    I      = int(np.ceil(g2time_ps / timebin_ps))
    n_bins = 2 * I + 1
    hist   = np.zeros(n_bins, dtype=np.int64)
    edges  = (np.arange(n_bins + 1, dtype=np.int64) - I) * timebin_ps

    ch0 = np.sort(ch0.astype(np.int64))
    ch1 = np.sort(ch1.astype(np.int64))

    for start in range(0, len(ch0), chunk):
        ch0c   = ch0[start:start + chunk]
        lo     = np.searchsorted(ch1, ch0c - g2time_ps, side='left')
        hi     = np.searchsorted(ch1, ch0c + g2time_ps, side='right')
        counts = (hi - lo).astype(np.int64)
        total  = int(counts.sum())
        if total == 0:
            continue

        starts  = np.zeros(len(ch0c), dtype=np.int64)
        np.cumsum(counts[:-1], out=starts[1:])
        offsets = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
        t1_idx  = np.repeat(lo.astype(np.int64), counts) + offsets
        dt      = ch1[t1_idx] - np.repeat(ch0c, counts)

        bins  = np.searchsorted(edges, dt, side='left').astype(np.int64) - 1
        valid = (bins >= 0) & (bins < n_bins)
        np.add.at(hist, bins[valid], 1)

    return hist


def _fit_g2(tau, g2):
    """Multi-start grid search for g2 fit (g0 fixed, only a, b, T1, T2 free)."""
    best_popt, best_res = None, np.inf
    for b0 in [0.3, 0.5, 0.7, 0.9]:
        for T1_0 in [0.5, 1, 3, 10]:
            for a0 in [0, 1]:
                try:
                    popt, _ = curve_fit(
                        _model_g2, tau, g2,
                        p0=[a0, b0, T1_0, 500.0],
                        bounds=([0, 0, 0.05, 1.0],
                                [10.0, G0_FIXED * 1.5 + 0.5, 100.0, 1e5]),
                        maxfev=10_000
                    )
                    res = float(np.sum((_model_g2(tau, *popt) - g2) ** 2))
                    if res < best_res:
                        best_res, best_popt = res, popt
                except Exception:
                    continue
    return best_popt


def calculate_g2_50ns(ch0, ch1):
    """
    Calculate g2 with 50 ns window from raw channel data.
    Returns dict with tau, g2, popt (fit params), g2_0 (fit value at tau=0).
    """
    g2time_ps  = int(round(G2TIME_NS  * 1000))
    timebin_ps = int(round(TIMEBIN_NS * 1000))
    I          = int(np.ceil(g2time_ps / timebin_ps))
    n_bins     = 2 * I + 1
    tau_ns     = (np.arange(n_bins) - I) * timebin_ps / 1000.0

    # Build histogram
    hist = _cross_correlation_hist(ch0, ch1, g2time_ps, timebin_ps)

    # Far-wing normalisation
    wing_low  = G2TIME_NS * WING_FRAC_LOW
    wing_high = G2TIME_NS * WING_FRAC_HIGH
    wing_mask = (np.abs(tau_ns) >= wing_low) & (np.abs(tau_ns) <= wing_high)
    wing_vals = hist[wing_mask]
    if wing_vals.size == 0:
        return None
    c_wing    = float(wing_vals.mean())
    if c_wing <= 0:
        return None

    g2_arr = hist.astype(float) / c_wing

    # Fit: exclude afterflash region and edges
    af_mask   = (np.abs(tau_ns) >= AFTERFLASH_LOW_NS) & (np.abs(tau_ns) <= AFTERFLASH_HIGH_NS)
    edge_mask = (np.arange(n_bins) > 0) & (np.arange(n_bins) < n_bins - 1)
    fit_mask  = edge_mask & ~af_mask

    popt = _fit_g2(tau_ns[fit_mask], g2_arr[fit_mask])

    g2_0 = None
    if popt is not None:
        a, b, T1, T2 = popt
        g2_0 = float(G0_FIXED - b)
    
    return {
        'tau': tau_ns,
        'g2': g2_arr,
        'popt': popt,
        'g2_0': g2_0,
        'wing_level': c_wing,
    }


# ── Data extraction ───────────────────────────────────────────────────────────

_RUN_RE   = re.compile(r'(?P<date>\d{8})-PLSPC-HT-Ch(?P<chip>\w+)-f(?P<field>\d+)-')
_COORD_RE = re.compile(r'_x(?P<x>-?[\d.]+)_y(?P<y>-?[\d.]+)')


def _parse_run_meta(run_name):
    m = _RUN_RE.search(run_name)
    if not m:
        return {'chip': '?', 'field': '?', 'date': run_name[:8]}
    return {'chip': m.group('chip'), 'field': m.group('field'), 'date': m.group('date')}


def iter_emitters(data_dir, verbose=False):
    """Yield one dict per emitter that has raw g2 data (.npz) and a matching spectrum.
    
    Recalculates g2 with 50 ns window from raw data, excluding processed files.
    """
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
                g2_result = calculate_g2_50ns(ch0, ch1)
                if g2_result is None:
                    continue
                
                popt = g2_result['popt']
                g2_0_norm = g2_result['g2_0']
                
                if popt is None or g2_0_norm is None:
                    continue
                    
                a, b, T1, T2 = popt
                
            except Exception as e:
                if verbose:
                    print(f"Skipped {run_name}/{coord_str}: {e}")
                continue

            # ── filtered spectrum → ZPL + FWHM (Gaussian fit, 560-630 nm) ──
            lf_path = os.path.join(run_path, lf_dir)
            try:
                wl       = np.load(os.path.join(lf_path, 'wl.npy'))
                out      = np.load(os.path.join(lf_path, 'out.npy'))
                spectrum = out[0, 0, :]
            except (FileNotFoundError, IndexError):
                continue

            label = f'{run_name}/{coord_str}'

            zpl, fwhm, zpl_area = _fit_zpl_gaussian(spectrum, wl)
            if zpl is None:
                continue

            psb_nm, psb_fwhm_nm, psb_area = _fit_psb_gaussian(
                spectrum, wl, zpl, label=label, verbose=verbose)
            dwf = _debye_waller(zpl_area, psb_area)

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
                'DWF':         dwf,
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

    _scatter(axes[0, 0], df, 'ZPL_nm',  'g2_0',  'ZPL (nm)',      'g²(0)', cmap)
    _scatter(axes[0, 1], df, 'FWHM_nm', 'g2_0',  'ZPL FWHM (nm)', 'g²(0)', cmap)
    _scatter(axes[0, 2], df, 'DWF',     'g2_0',  'Debye-Waller factor', 'g²(0)', cmap)
    _scatter(axes[1, 0], df, 'ZPL_nm',  'T1_ns', 'ZPL (nm)',      'T₁ (ns)', cmap)
    _scatter(axes[1, 1], df, 'FWHM_nm', 'T1_ns', 'ZPL FWHM (nm)', 'T₁ (ns)', cmap)
    axes[1, 2].axis('off')   # unused panel

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
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    fig.suptitle('Emitter property distributions', fontsize=13)

    axes[0].hist(df['ZPL_nm'].dropna(),  bins=20, edgecolor='k', color='steelblue')
    axes[0].set_xlabel('ZPL (nm)');  axes[0].set_ylabel('Count')
    axes[0].set_title('ZPL')

    axes[1].hist(df['FWHM_nm'].dropna(), bins=20, edgecolor='k', color='seagreen')
    axes[1].set_xlabel('FWHM (nm)'); axes[1].set_ylabel('Count')
    axes[1].set_title('FWHM')

    axes[2].hist(df['DWF'].dropna(), bins=20, edgecolor='k', color='goldenrod')
    axes[2].set_xlabel('Debye-Waller factor'); axes[2].set_ylabel('Count')
    axes[2].set_title('DWF')

    axes[3].hist(df['g2_0'].dropna(),    bins=20, edgecolor='k', color='salmon')
    axes[3].axvline(0.5, ls='--', color='red', lw=1.2, label='g²(0) = 0.5')
    axes[3].set_xlabel('g²(0)');     axes[3].set_ylabel('Count')
    axes[3].set_title('g²(0)')
    axes[3].legend()

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
    ap.add_argument('--verbose', action='store_true',
                     help='Print the reason each PSB (DWF) fit was skipped')
    args = ap.parse_args()

    print('Scanning data folders...')
    rows = list(iter_emitters(args.data_dir, verbose=args.verbose))

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

    cols = ['chip', 'field', 'x', 'y', 'ZPL_nm', 'FWHM_nm', 'DWF', 'g2_0', 'T1_ns']
    print(df[cols].to_string(index=False, float_format=lambda v: f'{v:.3f}' if pd.notna(v) else 'None'))

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'emitter_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}')

    make_scatter_plots(df, args.out_dir)
    make_histogram_plots(df, args.out_dir)


if __name__ == '__main__':
    main()