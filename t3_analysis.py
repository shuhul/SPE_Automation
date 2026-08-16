"""
emitter_T3_analysis.py — for every CONFIRMED single-photon emitter, save the
dual g2 plot (linear + log, with the multiscale fit and T1/T2/T3 annotated),
plus the on/off blinking scan at 0.1xT3/T3/10xT3, into a folder tree that
mirrors your actual data layout:

    <OUT_DIR>/<run folder name>/<g2 folder name>/
        dual_g2_plot.png
        g2_fit_summary.txt
        blinking_at_T3.png            (only if T3 resolved)
        blinking_at_T3_summary.txt    (only if T3 resolved)



Only CONFIRMED single emitters (g2(0) < SPE_THRESHOLD, wing normalisation
reliable) get a folder at all — non-SPE emitters still contribute to the
T3-vs-ZPL/FWHM correlation plots and the summary CSV, but don't get the
per-emitter plot tree, since that's specifically what you asked for.

Fit algorithm — ported from the reference dual-plot script: a single
stage-2 fit over all 6 parameters (a, b, T1, T2, c, T3), seeded from a
dip-only short-regime fit. T1/T2 are never swapped in the stored
parameters (only sorted for display) — swapping without adjusting `a`
would corrupt the model curve.

NOTE: the reference script itself had MID_TIME_NS = LONG_TIME_NS = 50.0
with 0.10 ns bins everywhere, which collapses all three regimes to the
same window and makes T2/T3 unresolvable. This script uses the reference's
fit ALGORITHM but with proper multiscale windows (50 ns / 100 us / 10 ms)
so T2/T3 stay resolvable — worth checking that config on your end too.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

DATA_DIR = 'data'
OUT_DIR  = 'analysis_output'

# ── Multiscale g2 config (proper 3-regime windows) ────────────────────────────
SHORT_TIME_NS       = 50.0
SHORT_BIN_NS        = 0.250

MID_TIME_NS         = 100_000.0        # 100 us
MID_BIN_NS          = 200.0            # 0.2 us

LONG_TIME_NS        = 10_000_000.0     # 10 ms
LONG_BIN_NS         = 6000.0           # 6 us

AFTERFLASH_LOW      = 10.0
AFTERFLASH_HIGH     = 35.0

WING_FRAC_LOW_LONG  = 0.90
WING_FRAC_HIGH_LONG = 0.98

LINEAR_XLIM_NS      = SHORT_TIME_NS
LOG_XLIM_LOW_NS      = 1e-1

SPE_THRESHOLD       = 0.5
G0_FIXED            = 1.0

BOUND_PIN_TOL       = 0.02
WING_ZERO_FRAC_WARN = 0.5
WING_RATIO_WARN     = 0.25

T1_BOUNDS = (0.01, 100.0)
T2_BOUNDS = (1.0, 1e5)
T3_BOUNDS = (1.0, 1e8)

# ── Spectral fit config ───────────────────────────────────────────────────────
MAX_FWHM_NM = 30.0
_FWHM_K = 2.0 * np.sqrt(2.0 * np.log(2.0))

# ── Blinking-scan-at-T3 safety cap ────────────────────────────────────────────
MAX_BLINKING_BINS = 20_000_000


# =============================================================================
# ZPL spectral fit
# =============================================================================
def _gaussian1d(x, A, mu, sigma, bkg):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + bkg


def _fit_zpl_gaussian(spectrum, wl, wl_min_nm=560.0, wl_max_nm=630.0,
                       window_nm=15.0, min_snr=3.0, max_fwhm_nm=45.0):
    mask = (wl > wl_min_nm) & (wl < wl_max_nm)
    if not mask.any():
        return None, None
    wl_m, sp_m = wl[mask], spectrum[mask]
    pk       = int(np.argmax(sp_m))
    peak_wl  = float(wl_m[pk])
    peak_val = float(sp_m[pk])

    wing  = np.abs(wl_m - peak_wl) > window_nm
    bkg0  = float(np.median(sp_m[wing])) if wing.any() else float(np.percentile(sp_m, 10))
    noise = float(np.std(sp_m[wing]))    if wing.sum() > 5 else 1.0
    if (peak_val - bkg0) < min_snr * max(noise, 1.0):
        return None, None

    win = np.abs(wl_m - peak_wl) <= window_nm
    if win.sum() < 5:
        return None, None

    x, y = wl_m[win], sp_m[win]
    A0   = peak_val - bkg0
    try:
        popt, _ = curve_fit(
            _gaussian1d, x, y, p0=[A0, peak_wl, 1.5, bkg0],
            bounds=([0,      max(peak_wl - window_nm, wl_min_nm), 0.05,                  -np.inf],
                    [A0 * 3, min(peak_wl + window_nm, wl_max_nm), max_fwhm_nm / _FWHM_K, peak_val]),
            maxfev=3000,
        )
        A, mu, sigma, _ = popt
        if A <= 0:
            return None, None
        fwhm = float(_FWHM_K * sigma)
        if fwhm > MAX_FWHM_NM:
            return None, None
        return float(mu), fwhm
    except Exception:
        return None, None


# =============================================================================
# Multiscale g2 fit — algorithm from the reference dual-plot script
# =============================================================================
def _cross_correlation_hist(ch0, ch1, g2time_ps, timebin_ps, chunk=200_000):
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


def _build_regime(ch0, ch1, time_ns, bin_ns):
    g2time_ps  = int(round(time_ns * 1000))
    timebin_ps = int(round(bin_ns  * 1000))
    I          = int(np.ceil(g2time_ps / timebin_ps))
    n_bins     = 2 * I + 1
    tau_ns     = (np.arange(n_bins) - I) * timebin_ps / 1000.0
    hist       = _cross_correlation_hist(ch0, ch1, g2time_ps, timebin_ps)
    return tau_ns, hist, timebin_ps


def _model3(x, a, b, T1, T2, c, T3):
    fast = (1 + a) * np.exp(-np.abs(x) / T1) - a * np.exp(-np.abs(x) / T2)
    slow = 1 - np.exp(-np.abs(x) / T3)
    return G0_FIXED - b * fast - c * slow


def _model3_dip_only(x, a, b, T1, T2):
    fast = (1 + a) * np.exp(-np.abs(x) / T1) - a * np.exp(-np.abs(x) / T2)
    return G0_FIXED - b * fast


def _fit_g2_multiscale(tau_short, g2_short, tau_mid, g2_mid, tau_long, g2_long):
    finite  = g2_short[np.isfinite(g2_short)]
    ceiling = float(np.nanmax(finite)) * 1.5 if finite.size else 2.0
    ceiling = max(ceiling, 2.0)

    dip_idx  = int(np.nanargmin(g2_short))
    dip_min  = float(g2_short[dip_idx])
    b0_guess = max(0.05, G0_FIXED - dip_min)
    T1_guess = max(0.2, float(np.abs(tau_short[dip_idx])) + 0.5)

    stage1_seeds = [
        (0.3, b0_guess, T1_guess, 500.0),
        (0.5, b0_guess * 1.2, T1_guess * 3, 5000.0),
        (0.1, b0_guess, T1_guess, 100.0),
    ]
    best1, best1_res = None, np.inf
    for a0, b0, T1_0, T2_0 in stage1_seeds:
        try:
            popt, _ = curve_fit(
                _model3_dip_only, tau_short, g2_short,
                p0=[a0, b0, T1_0, T2_0],
                bounds=([0, 0, T1_BOUNDS[0], T2_BOUNDS[0]],
                        [10.0, ceiling, T1_BOUNDS[1], T2_BOUNDS[1]]),
                maxfev=5000)
            res = float(np.sum((_model3_dip_only(tau_short, *popt) - g2_short) ** 2))
            if res < best1_res:
                best1_res, best1 = res, popt
        except Exception:
            continue
    if best1 is None:
        return None, None
    a0, b0, T1_0, T2_0 = best1

    tau_all = np.concatenate([tau_short, tau_mid, tau_long])
    g2_all  = np.concatenate([g2_short,  g2_mid,  g2_long])
    finite_mask = np.isfinite(tau_all) & np.isfinite(g2_all)
    tau_all, g2_all = tau_all[finite_mask], g2_all[finite_mask]

    long_finite = g2_long[np.isfinite(g2_long)]
    c0_guess = max(0.02, G0_FIXED - float(np.nanmin(long_finite))) if long_finite.size else 0.1

    stage2_seeds = [
        (a0, b0, T1_0, T2_0, c0_guess,       1e5),
        (a0, b0, T1_0, T2_0, c0_guess,       1e6),
        (a0, b0, T1_0, T2_0, c0_guess * 0.5, 5e5),
        (a0, b0, T1_0, T2_0, 0.05,           2e6),
    ]
    best2, best2_res = None, np.inf
    for a1, b1, T1_1, T2_1, c1, T3_1 in stage2_seeds:
        try:
            popt, _ = curve_fit(
                _model3, tau_all, g2_all,
                p0=[a1, b1, T1_1, T2_1, c1, T3_1],
                bounds=([0, 0, T1_BOUNDS[0], T2_BOUNDS[0], 0, T3_BOUNDS[0]],
                        [10.0, ceiling, T1_BOUNDS[1], T2_BOUNDS[1], 1.0, T3_BOUNDS[1]]),
                maxfev=8000)
            res = float(np.sum((_model3(tau_all, *popt) - g2_all) ** 2))
            if res < best2_res:
                best2_res, best2 = res, popt
        except Exception:
            continue
    if best2 is None:
        return None, None

    a, b, T1, T2, c, T3 = best2

    def pinned(val, lo, hi, tol=BOUND_PIN_TOL):
        near_lo = lo > 0 and (val - lo) < tol * lo
        near_hi = hi > 0 and (hi - val) < tol * hi
        return near_lo or near_hi

    bound_warnings = {
        'T1_pinned': pinned(T1, *T1_BOUNDS),
        'T2_pinned': pinned(T2, *T2_BOUNDS),
        'T3_pinned': pinned(T3, *T3_BOUNDS),
    }
    return list(best2), bound_warnings


def _compute_multiscale_g2(ch0, ch1):
    """Full short/mid/long g2 pipeline. Returns a dict with the fit result
    AND the plot-ready tau/g2 arrays (so the dual plot can be drawn later
    without rebuilding the histograms), or None if it fails outright."""
    tau_l, hist_l, timebin_ps_l = _build_regime(ch0, ch1, LONG_TIME_NS, LONG_BIN_NS)
    n_bins_l = len(tau_l)

    wing_low_l  = LONG_TIME_NS * WING_FRAC_LOW_LONG
    wing_high_l = LONG_TIME_NS * WING_FRAC_HIGH_LONG
    wing_mask_l = (np.abs(tau_l) >= wing_low_l) & (np.abs(tau_l) <= wing_high_l)
    wing_vals_l = hist_l[wing_mask_l]

    wing_reliable = True
    if wing_vals_l.size > 0:
        c_wing_l  = float(np.mean(wing_vals_l))
        zero_frac = float(np.mean(wing_vals_l == 0))
        if zero_frac > WING_ZERO_FRAC_WARN or c_wing_l <= 0:
            wing_reliable = False
    else:
        N1, N2 = len(ch0), len(ch1)
        T_acq  = int(max(ch0[-1], ch1[-1])) / 1e12
        c_wing_l = (N1 / T_acq) * (N2 / T_acq) * (timebin_ps_l * 1e-12) * T_acq if T_acq > 0 else 0
        wing_reliable = False

    if c_wing_l <= 0:
        return None

    N1, N2 = len(ch0), len(ch1)
    T_acq  = int(max(ch0[-1], ch1[-1])) / 1e12
    if T_acq > 0:
        R1, R2 = N1 / T_acq, N2 / T_acq
        c_wing_l_analytic = R1 * R2 * (timebin_ps_l * 1e-12) * T_acq
        wing_ratio = c_wing_l / c_wing_l_analytic if c_wing_l_analytic > 0 else np.nan
        if not np.isfinite(wing_ratio) or abs(wing_ratio - 1.0) > WING_RATIO_WARN:
            wing_reliable = False

    g2_l = hist_l.astype(float) / c_wing_l
    af_mask_l   = (np.abs(tau_l) >= AFTERFLASH_LOW) & (np.abs(tau_l) <= AFTERFLASH_HIGH)
    edge_mask_l = (np.arange(n_bins_l) > 0) & (np.arange(n_bins_l) < n_bins_l - 1)
    g2_plot_l               = g2_l.copy()
    g2_plot_l[af_mask_l]    = np.nan
    g2_plot_l[~edge_mask_l] = np.nan

    timebin_ps_m = int(round(MID_BIN_NS   * 1000))
    timebin_ps_s = int(round(SHORT_BIN_NS * 1000))
    c_wing_m = c_wing_l * (timebin_ps_m / timebin_ps_l)
    c_wing_s = c_wing_l * (timebin_ps_s / timebin_ps_l)

    tau_m, hist_m, _ = _build_regime(ch0, ch1, MID_TIME_NS, MID_BIN_NS)
    n_bins_m = len(tau_m)
    g2_m = hist_m.astype(float) / c_wing_m
    af_mask_m   = (np.abs(tau_m) >= AFTERFLASH_LOW) & (np.abs(tau_m) <= AFTERFLASH_HIGH)
    edge_mask_m = (np.arange(n_bins_m) > 0) & (np.arange(n_bins_m) < n_bins_m - 1)
    g2_plot_m               = g2_m.copy()
    g2_plot_m[af_mask_m]    = np.nan
    g2_plot_m[~edge_mask_m] = np.nan

    tau_s, hist_s, _ = _build_regime(ch0, ch1, SHORT_TIME_NS, SHORT_BIN_NS)
    n_bins_s = len(tau_s)
    g2_s = hist_s.astype(float) / c_wing_s
    af_mask_s   = (np.abs(tau_s) >= AFTERFLASH_LOW) & (np.abs(tau_s) <= AFTERFLASH_HIGH)
    edge_mask_s = (np.arange(n_bins_s) > 0) & (np.arange(n_bins_s) < n_bins_s - 1)
    fit_mask_s  = edge_mask_s & ~af_mask_s
    g2_plot_s               = g2_s.copy()
    g2_plot_s[af_mask_s]    = np.nan
    g2_plot_s[~edge_mask_s] = np.nan

    fit_mask_m = (np.abs(tau_m) > SHORT_TIME_NS) & edge_mask_m & ~af_mask_m
    fit_mask_l = (np.abs(tau_l) > MID_TIME_NS)   & edge_mask_l & ~af_mask_l

    popt, bound_warnings = _fit_g2_multiscale(
        tau_s[fit_mask_s], g2_s[fit_mask_s],
        tau_m[fit_mask_m], g2_m[fit_mask_m],
        tau_l[fit_mask_l], g2_l[fit_mask_l])

    if popt is None:
        return dict(popt=None, bound_warnings=None, g2_0=None,
                    wing_reliable=wing_reliable)

    a, b, T1, T2, c, T3 = popt
    g2_0 = float(G0_FIXED - b)

    return dict(
        popt=popt, bound_warnings=bound_warnings, g2_0=g2_0,
        wing_reliable=wing_reliable,
        tau_s=tau_s, g2_plot_s=g2_plot_s,
        tau_m=tau_m, g2_plot_m=g2_plot_m,
        tau_l=tau_l, g2_plot_l=g2_plot_l,
        timebin_ps_s=timebin_ps_s,
    )


def plot_dual_g2(label, fit, out_path):
    """Draw the linear + log dual g2 plot for one emitter and save it."""
    popt, bound_warnings, g2_0, wing_reliable = (
        fit['popt'], fit['bound_warnings'], fit['g2_0'], fit['wing_reliable'])
    tau_s, g2_plot_s = fit['tau_s'], fit['g2_plot_s']
    tau_m, g2_plot_m = fit['tau_m'], fit['g2_plot_m']
    tau_l, g2_plot_l = fit['tau_l'], fit['g2_plot_l']
    timebin_ps_s = fit['timebin_ps_s']

    a, b, T1, T2, c, T3 = popt
    g2_inf = float(G0_FIXED - c)
    T1_disp, T2_disp = min(T1, T2), max(T1, T2)
    result_reliable = (wing_reliable and not bound_warnings['T1_pinned']
                        and not bound_warnings['T2_pinned'])

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle(f'g\u00b2(\u03c4) \u2014 {label}', fontsize=13)

    all_visible = np.concatenate([
        g2_plot_s[np.isfinite(g2_plot_s)],
        g2_plot_m[np.isfinite(g2_plot_m)],
        g2_plot_l[np.isfinite(g2_plot_l)],
    ])
    ylim_top = max(1.5, float(np.percentile(all_visible, 99)) * 1.3) if all_visible.size else 1.5

    def draw_common(ax):
        for sign in [+1, -1]:
            ax.axvspan(sign * AFTERFLASH_LOW, sign * AFTERFLASH_HIGH,
                       color='#f0e6ff', alpha=0.85, zorder=0,
                       label=' ' if sign == 1 else None)
        ax.axvline(0, ls=':', color='#bbb', lw=0.8, zorder=1)
        ax.axhline(G0_FIXED,      ls='--', color='#555',     lw=0.9, label=f'g\u2080 = {G0_FIXED:.1f} ')
        ax.axhline(SPE_THRESHOLD, ls='-.', color='#e74c3c',  lw=1.1, label=f'g\u00b2=0.{int(SPE_THRESHOLD*10)} ')

    draw_common(ax_lin)
    lin_mask = (tau_s >= -LINEAR_XLIM_NS) & (tau_s <= LINEAR_XLIM_NS)
    ax_lin.plot(tau_s[lin_mask], g2_plot_s[lin_mask],
                color=[0.55, 0.55, 0.55], lw=1.1, label='g\u00b2(\u03c4)', zorder=2)

    tf_lin     = np.linspace(-LINEAR_XLIM_NS, LINEAR_XLIM_NS, 3000)
    g2_fit_lin = _model3(tf_lin, *popt).copy()
    g2_fit_lin[(np.abs(tf_lin) >= AFTERFLASH_LOW) &
               (np.abs(tf_lin) <= AFTERFLASH_HIGH)] = np.nan
    ax_lin.plot(tf_lin, g2_fit_lin, 'k', lw=1.8, zorder=3,
                label=f'Fit  g\u00b2(0) = {g2_0:.3f}')

    for sign in [+1, -1]:
        ax_lin.text(sign * (AFTERFLASH_LOW + AFTERFLASH_HIGH) / 2,
                    0.04, 'afterflash\nremoved',
                    ha='center', va='bottom', fontsize=7.5,
                    color='#7b3fa0', style='italic',
                    transform=ax_lin.get_xaxis_transform())

    ax_lin.set_ylim(0, ylim_top)
    ax_lin.set_xlim(-LINEAR_XLIM_NS, LINEAR_XLIM_NS)
    ax_lin.set_xlabel('\u03c4 (ns)', fontsize=14)
    ax_lin.set_ylabel('g\u00b2(\u03c4)',  fontsize=14)
    ax_lin.set_title('Linear scale  (antibunching dip)', fontsize=11)
    ax_lin.legend(fontsize=9, loc='lower right')

    draw_common(ax_log)
    data_floor_ns = timebin_ps_s / 1000.0

    mask_s = (tau_s > data_floor_ns)  & (tau_s <= SHORT_TIME_NS)
    mask_m = (tau_m > SHORT_TIME_NS)  & (tau_m <= MID_TIME_NS)
    mask_l = (tau_l > MID_TIME_NS)    & (tau_l <= LONG_TIME_NS)

    tau_log_data = np.concatenate([tau_s[mask_s], tau_m[mask_m], tau_l[mask_l]])
    g2_log_data  = np.concatenate([g2_plot_s[mask_s], g2_plot_m[mask_m], g2_plot_l[mask_l]])
    order        = np.argsort(tau_log_data)
    tau_log_data = tau_log_data[order]
    g2_log_data  = g2_log_data[order]

    ax_log.plot(tau_log_data, g2_log_data,
                color=[0.55, 0.55, 0.55], lw=0.8, label='g\u00b2(\u03c4)', zorder=2)

    tf_log     = np.logspace(np.log10(LOG_XLIM_LOW_NS), np.log10(LONG_TIME_NS), 8000)
    g2_fit_log = _model3(tf_log, *popt).copy()
    g2_fit_log[(tf_log >= AFTERFLASH_LOW) & (tf_log <= AFTERFLASH_HIGH)] = np.nan
    ax_log.plot(tf_log, g2_fit_log, 'k', lw=1.8, zorder=3,
                label=f'Fit  g\u00b2(0) = {g2_0:.3f},  g\u00b2(\u221e) = {g2_inf:.3f}')

    for T_val, T_lbl, is_pinned in [
            (T1_disp, 'T\u2081', bound_warnings.get('T1_pinned')),
            (T2_disp, 'T\u2082', bound_warnings.get('T2_pinned')),
            (T3,      'T\u2083', bound_warnings.get('T3_pinned'))]:
        if LOG_XLIM_LOW_NS < T_val < LONG_TIME_NS:
            color  = 'crimson' if is_pinned else 'steelblue'
            suffix = ' (pinned!)' if is_pinned else ''
            ax_log.axvline(T_val, ls=':', color=color, lw=1.0, alpha=0.7)
            ax_log.text(T_val * 1.15, ylim_top * 0.6,
                        f'{T_lbl} = {T_val:.1f} ns{suffix}',
                        fontsize=8, color=color, rotation=90, va='top')

    reliability_line = '' if result_reliable else '\n\u26a0 provisional \u2014 see console'
    info = (f'T\u2081 = {T1_disp:.3f} ns\n'
            f'T\u2082 = {T2_disp:.1f} ns\n'
            f'T\u2083 = {T3:.1f} ns\n'
            f'g\u00b2(0)  = {g2_0:.4f}\n'
            f'g\u00b2(\u221e)  = {g2_inf:.4f}\n'
            f'\u2713 Single emitter'
            f'{reliability_line}')
    ax_log.text(0.02, 0.97, info, transform=ax_log.transAxes,
                ha='left', va='top', fontsize=9,
                fontfamily='monospace',
                bbox=dict(facecolor='white', edgecolor='#ccc', boxstyle='round,pad=0.4'))

    ax_log.set_xscale('log')
    ax_log.set_xlim(LOG_XLIM_LOW_NS, LONG_TIME_NS)
    ax_log.set_ylim(0, ylim_top)
    ax_log.set_xlabel('\u03c4 (ns)  [log scale]', fontsize=14)
    ax_log.set_ylabel('g\u00b2(\u03c4)',               fontsize=14)
    ax_log.set_title(f'Log-linear {LOG_XLIM_LOW_NS:.1e}\u2013{LONG_TIME_NS/1e6:.1f} ms', fontsize=11)
    ax_log.legend(fontsize=9, loc='upper right')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return dict(T1_ns=T1_disp, T2_ns=T2_disp, T3_ns=T3, g2_0=g2_0, g2_inf=g2_inf,
                result_reliable=result_reliable)


# =============================================================================
# Blinking on/off classification
# =============================================================================
def _double_gaussian(x, w1, mu1, sig1, mu2, sig2):
    w2 = 1 - w1
    g1 = w1 * np.exp(-0.5 * ((x - mu1) / sig1) ** 2) / (sig1 * np.sqrt(2 * np.pi))
    g2 = w2 * np.exp(-0.5 * ((x - mu2) / sig2) ** 2) / (sig2 * np.sqrt(2 * np.pi))
    return g1 + g2


def _find_crossing(mu1, sig1, w1, mu2, sig2, w2, n=2000):
    x = np.linspace(min(mu1, mu2), max(mu1, mu2), n)
    f1 = w1 * np.exp(-0.5 * ((x - mu1) / sig1) ** 2) / sig1
    f2 = w2 * np.exp(-0.5 * ((x - mu2) / sig2) ** 2) / sig2
    diff = f1 - f2
    sc = np.where(np.diff(np.sign(diff)) != 0)[0]
    return float(x[sc[0]]) if len(sc) else (mu1 + mu2) / 2.0


def _bin_counts(ch0, ch1, bin_ps, t_end_ps):
    edges = np.arange(0, t_end_ps + bin_ps, bin_ps, dtype=np.int64)
    c0, _ = np.histogram(ch0, bins=edges)
    c1, _ = np.histogram(ch1, bins=edges)
    return (c0 + c1).astype(float)


def _classify_blinking(counts):
    hist_vals, hist_edges = np.histogram(counts, bins=60, density=True)
    hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2.0
    lo_g, hi_g = np.percentile(counts, 10), np.percentile(counts, 90)
    p0 = [0.5, lo_g, max(counts.std() * 0.5, 0.5), hi_g, max(counts.std() * 0.5, 0.5)]
    try:
        popt, _ = curve_fit(
            _double_gaussian, hist_centers, hist_vals, p0=p0,
            bounds=([0.01, counts.min(), 0.1, counts.min(), 0.1],
                    [0.99, counts.max(), counts.max(), counts.max(), counts.max()]),
            maxfev=10000)
        w1, mu1, sig1, mu2, sig2 = popt
        if mu1 > mu2:
            w1, mu1, sig1, mu2, sig2 = (1 - w1), mu2, sig2, mu1, sig1
        w2 = 1 - w1
        sep = (mu2 - mu1) / np.sqrt(sig1 ** 2 + sig2 ** 2)
        thr = _find_crossing(mu1, sig1, w1, mu2, sig2, w2)
        return thr, sep, (w1, mu1, sig1, mu2, sig2)
    except Exception:
        return None, 0.0, None


def blinking_scan_at_T3(ch0, ch1, T3_ns, out_dir, label):
    T_end_ps = int(max(ch0[-1], ch1[-1]))
    scales = [('0.1x T3', 0.1 * T3_ns), ('T3', T3_ns), ('10x T3', 10.0 * T3_ns)]

    results = []
    for name, t3_scaled_ns in scales:
        bin_ps = int(round(t3_scaled_ns * 1000))
        if bin_ps <= 0:
            results.append((name, t3_scaled_ns, None))
            continue
        n_bins_est = T_end_ps // bin_ps + 1
        if n_bins_est > MAX_BLINKING_BINS:
            print(f"    [{label}] skipping {name} ({t3_scaled_ns:.3g} ns) — "
                  f"would need {n_bins_est:,} bins, over the safety cap")
            results.append((name, t3_scaled_ns, None))
            continue
        counts = _bin_counts(ch0, ch1, bin_ps, T_end_ps)
        thr, sep, popt = _classify_blinking(counts)
        results.append((name, t3_scaled_ns, (thr, sep, popt, counts)))

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2))
    if n == 1:
        axes = [axes]

    summary_lines = [f"Blinking scan at T3 for {label}  (T3 = {T3_ns:.3g} ns)\n"]
    for ax, (name, t3_scaled_ns, data) in zip(axes, results):
        if data is None:
            ax.set_title(f'{name}\n({t3_scaled_ns:.3g} ns)\nskipped/failed', fontsize=9)
            summary_lines.append(f"  {name} ({t3_scaled_ns:.3g} ns): skipped or fit failed")
            continue
        thr, sep, popt, counts = data
        ax.hist(counts, bins=60, density=True, color='gray', alpha=0.6)
        if popt is not None:
            xf = np.linspace(counts.min(), counts.max(), 400)
            ax.plot(xf, _double_gaussian(xf, *popt), 'k-', lw=1.5)
            ax.axvline(thr, ls='--', color='crimson', lw=1.2)
            verdict = 'bimodal (blinking)' if sep > 2 else 'unimodal / washed out'
            ax.set_title(f'{name} ({t3_scaled_ns:.3g} ns)\nseparation = {sep:.2f}\u03c3', fontsize=9)
            summary_lines.append(f"  {name} ({t3_scaled_ns:.3g} ns): separation={sep:.2f} sigma, "
                                  f"threshold={thr:.2f}, {verdict}")
        else:
            ax.set_title(f'{name} ({t3_scaled_ns:.3g} ns)\nfit failed', fontsize=9)
            summary_lines.append(f"  {name} ({t3_scaled_ns:.3g} ns): two-Gaussian fit failed")
        ax.set_xlabel('Counts / bin', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)

    fig.suptitle(f'Blinking scan at T3 \u2014 {label}', fontsize=12)
    fig.tight_layout()
    png_path = os.path.join(out_dir, 'blinking_at_T3.png')
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    txt_path = os.path.join(out_dir, 'blinking_at_T3_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines) + '\n')


# =============================================================================
# Data extraction
# =============================================================================
_RUN_RE   = re.compile(r'(?P<date>\d{8})-PLSPC-HT-Ch(?P<chip>\w+)-f(?P<field>\d+)-')
_COORD_RE = re.compile(r'_x(?P<x>-?[\d.]+)_y(?P<y>-?[\d.]+)')


def _parse_run_meta(run_name):
    m = _RUN_RE.search(run_name)
    if not m:
        return {'chip': '?', 'field': '?', 'date': run_name[:8]}
    return {'chip': m.group('chip'), 'field': m.group('field'), 'date': m.group('date')}


def iter_emitters_T3(data_dir, out_dir, from_date=None, to_date=None, verbose=False):
    """from_date / to_date: 'YYYYMMDD' strings (inclusive). Run folders whose
    leading date falls outside this range are skipped entirely."""
    run_pattern = re.compile(r'.*HT.*fullauto.*', re.IGNORECASE)
    n_fwhm_rejected = 0
    n_t3_unresolved = 0
    n_fit_failed    = 0
    n_spe_folders   = 0
    n_skipped_date  = 0

    for run_name in sorted(os.listdir(data_dir)):
        if not run_pattern.match(run_name):
            continue
        run_path = os.path.join(data_dir, run_name)
        if not os.path.isdir(run_path):
            continue

        date_m   = re.match(r'^(\d{8})', run_name)
        run_date = date_m.group(1) if date_m else None
        if run_date:
            if from_date and run_date < from_date:
                n_skipped_date += 1
                continue
            if to_date and run_date > to_date:
                n_skipped_date += 1
                continue

        subfolders = set(os.listdir(run_path))
        meta = _parse_run_meta(run_name)

        for subdir in sorted(subfolders):
            if not subdir.startswith('g2_'):
                continue
            coord_str = subdir[2:]
            lf_dir = 'long' + coord_str
            if lf_dir not in subfolders:
                continue

            g2_path = os.path.join(run_path, subdir)
            raw_files = sorted(f for f in os.listdir(g2_path)
                                if f.endswith('.npz') and '_processed' not in f)
            if not raw_files:
                continue

            label = f'{run_name}/{subdir}'
            try:
                npz = np.load(os.path.join(g2_path, raw_files[-1]), allow_pickle=True)
                ch0 = npz['ch0'].astype(np.int64)
                ch1 = npz['ch1'].astype(np.int64)
                fit = _compute_multiscale_g2(ch0, ch1)
                if fit is None or fit['popt'] is None:
                    n_fit_failed += 1
                    continue
            except Exception as e:
                if verbose:
                    print(f"Skipped {label}: {e}")
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
                n_fwhm_rejected += 1
                continue

            g2_0 = fit['g2_0']
            bound_warnings = fit['bound_warnings']
            wing_reliable = fit['wing_reliable']
            a, b, T1, T2, c, T3 = fit['popt']
            is_spe = (g2_0 < SPE_THRESHOLD) and wing_reliable

            t3_out = np.nan
            if is_spe and not bound_warnings['T3_pinned']:
                t3_out = float(T3)
            elif is_spe:
                n_t3_unresolved += 1

            # ── Only confirmed SPEs get a folder: dual plot + (if T3
            #     resolved) the blinking-at-T3 scan ──────────────────────
            if is_spe:
                emitter_out_dir = os.path.join(out_dir, run_name, subdir)
                os.makedirs(emitter_out_dir, exist_ok=True)

                print(f"  [{label}] confirmed SPE (g2(0)={g2_0:.3f}) — saving dual plot...")
                dual_path = os.path.join(emitter_out_dir, 'dual_g2_plot.png')
                fit_summary = plot_dual_g2(label, fit, dual_path)

                with open(os.path.join(emitter_out_dir, 'g2_fit_summary.txt'), 'w',
                          encoding='utf-8') as f:
                    f.write(f"{label}\n")
                    f.write(f"g2(0)  = {fit_summary['g2_0']:.4f}\n")
                    f.write(f"g2(inf)= {fit_summary['g2_inf']:.4f}\n")
                    f.write(f"T1     = {fit_summary['T1_ns']:.3f} ns\n")
                    f.write(f"T2     = {fit_summary['T2_ns']:.1f} ns\n")
                    f.write(f"T3     = {fit_summary['T3_ns']:.1f} ns"
                            f"{'  (PINNED / unresolved)' if bound_warnings['T3_pinned'] else ''}\n")
                    f.write(f"reliable = {fit_summary['result_reliable']}\n")

                if np.isfinite(t3_out):
                    print(f"  [{label}] running blinking scan at T3={t3_out:.3g} ns...")
                    blinking_scan_at_T3(ch0, ch1, t3_out, emitter_out_dir, label)

                n_spe_folders += 1

            cm = _COORD_RE.search(coord_str)
            x  = float(cm.group('x')) if cm else None
            y  = float(cm.group('y')) if cm else None

            yield {
                'run':      run_name,
                **meta,
                'x':        x,
                'y':        y,
                'ZPL_nm':   zpl,
                'FWHM_nm':  fwhm,
                'g2_0':     g2_0,
                'T3_ns':    t3_out,
            }

    print(f'\nExclusions / notes:')
    if n_skipped_date:
        print(f'  {n_skipped_date} run folder(s) skipped — outside the date range')
    if n_fit_failed:
        print(f'  {n_fit_failed} emitter(s) — multiscale g2 fit failed to converge')
    if n_fwhm_rejected:
        print(f'  {n_fwhm_rejected} emitter(s) — ZPL fit FWHM > {MAX_FWHM_NM} nm (or no clear ZPL)')
    if n_t3_unresolved:
        print(f'  {n_t3_unresolved} confirmed SPE(s) got a dual plot, but no blinking scan — '
              f'T3 pinned at fit bound (unresolved within the 10 ms window)')
    print(f'  {n_spe_folders} confirmed SPE folder(s) created under {out_dir}/<run>/<g2 folder>/')


# =============================================================================
# Plotting — T3 vs ZPL, T3 vs FWHM (batch-level correlation, unchanged)
# =============================================================================
def _chip_palette(chips):
    unique = sorted(set(chips))
    colors = plt.cm.tab10.colors
    return {ch: colors[i % len(colors)] for i, ch in enumerate(unique)}


def make_T3_plots(df, out_dir):
    chips = df['chip'].fillna('?').astype(str)
    cmap  = _chip_palette(chips.unique())
    colors = chips.map(cmap)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('T\u2083 (slow metastable timescale) \u2014 confirmed SPEs only', fontsize=13)

    for ax, xcol, xlabel in [(axes[0], 'ZPL_nm', 'ZPL (nm)'),
                              (axes[1], 'FWHM_nm', 'ZPL FWHM (nm)')]:
        sub = df[[xcol, 'T3_ns']].copy()
        valid = sub.notna().all(axis=1)
        ax.scatter(sub.loc[valid, xcol], sub.loc[valid, 'T3_ns'],
                   c=colors[valid], s=30, edgecolors='k', linewidths=0.4)
        ax.set_yscale('log')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('T\u2083 (ns)  [confirmed SPE, resolved only]', fontsize=11)
        ax.grid(True, alpha=0.3)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=c, markeredgecolor='k', markersize=9, label=f'Ch{ch}')
        for ch, c in cmap.items()
    ]
    fig.legend(handles=handles, title='Chip', loc='lower center',
               ncol=len(cmap), fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    out_path = os.path.join(out_dir, 'T3_correlations.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    ap = argparse.ArgumentParser(description='Multiscale T3 analysis for confirmed SPEs.')
    ap.add_argument('--data-dir', default=DATA_DIR, help='Root data folder to scan')
    ap.add_argument('--out-dir',  default=OUT_DIR,  help='Output folder for CSV and plots')
    ap.add_argument('--from-date', default='20260615', metavar='YYYYMMDD',
                     help='Only process runs on or after this date (default: 20260615)')
    ap.add_argument('--to-date',   default='20260727', metavar='YYYYMMDD',
                     help='Only process runs on or before this date (default: 20260727)')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Scanning data folders from {args.from_date} to {args.to_date} '
          '(multiscale g2 fit + dual plot + blinking-at-T3 scan for every '
          'confirmed SPE — this is the slowest pipeline in the set)...')
    rows = list(iter_emitters_T3(args.data_dir, args.out_dir,
                                  from_date=args.from_date, to_date=args.to_date,
                                  verbose=args.verbose))

    if not rows:
        print('No emitters found with both g2 and long_filter spectrum data '
              'in this date range.')
        return

    df = pd.DataFrame(rows)
    n_spe = int((df['g2_0'] < SPE_THRESHOLD).sum())
    n_t3  = int(df['T3_ns'].notna().sum())
    print(f'\nFound {len(df)} emitters across {df["run"].nunique()} runs '
          f'({n_spe} confirmed SPEs, {n_t3} with a resolved T3).\n')

    print(df[['chip', 'field', 'x', 'y', 'ZPL_nm', 'FWHM_nm', 'g2_0', 'T3_ns']]
          .to_string(index=False, float_format=lambda v: f'{v:.3f}' if pd.notna(v) else 'None'))

    csv_path = os.path.join(args.out_dir, 'T3_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}')

    make_T3_plots(df, args.out_dir)


if __name__ == '__main__':
    main()