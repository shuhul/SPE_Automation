import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import re
import argparse

# Ensure Unicode print output works on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


SHORT_TIME_NS        = 50.0
SHORT_BIN_NS         = 0.250

MID_TIME_NS          = 100_000.0        # 100 µs
MID_BIN_NS           = 200.0            # 0.2 µs

LONG_TIME_NS         = 10_000_000.0     # 10 ms
LONG_BIN_NS          = 6000.0           # 6 µs

AFTERFLASH_LOW       = 10.0
AFTERFLASH_HIGH      = 35.0

WING_FRAC_LOW_LONG   = 0.90
WING_FRAC_HIGH_LONG  = 0.98

LINEAR_XLIM_NS       = SHORT_TIME_NS
LOG_XLIM_LOW_NS      = 1e-1

SPE_THRESHOLD        = 0.5
G0_FIXED             = 1.0

BOUND_PIN_TOL        = 0.02
WING_ZERO_FRAC_WARN  = 0.5
WING_RATIO_WARN      = 0.25

_RUN_RE = re.compile(r'.*HT.*fullauto.*', re.IGNORECASE)


# =============================================================================
# Cross-correlation histogram (all-pairs, vectorised, right-closed bins)
# =============================================================================
def cross_correlation_hist(ch0, ch1, g2time_ps, timebin_ps, chunk=200_000):
    I      = int(np.ceil(g2time_ps / timebin_ps))
    n_bins = 2 * I + 1
    hist   = np.zeros(n_bins, dtype=np.int64)
    edges  = (np.arange(n_bins + 1, dtype=np.int64) - I) * timebin_ps

    ch0 = np.sort(ch0.astype(np.int64))
    ch1 = np.sort(ch1.astype(np.int64))

    # Adaptive chunk to keep peak memory ~160 MB regardless of window size
    if len(ch0) > 0 and len(ch1) > 0:
        t_span         = max(int(ch1[-1]) - int(ch1[0]), 1)
        density_ch1    = len(ch1) / t_span
        avg_pairs      = max(1, int(2 * g2time_ps * density_ch1) + 1)
        max_pairs      = 20_000_000
        chunk = max(1, min(chunk, max_pairs // avg_pairs))

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
        hist += np.bincount(bins[valid].astype(np.intp), minlength=n_bins)

    return hist


def build_regime(ch0, ch1, time_ns, bin_ns):
    g2time_ps  = int(round(time_ns * 1000))
    timebin_ps = int(round(bin_ns  * 1000))
    I          = int(np.ceil(g2time_ps / timebin_ps))
    n_bins     = 2 * I + 1
    tau_ns     = (np.arange(n_bins) - I) * timebin_ps / 1000.0
    hist       = cross_correlation_hist(ch0, ch1, g2time_ps, timebin_ps)
    return tau_ns, hist, timebin_ps


# =============================================================================
# Model — three exponentials
# =============================================================================
def model3(x, a, b, T1, T2, c, T3):
    fast = (1 + a) * np.exp(-np.abs(x) / T1) - a * np.exp(-np.abs(x) / T2)
    slow = 1 - np.exp(-np.abs(x) / T3)
    return G0_FIXED - b * fast - c * slow


def model3_dip_only(x, a, b, T1, T2):
    fast = (1 + a) * np.exp(-np.abs(x) / T1) - a * np.exp(-np.abs(x) / T2)
    return G0_FIXED - b * fast


def fit_g2(tau_short, g2_short, tau_mid, g2_mid, tau_long, g2_long):
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
    T1_bounds = (0.01, 100.0)
    T2_bounds = (1.0, 1e5)
    best1, best1_res = None, np.inf
    for a0, b0, T1_0, T2_0 in stage1_seeds:
        try:
            popt, _ = curve_fit(
                model3_dip_only, tau_short, g2_short,
                p0=[a0, b0, T1_0, T2_0],
                bounds=([0, 0, T1_bounds[0], T2_bounds[0]],
                        [10.0, ceiling, T1_bounds[1], T2_bounds[1]]),
                maxfev=5000
            )
            res = float(np.sum((model3_dip_only(tau_short, *popt) - g2_short) ** 2))
            if res < best1_res:
                best1_res, best1 = res, popt
        except Exception:
            continue

    if best1 is None:
        return None, None

    # Stage 1 locks T1 (antibunching width) and b (dip depth).
    # Stage 2 only re-fits a, T2, c, T3 on all three regimes.
    # This prevents the large number of LONG-regime points from pulling b upward
    # and causing the fit to lose the antibunching dip.
    T1_fixed = float(best1[2])
    b_fixed  = float(best1[1])
    a0_seed  = float(best1[0])
    T2_seed  = float(best1[3])

    def _model_T1b_fixed(x, a, T2, c, T3):
        fast = (1 + a) * np.exp(-np.abs(x) / T1_fixed) - a * np.exp(-np.abs(x) / T2)
        slow = 1 - np.exp(-np.abs(x) / T3)
        return G0_FIXED - b_fixed * fast - c * slow

    tau_all = np.concatenate([tau_short, tau_mid, tau_long])
    g2_all  = np.concatenate([g2_short,  g2_mid,  g2_long])
    finite_mask = np.isfinite(tau_all) & np.isfinite(g2_all)
    tau_all, g2_all = tau_all[finite_mask], g2_all[finite_mask]

    long_finite = g2_long[np.isfinite(g2_long)]
    c0_guess = max(0.02, G0_FIXED - float(np.nanmin(long_finite))) if long_finite.size else 0.1

    T3_bounds = (1.0, 1e8)
    stage2_seeds = [
        (a0_seed, T2_seed,  c0_guess,       1e5),
        (a0_seed, T2_seed,  c0_guess,       1e6),
        (a0_seed, T2_seed,  c0_guess * 0.5, 5e5),
        (a0_seed, 1e4,      0.05,           2e6),
    ]

    best2, best2_res = None, np.inf
    for a1, T2_1, c1, T3_1 in stage2_seeds:
        try:
            popt, _ = curve_fit(
                _model_T1b_fixed, tau_all, g2_all,
                p0=[a1, T2_1, c1, T3_1],
                bounds=([0,    T2_bounds[0], 0,   T3_bounds[0]],
                        [10.0, T2_bounds[1], 1.0, T3_bounds[1]]),
                maxfev=8000
            )
            res = float(np.sum((_model_T1b_fixed(tau_all, *popt) - g2_all) ** 2))
            if res < best2_res:
                best2_res, best2 = res, popt
        except Exception:
            continue

    if best2 is None:
        return None, None

    a2, T2_2, c2, T3_2 = best2
    full_popt = [a2, b_fixed, T1_fixed, T2_2, c2, T3_2]

    def pinned(val, lo, hi, tol=BOUND_PIN_TOL):
        near_lo = lo > 0 and (val - lo) < tol * lo
        near_hi = hi > 0 and (hi - val) < tol * hi
        return near_lo or near_hi

    warnings = {}
    warnings['T1_pinned'] = pinned(T1_fixed, *T1_bounds)
    warnings['T2_pinned'] = pinned(T2_2,    *T2_bounds)
    warnings['T3_pinned'] = pinned(T3_2,    *T3_bounds)

    return full_popt, warnings


# =============================================================================
# Main analysis for a single file
# =============================================================================
def run(data_path, out_path):
    npz    = np.load(data_path)
    ch0    = npz['ch0'].astype(np.int64)
    ch1    = npz['ch1'].astype(np.int64)
    N1, N2 = len(ch0), len(ch1)
    TT     = int(max(ch0[-1], ch1[-1]))
    T_acq  = TT / 1e12
    print(f"N1={N1:,}  N2={N2:,}  T={T_acq:.2f}s")
    print(f"Ch0: {N1/T_acq/1000:.2f} kHz   Ch1: {N2/T_acq/1000:.2f} kHz")

    print(f"\nBuilding LONG regime ({LONG_BIN_NS/1000:.1f} µs bins, "
          f"±{LONG_TIME_NS/1e6:.1f} ms window)...")
    tau_l, hist_l, timebin_ps_l = build_regime(ch0, ch1, LONG_TIME_NS, LONG_BIN_NS)
    print("Done.")

    n_bins_l    = len(tau_l)
    wing_low_l  = LONG_TIME_NS * WING_FRAC_LOW_LONG
    wing_high_l = LONG_TIME_NS * WING_FRAC_HIGH_LONG
    wing_mask_l = (np.abs(tau_l) >= wing_low_l) & (np.abs(tau_l) <= wing_high_l)
    wing_vals_l = hist_l[wing_mask_l]

    wing_reliable = True
    if wing_vals_l.size > 0:
        c_wing_l  = float(np.mean(wing_vals_l))
        zero_frac = float(np.mean(wing_vals_l == 0))
        print(f"Far-field wing (τ ∈ [{wing_low_l/1e6:.2f}, {wing_high_l/1e6:.2f}] ms): "
              f"{c_wing_l:.4f} counts/bin  (n_bins={wing_vals_l.size}, zero_frac={zero_frac:.2f})")
        if zero_frac > WING_ZERO_FRAC_WARN:
            wing_reliable = False
            print(f"  WARNING: {zero_frac:.0%} of wing bins are empty — c_wing is a noisy "
                  f"estimate. Consider longer LONG_TIME_NS or coarser LONG_BIN_NS.")
    else:
        c_wing_l = (N1 / T_acq) * (N2 / T_acq) * (timebin_ps_l * 1e-12) * T_acq
        wing_reliable = False
        print(f"Long wing empty — analytic fallback: {c_wing_l:.4f} counts/bin")

    R1, R2 = N1 / T_acq, N2 / T_acq
    c_wing_l_analytic = R1 * R2 * (timebin_ps_l * 1e-12) * T_acq
    wing_ratio = c_wing_l / c_wing_l_analytic if c_wing_l_analytic > 0 else np.nan
    print(f"Analytic accidental level: {c_wing_l_analytic:.4f} counts/bin "
          f"(ratio empirical/analytic = {wing_ratio:.3f})")
    if abs(wing_ratio - 1.0) > WING_RATIO_WARN:
        wing_reliable = False
        print(f"  WARNING: wing not flat — normalization may be biased. "
              f"Extend LONG_TIME_NS until ratio settles near 1.0.")

    g2_l = hist_l.astype(float) / c_wing_l
    af_mask_l    = (np.abs(tau_l) >= AFTERFLASH_LOW) & (np.abs(tau_l) <= AFTERFLASH_HIGH)
    edge_mask_l  = (np.arange(n_bins_l) > 0) & (np.arange(n_bins_l) < n_bins_l - 1)
    g2_plot_l               = g2_l.copy()
    g2_plot_l[af_mask_l]    = np.nan
    g2_plot_l[~edge_mask_l] = np.nan

    timebin_ps_m = int(round(MID_BIN_NS   * 1000))
    timebin_ps_s = int(round(SHORT_BIN_NS * 1000))
    c_wing_m = c_wing_l * (timebin_ps_m / timebin_ps_l)
    c_wing_s = c_wing_l * (timebin_ps_s / timebin_ps_l)

    print(f"\nBuilding INTERMEDIATE regime ({MID_BIN_NS/1000:.2f} µs bins, "
          f"±{MID_TIME_NS/1000:.0f} µs window)...")
    tau_m, hist_m, _ = build_regime(ch0, ch1, MID_TIME_NS, MID_BIN_NS)
    print("Done.")

    n_bins_m = len(tau_m)
    g2_m = hist_m.astype(float) / c_wing_m
    af_mask_m   = (np.abs(tau_m) >= AFTERFLASH_LOW) & (np.abs(tau_m) <= AFTERFLASH_HIGH)
    edge_mask_m = (np.arange(n_bins_m) > 0) & (np.arange(n_bins_m) < n_bins_m - 1)
    g2_plot_m               = g2_m.copy()
    g2_plot_m[af_mask_m]    = np.nan
    g2_plot_m[~edge_mask_m] = np.nan

    print(f"\nBuilding SHORT regime ({SHORT_BIN_NS:.3f} ns bins, "
          f"±{SHORT_TIME_NS:.0f} ns window)...")
    tau_s, hist_s, _ = build_regime(ch0, ch1, SHORT_TIME_NS, SHORT_BIN_NS)
    print("Done.")

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

    popt, bound_warnings = fit_g2(tau_s[fit_mask_s], g2_s[fit_mask_s],
                                   tau_m[fit_mask_m], g2_m[fit_mask_m],
                                   tau_l[fit_mask_l], g2_l[fit_mask_l])

    g2_0 = g2_inf = None
    result_reliable = wing_reliable
    if popt is not None:
        a, b, T1, T2, c, T3 = popt
        T1_disp = min(T1, T2)
        T2_disp = max(T1, T2)
        g2_0   = float(G0_FIXED - b)
        g2_inf = float(G0_FIXED - c)
        print(f"\nT1 (antibunching)    = {T1_disp:.3f} ns"
              + ("   *** PINNED AT BOUND ***" if bound_warnings.get('T1_pinned') else ""))
        print(f"T2 (fast metastable) = {T2_disp:.1f} ns"
              + ("   *** PINNED AT BOUND — unresolved ***"
                 if bound_warnings.get('T2_pinned') else ""))
        print(f"T3 (slow metastable) = {T3:.1f} ns"
              + ("   *** PINNED AT BOUND — unresolved ***"
                 if bound_warnings.get('T3_pinned') else ""))
        print(f"g0 (fixed)           = {G0_FIXED:.4f}")
        print(f"g²(0)                = {g2_0:.4f}")
        print(f"g²(∞) [true baseline]= {g2_inf:.4f}  (should be ≈ 1)")

        if bound_warnings.get('T1_pinned') or bound_warnings.get('T2_pinned') or not wing_reliable:
            result_reliable = False
            print("\n  *** RELIABILITY WARNING (g²(0)) ***")
            print("  T1/T2 window-limited and/or wing normalization unsettled.")

        if bound_warnings.get('T3_pinned'):
            print("\n  NOTE: T3 window-limited — very-long dynamics not trustworthy.")

        verdict    = 'SINGLE EMITTER' if g2_0 < SPE_THRESHOLD else 'NOT single emitter'
        confidence = '' if result_reliable else '  (LOW CONFIDENCE — see warning above)'
        print(f"g²(0) < {SPE_THRESHOLD} ?  →  {verdict}{confidence}")
    else:
        print("Fit did not converge.")

    # =========================================================================
    # Plot
    # =========================================================================
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle(f'g²(τ) — {os.path.basename(data_path)}', fontsize=13)

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
        ax.axhline(G0_FIXED,      ls='--', color='#555',     lw=0.9, label=f'g₀ = {G0_FIXED:.1f} ')
        ax.axhline(SPE_THRESHOLD, ls='-.', color='#e74c3c',  lw=1.1, label=f'g²=0.{int(SPE_THRESHOLD*10)} ')

    draw_common(ax_lin)
    lin_mask = (tau_s >= -LINEAR_XLIM_NS) & (tau_s <= LINEAR_XLIM_NS)
    ax_lin.plot(tau_s[lin_mask], g2_plot_s[lin_mask],
                color=[0.55, 0.55, 0.55], lw=1.1, label='g²(τ)', zorder=2)

    if popt is not None:
        tf_lin     = np.linspace(-LINEAR_XLIM_NS, LINEAR_XLIM_NS, 3000)
        g2_fit_lin = model3(tf_lin, *popt).copy()
        g2_fit_lin[(np.abs(tf_lin) >= AFTERFLASH_LOW) &
                   (np.abs(tf_lin) <= AFTERFLASH_HIGH)] = np.nan
        ax_lin.plot(tf_lin, g2_fit_lin, 'k', lw=1.8, zorder=3,
                    label=f'Fit  g²(0) = {g2_0:.3f}')

    for sign in [+1, -1]:
        ax_lin.text(sign * (AFTERFLASH_LOW + AFTERFLASH_HIGH) / 2,
                    0.04, 'afterflash\nremoved',
                    ha='center', va='bottom', fontsize=7.5,
                    color='#7b3fa0', style='italic',
                    transform=ax_lin.get_xaxis_transform())

    ax_lin.set_ylim(0, ylim_top)
    ax_lin.set_xlim(-LINEAR_XLIM_NS, LINEAR_XLIM_NS)
    ax_lin.set_xlabel('τ (ns)', fontsize=14)
    ax_lin.set_ylabel('g²(τ)',  fontsize=14)
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
                color=[0.55, 0.55, 0.55], lw=0.8, label='g²(τ)', zorder=2)

    if popt is not None:
        tf_log     = np.logspace(np.log10(LOG_XLIM_LOW_NS), np.log10(LONG_TIME_NS), 8000)
        g2_fit_log = model3(tf_log, *popt).copy()
        g2_fit_log[(tf_log >= AFTERFLASH_LOW) & (tf_log <= AFTERFLASH_HIGH)] = np.nan
        ax_log.plot(tf_log, g2_fit_log, 'k', lw=1.8, zorder=3,
                    label=f'Fit  g²(0) = {g2_0:.3f},  g²(∞) = {g2_inf:.3f}')

        for T_val, T_lbl, is_pinned in [
                (T1_disp, 'T₁', bound_warnings.get('T1_pinned')),
                (T2_disp, 'T₂', bound_warnings.get('T2_pinned')),
                (T3,      'T₃', bound_warnings.get('T3_pinned'))]:
            if LOG_XLIM_LOW_NS < T_val < LONG_TIME_NS:
                color  = 'crimson' if is_pinned else 'steelblue'
                suffix = ' (pinned!)' if is_pinned else ''
                ax_log.axvline(T_val, ls=':', color=color, lw=1.0, alpha=0.7)
                ax_log.text(T_val * 1.15, ylim_top * 0.6,
                            f'{T_lbl} = {T_val:.1f} ns{suffix}',
                            fontsize=8, color=color, rotation=90, va='top')

        reliability_line = '' if result_reliable else '\n⚠ provisional — see console'
        info = (f'T₁ = {T1_disp:.3f} ns\n'
                f'T₂ = {T2_disp:.1f} ns\n'
                f'T₃ = {T3:.1f} ns\n'
                f'g²(0)  = {g2_0:.4f}\n'
                f'g²(∞)  = {g2_inf:.4f}\n'
                f'{"✓ Single emitter" if g2_0 < SPE_THRESHOLD else "✗ Not single emitter"}'
                f'{reliability_line}')
        ax_log.text(0.02, 0.97, info, transform=ax_log.transAxes,
                    ha='left', va='top', fontsize=9,
                    fontfamily='monospace',
                    bbox=dict(facecolor='white', edgecolor='#ccc', boxstyle='round,pad=0.4'))

    ax_log.set_xscale('log')
    ax_log.set_xlim(LOG_XLIM_LOW_NS, LONG_TIME_NS)
    ax_log.set_ylim(0, ylim_top)
    ax_log.set_xlabel('τ (ns)  [log scale]', fontsize=14)
    ax_log.set_ylabel('g²(τ)',               fontsize=14)
    ax_log.set_title(f'Log-linear {LOG_XLIM_LOW_NS:.1e}–{LONG_TIME_NS/1e6:.1f} ms', fontsize=11)
    ax_log.legend(fontsize=9, loc='upper right')

    fig.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Save fit parameters so analyse_metastable.py can read them without re-running
    results_path = out_path.replace('_multiscale.png', '_multiscale_results.npz')
    bw = bound_warnings or {}
    np.savez(results_path,
             popt            = np.array(popt, dtype=float) if popt is not None else np.zeros(0, dtype=float),
             fit_converged   = np.array([popt is not None]),
             g2_0            = np.array([g2_0   if g2_0   is not None else np.nan]),
             g2_inf          = np.array([g2_inf  if g2_inf  is not None else np.nan]),
             wing_reliable   = np.array([wing_reliable]),
             result_reliable = np.array([result_reliable]),
             T1_pinned       = np.array([bool(bw.get('T1_pinned', True))]),
             T2_pinned       = np.array([bool(bw.get('T2_pinned', True))]),
             T3_pinned       = np.array([bool(bw.get('T3_pinned', True))]),
             )
    print(f"Saved: {results_path}")

    return dict(tau_s=tau_s, tau_m=tau_m, tau_l=tau_l,
                g2_s=g2_s, g2_m=g2_m, g2_l=g2_l,
                popt=popt, g2_0=g2_0, g2_inf=g2_inf,
                result_reliable=result_reliable)


# =============================================================================
# Batch mode — scan data/ folder for all g2_ subfolders
# =============================================================================
def _find_raw_npz(folder):
    """Return path to raw photon .npz in folder (skips *_processed.npz and
    pre-processed histogram files that lack ch0/ch1 keys)."""
    for f in sorted(os.listdir(folder)):
        if not f.endswith('.npz') or f.endswith('_processed.npz'):
            continue
        path = os.path.join(folder, f)
        try:
            keys = np.load(path, allow_pickle=True).files
            if 'ch0' in keys and 'ch1' in keys:
                return path
        except Exception:
            continue
    return None


def batch_run(data_dir='data', from_date=None, to_date=None):
    """Process every g2_* subfolder found under HT fullauto run folders.

    from_date / to_date: 'YYYYMMDD' strings (inclusive).  Run folders whose
    leading date falls outside this range are skipped.
    """
    jobs = []
    skipped_date = []
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
                skipped_date.append(run_name)
                continue
            if to_date and run_date > to_date:
                skipped_date.append(run_name)
                continue

        for subdir in sorted(os.listdir(run_path)):
            if not subdir.startswith('g2_'):
                continue
            g2_folder = os.path.join(run_path, subdir)
            raw_npz = _find_raw_npz(g2_folder)
            if raw_npz is None:
                continue
            stem     = os.path.splitext(os.path.basename(raw_npz))[0]
            out_path = os.path.join(g2_folder, stem + '_multiscale.png')
            jobs.append((run_name, subdir, raw_npz, out_path))

    if skipped_date:
        print(f'Skipped {len(skipped_date)} run(s) outside date range.')

    if not jobs:
        print(f'No raw photon .npz files found under {data_dir}/')
        return

    print(f'Found {len(jobs)} g2 measurement(s) to process.\n')
    for i, (run_name, subdir, raw_npz, out_path) in enumerate(jobs, 1):
        results_path = out_path.replace('_multiscale.png', '_multiscale_results.npz')
        print(f'\n{"="*60}')
        print(f'[{i}/{len(jobs)}]  {run_name}  /  {subdir}')
        if os.path.exists(results_path):
            print(f'  [skip] results already exist: {os.path.basename(results_path)}')
            continue
        print(f'  input : {os.path.basename(raw_npz)}')
        print(f'  output: {os.path.basename(out_path)}')
        try:
            run(raw_npz, out_path)
        except Exception as e:
            import traceback
            print(f'  ERROR: {e}')
            traceback.print_exc()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Multi-scale g²(τ) analysis.\n'
                    'Default: batch-processes all g2_* folders under data/.\n\n'
                    'Examples:\n'
                    '  python g2_multiscale.py                          # all runs\n'
                    '  python g2_multiscale.py --from-date 20260617     # June 17 onwards\n'
                    '  python g2_multiscale.py --from-date 20260617 --to-date 20260629\n'
                    '  python g2_multiscale.py --file data/.../20260629_150657.npz',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default='data',
                    help='Root data folder to scan (default: data/)')
    ap.add_argument('--from-date', default=None, metavar='YYYYMMDD',
                    help='Only process runs on or after this date')
    ap.add_argument('--to-date', default=None, metavar='YYYYMMDD',
                    help='Only process runs on or before this date')
    ap.add_argument('--file', metavar='NPZ',
                    help='Process a single raw-photon .npz file instead of batch mode')
    ap.add_argument('--out', metavar='PNG',
                    help='Output plot path for --file mode '
                         '(default: <stem>_multiscale.png in same folder)')
    args = ap.parse_args()

    if args.file:
        stem = os.path.splitext(os.path.abspath(args.file))[0]
        out  = args.out or (stem + '_multiscale.png')
        run(args.file, out)
    else:
        batch_run(args.data_dir,
                  from_date=args.from_date,
                  to_date=args.to_date)
