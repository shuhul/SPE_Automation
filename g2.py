"""
g2.py — G2 photon correlation analysis for the automated SPE pipeline.

Algorithm  : all-pairs cross-correlation (matches original MATLAB exactly)
Afterflash : NaN-blanked in plot and excluded from fit — no stochastic
             photon removal (cleaner, reproducible, no seed dependency)
Normalise  : wing mean (far-tau region where channels are uncorrelated)
Fit        : double-exponential with free baseline g0, multi-start grid

Interface used by automate.py:
    result = g2mod.run(npz_path, out_folder=..., g2time_ns=..., timebin_ns=...)
    result['g2_0_norm']   # (g0-b)/g0 — use for single-emitter test
    result['popt']        # None if fit failed
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================================================================
# Module-level constants  (edit these to tune afterflash / wing regions)
# =============================================================================
AFTERFLASH_LOW_NS  = 15.0   # ns  start of afterflash band to blank
AFTERFLASH_HIGH_NS = 35.0   # ns  end   of afterflash band to blank
WING_FRAC_LOW      = 0.90   # wing starts at this fraction of g2time_ns
WING_FRAC_HIGH     = 0.95   # wing ends   at this fraction of g2time_ns


# =============================================================================
# Model
# =============================================================================

def _model(x, a, b, T1, T2, g0):
    """
    Double-exponential antibunching model with free baseline g0.
    g2(0) absolute   = g0 - b
    g2(0) normalised = (g0 - b) / g0   <- use this for single-emitter test
    """
    return g0 - b * ((1 + a) * np.exp(-np.abs(x) / T1)
                         - a  * np.exp(-np.abs(x) / T2))


# =============================================================================
# All-pairs cross-correlation histogram  (vectorised, chunked)
# =============================================================================

def _cross_correlation_hist(ch0, ch1, g2time_ps, timebin_ps, chunk=200_000):
    """
    All-pairs cross-correlation histogram matching the original MATLAB algorithm.

    Binning convention — right-closed (matches MATLAB exactly):
        bin k  iff  edges[k] < dt <= edges[k+1]
    implemented with np.searchsorted(..., side='left') - 1.

    Vectorised with chunked searchsorted — seconds on tens of millions of photons.
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


# =============================================================================
# Fit
# =============================================================================

def _fit(tau, g2):
    """Multi-start grid fit. Returns best popt (a, b, T1, T2, g0) or None."""
    g0_guess  = float(np.mean(g2[np.abs(tau) > 0.6 * np.abs(tau).max()]))
    best_popt, best_res = None, np.inf

    for b0 in [0.3, 0.5, 0.7, 0.9]:
        for T1_0 in [1, 3, 10, 30]:
            for a0 in [0, 1]:
                try:
                    popt, _ = curve_fit(
                        _model, tau, g2,
                        p0=[a0, b0, T1_0, 5000, g0_guess],
                        bounds=([0, 0, 0.1, 10,  0],
                                [np.inf] * 5),
                        maxfev=10_000
                    )
                    res = float(np.sum((_model(tau, *popt) - g2) ** 2))
                    if res < best_res:
                        best_res, best_popt = res, popt
                except Exception:
                    continue

    return best_popt


# =============================================================================
# Core computation
# =============================================================================

def _compute_g2(ch0, ch1, g2time_ns, timebin_ns):
    """
    Compute g2 from raw photon timestamp arrays ch0, ch1 (ps, int64).
    Returns result dict.
    """
    g2time_ps  = int(round(g2time_ns  * 1000))
    timebin_ps = int(round(timebin_ns * 1000))
    I          = int(np.ceil(g2time_ps / timebin_ps))
    n_bins     = 2 * I + 1
    tau_ns     = (np.arange(n_bins) - I) * timebin_ps / 1000.0

    N1 = len(ch0)
    N2 = len(ch1)
    TT = int(max(ch0[-1], ch1[-1]))

    print("  Building histogram...")
    hist = _cross_correlation_hist(ch0, ch1, g2time_ps, timebin_ps)
    print("  Done.")

    # Normalise by far-tau wing mean
    wing_low  = g2time_ns * WING_FRAC_LOW
    wing_high = g2time_ns * WING_FRAC_HIGH
    wing_mask = (np.abs(tau_ns) >= wing_low) & (np.abs(tau_ns) <= wing_high)
    c_wing    = float(hist[wing_mask].mean()) if wing_mask.any() else 1.0
    g2_arr    = hist.astype(float) / c_wing

    # Afterflash band mask
    af_mask = (np.abs(tau_ns) >= AFTERFLASH_LOW_NS) & \
              (np.abs(tau_ns) <= AFTERFLASH_HIGH_NS)

    # Edge bin mask (boundary artifact at +/- g2time_ns)
    edge_mask = (np.arange(n_bins) > 0) & (np.arange(n_bins) < n_bins - 1)

    # Fit on clean data: exclude afterflash band and edge bins
    fit_mask = edge_mask & ~af_mask
    popt = _fit(tau_ns[fit_mask], g2_arr[fit_mask])

    g2_0 = g2_0_norm = None
    if popt is not None:
        a, b, T1, T2, g0 = popt
        g2_0      = float(g0 - b)
        g2_0_norm = float((g0 - b) / g0)

    return dict(
        tau=tau_ns, g2=g2_arr, c=hist,
        N1=N1, N2=N2, TT=TT,
        wing_level=c_wing,
        popt=popt, g2_0=g2_0, g2_0_norm=g2_0_norm
    )


# =============================================================================
# Public API  —  called by automate.py as g2mod.run(...)
# =============================================================================

def run(path, out_folder='g2_data', g2time_ns=100.0, timebin_ns=1.0, seed=0):
    """
    Full pipeline: load raw photon .npz (ch0/ch1 in ps), compute g2,
    save result .npz and .png.

    Args:
        path       : path to raw photon .npz (ch0, ch1 arrays in ps)
        out_folder : folder to save outputs
        g2time_ns  : correlation half-window in ns
        timebin_ns : bin width in ns
        seed       : unused (kept for API compatibility with automate.py)

    Returns result dict. Key values for automate.py:
        g2_0_norm  : (g0-b)/g0 — use for single-emitter test (threshold < 0.5)
        popt       : (a, b, T1, T2, g0) or None if fit failed
    """
    os.makedirs(out_folder, exist_ok=True)
    stem   = os.path.splitext(os.path.basename(path))[0]
    prefix = os.path.join(out_folder, stem)

    print(f"Running g2 on {path}...")
    npz = np.load(path)
    ch0 = npz['ch0'].astype(np.int64)
    ch1 = npz['ch1'].astype(np.int64)

    result = _compute_g2(ch0, ch1, g2time_ns, timebin_ns)

    T_acq = result['TT'] / 1e12
    print(f"  T={T_acq:.2f}s  "
          f"N1={result['N1']:,} ({result['N1']/T_acq/1e3:.1f} kcps)  "
          f"N2={result['N2']:,} ({result['N2']/T_acq/1e3:.1f} kcps)")
    print(f"  Wing level: {result['wing_level']:.1f} counts/bin")

    if result['popt'] is not None:
        a, b, T1, T2, g0 = result['popt']
        print(f"  g2(0) = {result['g2_0_norm']:.3f}  "
              f"T1 = {T1:.2f} ns  baseline = {g0:.3f}")
        print(f"  {'SINGLE EMITTER' if result['g2_0_norm'] < 0.5 else 'Not a single emitter'}"
              f"  [threshold g2(0) < 0.5]")
    else:
        print("  Fit did not converge.")

    # ── Save result npz ───────────────────────────────────────────────────────
    np.savez(
        prefix + '_processed.npz',
        tau      =result['tau'],
        g2       =result['g2'],
        c        =result['c'],
        N1       =result['N1'],
        N2       =result['N2'],
        TT       =result['TT'],
        wing_level=result['wing_level'],
        popt     =(result['popt'] if result['popt'] is not None else np.array([])),
        g2_0     =result['g2_0']      if result['g2_0']      is not None else np.nan,
        g2_0_norm=result['g2_0_norm'] if result['g2_0_norm'] is not None else np.nan,
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    tau_ns  = result['tau']
    g2_arr  = result['g2']
    popt    = result['popt']
    n_bins  = len(tau_ns)

    af_mask   = (np.abs(tau_ns) >= AFTERFLASH_LOW_NS) & \
                (np.abs(tau_ns) <= AFTERFLASH_HIGH_NS)
    edge_mask = (np.arange(n_bins) > 0) & (np.arange(n_bins) < n_bins - 1)

    # NaN out afterflash band and edge bins for display
    g2_plot             = g2_arr.copy()
    g2_plot[af_mask]    = np.nan
    g2_plot[~edge_mask] = np.nan

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Shaded afterflash bands
    for sign in [+1, -1]:
        ax.axvspan(sign * AFTERFLASH_LOW_NS, sign * AFTERFLASH_HIGH_NS,
                   color='#f0e6ff', alpha=0.85, zorder=0,
                   label='Afterflash removed' if sign == 1 else None)

    ax.plot(tau_ns, g2_plot, color=[0.55, 0.55, 0.55], lw=1.1,
            label='g²(τ)', zorder=2)

    if popt is not None:
        a, b, T1, T2, g0 = popt
        tf       = np.linspace(tau_ns.min(), tau_ns.max(), 6000)
        g2_model = _model(tf, *popt).copy()
        g2_model[(np.abs(tf) >= AFTERFLASH_LOW_NS) &
                 (np.abs(tf) <= AFTERFLASH_HIGH_NS)] = np.nan

        ax.plot(tf, g2_model, 'k', lw=1.8, zorder=3,
                label=f'Fit  g²(0) = {result["g2_0_norm"]:.3f}')
        ax.axhline(g0,     ls='--', color='#555', lw=0.9, zorder=1,
                   label=f'Baseline g₀ = {g0:.3f}')
        ax.axhline(g0*0.5, ls='-.', color='#e74c3c', lw=1.1, zorder=1,
                   label=f'½-baseline = {g0*0.5:.3f}')
    else:
        ax.axhline(1.0, ls='--', color='#555',     lw=0.9, label='g²=1')
        ax.axhline(0.5, ls='-.', color='#e74c3c',  lw=1.1, label='g²=0.5')

    ax.axvline(0, ls=':', color='#bbb', lw=0.8, zorder=1)

    ylim_top = max(1.5, float(np.nanmax(g2_plot)) * 1.1)
    ax.set_ylim(0, ylim_top)

    for sign in [+1, -1]:
        ax.text(sign * (AFTERFLASH_LOW_NS + AFTERFLASH_HIGH_NS) / 2,
                ylim_top * 0.93,
                'afterflash\nremoved',
                ha='center', va='top', fontsize=8,
                color='#7b3fa0', style='italic')

    ax.set_xlim(-100, 100)
    ax.set_xlabel('τ (ns)', fontsize=14)
    ax.set_ylabel('g²(τ)',  fontsize=14)
    ax.legend(fontsize=10, loc='lower right')
    fig.tight_layout()
    fig.savefig(prefix + '_plot.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: {prefix}_plot.png")

    return result
        
# # =============================================================================
# # G2 — eff2 start-stop algorithm
# # =============================================================================

# def _clean_channel_afterflashes(times, min_dt_ps=9000, max_dt_ps=35000, seed=0):
#     """
#     Performs afterflash removal on a single channel's absolute timeline.
#     Removes stochastically anomalous successive triggers within the detector dead/trap time.
#     """
#     if times.size <= 1:
#         return times
#     # Calculate time differences between successive photons on the SAME detector
#     dt = times[1:] - times[:-1]
    
#     # Identify indices where an afterflash candidate exists
#     flash_candidates = (dt >= min_dt_ps) & (dt <= max_dt_ps)
    
#     if not flash_candidates.any():
#         return times

#     # Calculate the baseline random coincidence rate for this channel's internal density
#     # to establish an expected random background count
#     avg_dt = np.mean(dt)
#     expected_random_fraction = (max_dt_ps - min_dt_ps) / avg_dt if avg_dt > 0 else 0.0
    
#     # Stochastic filter: if candidates exceed expected random distribution, flag them
#     rng = np.random.default_rng(seed)
#     random_thresholds = rng.uniform(size=flash_candidates.sum())
    
#     # Keep pairs that match expected random statistics, drop anomalous bursts
#     # (If the local density matches random chance, we keep it; if it's an afterflash burst, we drop)
#     drop_mask = np.zeros(times.size, dtype=bool)
#     candidate_indices = np.where(flash_candidates)[0] + 1
    
#     # Prune out the afterflashes
#     to_drop = candidate_indices[random_thresholds > expected_random_fraction]
#     drop_mask[to_drop] = True

#     return times[~drop_mask]


# def _model(x, a, b, T1, T2):
#     return 1 - b * ((1 + a) * np.exp(-np.abs(x) / T1) - a * np.exp(-np.abs(x) / T2))

# # =============================================================================
# # Public API
# # =============================================================================

# def _compute_g2_multihit(ch0, ch1, g2time_ns, timebin_ns, seed):
#     """
#     Computes g2 via full multi-hit cross-correlation.
#     Includes independent-channel afterflash removal and accurate normalization.
#     """
#     g2time_ps  = int(round(g2time_ns * 1000))
#     timebin_ps = int(round(timebin_ns * 1000))
#     I          = int(np.ceil(g2time_ps / timebin_ps))
#     tau_ns     = (np.arange(2 * I + 1) - I) * timebin_ps / 1000.0

#     # 1. Run afterflash removal independently per detector channel before cross-correlating
#     ch0_clean = _clean_channel_afterflashes(ch0, seed=seed)
#     ch1_clean = _clean_channel_afterflashes(ch1, seed=seed + 1)

#     # 2. Compute raw multi-hit histogram directly from the cleaned timelines
#     c = full_cross_correlation_hist(ch0_clean, ch1_clean, g2time_ps, timebin_ps, I)
    
#     N1 = int(ch0_clean.size)
#     N2 = int(ch1_clean.size)
    
#     # 3. Calculate the precise observation timeline window (T_total)
#     if N1 > 0 and N2 > 0:
#         min_time = min(ch0_clean.min(), ch1_clean.min())
#         max_time = max(ch0_clean.max(), ch1_clean.max())
#         TT = int(max_time - min_time)
#     else:
#         TT = 1

#     # 4. Standard analytical normalization denominator for full cross-correlation
#     A = N1 * N2 * timebin_ps / TT
#     g2_arr = c / A if A > 0 else np.zeros_like(c, dtype=float)

#     # 5. Handle fitting
#     try:
#         popt, _ = curve_fit(
#             _model, tau_ns, g2_arr,
#             p0=[1, 0.8, 10, 5000],
#             bounds=([0, -1, 0.1, 10], [np.inf, 1, np.inf, np.inf]),
#             maxfev=20000
#         )
#     except Exception:
#         popt = None

#     # Keeping matching dictionary keys for downstream pipeline compatibility
#     return dict(tau=tau_ns, g2=g2_arr, c=c, c_raw=c,
#                 N1=N1, N2=N2, TT=TT, A=A, popt=popt)


# def full_cross_correlation_hist(ch0_times, ch1_times, g2time_ps, timebin_ps, I):
#     """
#     Calculates a full multi-hit cross-correlation histogram between Ch0 and Ch1.
#     Eliminates nearest-neighbor artifacts.
#     """
#     #Empty array for histogram
#     c = np.zeros(2 * I + 1, dtype=np.int64)
    
#     #Returns the empty histogram if either channel is empty
#     if ch0_times.size == 0 or ch1_times.size == 0:
#         return c

#     # Find where each Ch0 photon would land in the sorted Ch1 timeline to bound the search window
#     left_indices = np.searchsorted(ch1_times, ch0_times - g2time_ps, side='left')
#     right_indices = np.searchsorted(ch1_times, ch0_times + g2time_ps, side='right')

#     # Loop over Ch0 photons (efficient in Python when vectorized internally)
#     for i, ch0_t in enumerate(ch0_times):
#         start_idx = left_indices[i]
#         end_idx = right_indices[i]
        
#         # start_idx == end_idx when no photons landed on Channel 1 within the window around this specific Channel 0 photon. Skips directly to the next photon to save time.
#         if start_idx == end_idx:
#             continue
            
#         # Get ALL Ch1 photons within the time window for this specific Ch0 photon
#         ch1_matches = ch1_times[start_idx:end_idx]
#         dt = ch1_matches - ch0_t  # positive if Ch1 came after Ch0
        
#         # Convert time differences to histogram bin indices
#         # Center bin is at index I
#         idx = I + np.floor(dt / timebin_ps).astype(np.int64)
        
#         # Filter out any edge cases falling outside the histogram bounds
#         valid = (idx >= 0) & (idx < 2 * I + 1)
#         np.add.at(c, idx[valid], 1)
        
#     return c


# def _from_npz(npz_path, g2time_ns=100.0, timebin_ns=1.0, seed=0):
#     """
#     Updated wrapper that reads raw arrays and routes them directly 
#     to the robust multi-hit processing engine.
#     """
#     npz = np.load(npz_path)
#     ch0 = npz['ch0'].astype(np.int64)
#     ch1 = npz['ch1'].astype(np.int64)
    
#     return _compute_g2_multihit(ch0, ch1, g2time_ns, timebin_ns, seed)


# def plot_g2(result, out_path):
#     """Save a g2(tau) plot with optional fit to out_path."""
#     tau, g2_arr, popt = result['tau'], result['g2'], result['popt']

#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.plot(tau, g2_arr, color=[.8, .8, .8], lw=1, label='g²(τ)')
#     if popt is not None:
#         tf = np.linspace(tau.min(), tau.max(), 3000)
#         g2_0 = _model(0, *popt)
#         ax.plot(tf, _model(tf, *popt), 'k', lw=1.5, label=f'Fit  g²(0)={g2_0:.3f}')
#     ax.axhline(0.5, ls='-.', color='r', lw=1, label='g²=0.5')
#     ax.axhline(1.0, ls='--', color='#888888', lw=0.9, label='g²=1')
#     ax.axvline(0,   ls=':',  color='#cccccc', lw=0.8)
#     ax.set_xlim(-30, 30)
#     ax.set_xlabel('τ (ns)', fontsize=14)
#     ax.set_ylabel('g²(τ)',  fontsize=14)
#     ax.legend(fontsize=10)
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=150)
#     plt.close(fig)
#     print(f"Saved {out_path}")


# def run(path, out_folder='g2_data', g2time_ns=100.0, timebin_ns=1.0, seed=0):
#     """
#     Full pipeline: parse .npz, compute g2, save results as .npz and .png.

#     Args:
#         path        : path to .npz file
#         out_folder  : folder to save outputs (default 'g2_data')
#         g2time_ns   : correlation half-window in ns
#         timebin_ns  : bin width in ns
#         seed        : random seed for afterflash removal

#     Returns result dict.
#     """
#     eff2_from_npz = _from_npz  # Backwards compatibility alias
#     os.makedirs(out_folder, exist_ok=True)
#     stem   = os.path.splitext(os.path.basename(path))[0]
#     prefix = os.path.join(out_folder, stem)

#     print(f"Running g2 on {path}...")
#     result = _from_npz(path, g2time_ns=g2time_ns, timebin_ns=timebin_ns, seed=seed)

#     print(f"  N1={result['N1']:,}  N2={result['N2']:,}")
#     if result['popt'] is not None:
#         a, b, T1, T2 = result['popt']
#         print(f"  Fit: a={a:.3g}  b={b:.3g}  T1={T1:.3g}ns  T2={T2:.3g}ns  g2(0)={_model(0, *result['popt']):.3f}")
#     else:
#         print("  Fit did not converge.")

#     np.savez(prefix + '_processed.npz',
#              **{k: v for k, v in result.items() if k != 'popt'},
#              popt=(result['popt'] if result['popt'] is not None else np.array([])))
#     plot_g2(result, prefix + '_plot.png')

#     return result
