import os
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================================================================
# Constants
# =============================================================================
WRAPAROUND      = 2 ** 28
RESOLUTION_PS   = 4
# =============================================================================
# Model
# =============================================================================
def _model(x, a, b, T1, T2, g0):
    """
    Double exponential antibunching model with free baseline g0.
    """
    return g0 - b * ((1+a) *np.exp(-np.abs(x) / T1) - a * np.exp(-np.abs(x) / T2))

# =============================================================================
# Coincidence histogram, vectorised all pairs
# =============================================================================

def _cross_correlation_hist(ch0, ch1, g2time_ps, timebin_ps, chunk=200_000):
    """
    Vectorised all-pairs cross-correlation histogram.
    """
    I = int(np.ceil(g2time_ps / timebin_ps))
    hist = np.zeros(2 * I + 1, dtype=np.int64)

    ch0 = np.sort(ch0.astype(np.int64))
    ch1 = np.sort(ch1.astype(np.int64))
     
    for start in range(0, len(ch0), chunk):
        ch0c = ch0[start:start + chunk]
        lo = np.searchsorted(ch1, ch0c - g2time_ps, side='left')
        hi = np.searchsorted(ch1, ch0c + g2time_ps, side='right')
        counts = (hi - lo).astype(np.int64)
        total = int(counts.sum())
        if total == 0:
            continue

        # build flat index array
        starts  = np.zeros(len(ch0c), dtype=np.int64)
        np.cumsum(counts[:-1], out=starts[1:])
        offsets = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
        t1_idx  = np.repeat(lo.astype(np.int64), counts) + offsets
        dt      = ch1[t1_idx] - np.repeat(ch0c, counts)

        bins  = np.floor(dt / timebin_ps).astype(np.int64) + I
        valid = (bins >= 0) & (bins <= 2 * I)
        np.add.at(hist, bins[valid], 1)

    return hist


# =============================================================================
# Afterflash removal  —  cross-channel consecutive pairs
# =============================================================================

def _afterflash_remove(chan, times, c, cavg, I, g2time_ps, timebin_ps, rng):
    """
    Fix 1: operates on cross-channel consecutive pairs in the time-sorted
    stream (ch0->ch1 or ch1->ch0), NOT on same-channel photon pairs.

    Afterpulsing: a detector fires a spurious photon shortly after a real
    detection. In a HBT setup this creates fake ch0->ch1 (or ch1->ch0)
    coincidences at |tau| = 9-35 ns, showing as peaks in the g2 histogram.
    We stochastically remove pairs in that window where the local count
    c[bin] exceeds the expected background cavg.
    """
    tau_ns = (np.arange(2 * I + 1) - I) * timebin_ps / 1000.0
    dt     = times[1:] - times[:-1]

    pair01 = (chan[:-1] == 0) & (chan[1:] == 1) & (dt < g2time_ps)
    idx01  = I + (dt // timebin_ps).astype(np.int64)
    pair10 = (chan[:-1] == 1) & (chan[1:] == 0) & (dt <= g2time_ps)
    idx10  = I - np.ceil(dt / timebin_ps).astype(np.int64)

    s01 = np.clip(idx01, 0, 2 * I)
    s10 = np.clip(idx10, 0, 2 * I)

    mask01 = pair01 & (np.abs(tau_ns[s01]) > 9) & (np.abs(tau_ns[s01]) < 35)
    mask10 = pair10 & (np.abs(tau_ns[s10]) > 9) & (np.abs(tau_ns[s10]) < 35)
    cand   = np.where(mask01 | mask10)[0]
    if cand.size == 0:
        return chan, times

    cand_idx = np.where(mask01[cand], idx01[cand], idx10[cand])
    c_at     = c[np.clip(cand_idx, 0, 2 * I)]

    p1 = rng.poisson(cavg, size=cand.size).astype(np.float64)
    p2 = rng.poisson(c_at).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        crat = p1 / np.where(p2 == 0, np.inf, p2)

    to_delete       = cand[rng.uniform(size=cand.size) > crat] + 1
    keep            = np.ones(len(times), dtype=bool)
    keep[to_delete] = False
    return chan[keep], times[keep]


# =============================================================================
# Fit
# =============================================================================

def _fit(tau, g2):
    """Multi-start fit to avoid local minima. Returns popt or None."""
    g0_guess  = float(np.mean(g2[np.abs(tau) > 0.6 * np.abs(tau).max()]))
    best_popt, best_res = None, np.inf
    for b0 in [0.3, 0.5, 0.7, 0.9]:
        for T1_0 in [1, 3, 10, 30]:
            for a0 in [0, 1]:
                try:
                    popt, _ = curve_fit(
                        _model, tau, g2,
                        p0=[a0, b0, T1_0, 5000, g0_guess],
                        bounds=([0, 0, 0.1, 10, 0],
                                [np.inf, np.inf, np.inf, np.inf, np.inf]),
                        maxfev=10000
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

def _compute_g2(ch0, ch1, g2time_ns, timebin_ns, seed):
    """
    Compute g2 from raw photon timestamp arrays ch0, ch1 (ps, int64).
    Returns result dict.
    """
    g2time_ps  = int(round(g2time_ns  * 1000))
    timebin_ps = int(round(timebin_ns * 1000))
    I          = int(np.ceil(g2time_ps / timebin_ps))
    tau_ns     = (np.arange(2 * I + 1) - I) * timebin_ps / 1000.0

    # Raw histogram before afterflash removal
    c_raw = _cross_correlation_hist(ch0, ch1, g2time_ps, timebin_ps)

    # Background level from far-tau wings (used for afterflash removal criterion)
    wings = (tau_ns > 40) & (tau_ns < 90)
    cavg  = float(c_raw[wings].mean()) if wings.any() else 0.0

    # Afterflash removal operates on the time-sorted merged stream
    chan  = np.concatenate([np.zeros(len(ch0), dtype=np.int8),
                            np.ones( len(ch1), dtype=np.int8)])
    times = np.concatenate([ch0, ch1])
    order = np.argsort(times, kind='stable')
    chan, times = chan[order], times[order]

    rng = np.random.default_rng(seed)
    chan, times = _afterflash_remove(chan, times, c_raw, cavg, I,
                                     g2time_ps, timebin_ps, rng)

    ch0_c = times[chan == 0]
    ch1_c = times[chan == 1]

    # Histogram after afterflash removal
    c  = _cross_correlation_hist(ch0_c, ch1_c, g2time_ps, timebin_ps)
    N1 = int(ch0_c.size)
    N2 = int(ch1_c.size)

    # Fix 3: TT = absolute end time, not span
    TT = int(times[-1]) if len(times) > 0 else 1

    # Normalization: N1*N2*timebin/TT is correct for the all-pairs algorithm
    # (every possible pair is counted, unlike adjacent-pair eff2)
    A      = N1 * N2 * timebin_ps / TT if TT > 0 else 1.0
    g2_arr = c / A if A > 0 else np.zeros_like(c, dtype=float)

    # Fit on trimmed data (drop edge bins — boundary artifact at +-g2time_ns)
    tau_fit = tau_ns[1:-1]
    g2_fit  = g2_arr[1:-1]
    popt    = _fit(tau_fit, g2_fit)

    # Fix 6: compute and store g2_0_norm so automate.py does not crash
    g2_0 = g2_0_norm = None
    if popt is not None:
        a, b, T1, T2, g0 = popt
        g2_0      = float(g0 - b)
        g2_0_norm = float((g0 - b) / g0)

    return dict(
        tau=tau_ns, g2=g2_arr, c=c, c_raw=c_raw,   # Fix 5: c_raw is separate
        cavg=cavg, N1=N1, N2=N2, TT=TT, A=A,
        popt=popt, g2_0=g2_0, g2_0_norm=g2_0_norm
    )


# =============================================================================
# Public API
# =============================================================================

def run(path, out_folder='g2_data', g2time_ns=100.0, timebin_ns=1.0, seed=0):
    """
    Full pipeline: load raw photon .npz (ch0/ch1), compute g2,
    save result .npz and .png.

    Args:
        path       : path to raw photon .npz file (ch0, ch1 arrays in ps)
        out_folder : folder to save outputs
        g2time_ns  : correlation half-window in ns
        timebin_ns : bin width in ns
        seed       : random seed for afterflash removal

    Returns result dict. Key values for automate.py:
        g2_0_norm  : correctly normalized g2(0) — use this for single-emitter test
        popt       : (a, b, T1, T2, g0)
    """
    os.makedirs(out_folder, exist_ok=True)
    stem   = os.path.splitext(os.path.basename(path))[0]
    prefix = os.path.join(out_folder, stem)

    print(f"Running g2 on {path}...")

    npz = np.load(path)
    ch0 = npz['ch0'].astype(np.int64)
    ch1 = npz['ch1'].astype(np.int64)

    result = _compute_g2(ch0, ch1, g2time_ns, timebin_ns, seed)

    T_acq = result['TT'] / 1e12
    print(f"  T={T_acq:.1f}s  "
          f"N1={result['N1']:,} ({result['N1']/T_acq/1e3:.1f} kcps)  "
          f"N2={result['N2']:,} ({result['N2']/T_acq/1e3:.1f} kcps)")

    if result['popt'] is not None:
        a, b, T1, T2, g0 = result['popt']
        print(f"  g2(0) = {result['g2_0_norm']:.3f}  "
              f"T1 = {T1:.2f} ns  baseline = {g0:.3f}")
        print(f"  {'SINGLE EMITTER' if result['g2_0_norm'] < 0.5 else 'Not single emitter'}"
              f"  [g2(0) {'<' if result['g2_0_norm'] < 0.5 else '>='} 0.5]")
    else:
        print("  Fit did not converge.")

    # Save result npz
    np.savez(
        prefix + '_processed.npz',
        **{k: v for k, v in result.items()
           if k not in ('popt', 'g2_0', 'g2_0_norm')},
        popt     =(result['popt'] if result['popt'] is not None else np.array([])),
        g2_0     =result['g2_0']      if result['g2_0']      is not None else np.nan,
        g2_0_norm=result['g2_0_norm'] if result['g2_0_norm'] is not None else np.nan,
    )

    # Plot
    tau_plot = result['tau'][1:-1]
    g2_plot  = result['g2'][1:-1]
    popt     = result['popt']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tau_plot, g2_plot, color=[.7, .7, .7], lw=1, label='g²(τ)')

    if popt is not None:
        a, b, T1, T2, g0 = popt
        tf = np.linspace(tau_plot.min(), tau_plot.max(), 3000)
        ax.plot(tf, _model(tf, *popt), 'k', lw=1.8,
                label=f'Fit  g²(0) = {result["g2_0_norm"]:.3f}')
        ax.axhline(g0,       ls='--', color='#888888', lw=0.9,
                   label=f'Baseline = {g0:.3f}')
        ax.axhline(g0 * 0.5, ls='-.', color='r', lw=1.0,
                   label=f'Half-baseline = {g0*0.5:.3f}')
    else:
        ax.axhline(1.0, ls='--', color='#888888', lw=0.9, label='g²=1')
        ax.axhline(0.5, ls='-.', color='r',       lw=1.0, label='g²=0.5')

    ax.axvline(0, ls=':', color='#cccccc', lw=0.8)
    # Fix 7: x-axis uses full data range, not hardcoded +-30
    ax.set_xlim(tau_plot.min(), tau_plot.max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel('τ (ns)', fontsize=14)
    ax.set_ylabel('g²(τ)',  fontsize=14)
    ax.legend(fontsize=10)
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
