"""G2 photon correlation analysis. PTU parsing + eff2 start-stop algorithm."""
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
WRAPAROUND      = 210698240
RESOLUTION_PS   = 4
RT_PICOHARP_T2  = 0x00010203

TY_EMPTY8    = 0xFFFF0008
TY_BOOL8     = 0x00000008
TY_INT8      = 0x10000008
TY_BITSET64  = 0x11000008
TY_COLOR8    = 0x12000008
TY_FLOAT8    = 0x20000008
TY_TDATETIME = 0x21000008
TY_FLOATARR  = 0x2001FFFF
TY_ANSISTR   = 0x4001FFFF
TY_WIDESTR   = 0x4002FFFF
TY_BLOB      = 0xFFFFFFFF

# =============================================================================
# PTU Parsing
# =============================================================================

def read_ptu(path):
    """Parse PTU file header and return raw TTTR records + tag dict."""
    tags = {}
    with open(path, 'rb') as f:
        if b'PQTTTR' not in f.read(8):
            raise ValueError("Not a valid PTU file")
        f.read(8)
        while True:
            ident = f.read(32).rstrip(b'\x00').decode('ascii', errors='ignore')
            _idx, typ = struct.unpack('<iI', f.read(8))
            if typ == TY_EMPTY8:
                f.read(8)
            elif typ in (TY_BOOL8, TY_INT8, TY_BITSET64, TY_COLOR8):
                tags[ident] = struct.unpack('<q', f.read(8))[0]
            elif typ in (TY_FLOAT8, TY_TDATETIME):
                tags[ident] = struct.unpack('<d', f.read(8))[0]
            elif typ == TY_FLOATARR:
                n = struct.unpack('<q', f.read(8))[0]; f.read(n)
            elif typ in (TY_ANSISTR, TY_WIDESTR):
                n = struct.unpack('<q', f.read(8))[0]
                tags[ident] = f.read(n).rstrip(b'\x00').decode('ascii', errors='ignore')
            elif typ == TY_BLOB:
                n = struct.unpack('<q', f.read(8))[0]; f.read(n)
            else:
                raise ValueError(f"Unknown tag type 0x{typ:08X}")
            if ident == 'Header_End':
                break
        n_records = tags.get('TTResult_NumberOfRecords', 0)
        raw = np.frombuffer(f.read(4 * n_records), dtype=np.uint32, count=n_records)

    if tags.get('TTResultFormat_TTTRRecType') != RT_PICOHARP_T2:
        raise ValueError("Only PicoHarp T2 format is supported")
    return raw, tags


def parse_pt2(raw):
    """Convert raw TTTR records to (channel, absolute_time_ps) arrays for photons."""
    t2time  = (raw & 0x0FFFFFFF).astype(np.int64)
    chan    = ((raw >> 28) & 0xF).astype(np.int32)
    markers = (raw & 0xF).astype(np.int32)

    overflow      = (chan == 15) & (markers == 0)
    ofl_before    = np.cumsum(overflow.astype(np.int64)) - overflow.astype(np.int64)
    abs_ps        = (ofl_before * WRAPAROUND + t2time) * RESOLUTION_PS

    photon = (chan >= 0) & (chan <= 4) & ~overflow
    return chan[photon].astype(np.int8), abs_ps[photon]

# =============================================================================
# G2 — eff2 start-stop algorithm
# =============================================================================

def _start_stop_hist(chan, times, g2time_ps, timebin_ps, I):
    dt   = times[1:] - times[:-1]
    c01  = (chan[:-1] == 0) & (chan[1:] == 1) & (dt <= g2time_ps)
    c10  = (chan[:-1] == 1) & (chan[1:] == 0) & (dt <= g2time_ps)
    c    = np.zeros(2 * I + 1, dtype=np.int64)
    if c01.any():
        idx = I + (dt[c01] // timebin_ps).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < 2 * I + 1)]
        np.add.at(c, idx, 1)
    if c10.any():
        idx = I - np.ceil(dt[c10] / timebin_ps).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < 2 * I + 1)]
        np.add.at(c, idx, 1)
    return c


def _afterflash_remove(chan, times, c, cavg, I, g2time_ps, timebin_ps, rng):
    """Stochastically remove afterflash pairs with |τ| in (9, 35) ns."""
    tau_ns   = (np.arange(2 * I + 1) - I) * timebin_ps / 1000.0
    dt       = times[1:] - times[:-1]

    pair01   = (chan[:-1] == 0) & (chan[1:] == 1) & (dt < g2time_ps)
    idx01    = I + (dt // timebin_ps).astype(np.int64)
    pair10   = (chan[:-1] == 1) & (chan[1:] == 0) & (dt <= g2time_ps)
    idx10    = I - np.ceil(dt / timebin_ps).astype(np.int64)

    idx_safe01 = np.clip(idx01, 0, 2 * I)
    idx_safe10 = np.clip(idx10, 0, 2 * I)

    mask01 = pair01 & (np.abs(tau_ns[idx_safe01]) > 9) & (np.abs(tau_ns[idx_safe01]) < 35)
    mask10 = pair10 & (np.abs(tau_ns[idx_safe10]) > 9) & (np.abs(tau_ns[idx_safe10]) < 35)
    cand   = np.where(mask01 | mask10)[0]
    if cand.size == 0:
        return chan, times

    cand_idx = np.where(mask01[cand], idx01[cand], idx10[cand])
    c_at     = c[np.clip(cand_idx, 0, 2 * I)]
    p1       = rng.poisson(cavg, size=cand.size).astype(np.float64)
    p2       = rng.poisson(c_at).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        crat = p1 / np.where(p2 == 0, np.inf, p2)
    to_delete    = cand[rng.uniform(size=cand.size) > crat] + 1
    keep         = np.ones(len(times), dtype=bool)
    keep[to_delete] = False
    return chan[keep], times[keep]


def _model(x, a, b, T1, T2):
    return 1 - b * ((1 + a) * np.exp(-np.abs(x) / T1) - a * np.exp(-np.abs(x) / T2))

# =============================================================================
# Public API
# =============================================================================

def _compute_g2(chan, times, g2time_ns, timebin_ns, seed):
    """Core eff2 algorithm on pre-sorted (chan, times) arrays. Returns result dict."""
    g2time_ps  = int(round(g2time_ns * 1000))
    timebin_ps = int(round(timebin_ns * 1000))
    I          = int(np.ceil(g2time_ps / timebin_ps))
    tau_ns     = (np.arange(2 * I + 1) - I) * timebin_ps / 1000.0

    c_raw = _start_stop_hist(chan, times, g2time_ps, timebin_ps, I)
    wings = (tau_ns > 40) & (tau_ns < 90)
    cavg  = float(c_raw[wings].mean()) if wings.any() else 0.0

    rng = np.random.default_rng(seed)
    chan, times = _afterflash_remove(chan, times, c_raw, cavg, I, g2time_ps, timebin_ps, rng)

    c  = _start_stop_hist(chan, times, g2time_ps, timebin_ps, I)
    N1 = int((chan == 0).sum())
    N2 = int((chan == 1).sum())
    TT = int(times[-1])
    A  = N1 * N2 * timebin_ps / TT
    g2_arr = c / A

    try:
        popt, _ = curve_fit(
            _model, tau_ns, g2_arr,
            p0=[1, 0.8, 10, 5000],
            bounds=([0, -1, 0.1, 10], [np.inf, 1, np.inf, np.inf]),
            maxfev=20000
        )
    except Exception:
        popt = None

    return dict(tau=tau_ns, g2=g2_arr, c=c, c_raw=c_raw,
                cavg=cavg, N1=N1, N2=N2, TT=TT, A=A, popt=popt)


def eff2(ptu_path, g2time_ns=100.0, timebin_ns=1.0, seed=0):
    """
    Run eff2 g2 analysis on a PTU file.

    Returns a dict with keys:
        tau     : delay axis (ns)
        g2      : normalised g2(tau)
        c       : coincidence histogram (after afterflash removal)
        c_raw   : coincidence histogram (before afterflash removal)
        cavg    : background level used for afterflash removal
        N1, N2  : photon counts per channel
        TT      : total acquisition time (ps)
        A       : normalisation factor
        popt    : fit parameters (a, b, T1, T2) or None if fit failed
    """
    raw, _ = read_ptu(ptu_path)
    chan, times = parse_pt2(raw)
    order  = np.argsort(times, kind='stable')
    return _compute_g2(chan[order].astype(np.int8), times[order].astype(np.int64),
                       g2time_ns, timebin_ns, seed)


def eff2_from_npz(npz_path, g2time_ns=100.0, timebin_ns=1.0, seed=0):
    """
    Run eff2 g2 analysis on a .npz file saved by picoharp.ph_acquire().
    The .npz must contain arrays 'ch0' and 'ch1' (absolute photon times in ps).
    Returns the same result dict as eff2().
    """
    npz   = np.load(npz_path)
    ch0   = npz['ch0'].astype(np.int64)
    ch1   = npz['ch1'].astype(np.int64)
    chan  = np.concatenate([np.zeros(len(ch0), dtype=np.int8),
                            np.ones( len(ch1), dtype=np.int8)])
    times = np.concatenate([ch0, ch1])
    order = np.argsort(times, kind='stable')
    return _compute_g2(chan[order], times[order], g2time_ns, timebin_ns, seed)


def plot_g2(result, out_path):
    """Save a g2(tau) plot with optional fit to out_path."""
    tau, g2_arr, popt = result['tau'], result['g2'], result['popt']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tau, g2_arr, color=[.8, .8, .8], lw=1, label='g²(τ)')
    if popt is not None:
        tf = np.linspace(tau.min(), tau.max(), 3000)
        g2_0 = _model(0, *popt)
        ax.plot(tf, _model(tf, *popt), 'k', lw=1.5, label=f'Fit  g²(0)={g2_0:.3f}')
    ax.axhline(0.5, ls='-.', color='r', lw=1, label='g²=0.5')
    ax.axhline(1.0, ls='--', color='#888888', lw=0.9, label='g²=1')
    ax.axvline(0,   ls=':',  color='#cccccc', lw=0.8)
    ax.set_xlim(-30, 30)
    ax.set_xlabel('τ (ns)', fontsize=14)
    ax.set_ylabel('g²(τ)',  fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def run(path, out_folder='g2_data', g2time_ns=100.0, timebin_ns=1.0, seed=0):
    """
    Full pipeline: parse .ptu or .npz, compute g2, save results as .npz and .png.

    Args:
        path        : path to .ptu or .npz file
        out_folder  : folder to save outputs (default 'g2_data')
        g2time_ns   : correlation half-window in ns
        timebin_ns  : bin width in ns
        seed        : random seed for afterflash removal

    Returns result dict.
    """
    os.makedirs(out_folder, exist_ok=True)
    stem   = os.path.splitext(os.path.basename(path))[0]
    prefix = os.path.join(out_folder, stem)

    print(f"Running g2 on {path}...")
    if path.endswith('.npz'):
        result = eff2_from_npz(path, g2time_ns=g2time_ns, timebin_ns=timebin_ns, seed=seed)
    else:
        result = eff2(path, g2time_ns=g2time_ns, timebin_ns=timebin_ns, seed=seed)

    print(f"  N1={result['N1']:,}  N2={result['N2']:,}  cavg={result['cavg']:.2f}")
    if result['popt'] is not None:
        a, b, T1, T2 = result['popt']
        print(f"  Fit: a={a:.3g}  b={b:.3g}  T1={T1:.3g}ns  T2={T2:.3g}ns  g2(0)={_model(0, *result['popt']):.3f}")
    else:
        print("  Fit did not converge.")

    np.savez(prefix + '.npz',
             **{k: v for k, v in result.items() if k != 'popt'},
             popt=(result['popt'] if result['popt'] is not None else np.array([])))
    plot_g2(result, prefix + '.png')

    return result
