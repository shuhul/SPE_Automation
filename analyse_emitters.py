import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

_trapz = getattr(np, 'trapezoid', None) or np.trapz

DATA_DIR = 'data'
OUT_DIR  = 'analysis_output'

# ── g2 calculation config ─────────────────────────────────────────────────────
# Window widened from the original 50 ns to 400 ns: this defect isn't
# carbon-doped, and literature for non-carbon hBN defects reports metastable
# (T2) lifetimes more like ~100-300 ns rather than the tens-of-microseconds
# seen for carbon-related centers. A 50 ns window has almost no leverage on
# a ~200 ns decay; 400 ns gives several T2 time constants of margin before
# the wing region, so the far-wing normalisation actually sits past the
# metastable decay instead of extrapolating through it.
G2TIME_NS          = 400.0    # correlation half-window (ns) — was 50.0
TIMEBIN_NS         = 0.25     # bin width (ns) — unchanged, still resolves T1 fine
AFTERFLASH_LOW_NS  = 15.0
AFTERFLASH_HIGH_NS = 35.0
WING_FRAC_LOW      = 0.90     # wing now sits at 360-380 ns — check this is
WING_FRAC_HIGH     = 0.95     # past your actual T2 once you see real fits
G0_FIXED           = 1.0
SPE_THRESHOLD      = 0.5

# ── Quality-control restrictions ──────────────────────────────────────────────
MAX_FWHM_NM = 30.0   # reject ZPL fits wider than this — not a physically
                     # credible narrow ZPL, almost certainly a fit that
                     # locked onto background/noise instead of the real line.

MAX_T1_NS   = 20.0   # reject antibunching lifetimes above this as unphysical

MIN_T2_NS   = 1.0    # reject metastable timescales at or below this — a T2
                     # this close to T1 isn't a distinct metastable process,
                     # it's the fit failing to separate T1 and T2.

# ── DWF / PSB config  (see the block comment below on why this changed) ──────
#
# DWF is now computed by INTEGRATION from the FINE scan cube, not by fitting
# a Gaussian to the PSB in the long_ spectrum. Three reasons:
#
#   1. The PSB is not Gaussian. In hBN it is broad, asymmetric and usually
#      multi-peaked (several phonon modes), so a single Gaussian is the wrong
#      model regardless of SNR. DWF = I_ZPL/(I_ZPL+I_PSB) is defined with
#      INTEGRATED intensities anyway — the PSB never needed to be fitted.
#
#   2. long_ scans (600 g/mm @ 595 nm) truncate the PSB for most emitters.
#      Measured DWF success rate by ZPL: 39% for 560-570 nm, 4% for 570-585,
#      0% for 585-600 — a clean signature of the sideband running off the
#      detector. fine_ scans (150 g/mm @ 700 nm) cover ~415-980 nm, so the
#      whole PSB and a clean red background anchor are both present.
#
#   3. BACKGROUND CHOICE DOMINATES THE ANSWER. On one real emitter the same
#      spectrum gave DWF = 0.79 / 0.68 / 0.51 for a flat high baseline / a
#      sloping baseline / a flat dark baseline. That spread is larger than
#      any emitter-to-emitter variation you would be trying to measure, so
#      the background is now fixed with two anchors and interpolated.
#      By contrast the other choices barely matter: integrating the PSB to
#      700 vs 900 nm moved DWF by 0.015.
#
# NOTE ON RESOLUTION: ZPL_nm and FWHM_nm still come from the long_ spectrum
# (600 g/mm, 10 s) because it resolves the linewidth far better than the
# fine cube (150 g/mm, 1 s). Only the DWF integration uses fine_. The fine
# cube's own ZPL fit is reported separately as ZPL_fine_nm / FWHM_fine_nm so
# the two can be cross-checked — do NOT use FWHM_fine_nm as a linewidth.

LASER_CUTOFF_NM   = 556.0            # ignore bluer: 532 laser leak + 550 LP edge
# The blue background anchor is placed RELATIVE to the fitted ZPL, not at
# fixed wavelengths. A fixed window (e.g. 556-578 nm) silently overlaps the
# ZPL for any emitter bluer than ~585 nm: it then reads the ZPL's own blue
# flank as "background", subtracts it from the whole spectrum, and wipes the
# sideband out entirely (observed: a 575 nm emitter returned I_psb = 0 and
# DWF = 1.0). Anchor spans [mu - BLUE_ANCHOR_SIGMA_FAR*sigma,
# mu - BLUE_ANCHOR_SIGMA_NEAR*sigma], clipped at LASER_CUTOFF_NM.
BLUE_ANCHOR_SIGMA_NEAR = 4.0         # inner edge, in sigma from the ZPL centre
BLUE_ANCHOR_SIGMA_FAR  = 9.0         # outer edge
BLUE_ANCHOR_MIN_NM     = 4.0         # need at least this wide a clean window
RED_ANCHOR_NM     = (780.0, 950.0)   # background anchor where emission has decayed
ZPL_INT_SIGMA     = 3.0              # ZPL integration window = mu +/- this many sigma
PSB_END_NM        = 760.0            # integrate PSB out to here
FINE_MATCH_TOL_UM = 0.30             # g2 coord must land within this of a fine pixel
TRUNCATION_FRAC   = 0.15             # PSB still above this fraction of peak = truncated
DWF_MIN_SNR       = 3.0

_FWHM_K = 2.0 * np.sqrt(2.0 * np.log(2.0))   # 2.3548
_AREA_K = np.sqrt(2.0 * np.pi)


# ── Spectrum helpers ──────────────────────────────────────────────────────────

def _gaussian1d(x, A, mu, sigma, bkg):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + bkg


def _fit_zpl_gaussian(spectrum, wl, wl_min_nm=560.0, wl_max_nm=630.0,
                       window_nm=15.0, min_snr=3.0, max_fwhm_nm=45.0,
                       enforce_max_fwhm=True):
    """Fit a Gaussian to the ZPL peak, restricted to wl in [wl_min_nm, wl_max_nm].

    Returns (zpl_nm, fwhm_nm, area) or (None, None, None).
    Also rejects fits whose peak lands within EDGE_TOL_NM of the search-window
    edge — those are edge-clipped, not measured (this was producing a cluster
    of emitters reported at exactly 560.000 nm).
    """
    EDGE_TOL_NM = 1.0
    mask = (wl > wl_min_nm) & (wl < wl_max_nm)
    if not mask.any():
        return None, None, None
    wl_m, sp_m  = wl[mask], spectrum[mask]
    pk          = int(np.argmax(sp_m))
    peak_wl     = float(wl_m[pk])
    peak_val    = float(sp_m[pk])

    wing   = np.abs(wl_m - peak_wl) > window_nm
    bkg0   = float(np.median(sp_m[wing])) if wing.any() else float(np.percentile(sp_m, 10))
    noise  = float(np.std(sp_m[wing]))    if wing.sum() > 5 else 1.0

    if (peak_val - bkg0) < min_snr * max(noise, 1.0):
        return None, None, None

    win = np.abs(wl_m - peak_wl) <= window_nm
    if win.sum() < 5:
        return None, None, None

    x, y = wl_m[win], sp_m[win]
    A0   = peak_val - bkg0
    try:
        popt, _ = curve_fit(
            _gaussian1d, x, y, p0=[A0, peak_wl, 1.5, bkg0],
            bounds=([0,      max(peak_wl - window_nm, wl_min_nm), 0.05,            -np.inf],
                    [A0 * 3, min(peak_wl + window_nm, wl_max_nm), max_fwhm_nm / _FWHM_K, peak_val]),
            maxfev=3000,
        )
        A, mu, sigma, _ = popt
        if A <= 0:
            return None, None, None
        # edge-clipped fit: the "peak" is pinned at the search boundary
        if (mu - wl_min_nm) < EDGE_TOL_NM or (wl_max_nm - mu) < EDGE_TOL_NM:
            return None, None, None
        fwhm = float(_FWHM_K * sigma)
        if enforce_max_fwhm and fwhm > MAX_FWHM_NM:
            return None, None, None
        area = float(A * sigma * _AREA_K)
        return float(mu), fwhm, area
    except Exception:
        return None, None, None


# ── DWF from the FINE scan cube ───────────────────────────────────────────────

def _find_fine_pixel(run_path, tx, ty, tol_um=FINE_MATCH_TOL_UM, verbose=False):
    """Find which fine_* cube contains the g2 target (tx, ty) and which pixel.

    A fine cube routinely holds SEVERAL flagged emitters (classified.npy can
    have many 1s). Matching on the g2 coordinate is what guarantees the
    spectrum analysed is the one that produced the g2 being correlated
    against, rather than whichever emitter happens to be brightest.

    Returns dict(folder, ix, iy, x, y, dist_um, n_flagged, is_flagged) or None.
    """
    best = None
    for sub in sorted(os.listdir(run_path)):
        if not sub.startswith('fine_'):
            continue
        folder = os.path.join(run_path, sub)
        try:
            xs = np.load(os.path.join(folder, 'xs.npy'))
            ys = np.load(os.path.join(folder, 'ys.npy'))
        except (FileNotFoundError, OSError):
            continue
        ix = int(np.argmin(np.abs(xs - tx)))
        iy = int(np.argmin(np.abs(ys - ty)))
        dist = float(np.hypot(xs[ix] - tx, ys[iy] - ty))
        if best is None or dist < best['dist_um']:
            n_flag, is_flag = 0, None
            cpath = os.path.join(folder, 'classified.npy')
            if os.path.exists(cpath):
                try:
                    cls = np.load(cpath)
                    n_flag = int(cls.sum())
                    if iy < cls.shape[0] and ix < cls.shape[1]:
                        is_flag = bool(cls[iy, ix] == 1)
                except Exception:
                    pass
            best = dict(folder=folder, ix=ix, iy=iy,
                        x=float(xs[ix]), y=float(ys[iy]), dist_um=dist,
                        n_flagged=n_flag, is_flagged=is_flag)

    if best is None:
        return None
    if best['dist_um'] > tol_um:
        if verbose:
            print(f'    fine-pixel match {best["dist_um"]:.2f} um from g2 target '
                  f'({tx:.2f},{ty:.2f}) — outside tol {tol_um} um')
        return None
    return best


def _dwf_by_integration(wl, sp, verbose=False, label=None):
    """ZPL fit + sloping background + numeric PSB integration.

    Returns dict(dwf, I_zpl, I_psb, zpl_fine_nm, fwhm_fine_nm, truncated,
    bkg_blue, bkg_red, note). dwf is None when it cannot be computed, with
    `note` saying why.
    """
    tag = f'[{label}] ' if label else ''
    wl = np.asarray(wl, float); sp = np.asarray(sp, float)

    # 1. fit the ZPL in the fine cube — needed for the integration window.
    #    Never touch < LASER_CUTOFF_NM (532 laser leak + longpass edge).
    m = (wl > LASER_CUTOFF_NM) & (wl < 640.0)
    if m.sum() < 10:
        return dict(dwf=None, note='fine cube has no data in ZPL range')
    wl_m, sp_m = wl[m], sp[m]
    pk = int(np.argmax(sp_m))
    peak_wl, peak_val = float(wl_m[pk]), float(sp_m[pk])

    off = np.abs(wl_m - peak_wl) > 13.0
    bkg0  = float(np.median(sp_m[off])) if off.sum() > 5 else float(np.percentile(sp_m, 10))
    noise = float(np.std(sp_m[off]))    if off.sum() > 5 else 1.0
    snr = (peak_val - bkg0) / max(noise, 1.0)
    if snr < DWF_MIN_SNR:
        if verbose:
            print(f'{tag}DWF skip: fine-cube ZPL SNR={snr:.1f} < {DWF_MIN_SNR}')
        return dict(dwf=None, note=f'fine-cube ZPL SNR too low ({snr:.1f})')

    win = np.abs(wl_m - peak_wl) <= 13.0
    if win.sum() < 6:
        return dict(dwf=None, note='too few points around fine-cube ZPL')
    try:
        popt, _ = curve_fit(
            _gaussian1d, wl_m[win], sp_m[win],
            p0=[peak_val - bkg0, peak_wl, 4.0, bkg0],
            bounds=([0, peak_wl - 13.0, 0.3, -np.inf],
                    [(peak_val - bkg0) * 3, peak_wl + 13.0, 25.0, peak_val]),
            maxfev=8000)
    except Exception as exc:
        if verbose:
            print(f'{tag}DWF skip: fine-cube ZPL fit failed ({exc})')
        return dict(dwf=None, note=f'fine-cube ZPL fit failed ({exc})')
    A, mu, sigma, _ = popt
    if A <= 0:
        return dict(dwf=None, note='fine-cube ZPL amplitude non-positive')
    fwhm_fine = float(_FWHM_K * sigma)

    # 2. background — anchors placed RELATIVE to the fitted ZPL
    rm = (wl >= RED_ANCHOR_NM[0]) & (wl <= RED_ANCHOR_NM[1])
    if rm.sum() < 5:
        if verbose:
            print(f'{tag}DWF skip: red anchor empty — data ends at {wl.max():.0f} nm, '
                  f'need >= {RED_ANCHOR_NM[0]:.0f} nm')
        return dict(dwf=None, zpl_fine_nm=float(mu), fwhm_fine_nm=fwhm_fine,
                    note=f'red background anchor empty (data ends {wl.max():.0f} nm)')
    xr, yr = float(wl[rm].mean()), float(np.median(sp[rm]))

    b_hi = mu - BLUE_ANCHOR_SIGMA_NEAR * sigma
    b_lo = max(LASER_CUTOFF_NM, mu - BLUE_ANCHOR_SIGMA_FAR * sigma)
    bm = (wl >= b_lo) & (wl <= b_hi)
    if (b_hi - b_lo) >= BLUE_ANCHOR_MIN_NM and bm.sum() >= 5:
        xb, yb = float(wl[bm].mean()), float(np.median(sp[bm]))
        corr = sp - (yb + (yr - yb) / (xr - xb) * (wl - xb))
        bg_mode = f'sloping ({b_lo:.0f}-{b_hi:.0f} nm -> red)'
    else:
        # No clean gap between the longpass edge and the ZPL: fall back to a
        # flat baseline from the red anchor only. This can UNDER-subtract any
        # diffuse flake PL, which inflates the broad PSB more than the narrow
        # ZPL and therefore biases DWF DOWNWARD — the opposite direction from
        # truncation. Flagged so it can be filtered later.
        yb = float('nan')
        corr = sp - yr
        bg_mode = 'flat-from-red (no clean blue window)'
        if verbose:
            print(f'{tag}DWF: blue anchor would overlap the ZPL '
                  f'(mu={mu:.1f}, sigma={sigma:.1f}) — using {bg_mode}')

    # 3. integrate — no PSB model at all
    zlo, zhi = mu - ZPL_INT_SIGMA * sigma, mu + ZPL_INT_SIGMA * sigma
    phi = min(PSB_END_NM, float(wl.max()))
    zm = (wl >= zlo) & (wl <= zhi)
    pm = (wl > zhi) & (wl <= phi)
    if zm.sum() < 3 or pm.sum() < 5:
        return dict(dwf=None, zpl_fine_nm=float(mu), fwhm_fine_nm=fwhm_fine,
                    note='ZPL or PSB integration window too small')
    I_zpl = float(_trapz(np.clip(corr[zm], 0, None), wl[zm]))
    I_psb = float(_trapz(np.clip(corr[pm], 0, None), wl[pm]))
    if I_zpl + I_psb <= 0:
        return dict(dwf=None, zpl_fine_nm=float(mu), fwhm_fine_nm=fwhm_fine,
                    note='no net intensity after background subtraction')

    # 4. truncation flag — a clipped PSB inflates DWF, so mark it as a bound
    tail = corr[(wl > phi - 15) & (wl <= phi)]
    psb_peak = float(np.max(corr[pm])) if pm.any() else 0.0
    truncated = bool(psb_peak > 0 and tail.size and
                     np.median(tail) > TRUNCATION_FRAC * psb_peak)

    dwf = I_zpl / (I_zpl + I_psb)
    if I_psb <= 0 or dwf > 0.995:
        return dict(dwf=None, zpl_fine_nm=float(mu), fwhm_fine_nm=fwhm_fine,
                    I_zpl=I_zpl, I_psb=I_psb, bg_mode=bg_mode,
                    note='PSB vanished after background subtraction — '
                         'background over-subtracted, DWF rejected')
    note = 'ok'
    if truncated:
        note = (f'TRUNCATED: PSB still {100*np.median(tail)/psb_peak:.0f}% of peak '
                f'at {phi:.0f} nm — DWF is an upper bound')
        if verbose:
            print(f'{tag}{note}')

    return dict(dwf=float(dwf), I_zpl=I_zpl, I_psb=I_psb,
                zpl_fine_nm=float(mu), fwhm_fine_nm=fwhm_fine,
                truncated=truncated, bkg_blue=yb, bkg_red=yr,
                bg_mode=bg_mode, note=note)


def _dwf_from_fine(run_path, tx, ty, verbose=False, label=None):
    """g2 target coordinate -> matching fine-cube pixel -> DWF by integration."""
    hit = _find_fine_pixel(run_path, tx, ty, verbose=verbose)
    if hit is None:
        return dict(dwf=None, note='no matching fine-scan pixel')
    try:
        cube = np.load(os.path.join(hit['folder'], 'out.npy'))
        wl   = np.load(os.path.join(hit['folder'], 'wl.npy'))
    except (FileNotFoundError, OSError):
        return dict(dwf=None, note='fine cube missing out.npy/wl.npy')
    if cube.ndim != 3 or hit['iy'] >= cube.shape[0] or hit['ix'] >= cube.shape[1]:
        return dict(dwf=None, note='fine cube shape unexpected')
    res = _dwf_by_integration(wl, cube[hit['iy'], hit['ix'], :].astype(float),
                              verbose=verbose, label=label)
    res.update(fine_folder=os.path.basename(hit['folder']),
               fine_match_um=hit['dist_um'],
               fine_n_flagged=hit['n_flagged'],
               fine_is_flagged=hit['is_flagged'])
    return res


# ── g2 calculation ────────────────────────────────────────────────────────────

def _model_g2(x, a, b, T1, T2):
    return G0_FIXED - b * ((1 + a) * np.exp(-np.abs(x) / T1)
                           - a * np.exp(-np.abs(x) / T2))


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


def _fit_g2(tau, g2):
    """Multi-start grid search (g0 fixed, only a, b, T1, T2 free)."""
    best_popt, best_res = None, np.inf
    for b0 in [0.3, 0.5, 0.7, 0.9]:
        for T1_0 in [0.5, 1, 3, 10]:
            for a0 in [0, 1]:
                for T2_0 in [50.0, 150.0, 500.0, 2000.0]:
                    try:
                        popt, _ = curve_fit(
                            _model_g2, tau, g2, p0=[a0, b0, T1_0, T2_0],
                            bounds=([0, 0, 0.05, 1.0],
                                    [10.0, G0_FIXED * 1.5 + 0.5, 100.0, 1e5]),
                            maxfev=10_000)
                        res = float(np.sum((_model_g2(tau, *popt) - g2) ** 2))
                        if res < best_res:
                            best_res, best_popt = res, popt
                    except Exception:
                        continue
    return best_popt


def calculate_g2(ch0, ch1):
    """Calculate g2 over the G2TIME_NS window from raw channel data."""
    g2time_ps  = int(round(G2TIME_NS  * 1000))
    timebin_ps = int(round(TIMEBIN_NS * 1000))
    I          = int(np.ceil(g2time_ps / timebin_ps))
    n_bins     = 2 * I + 1
    tau_ns     = (np.arange(n_bins) - I) * timebin_ps / 1000.0

    hist = _cross_correlation_hist(ch0, ch1, g2time_ps, timebin_ps)

    wing_mask = (np.abs(tau_ns) >= G2TIME_NS * WING_FRAC_LOW) & \
                (np.abs(tau_ns) <= G2TIME_NS * WING_FRAC_HIGH)
    wing_vals = hist[wing_mask]
    if wing_vals.size == 0:
        return None
    c_wing = float(wing_vals.mean())
    if c_wing <= 0:
        return None

    g2_arr = hist.astype(float) / c_wing
    af_mask   = (np.abs(tau_ns) >= AFTERFLASH_LOW_NS) & (np.abs(tau_ns) <= AFTERFLASH_HIGH_NS)
    edge_mask = (np.arange(n_bins) > 0) & (np.arange(n_bins) < n_bins - 1)
    fit_mask  = edge_mask & ~af_mask

    popt = _fit_g2(tau_ns[fit_mask], g2_arr[fit_mask])
    g2_0 = float(G0_FIXED - popt[1]) if popt is not None else None
    return dict(tau=tau_ns, g2=g2_arr, popt=popt, g2_0=g2_0, wing_level=c_wing)


# ── Data extraction ───────────────────────────────────────────────────────────

_RUN_RE   = re.compile(r'(?P<date>\d{8})-PLSPC-HT-Ch(?P<chip>\w+)-f(?P<field>\d+)-')
_COORD_RE = re.compile(r'_x(?P<x>-?[\d.]+)_y(?P<y>-?[\d.]+)')


def _parse_run_meta(run_name):
    m = _RUN_RE.search(run_name)
    if not m:
        return {'chip': '?', 'field': '?', 'date': run_name[:8]}
    return {'chip': m.group('chip'), 'field': m.group('field'), 'date': m.group('date')}


def iter_emitters(data_dir, verbose=False):
    """Yield one dict per emitter with raw g2 data and a matching spectrum.

    ZPL_nm / FWHM_nm come from the long_ spectrum (600 g/mm, 10 s — resolves
    the linewidth). DWF comes from the fine_ cube (150 g/mm, ~415-980 nm —
    the only place the whole PSB is actually recorded), matched to this
    emitter's g2 coordinate so multi-emitter fine cubes resolve correctly.

    T1_ns / T2_ns are only populated for CONFIRMED single emitters
    (g2_0 < SPE_THRESHOLD) that also pass MAX_T1_NS / MIN_T2_NS.
    """
    n_zpl_rejected  = 0
    n_t1_excluded   = 0
    n_t2_excluded   = 0
    n_dwf_ok        = 0
    n_dwf_trunc     = 0
    dwf_fail_reason = {}

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

            g2_path = os.path.join(run_path, subdir)
            raw_files = sorted(f for f in os.listdir(g2_path)
                               if f.endswith('.npz') and '_processed' not in f)
            if not raw_files:
                continue

            label = f'{run_name}/{coord_str}'

            try:
                npz = np.load(os.path.join(g2_path, raw_files[-1]))
                ch0 = npz['ch0'].astype(np.int64)
                ch1 = npz['ch1'].astype(np.int64)
                T_acq_s  = int(max(ch0[-1], ch1[-1])) / 1e12
                rate_khz = (len(ch0) + len(ch1)) / T_acq_s / 1000.0 if T_acq_s > 0 else np.nan
                g2_result = calculate_g2(ch0, ch1)
                if g2_result is None or g2_result['popt'] is None or g2_result['g2_0'] is None:
                    continue
                popt = g2_result['popt']
                g2_0_norm = g2_result['g2_0']
                a, b, T1, T2 = popt
            except Exception as e:
                if verbose:
                    print(f'Skipped {label}: {e}')
                continue

            # ── ZPL + FWHM from the long_ spectrum (high resolution) ────────
            lf_path = os.path.join(run_path, lf_dir)
            try:
                wl_long  = np.load(os.path.join(lf_path, 'wl.npy'))
                out_long = np.load(os.path.join(lf_path, 'out.npy'))
                spectrum = out_long[0, 0, :]
            except (FileNotFoundError, IndexError, OSError):
                continue

            zpl, fwhm, zpl_area = _fit_zpl_gaussian(spectrum, wl_long)
            if zpl is None:
                n_zpl_rejected += 1
                continue

            # ── DWF from the fine_ cube, matched on the g2 coordinate ───────
            cm = _COORD_RE.search(coord_str)
            x  = float(cm.group('x')) if cm else None
            y  = float(cm.group('y')) if cm else None

            dwf_res = dict(dwf=None, note='no coords parsed')
            if x is not None and y is not None:
                dwf_res = _dwf_from_fine(run_path, x, y, verbose=verbose, label=label)
            if dwf_res.get('dwf') is not None:
                n_dwf_ok += 1
                if dwf_res.get('truncated'):
                    n_dwf_trunc += 1
            else:
                key = str(dwf_res.get('note', 'unknown')).split('(')[0].strip()
                dwf_fail_reason[key] = dwf_fail_reason.get(key, 0) + 1

            # ── T1/T2 gating ───────────────────────────────────────────────
            is_spe = g2_0_norm < SPE_THRESHOLD
            if is_spe and T1 < MAX_T1_NS:
                t1_out = float(T1)
            else:
                t1_out = np.nan
                if is_spe:
                    n_t1_excluded += 1

            if is_spe and T2 > MIN_T2_NS:
                t2_out = float(T2)
            else:
                t2_out = np.nan
                if is_spe:
                    n_t2_excluded += 1

            yield {
                'run':            run_name,
                **meta,
                'x':              x,
                'y':              y,
                'ZPL_nm':         zpl,                       # from long_ (600 g/mm)
                'FWHM_nm':        fwhm,                      # from long_ (600 g/mm)
                'DWF':            dwf_res.get('dwf'),        # from fine_ by integration
                'DWF_truncated':  dwf_res.get('truncated'),
                'I_zpl':          dwf_res.get('I_zpl'),
                'I_psb':          dwf_res.get('I_psb'),
                'ZPL_fine_nm':    dwf_res.get('zpl_fine_nm'),   # cross-check only
                'FWHM_fine_nm':   dwf_res.get('fwhm_fine_nm'),  # NOT a linewidth (150 g/mm)
                'fine_folder':    dwf_res.get('fine_folder'),
                'fine_match_um':  dwf_res.get('fine_match_um'),
                'fine_n_flagged': dwf_res.get('fine_n_flagged'),
                'fine_is_flagged': dwf_res.get('fine_is_flagged'),
                'dwf_bg_mode':    dwf_res.get('bg_mode'),
                'dwf_note':       dwf_res.get('note'),
                'g2_0':           g2_0_norm,
                'T1_ns':          t1_out,
                'T2_ns':          t2_out,
                'ZPL_intensity':  zpl_area,
                'rate_kHz':       rate_khz,
            }

    print('\nQuality-control exclusions:')
    if n_zpl_rejected:
        print(f'  {n_zpl_rejected} emitter(s) dropped — no clear ZPL, FWHM > {MAX_FWHM_NM} nm, '
              f'or fit pinned at the search-window edge')
    if n_t1_excluded:
        print(f'  {n_t1_excluded} confirmed SPE(s) kept, but T1 excluded (>= {MAX_T1_NS} ns)')
    if n_t2_excluded:
        print(f'  {n_t2_excluded} confirmed SPE(s) kept, but T2 excluded (<= {MIN_T2_NS} ns)')
    print(f'\nDWF (from fine_ cube, by integration):')
    print(f'  {n_dwf_ok} computed  ({n_dwf_trunc} of them TRUNCATED — upper bounds only)')
    if dwf_fail_reason:
        print('  failures:')
        for k, v in sorted(dwf_fail_reason.items(), key=lambda kv: -kv[1]):
            print(f'    {v:3d}  {k}')


# ── Plotting ──────────────────────────────────────────────────────────────────

_ORIGIN_VISIBLE_COLS = {'FWHM_nm', 'g2_0', 'T1_ns', 'T2_ns', 'rate_kHz',
                        'ZPL_intensity', 'DWF'}
_POINT_COLOR = '#4C72B0'


def _scatter(ax, df, xcol, ycol, xlabel, ylabel):
    sub   = df[[xcol, ycol]].copy()
    valid = sub.notna().all(axis=1)
    sub   = sub[valid]
    if sub.empty:
        ax.text(0.5, 0.5, f'no data\n({xcol} vs {ycol})', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='0.5')
        ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
        return

    ax.scatter(sub[xcol], sub[ycol], c=_POINT_COLOR, s=25,
               edgecolors='k', linewidths=0.4, zorder=3)

    se = df.loc[valid, 'g2_0'] < 0.5
    if se.any():
        ax.scatter(sub.loc[se, xcol], sub.loc[se, ycol], marker='*', s=80,
                   c=_POINT_COLOR, edgecolors='k', linewidths=0.4, zorder=4,
                   label='single emitter (g²(0) < 0.5)')

    if ycol == 'g2_0':
        ax.axhline(0.5, ls='--', color='#e74c3c', lw=1.2, zorder=2, label='g²(0) = 0.5')

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.margins(0.08)
    if xcol in _ORIGIN_VISIBLE_COLS:
        right = ax.get_xlim()[1]
        ax.set_xlim(left=-0.05 * right, right=right * 1.05)
    if ycol in _ORIGIN_VISIBLE_COLS:
        top = ax.get_ylim()[1]
        ax.set_ylim(bottom=-0.05 * top, top=top * 1.1)
    ax.legend(fontsize=8, loc='best')


def make_scatter_plots(df, out_dir):
    # 4 rows x 3 cols. Row layout is consistent: the quantity being explained
    # is always on Y, the explanatory variable on X (ZPL / FWHM / other).
    fig, axes = plt.subplots(4, 3, figsize=(16, 17))
    fig.suptitle('Emitter correlation analysis', fontsize=14, fontweight='bold')

    _scatter(axes[0, 0], df, 'ZPL_nm',   'g2_0',  'ZPL (nm)',            'g²(0)')
    _scatter(axes[0, 1], df, 'FWHM_nm',  'g2_0',  'ZPL FWHM (nm)',       'g²(0)')
    _scatter(axes[0, 2], df, 'rate_kHz', 'g2_0',  'Emission rate (kHz)', 'g²(0)')

    _scatter(axes[1, 0], df, 'ZPL_nm',   'T1_ns', 'ZPL (nm)',            'T₁ (ns)  [confirmed SPE only]')
    _scatter(axes[1, 1], df, 'FWHM_nm',  'T1_ns', 'ZPL FWHM (nm)',       'T₁ (ns)  [confirmed SPE only]')
    _scatter(axes[1, 2], df, 'rate_kHz', 'T1_ns', 'Emission rate (kHz)', 'T₁ (ns)  [confirmed SPE only]')

    _scatter(axes[2, 0], df, 'ZPL_nm',   'T2_ns', 'ZPL (nm)',            'T₂ (ns)  [confirmed SPE only]')
    _scatter(axes[2, 1], df, 'FWHM_nm',  'T2_ns', 'ZPL FWHM (nm)',       'T₂ (ns)  [confirmed SPE only]')
    axes[2, 2].axis('off')

    # DWF row. DWF vs FWHM and DWF vs T1 are the direct tests of whether
    # electron-phonon coupling is the common cause behind the T1-FWHM trend:
    # stronger coupling should broaden the ZPL, lower the DWF, and shorten T1.
    _scatter(axes[3, 0], df, 'ZPL_nm',   'DWF',   'ZPL (nm)',            'Debye-Waller factor')
    _scatter(axes[3, 1], df, 'FWHM_nm',  'DWF',   'ZPL FWHM (nm)',       'Debye-Waller factor')
    _scatter(axes[3, 2], df, 'T1_ns',    'DWF',   'T₁ (ns)  [confirmed SPE only]', 'Debye-Waller factor')

    fig.tight_layout()
    out_path = os.path.join(out_dir, 'emitter_correlations.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {out_path}')


def make_histogram_plots(df, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    fig.suptitle('Emitter property distributions', fontsize=13)

    axes[0].hist(df['ZPL_nm'].dropna(), bins=20, edgecolor='k', color='steelblue')
    axes[0].set_xlabel('ZPL (nm)'); axes[0].set_ylabel('Count'); axes[0].set_title('ZPL')

    axes[1].hist(df['FWHM_nm'].dropna(), bins=20, edgecolor='k', color='seagreen')
    axes[1].set_xlabel('FWHM (nm)'); axes[1].set_ylabel('Count')
    axes[1].set_title(f'FWHM  (<= {MAX_FWHM_NM} nm)')

    dwf_ok = df.loc[df['DWF_truncated'] == False, 'DWF'].dropna()
    dwf_tr = df.loc[df['DWF_truncated'] == True,  'DWF'].dropna()
    if len(dwf_ok) or len(dwf_tr):
        bins = np.linspace(0, 1, 21)
        axes[2].hist([dwf_ok, dwf_tr], bins=bins, stacked=True, edgecolor='k',
                     color=['goldenrod', '0.75'],
                     label=[f'measured (n={len(dwf_ok)})', f'upper bound (n={len(dwf_tr)})'])
        axes[2].legend(fontsize=7)
    axes[2].set_xlabel('Debye-Waller factor'); axes[2].set_ylabel('Count')
    axes[2].set_title('DWF (integration, fine_ cube)')

    axes[3].hist(df['g2_0'].dropna(), bins=20, edgecolor='k', color='salmon')
    axes[3].axvline(0.5, ls='--', color='red', lw=1.2, label='g²(0) = 0.5')
    axes[3].set_xlabel('g²(0)'); axes[3].set_ylabel('Count'); axes[3].set_title('g²(0)')
    axes[3].legend()

    fig.tight_layout()
    out_path = os.path.join(out_dir, 'emitter_histograms.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Analyse HT fullauto emitter data.')
    ap.add_argument('--data-dir', default=DATA_DIR)
    ap.add_argument('--out-dir',  default=OUT_DIR)
    ap.add_argument('--verbose', action='store_true',
                    help='Print per-emitter reasons for DWF/fine-match failures')
    args = ap.parse_args()

    print('Scanning data folders...')
    rows = list(iter_emitters(args.data_dir, verbose=args.verbose))

    if not rows:
        print('No emitters found with both g2 and long_ spectrum data.')
        return

    df = pd.DataFrame(rows)

    bad = df[df['g2_0'] < 0]
    if not bad.empty:
        print('\nIgnored (g²(0) < 0 — unphysical fit):')
        for _, row in bad.iterrows():
            print(f'  {row["run"]}  /  g2_{row["x"]}_{row["y"]}  g²(0) = {row["g2_0"]:.3f}')
        df = df[df['g2_0'] >= 0].reset_index(drop=True)

    n_single = int((df['g2_0'] < 0.5).sum())
    print(f'\nFound {len(df)} valid emitters across {df["run"].nunique()} runs '
          f'({n_single} single emitters, '
          f'{int(df.T1_ns.notna().sum())} with T1, {int(df.T2_ns.notna().sum())} with T2, '
          f'{int(df.DWF.notna().sum())} with DWF).\n')

    cols = ['chip', 'field', 'x', 'y', 'ZPL_nm', 'FWHM_nm', 'DWF', 'DWF_truncated',
            'g2_0', 'T1_ns', 'T2_ns', 'rate_kHz']
    print(df[cols].to_string(index=False,
          float_format=lambda v: f'{v:.3f}' if pd.notna(v) else 'None'))

    # sanity: the fine cube's own ZPL should agree with the long_ ZPL
    chk = df[['ZPL_nm', 'ZPL_fine_nm']].dropna()
    if len(chk) >= 3:
        d = (chk.ZPL_fine_nm - chk.ZPL_nm).abs()
        print(f'\nZPL cross-check (long_ vs fine_): median |diff| = {d.median():.2f} nm, '
              f'max = {d.max():.2f} nm  (n={len(chk)})')
        if d.max() > 5:
            print('  WARNING: some emitters disagree by >5 nm — check the fine-pixel match')

    flagged = df['fine_is_flagged'].dropna()
    if len(flagged):
        print(f'Fine-pixel match landed on a flagged (classified==1) pixel for '
              f'{int(flagged.sum())}/{len(flagged)} emitters')

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'emitter_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}')

    make_scatter_plots(df, args.out_dir)
    make_histogram_plots(df, args.out_dir)


if __name__ == '__main__':
    main()