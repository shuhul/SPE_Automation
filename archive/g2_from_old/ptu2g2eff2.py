"""Fast Python port of ptu2g2eff2.m — start-stop g2 with afterflash removal + fit."""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from ptu2g2 import read_ptu, parse_pt2


def start_stop_hist(chan, times, g2time_ps, timebin_ps, I):
    """Adjacent-pair cross-correlation histogram matching ptu2g2eff2.m."""
    dt = times[1:] - times[:-1]  # ps, non-negative (time-sorted)
    c01 = (chan[:-1] == 0) & (chan[1:] == 1) & (dt <= g2time_ps)
    c10 = (chan[:-1] == 1) & (chan[1:] == 0) & (dt <= g2time_ps)
    c = np.zeros(2 * I + 1, dtype=np.int64)
    if c01.any():
        idx = I + (dt[c01] // timebin_ps).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < 2 * I + 1)]
        np.add.at(c, idx, 1)
    if c10.any():
        idx = I - np.ceil(dt[c10] / timebin_ps).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < 2 * I + 1)]
        np.add.at(c, idx, 1)
    return c


def afterflash_remove(chan, times, c, cavg, I, g2time_ps, timebin_ps, rng):
    """Stochastic removal of pairs with |τ|∈(9,35)ns, mirroring eff2."""
    tau_ns = (np.arange(2 * I + 1) - I) * timebin_ps / 1000.0
    dt = times[1:] - times[:-1]

    # ch0 -> ch1 (original uses strict < for afterflash branch)
    pair01 = (chan[:-1] == 0) & (chan[1:] == 1) & (dt < g2time_ps)
    idx01 = I + (dt // timebin_ps).astype(np.int64)
    # ch1 -> ch0 (original uses <=)
    pair10 = (chan[:-1] == 1) & (chan[1:] == 0) & (dt <= g2time_ps)
    idx10 = I - np.ceil(dt / timebin_ps).astype(np.int64)

    idx_safe01 = np.clip(idx01, 0, 2 * I)
    idx_safe10 = np.clip(idx10, 0, 2 * I)

    mask01 = pair01 & (np.abs(tau_ns[idx_safe01]) > 9) & (np.abs(tau_ns[idx_safe01]) < 35)
    mask10 = pair10 & (np.abs(tau_ns[idx_safe10]) > 9) & (np.abs(tau_ns[idx_safe10]) < 35)
    cand = np.where(mask01 | mask10)[0]
    if cand.size == 0:
        return chan, times

    cand_idx = np.where(mask01[cand], idx01[cand], idx10[cand])
    c_at = c[np.clip(cand_idx, 0, 2 * I)]

    p1 = rng.poisson(cavg, size=cand.size).astype(np.float64)
    p2 = rng.poisson(c_at).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        crat = p1 / np.where(p2 == 0, np.inf, p2)
    u = rng.uniform(size=cand.size)
    to_delete = cand[u > crat] + 1

    keep = np.ones(len(times), dtype=bool)
    keep[to_delete] = False
    return chan[keep], times[keep]


def _model(x, a, b, T1, T2):
    return 1 - b * ((1 + a) * np.exp(-np.abs(x) / T1) - a * np.exp(-np.abs(x) / T2))


def eff2(ptu_path, g2time_ns=100.0, timebin_ns=1.0, seed=0):
    raw, _ = read_ptu(ptu_path)
    chan, times = parse_pt2(raw)
    order = np.argsort(times, kind='stable')
    chan = chan[order].astype(np.int8)
    times = times[order].astype(np.int64)

    g2time_ps = int(round(g2time_ns * 1000))
    timebin_ps = int(round(timebin_ns * 1000))
    I = int(np.ceil(g2time_ps / timebin_ps))
    tau_ns = (np.arange(2 * I + 1) - I) * timebin_ps / 1000.0

    c_raw = start_stop_hist(chan, times, g2time_ps, timebin_ps, I)
    wings = (tau_ns > 40) & (tau_ns < 90)
    cavg = float(c_raw[wings].mean()) if wings.any() else 0.0

    rng = np.random.default_rng(seed)
    chan2, times2 = afterflash_remove(chan, times, c_raw, cavg, I,
                                      g2time_ps, timebin_ps, rng)

    c = start_stop_hist(chan2, times2, g2time_ps, timebin_ps, I)
    N1 = int((chan2 == 0).sum()); N2 = int((chan2 == 1).sum())
    TT = int(times2[-1])
    A = N1 * N2 * timebin_ps / TT
    g2 = c / A

    try:
        popt, _ = curve_fit(_model, tau_ns, g2, p0=[1, 0.8, 10, 5000],
                            bounds=([0, -1, 0.1, 10], [np.inf, 1, np.inf, np.inf]),
                            maxfev=20000)
    except Exception:
        popt = None

    return dict(tau=tau_ns, g2=g2, c=c, c_raw=c_raw, cavg=cavg,
                N1=N1, N2=N2, TT=TT, A=A, popt=popt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ptu', required=True)
    ap.add_argument('--g2time-ns', type=float, default=100.0)
    ap.add_argument('--timebin-ns', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-prefix', required=True)
    args = ap.parse_args()

    import time as _t
    s = _t.time()
    r = eff2(args.ptu, args.g2time_ns, args.timebin_ns, args.seed)
    print(f"eff2: {_t.time()-s:.2f}s  N1={r['N1']}  N2={r['N2']}  cavg={r['cavg']:.2f}  A={r['A']:.3f}")
    if r['popt'] is not None:
        a, b, T1, T2 = r['popt']
        print(f"fit: a={a:.3g} b={b:.3g} T1={T1:.3g}ns T2={T2:.3g}ns  g2(0)={_model(0,*r['popt']):.3f}")

    np.savez(args.out_prefix + '.npz', **{k: v for k, v in r.items() if k != 'popt'},
             popt=(r['popt'] if r['popt'] is not None else np.array([])))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r['tau'], r['g2'], color=[.8, .8, .8], lw=1, label='Raw g²(τ) Data')
    if r['popt'] is not None:
        tf = np.linspace(r['tau'].min(), r['tau'].max(), 3000)
        ax.plot(tf, _model(tf, *r['popt']), 'k', lw=1.5, label='Fitted g²(τ) Function')
    ax.axhline(0.5, ls='-.', color='r', label='g²(τ)=0.5 Threshold')
    ax.set_xlim(-30, 30)
    ax.set_xlabel('τ (ns)', fontsize=14); ax.set_ylabel('g²(τ)', fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out_prefix + '.png', dpi=150)
    print(f"saved {args.out_prefix}.png / .npz")


if __name__ == '__main__':
    main()
