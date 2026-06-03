"""Python port of ptu2g2.m — parses PicoHarp T2 PTU and computes g2."""
import os
import struct
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WRAPAROUND   = 210698240
RESOLUTION_PS = 4
RT_PICOHARP_T2 = 0x00010203

TY_EMPTY8   = 0xFFFF0008
TY_BOOL8    = 0x00000008
TY_INT8     = 0x10000008
TY_BITSET64 = 0x11000008
TY_COLOR8   = 0x12000008
TY_FLOAT8   = 0x20000008
TY_TDATETIME= 0x21000008
TY_FLOATARR = 0x2001FFFF
TY_ANSISTR  = 0x4001FFFF
TY_WIDESTR  = 0x4002FFFF
TY_BLOB     = 0xFFFFFFFF


def read_ptu(path, max_records=None):
    tags = {}
    with open(path, 'rb') as f:
        if b'PQTTTR' not in f.read(8):
            raise ValueError("Not a PTU file")
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
        n_records = tags.get('TTResult_NumberOfRecords')
        if max_records is not None:
            n_records = min(n_records, max_records)
        raw = np.frombuffer(f.read(4 * n_records), dtype=np.uint32, count=n_records)

    if tags.get('TTResultFormat_TTTRRecType') != RT_PICOHARP_T2:
        raise ValueError("Only PicoHarp T2 supported")
    return raw, tags


def parse_pt2(raw):
    """Return (chan, times_ps) for photon records, matching ptu2g2.m logic.
    Photons: chan in 0..4; chan==15 with marker bits==0 is overflow."""
    t2time = (raw & 0x0FFFFFFF).astype(np.int64)
    chan   = ((raw >> 28) & 0xF).astype(np.int32)
    markers = (raw & 0xF).astype(np.int32)

    overflow = (chan == 15) & (markers == 0)
    # cumulative overflow count BEFORE current record (matches matlab where
    # ofltime is updated only when overflow is seen, and photon records use
    # the current ofltime)
    ofl = np.cumsum(overflow.astype(np.int64))
    ofl_before = ofl - overflow.astype(np.int64)
    abs_ps = (ofl_before * WRAPAROUND + t2time) * RESOLUTION_PS

    photon = (chan >= 0) & (chan <= 4) & ~overflow
    return chan[photon].astype(np.int8), abs_ps[photon]


def g2_histogram(t0, t1, total_ns, bin_ps, chunk=200_000):
    """Cross-correlation histogram of (t1 - t0) for dt in (-T, T].
    Matches ptu2g2.m bin convention: bin k covers dt in (t[k], t[k+1]].
    Uses int64 ps throughout for exact arithmetic."""
    T_ps  = int(round(total_ns * 1000.0))
    bin_ps = int(bin_ps)
    tt    = int(np.ceil(T_ps / bin_ps))  # matches matlab
    edges = np.arange(-tt, tt + 1, dtype=np.int64) * bin_ps  # length 2*tt+1
    hist  = np.zeros(2 * tt, dtype=np.int64)

    t0 = np.sort(t0.astype(np.int64))
    t1 = np.sort(t1.astype(np.int64))

    for start in range(0, len(t0), chunk):
        t0c = t0[start:start + chunk]
        lo = np.searchsorted(t1, t0c - T_ps, side='right')  # dt > -T (strict)
        hi = np.searchsorted(t1, t0c + T_ps, side='right')  # dt <= T
        counts = (hi - lo).astype(np.int64)
        total = int(counts.sum())
        if total == 0:
            continue
        # build flat arrays of (t0_val, t1_idx) for each pair
        starts = np.zeros(len(t0c), dtype=np.int64)
        np.cumsum(counts[:-1], out=starts[1:])
        offsets = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
        t1_idx = np.repeat(lo.astype(np.int64), counts) + offsets
        dt = t1[t1_idx] - np.repeat(t0c, counts)
        # np.histogram uses [edge_k, edge_{k+1}) for k<last, [last-1,last] for last.
        # matlab wants (edge_k, edge_{k+1}] everywhere. Shift by flipping sign or
        # subtracting 1 ps (int). Equivalent trick: subtract 1 from dt then use
        # [edge_k, edge_{k+1}) so that dt==edge_{k+1} falls into bin k.
        h, _ = np.histogram(dt - 1, bins=edges)
        hist += h.astype(np.int64)

    centers_ps = (edges[:-1] + edges[1:]) // 2
    return centers_ps, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ptu', default=None)
    ap.add_argument('--totaltime-ns', type=float, default=100.0,
                    help='Correlation half-window in ns (default 100)')
    ap.add_argument('--timebin-ps', type=float, default=500.0)
    ap.add_argument('--max-records', type=int, default=None,
                    help='Truncate PTU to first N records (debug)')
    ap.add_argument('--out-prefix', default=None)
    ap.add_argument('--xlim-ns', type=float, default=10.0)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    ptu  = args.ptu or os.path.join(here, 'data_ph',
                                    'phenomch21_f009o009_e1_g2_2_500uw.ptu')

    print(f"Reading {ptu}" + (f" (first {args.max_records} records)"
                              if args.max_records else ""))
    raw, tags = read_ptu(ptu, max_records=args.max_records)
    print(f"Records: {len(raw):,}")
    chan, times = parse_pt2(raw)
    n0, n1 = int(np.sum(chan == 0)), int(np.sum(chan == 1))
    print(f"Photons: ch0={n0:,}  ch1={n1:,}  total={len(chan):,}")

    t0 = times[chan == 0]
    t1 = times[chan == 1]

    print(f"g2: window=±{args.totaltime_ns} ns, bin={args.timebin_ps} ps")
    import time as _t
    s = _t.time()
    centers_ps, hist = g2_histogram(t0, t1, args.totaltime_ns, args.timebin_ps)
    print(f"  done in {_t.time()-s:.2f}s  nbins={len(hist)}  total_pairs={int(hist.sum()):,}")

    stem = os.path.splitext(os.path.basename(ptu))[0]
    prefix = args.out_prefix or os.path.join(here, 'data_ph', stem + '_py_g2')
    tplot_ns = centers_ps / 1000.0
    np.savez(prefix + '.npz', tplot_ns=tplot_ns, c=hist)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tplot_ns, hist)
    ax.set_xlabel('time [ns]', fontsize=16)
    ax.set_ylabel('coincident counts', fontsize=16)
    ax.set_xlim(-args.xlim_ns, args.xlim_ns)
    ax.tick_params(labelsize=14)
    fig.tight_layout()
    fig.savefig(prefix + '.png', dpi=150)
    print(f"Saved {prefix}.npz and {prefix}.png")


if __name__ == '__main__':
    main()
