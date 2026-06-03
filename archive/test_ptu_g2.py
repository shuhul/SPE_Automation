import struct
import numpy as np
import matplotlib.pyplot as plt

PTU_FILE       = "data_ph/phenomch21_f009o009_e1_g2_2_500uw.ptu"
WRAPAROUND     = 210698240   # PicoHarp T2 PTU overflow period (confirmed from file)
RESOLUTION_PS  = 4

G2_RANGE_NS      = 100
TIMEBIN_NS       = 0.5
AFTERFLASH_LO_NS = 10
AFTERFLASH_HI_NS = 30

# =============================================================================
# 1. Read PTU Header
# =============================================================================
VAR_TAG_TYPES = {0x2001FFFF, 0x4001FFFF, 0x4002FFFF, 0xFFFFFFFF}

with open(PTU_FILE, 'rb') as f:
    magic = f.read(8)
    assert b'PQTTTR' in magic, "Not a valid PTU file"
    f.read(8)
    while True:
        ident = f.read(32).rstrip(b'\x00').decode('ascii', errors='ignore')
        idx, typ = struct.unpack('<iI', f.read(8))
        vb = f.read(8)
        if typ in VAR_TAG_TYPES:
            f.read(struct.unpack('<q', vb)[0])
        if ident == 'Header_End':
            break
    raw_bytes = f.read()

raw = np.frombuffer(raw_bytes, dtype=np.uint32)
print(f"Read {len(raw):,} TTTR records")

# =============================================================================
# 2. Parse T2 Records
# =============================================================================
channel_field = (raw >> 28) & 0xF
timetag_field =  raw        & 0x0FFFFFFF

overflow_mask        = channel_field == 0xF
cum_overflow         = np.cumsum(np.where(overflow_mask, 1, 0)) - np.where(overflow_mask, 1, 0)
abs_time_ps          = (cum_overflow.astype(np.int64) * WRAPAROUND + timetag_field) * RESOLUTION_PS

photon_mask = ~overflow_mask
channels    = channel_field[photon_mask]
times_ps    = abs_time_ps[photon_mask]

ch0_s = np.sort(times_ps[channels == 0])
ch1_s = np.sort(times_ps[channels == 1])
N1, N2 = len(ch0_s), len(ch1_s)
total_time_ps = times_ps[-1] - times_ps[0]

print(f"  Ch 0: {N1:,} photons | Ch 1: {N2:,} photons")
print(f"  Acquisition span: {total_time_ps/1e12:.3f} s")

# =============================================================================
# 3. G2 — all-pairs, vectorized chunked
# =============================================================================
G2_RANGE_PS  = G2_RANGE_NS * 1000
TIMEBIN_PS   = TIMEBIN_NS  * 1000
I            = int(np.ceil(G2_RANGE_PS / TIMEBIN_PS))
tau_ns       = np.arange(-I, I + 1) * TIMEBIN_NS
bin_edges_ps = (np.arange(-I, I + 2) - 0.5) * TIMEBIN_PS

print("Computing g2 (all-pairs)...")
c_g2  = np.zeros(2 * I + 1, dtype=np.int64)
CHUNK = 200_000

for start in range(0, N1, CHUNK):
    ch0_chunk = ch0_s[start:start + CHUNK]
    lo = np.searchsorted(ch1_s, ch0_chunk - G2_RANGE_PS, side='left')
    hi = np.searchsorted(ch1_s, ch0_chunk + G2_RANGE_PS, side='right')
    counts = hi - lo
    mask = counts > 0
    if not mask.any():
        continue
    lo_f, counts_f, ch0_f = lo[mask], counts[mask], ch0_chunk[mask]
    total = int(counts_f.sum())
    cs = np.concatenate([[0], np.cumsum(counts_f)[:-1]])
    off = np.arange(total, dtype=np.int64) - np.repeat(cs, counts_f)
    diffs = ch1_s[np.repeat(lo_f, counts_f) + off] - np.repeat(ch0_f, counts_f)
    c_g2 += np.histogram(diffs, bins=bin_edges_ps)[0]
    pct = min(start + CHUNK, N1) / N1 * 100
    print(f"\r  {pct:.0f}%", end='', flush=True)

print()

A    = N1 * N2 * TIMEBIN_PS / total_time_ps
g2   = c_g2 / A if A > 0 else c_g2.astype(float)
g2_0 = g2[I]
print(f"  g2(0) = {g2_0:.3f}")

# =============================================================================
# 4. Plot
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5))

for sign in [1, -1]:
    ax.axvspan(sign * AFTERFLASH_LO_NS, sign * AFTERFLASH_HI_NS,
               color='#ffd0d0', alpha=0.7, zorder=1)

ax.plot(tau_ns, g2, color='#aaaaaa', lw=0.8, zorder=2)
central = np.abs(tau_ns) <= AFTERFLASH_LO_NS
ax.plot(tau_ns[central], g2[central], color='steelblue', lw=1.4, zorder=3)

ax.axhline(1.0, color='#888888', ls='--', lw=0.9, label='g²=1')
ax.axhline(0.5, color='red',     ls='-.', lw=1.0, label='g²=0.5')
ax.axvline(0,   color='#cccccc', ls=':',  lw=0.8)

ax.set_xlabel("τ (ns)", fontsize=12)
ax.set_ylabel("g²(τ)",  fontsize=12)
ax.set_title(f"g²(τ)  [g²(0) = {g2_0:.3f}]", fontsize=12)
ax.set_xlim(-100, 100)
ax.legend(framealpha=0.75, fontsize=10)

plt.tight_layout()
plt.savefig("data_ph/ptu_g2_output.png", dpi=150, bbox_inches="tight")
print("Saved to 'data_ph/ptu_g2_output.png'")
plt.close()
