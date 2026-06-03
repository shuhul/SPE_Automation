import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Load Data — most recent .npz from data_ph/
# =============================================================================
npz_files = sorted(glob.glob("data_ph/*.npz"))
if not npz_files:
    raise FileNotFoundError("No .npz files found in data_ph/")
INPUT_FILE = npz_files[-1]

npz = np.load(INPUT_FILE)
ch0 = npz['ch0']   # arrival times in ps, sorted
ch1 = npz['ch1']   # arrival times in ps, sorted
print(f"Loaded '{INPUT_FILE}'")

N1, N2 = len(ch0), len(ch1)
t_min_ps     = min(ch0[0] if N1 else 0, ch1[0] if N2 else 0)
t_max_ps     = max(ch0[-1] if N1 else 0, ch1[-1] if N2 else 0)
total_time_ps = t_max_ps - t_min_ps

# =============================================================================
# G2 Calculation
# =============================================================================
G2_RANGE_NS      = 100
TIMEBIN_NS       = 0.5
AFTERFLASH_LO_NS = 10
AFTERFLASH_HI_NS = 30

G2_RANGE_PS = G2_RANGE_NS * 1000
TIMEBIN_PS  = TIMEBIN_NS  * 1000

I       = int(np.ceil(G2_RANGE_PS / TIMEBIN_PS))
tau_ns  = np.arange(-I, I + 1) * TIMEBIN_NS
bin_edges_ps = (np.arange(-I, I + 2) - 0.5) * TIMEBIN_PS

print("Computing g2...")
ch0_s = np.sort(ch0)
ch1_s = np.sort(ch1)

all_diffs = []
for t0 in ch0_s:
    lo = np.searchsorted(ch1_s, t0 - G2_RANGE_PS, side='left')
    hi = np.searchsorted(ch1_s, t0 + G2_RANGE_PS, side='right')
    if lo < hi:
        all_diffs.append(ch1_s[lo:hi] - t0)

if all_diffs:
    c_g2, _ = np.histogram(np.concatenate(all_diffs), bins=bin_edges_ps)
else:
    c_g2 = np.zeros(2 * I + 1, dtype=np.int64)

A    = N1 * N2 * TIMEBIN_PS / total_time_ps
g2   = c_g2 / A if A > 0 else c_g2.astype(float)
g2_0 = g2[I]
print(f"  g2(0) = {g2_0:.3f}")

# =============================================================================
# Plot
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
ax.set_title(f"g²(τ)    [g²(0) = {g2_0:.3f}]", fontsize=12)
ax.set_xlim(-G2_RANGE_NS, G2_RANGE_NS)
ax.legend(framealpha=0.75, fontsize=10)

plt.tight_layout()

out_path = os.path.splitext(INPUT_FILE)[0] + "_display.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to '{out_path}'")
plt.show()
