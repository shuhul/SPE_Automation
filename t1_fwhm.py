"""
T1_fwhm_fit.py — isolate the T1-vs-FWHM relationship near the origin and
fit an inverse curve (T1 = k / FWHM), consistent with a Fourier
transform-limit-style relationship between lifetime and linewidth.

Reads the emitter_summary.csv produced by emitter_analysis.py (already
has T1_ns SPE-gated — see that script's iter_emitters()).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

CSV_PATH  = 'analysis_output/emitter_summary.csv'
OUT_PATH  = 'analysis_output/T1_fwhm_fit.png'

# Near-origin zoom window — tune these to match what you're seeing.
T1_MAX_NS   = 3.0
FWHM_MAX_NM = 9.0


def _inverse_model(fwhm, k):
    return k / fwhm


def run(csv_path=CSV_PATH, out_path=OUT_PATH, t1_max_ns=T1_MAX_NS, fwhm_max_nm=FWHM_MAX_NM):
    df = pd.read_csv(csv_path)
    sub = df[['FWHM_nm', 'T1_ns']].dropna()
    print(f"{len(sub)} emitters with both FWHM and T1 (SPE-gated).")

    zoom = sub[(sub['FWHM_nm'] <= fwhm_max_nm) & (sub['T1_ns'] <= t1_max_ns)]
    print(f"{len(zoom)} of those have FWHM <= {fwhm_max_nm} nm AND T1 <= {t1_max_ns} ns "
          f"(the near-origin subset).")

    if len(zoom) < 3:
        print("Too few points in the zoom window to fit — widen T1_MAX_NS / FWHM_MAX_NM.")
        return

    popt, pcov = curve_fit(_inverse_model, zoom['FWHM_nm'], zoom['T1_ns'],
                            p0=[10.0], bounds=(0, np.inf))
    k = float(popt[0])
    k_err = float(np.sqrt(pcov[0, 0]))

    pred = _inverse_model(zoom['FWHM_nm'], k)
    ss_res = float(np.sum((zoom['T1_ns'] - pred) ** 2))
    ss_tot = float(np.sum((zoom['T1_ns'] - zoom['T1_ns'].mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    print(f"Fit: T1 = k / FWHM   with k = {k:.2f} +/- {k_err:.2f} (ns*nm)   R^2 = {r2:.3f}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(sub['FWHM_nm'], sub['T1_ns'],
               s=20, color='lightgray', edgecolors='k', linewidths=0.3,
               label='all confirmed SPEs')
    ax.scatter(zoom['FWHM_nm'], zoom['T1_ns'],
               s=35, color='steelblue', edgecolors='k', linewidths=0.5,
               label=f'near-origin subset (T\u2081\u2264{t1_max_ns:.0f}ns, FWHM\u2264{fwhm_max_nm:.0f}nm)',
               zorder=3)

    xf = np.linspace(0.5, fwhm_max_nm, 300)
    ax.plot(xf, _inverse_model(xf, k), 'k--', lw=1.5,
            label=f'fit: T\u2081 = {k:.1f} / FWHM   (R\u00b2={r2:.2f})')

    ax.set_xlim(0, fwhm_max_nm)
    ax.set_ylim(0, zoom['T1_ns'].max() * 1.3)
    ax.set_xlabel('ZPL FWHM (nm)', fontsize=12)
    ax.set_ylabel('T\u2081 (ns)  [confirmed SPE only]', fontsize=12)
    ax.set_title('T\u2081 vs FWHM \u2014 near origin, inverse fit', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    return dict(k=k, k_err=k_err, r2=r2)


if __name__ == '__main__':
    run()