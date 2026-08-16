# SPE Automation

Automated photoluminescence spectroscopy and single-photon emitter characterisation pipeline for a confocal microscope setup.

## Overview

The system scans a sample with a confocal microscope, identifies candidate single-photon emitters via PL spectroscopy, and characterises them with a g² photon-correlation measurement. The full pipeline runs from a PyQt6 GUI (`main.py`) or individual steps can be run manually from the notebook (`main.ipynb`).

### Automation Pipeline

```
Coarse PL Map → Classify Emitters → Fine PL Map (per emitter)
→ Classify → Long Integration Spectrum → Set Bandpass Filter
→ Verify Filter Calibration → g² Measurement → Next Emitter
```

## Project Structure

```
SPE_Automation/
├── main.py               # Full automation GUI
├── main.ipynb            # Manual control notebook
│
├── pl_init.py            # MATLAB / LightField initialisation
├── pl_spec.py            # Spectrometer scanning
├── sgd.py                # XY stage control
├── filter.py             # Bandpass filter flip mount + rotation stage
├── classifier.py         # Emitter classification from PL maps
├── plotter.py            # Heatmap and spectrum visualisation
├── g2.py                 # PTU parsing + g²(τ) eff2 analysis
├── filtercalibration.py  # Filter rotation stage calibration routine
├── verify_calibration.py # Pre-measurement calibration check
│
├── matlab/
│   └── pl_setup.m        # LightField initialisation (called via pl_init)
│
├── calibration/          # Saved calibration tables (angle → wavelength)
├── data/                 # PL scan data
├── g2_data/              # PicoHarp PTU files and g² outputs
└── focus/                # Focus scan data
```

## Hardware

| Device | Interface |
|---|---|
| XY Stage | `sgd.py` (custom) |
| Spectrometer (LightField) | `pl_spec.py` via MATLAB engine |
| Bandpass Filter Flip Mount (MFF101) | `filter.py` via Thorlabs Kinesis |
| Filter Rotation Stage (KDC101) | `filter.py` via Thorlabs Kinesis |
| PicoHarp 300 (g² detector) | PicoHarp software → `.ptu` files → `g2.py` |

## New User Setup

Python and the virtual environment are already installed in shared locations — you do not need to install Python or any packages. Just follow these steps.

### 1. Install VS Code

Download and install from https://code.visualstudio.com if you don't already have it.

### 2. Install the required VS Code extensions

Open VS Code, go to the Extensions panel (`Ctrl+Shift+X`), and install:
- **Python** (by Microsoft)
- **Jupyter** (by Microsoft)

### 3. Open the project folder

**File → Open Folder** → navigate to:
```
C:\Users\Public\Shared Confocal Files\SPE_Automation
```

### 4. Select the kernel in the notebook

1. Open `main.ipynb`
2. Click the kernel picker in the top-right corner (may say "Select Kernel")
3. Choose **Python Environments...**
4. Select **`.venv (Python 3.10.0)`**

You're ready to run cells.

---

### Running the notebook

Run cells top-to-bottom at the start of each session:

1. **Imports cell** — loads all modules
2. **Init cell** — launches MATLAB and connects to the spectrometer. Wait for `Ready for use!` before continuing.
3. **Remaining cells** — set scan parameters and run as needed.

> If MATLAB is already running from a previous session, use the **Reconnect cell** instead of the init cell.

---

### Troubleshooting

**`.venv` kernel not listed** — click **Select Kernel → Python Environments → Find Python Interpreter** and browse to:
```
C:\Users\Public\Shared Confocal Files\SPE_Automation\.venv\Scripts\python.exe
```

**`EngineError: Unable to connect to MATLAB session`** — the previous MATLAB session was lost. Run the init cell to launch a fresh one.

**`ipykernel not found` or the venv is broken** — rebuild it (admin PowerShell):
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& "C:\Users\Public\Shared Confocal Files\SPE_Automation\setup_venv.ps1"
```
This recreates the venv from `C:\Program Files\Python310\` and reinstalls all packages. If Python 3.10 itself is missing, reinstall it first as Administrator:
```powershell
python-3.10.0-amd64.exe /quiet InstallAllUsers=1 PrependPath=1
```

---

## Developer Setup (rebuilding from scratch)

**Requirements:**
- Python 3.10.0 installed for all users at `C:\Program Files\Python310\`
- MATLAB R2025b with Python engine installed
- Thorlabs Kinesis installed at `C:\Program Files\Thorlabs\Kinesis`
- LightField running with a shared MATLAB session

Run `setup_venv.ps1` (as Administrator) to create the venv and install all dependencies from `requirements.txt`.

**Filter calibration** (run once, or when filter is remounted):
```bash
python filtercalibration.py
```
Saves a `calibration_table.npy` to `calibration/<timestamp>/`.

## Usage

### Automation GUI
```bash
python main.py
```
Set scan dimensions, folder name, and long-scan parameters in the left panel, then click **START FULL AUTOMATION**.

### Manual Notebook
```bash
jupyter notebook main.ipynb
```
Run individual cells to control the stage, spectrometer, or filter directly.

### G² Analysis (standalone)
```python
import g2

# Full pipeline: parse PTU, compute g²(τ), save .npz + .png
result = g2.run('g2_data/myfile.ptu')

# Step by step
result = g2.eff2('g2_data/myfile.ptu', g2time_ns=100, timebin_ns=1.0)
g2.plot_g2(result, 'g2_data/myfile.png')
```

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| Coarse step | 0.5 µm | Spatial resolution of coarse map |
| Fine step | 0.25 µm | Spatial resolution of fine map |
| Long exposure | 10 s | Integration time for confirmation spectrum |
| g² window | ±100 ns | Correlation half-window |
| g² bin | 1 ns | Time bin width |
| Filter tolerance | ±2 nm | Acceptable calibration error |

#Install the latest PowerShell for new features and improvements! https://aka.ms/PSWindows

#PS C:\WINDOWS\system32> cd "C:\Users\Public\Shared Confocal Files\SPE_Automation"
#PS C:\Users\Public\Shared Confocal Files\SPE_Automation> claude --dangerously-skip-permissions



# emitter_analysis.py — README

## What this script does

Scans a folder of automated hBN SPE characterization runs, and for every
emitter that has both raw photon-correlation data (`g2_*` folder) and a
matching spectrum (`long_*` folder), it:

1. Recomputes **g²(τ)** directly from the raw photon timestamps (not from
   any pre-processed file) using a ±400 ns correlation window.
2. Fits the g²(τ) curve to a two-exponential model (antibunching + one
   metastable/shelving timescale), with the baseline fixed at g₀ = 1.
3. Fits the emitter's **ZPL** (zero-phonon line) in the filtered spectrum
   with a Gaussian, extracting wavelength, FWHM, and integrated intensity.
4. Applies quality-control gates (below) so junk fits don't leak into the
   summary or plots.
5. Outputs a CSV of every valid emitter, plus two summary figures.

Run it with:

```
python emitter_analysis.py --data-dir data --out-dir analysis_output
```

Optional flags: `--verbose` prints why each PSB (Debye-Waller) fit was
skipped, per emitter.

---

## Expected data layout

```
data/
  <run folder>/                      e.g. 20260727-PLSPC-HT-Ch5-f16-100uW-1s-fullauto-2
    g2_x<coord>_y<coord>/
      <raw>.npz                      must contain ch0, ch1 (int64 ps timestamps)
    long_x<coord>_y<coord>/
      wl.npy                         wavelength axis
      out.npy                        spectrum, indexed out[0, 0, :]
```

- Run folder names must match `.*HT.*fullauto.*` (case-insensitive) to be scanned at all.
- `g2_x.._y..` and `long_x.._y..` folders are matched by their shared coordinate suffix.
- Inside each `g2_` folder, the script picks the last raw `.npz` file
  alphabetically that is **not** named `*_processed.npz`.

If either the g2 folder or the matching `long_` folder is missing, that
emitter is silently skipped (not counted as an error).

## Output files

`<out_dir>/emitter_summary.csv` — one row per valid emitter:

| Column | Meaning |
|---|---|
| `run`, `chip`, `field`, `date`, `x`, `y` | Parsed from folder names |
| `ZPL_nm`, `FWHM_nm` | ZPL Gaussian fit results |
| `PSB_nm`, `PSB_FWHM_nm` | Phonon sideband Gaussian fit (if resolved) |
| `DWF` | Debye-Waller factor = ZPL area / (ZPL area + PSB area) |
| `g2_0` | g²(0) from the two-exponential fit |
| `T1_ns` | Antibunching lifetime — SPE-gated, capped at `MAX_T1_NS` |
| `T2_ns` | Metastable timescale — SPE-gated, no cap (see note above) |
| `ZPL_intensity` | Integrated ZPL Gaussian area (a brightness proxy) |
| `rate_kHz` | Average total photon count rate over the full acquisition (both channels), computed directly from the raw timestamps |

`<out_dir>/emitter_correlations.png` — 7-panel scatter grid:
g²(0) vs ZPL, g²(0) vs FWHM, T1 vs ZPL, T1 vs FWHM, **T1 vs emission
rate**, T2 vs ZPL, T2 vs FWHM. Confirmed single emitters (g²(0) < 0.5) are
marked with a star. All points use one uniform color — chip is no longer
color-coded (see Recent changes). Axes that have a physically meaningful
zero (FWHM, g²(0), T1, T2, rate, ZPL intensity) are forced to include the
origin; ZPL wavelength is deliberately left un-forced since 0 nm would be
meaningless clutter on that axis.

`<out_dir>/emitter_histograms.png` — 4-panel distribution histograms:
ZPL, FWHM, DWF, g²(0). Unchanged from earlier versions.


# SPE Automation — README

## What this does
Automates finding and characterizing single-photon emitters: coarse scan
→ fine scan → long spectrum → bandpass filter → g²(τ) measurement, for
every selected emitter/spot. Runs mostly unattended, with manual
checkpoints (mirror flips, emitter selection) where a human has to act.

Run: `python spe_automation.py`. Edit the parameters block at the top
before each session — no CLI.

## Rough pipeline (per emitter)
1. Coarse scan → select bright spot(s).
2. Fine scan on each → select spot(s).
3. Per spot: long spectrum → find ZPL → set bandpass filter to it →
   filtered scan for FWHM → g²(τ) via PicoHarp (needs a manual mirror
   flip, prompted via Telegram + terminal).
4. Summary table at the end: coords, ZPL, FWHM, filter angle, g²(0),
   status per spot.

## Controls
- `Ctrl+C` — finish current step, then stop
- `Ctrl+X` / `q` — emergency stop, saves partial data
- `s` — skip current emitter/spot
- `a` (at G2 start) — show live counts before acquiring

Windows only (`msvcrt`) — `s`/`Ctrl+X` won't work elsewhere, `Ctrl+C` still will.

## Needs
- Local modules: `lf_spec`, `sgd`, `filter`, `plotter`, `pl_spec_python`,
  `autofocus`, `picoharp`, `g2`.
- Calibration table at `calibration/<CAL_FOLDER>/calibration_table.npy`
  for the filter step (skipped, not fatal, if missing).
- PicoHarp/autofocus are optional — script warns and continues without
  them if init fails.
- `MANUAL_PLOT_INTERACTION = True` blocks on emitter selection plots.