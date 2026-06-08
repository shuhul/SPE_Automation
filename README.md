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