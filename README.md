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

## Setup

**Requirements:**
- Python 3.10+
- MATLAB R2025b with Python engine installed
- Thorlabs Kinesis installed at `C:\Program Files\Thorlabs\Kinesis`
- LightField running with a shared MATLAB session

**Python dependencies:**
```
pip install numpy matplotlib scipy tqdm PyQt6 pythonnet
```

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
