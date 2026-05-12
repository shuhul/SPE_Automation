import sys
import os
import time
import threading
import numpy as np
from datetime import datetime

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QLineEdit, QPushButton,
                             QLabel, QMessageBox, QGroupBox, QTabWidget)
from PyQt6.QtCore import QThread, pyqtSignal, Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

import pl_init
import sgd
import pl_spec
import plotter
import filter as fil
import picoharp
import g2 as g2mod

EXIT_SUCCESS       = 0
EXIT_STOPPED       = 1
EXIT_FOCUS_WARNING = 2

# ============================================================================
# 0. HARDWARE
# ============================================================================

def init_hardware():
    pl_init.pl_init()
    sgd.sgd_init()
    fil.filter_init()
    fil.filter_on()
    picoharp.ph_init()


def run_scan(xdim, ydim, dx, dy, foldername, current_user, center,
             scan_type, exposure_time, center_wavelength, stop_event, progress_signal):
    exit_code = pl_spec.pl_spec(
        xdim=xdim, ydim=ydim, dx=dx, dy=dy,
        foldername=foldername, current_user=current_user,
        center=center, stop_event=stop_event, progress_signal=progress_signal,
        scan_type=scan_type, exposure_time=exposure_time,
        center_wavelength=center_wavelength
    )
    plotter.save_plot(foldername=foldername, scan_type=scan_type)
    return exit_code

# ============================================================================
# 1. HELPERS
# ============================================================================

def find_emission_fwhm_center(spectrum, wl, laser_cutoff_nm=560):
    """Return the FWHM centre of the brightest emission peak above laser_cutoff_nm.
    Falls back to the peak wavelength if FWHM crossings cannot be found."""
    mask = wl > laser_cutoff_nm
    if not mask.any():
        return None
    wl_m = wl[mask]
    sp_m = spectrum[mask]
    peak_idx = int(np.argmax(sp_m))
    half_max = sp_m[peak_idx] / 2.0

    left_below  = np.where(sp_m[:peak_idx] < half_max)[0]
    right_below = np.where(sp_m[peak_idx:] < half_max)[0]

    if left_below.size == 0 or right_below.size == 0:
        return float(wl_m[peak_idx])

    # interpolate left crossing
    li = left_below[-1]
    if li + 1 < len(wl_m):
        x0, x1 = wl_m[li], wl_m[li + 1]
        y0, y1 = sp_m[li], sp_m[li + 1]
        left_wl = x0 + (half_max - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else (x0 + x1) / 2
    else:
        left_wl = wl_m[li]

    # interpolate right crossing
    ri = peak_idx + right_below[0]
    if ri > 0:
        x0, x1 = wl_m[ri - 1], wl_m[ri]
        y0, y1 = sp_m[ri - 1], sp_m[ri]
        right_wl = x0 + (half_max - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else (x0 + x1) / 2
    else:
        right_wl = wl_m[ri]

    return float((left_wl + right_wl) / 2.0)


def angle_for_wavelength(cal_folder, target_wl):
    """Look up the rotation stage angle for target_wl (nm) from the calibration table."""
    table_path = os.path.join('calibration', cal_folder, 'calibration_table.npy')
    if not os.path.exists(table_path):
        return None
    table = np.load(table_path)        # shape (N, 2): [angle, wavelength]
    valid = ~np.isnan(table[:, 1])
    if not valid.any():
        return None
    angles = table[valid, 0]
    wls    = table[valid, 1]
    return float(angles[np.argmin(np.abs(wls - target_wl))])


def _bandpass_slope(cal_folder, target_wl, angle_window=30.0):
    """Estimate dangle/dwl (degrees per nm) near target_wl from the calibration table.
    Filters to entries within angle_window degrees of the expected angle to exclude
    spurious calibration points (e.g. stray laser reflections at wrong angles)."""
    table_path = os.path.join('calibration', cal_folder, 'calibration_table.npy')
    if not os.path.exists(table_path):
        return None
    table = np.load(table_path)
    valid = ~np.isnan(table[:, 1])
    if valid.sum() < 2:
        return None
    angles = table[valid, 0]
    wls    = table[valid, 1]

    # find the expected angle for this wavelength, then restrict to entries near it
    expected_angle = float(angles[np.argmin(np.abs(wls - target_wl))])
    angle_diff = np.abs(((angles - expected_angle) + 180) % 360 - 180)
    clean = angle_diff <= angle_window
    if clean.sum() < 2:
        return None
    angles, wls = angles[clean], wls[clean]

    order  = np.argsort(wls)
    angles, wls = angles[order], wls[order]
    idx = int(np.argmin(np.abs(wls - target_wl)))
    i0  = max(0, idx - 1)
    i1  = min(len(wls) - 1, idx + 1)
    dwl = wls[i1] - wls[i0]
    if dwl == 0:
        return None
    raw_dangle = angles[i1] - angles[i0]
    dangle = (raw_dangle + 180) % 360 - 180
    return float(dangle / dwl)


def run_bandpass_setup(target_wl, cal_folder, current_user,
                       tolerance_nm=2.0, max_attempts=3):
    """Flip filter in, rotate to target_wl, verify FWHM centre with proportional feedback.

    Returns True if aligned within tolerance.
    Returns False after max_attempts — sends Telegram notification and flips filter back up.
    """
    angle = angle_for_wavelength(cal_folder, target_wl)
    if angle is None:
        print(f"No calibration data for {target_wl:.1f} nm — skipping filter setup.")
        return False

    slope = _bandpass_slope(cal_folder, target_wl)  # dangle/dwl, may be None

    if pl_spec.engine is None:
        pl_spec.connect_matlab()
    fil.flip_up()

    for attempt in range(max_attempts):
        print(f"Bandpass attempt {attempt + 1}/{max_attempts}: moving to {angle:.2f}°...")
        fil.rotation_move(angle)

        intensity, wl = pl_spec.pl_single_scan()
        measured_wl = find_emission_fwhm_center(np.array(intensity).flatten(),
                                                np.array(wl).flatten())
        if measured_wl is None:
            print("  Could not detect emission peak in filtered spectrum.")
            break

        error = target_wl - measured_wl
        print(f"  Target: {target_wl:.1f} nm  Measured: {measured_wl:.1f} nm  Error: {error:+.1f} nm")

        if abs(error) <= tolerance_nm:
            print("  Bandpass aligned.")
            return True

        if slope is not None:
            correction = error * slope
            angle += correction
            print(f"  Correction: {correction:+.2f} deg  new angle: {angle:.2f} deg")
        else:
            print("  No slope data — cannot correct angle.")
            break

    pl_spec.send_telegram_message(current_user,
        f"WARNING: Bandpass could not align to {target_wl:.1f} nm after {max_attempts} attempts.")
    fil.flip_down()
    return False


def latest_calibration_folder():
    """Return the most recently created subfolder in calibration/, or ''."""
    cal_dir = 'calibration'
    if not os.path.isdir(cal_dir):
        return ''
    folders = sorted([
        f for f in os.listdir(cal_dir)
        if os.path.isdir(os.path.join(cal_dir, f))
    ])
    return folders[-1] if folders else ''

# ============================================================================
# 2. PLOT WIDGET
# ============================================================================

class PlotPlayer:
    def __init__(self, canvas, ax_img, ax_spec, data_dict):
        self.canvas = canvas
        self.ax_img = ax_img
        self.ax_spec = ax_spec
        self.intensities = data_dict['intensities']
        self.wl = data_dict['wl']
        self.xs = data_dict['xs']
        self.ys = data_dict['ys']
        self.classified = data_dict.get('classified', None)

        self.dx = self.xs[1] - self.xs[0] if len(self.xs) > 1 else 1.0
        self.dy = self.ys[1] - self.ys[0] if len(self.ys) > 1 else 1.0
        self.n_rows, self.n_cols = len(self.ys), len(self.xs)
        self.ix, self.iy = 0, 0
        self.bg = None

        self.cursor_rect = patches.Rectangle(
            (0, 0), self.dx, self.dy, linewidth=3,
            edgecolor='red', facecolor='none', animated=True)
        self.ax_img.add_patch(self.cursor_rect)
        self.line, = self.ax_spec.plot(self.wl, self.intensities[0, 0, :], lw=2, animated=True)
        self.txt_coord = self.ax_spec.text(
            0.5, 0.95, '', transform=self.ax_spec.transAxes,
            ha='center', animated=True, bbox=dict(facecolor='white', alpha=0.9))
        self.cid = self.canvas.mpl_connect('draw_event', self.on_draw)

    def on_draw(self, event):
        if event is not None and event.canvas != self.canvas:
            return
        self.bg = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self.draw_animated_artists()

    def draw_animated_artists(self):
        self.iy = min(self.iy, self.n_rows - 1)
        self.ix = min(self.ix, self.n_cols - 1)
        spectrum = self.intensities[self.iy, self.ix, :]
        self.line.set_data(self.wl, spectrum)
        self.cursor_rect.set_xy((self.xs[self.ix] - self.dx / 2,
                                 self.ys[self.iy] - self.dy / 2))
        status = ''
        if (self.classified is not None
                and self.iy < self.classified.shape[0]
                and self.ix < self.classified.shape[1]
                and self.classified[self.iy, self.ix] == 1):
            status = '\n[TAGGED]'
        self.txt_coord.set_text(
            f"x={self.xs[self.ix]:.2f}, y={self.ys[self.iy]:.2f}{status}")
        self.ax_img.draw_artist(self.cursor_rect)
        self.ax_spec.draw_artist(self.line)
        self.ax_spec.draw_artist(self.txt_coord)

    def set_pos(self, ix, iy):
        if ix != self.ix or iy != self.iy:
            self.ix = max(0, min(self.n_cols - 1, ix))
            self.iy = max(0, min(self.n_rows - 1, iy))
            if self.bg:
                self.canvas.restore_region(self.bg)
                self.draw_animated_artists()
                self.canvas.blit(self.canvas.figure.bbox)
                self.canvas.flush_events()
            else:
                self.canvas.draw_idle()


class ScanTab(QWidget):
    def __init__(self, title):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(Figure(figsize=(10, 6)))
        self.ax_img, self.ax_spec = self.canvas.figure.subplots(1, 2)
        self.layout.addWidget(self.canvas)
        self.player = None
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def load_data(self, foldername, scan_type):
        path = f'data/{foldername}/{scan_type}'
        try:
            intensities = np.load(f'{path}/out.npy')
            wl          = np.load(f'{path}/wl.npy')
            xs          = np.load(f'{path}/xs.npy')
            ys          = np.load(f'{path}/ys.npy')
            classified  = (np.load(f'{path}/classified.npy')
                           if os.path.exists(f'{path}/classified.npy') else None)
        except Exception as e:
            print(f"Error loading data for {foldername}/{scan_type}: {e}")
            return

        self.ax_img.clear()
        self.ax_spec.clear()

        summed = np.sum(intensities, axis=-1)
        dx = (xs[1] - xs[0]) / 2 if len(xs) > 1 else 0.5
        dy = (ys[1] - ys[0]) / 2 if len(ys) > 1 else 0.5
        extent = [xs[0] - dx, xs[-1] + dx, ys[-1] + dy, ys[0] - dy]

        self.ax_img.imshow(summed, extent=extent, origin='upper',
                           cmap='viridis', aspect='equal')
        self.ax_img.set_title(f"{foldername} — {scan_type}")
        self.ax_img.set_xlabel("X (µm)")
        self.ax_img.set_ylabel("Y (µm)")
        self.ax_img.set_xlim(extent[0], extent[1])
        self.ax_img.set_ylim(extent[2], extent[3])

        if classified is not None:
            iys, ixs = np.where(classified == 1)
            self.ax_img.scatter(xs[ixs], ys[iys], facecolors='none',
                                edgecolors='white', s=150, alpha=0.9, linewidths=2)

        self.ax_spec.set_ylim(np.min(intensities), np.max(intensities) * 1.1 + 1)
        self.ax_spec.set_xlim(np.min(wl), np.max(wl))
        self.canvas.figure.tight_layout()

        data = {'intensities': intensities, 'wl': wl, 'xs': xs,
                'ys': ys, 'classified': classified}
        self.player = PlotPlayer(self.canvas, self.ax_img, self.ax_spec, data)
        self.canvas.draw()

    def on_mouse_move(self, event):
        if self.player and event.inaxes == self.ax_img:
            ix = np.argmin(np.abs(self.player.xs - event.xdata))
            iy = np.argmin(np.abs(self.player.ys - event.ydata))
            self.player.set_pos(ix, iy)

# ============================================================================
# 3. WORKER THREADS
# ============================================================================

class ScanWorker(QThread):
    finished        = pyqtSignal(int, str, str)  # exit_code, foldername, scan_type
    progress_signal = pyqtSignal(int)
    error           = pyqtSignal(str)

    def __init__(self, params, stop_event):
        super().__init__()
        self.params     = params
        self.stop_event = stop_event

    def run(self):
        try:
            exit_code = run_scan(
                xdim             = float(self.params.get('xdim', 0)),
                ydim             = float(self.params.get('ydim', 0)),
                dx               = float(self.params.get('dx', 1)),
                dy               = float(self.params.get('dy', 1)),
                foldername       = self.params['foldername'],
                current_user     = self.params['user'],
                center           = self.params.get('center', (0, 0)),
                scan_type        = self.params.get('scantype', 'coarse'),
                exposure_time    = float(self.params.get('exposuretime', 1)),
                center_wavelength= float(self.params.get('centerwavelength', 700)),
                stop_event       = self.stop_event,
                progress_signal  = self.progress_signal,
            )
            self.finished.emit(exit_code, self.params['foldername'],
                               self.params['scantype'])
        except Exception as e:
            self.error.emit(str(e))


class G2Worker(QThread):
    finished        = pyqtSignal(str)   # path to saved .npz
    progress_signal = pyqtSignal(int)
    error           = pyqtSignal(str)

    def __init__(self, target_records, out_folder):
        super().__init__()
        self.target_records = target_records
        self.out_folder     = out_folder

    def run(self):
        try:
            npz_path = picoharp.ph_acquire(
                target_records  = self.target_records,
                out_folder      = self.out_folder,
                progress_signal = self.progress_signal,
            )
            if npz_path:
                self.finished.emit(npz_path)
            else:
                self.error.emit("G2 acquisition returned no data.")
        except Exception as e:
            self.error.emit(str(e))

# ============================================================================
# 4. MAIN WINDOW
# ============================================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPE Detector")
        self.resize(1600, 900)

        self.stop_event      = threading.Event()
        self.fine_scan_queue = []

        central = QWidget()
        self.setCentralWidget(central)
        self.layout = QHBoxLayout(central)

        self.setup_left_panel()
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs, stretch=1)

        init_hardware()

    # -------------------------------------------------------------------------
    # GUI
    # -------------------------------------------------------------------------
    def setup_left_panel(self):
        self.panel = QWidget()
        self.panel.setFixedWidth(340)
        vbox = QVBoxLayout(self.panel)

        # 1. Coarse scan
        grp = QGroupBox("1. Coarse Scan")
        f   = QFormLayout()
        self.in_folder = QLineEdit(datetime.now().strftime('%Y%m%d') + '-Scan')
        self.in_user   = QLineEdit('shuhul')
        self.in_c_x    = QLineEdit('5')
        self.in_c_y    = QLineEdit('5')
        self.in_c_dx   = QLineEdit('0.5')
        self.in_c_dy   = QLineEdit('0.5')
        f.addRow("Folder Prefix:", self.in_folder)
        f.addRow("User:",          self.in_user)
        f.addRow("X Dim (µm):",    self.in_c_x)
        f.addRow("Y Dim (µm):",    self.in_c_y)
        f.addRow("dX (µm):",       self.in_c_dx)
        f.addRow("dY (µm):",       self.in_c_dy)
        grp.setLayout(f); vbox.addWidget(grp)

        # 2. Fine scan
        grp = QGroupBox("2. Fine Scan")
        f   = QFormLayout()
        self.in_f_x  = QLineEdit('0.5')
        self.in_f_y  = QLineEdit('0.5')
        self.in_f_dx = QLineEdit('0.25')
        self.in_f_dy = QLineEdit('0.25')
        f.addRow("X Dim (µm):", self.in_f_x)
        f.addRow("Y Dim (µm):", self.in_f_y)
        f.addRow("dX (µm):",    self.in_f_dx)
        f.addRow("dY (µm):",    self.in_f_dy)
        grp.setLayout(f); vbox.addWidget(grp)

        # 3. Long scan
        grp = QGroupBox("3. Long Scan")
        f   = QFormLayout()
        self.in_time   = QLineEdit('10')
        self.in_center = QLineEdit('700')
        f.addRow("Exposure (s):",       self.in_time)
        f.addRow("Center WL (nm):",     self.in_center)
        grp.setLayout(f); vbox.addWidget(grp)

        # 4. G² settings
        grp = QGroupBox("4. G² Settings")
        f   = QFormLayout()
        self.in_records = QLineEdit('100000')
        self.in_cal     = QLineEdit(latest_calibration_folder())
        f.addRow("Target Records:", self.in_records)
        f.addRow("Cal. Folder:",    self.in_cal)
        grp.setLayout(f); vbox.addWidget(grp)

        # Controls
        self.btn_run = QPushButton("START FULL AUTOMATION")
        self.btn_run.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 12px;")
        self.btn_run.clicked.connect(self.start_coarse_scan)
        vbox.addWidget(self.btn_run)

        self.btn_stop = QPushButton("STOP SCAN")
        self.btn_stop.setStyleSheet(
            "background-color: #f44336; color: white; font-weight: bold; padding: 12px;")
        self.btn_stop.clicked.connect(self.request_stop)
        self.btn_stop.setEnabled(False)
        vbox.addWidget(self.btn_stop)

        self.lbl_status = QLabel("Status: Ready")
        self.lbl_status.setWordWrap(True)
        vbox.addWidget(self.lbl_status)
        vbox.addStretch()
        self.layout.addWidget(self.panel)

    def request_stop(self):
        self.stop_event.set()
        self.lbl_status.setText("Status: Stop Requested...")

    def update_progress(self, value):
        self.lbl_status.setText(f"Status: Progress {value}%")

    # -------------------------------------------------------------------------
    # SCAN PIPELINE
    # -------------------------------------------------------------------------
    def start_coarse_scan(self):
        self.tabs.clear()
        self.stop_event.clear()
        self.fine_scan_queue = []
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("Status: Running Coarse Scan...")

        params = {
            'xdim': self.in_c_x.text(), 'ydim': self.in_c_y.text(),
            'dx':   self.in_c_dx.text(), 'dy':  self.in_c_dy.text(),
            'foldername': self.in_folder.text(),
            'scantype':   'coarse',
            'user':        self.in_user.text(),
            'center':      (0, 0),
            'exposuretime':    1.0,
            'centerwavelength': 700,
        }
        self.current_tab = ScanTab("Coarse Map")
        self.tabs.addTab(self.current_tab, "Coarse Map")
        self._start_scan_worker(params)

    def _start_scan_worker(self, params):
        self.worker = ScanWorker(params, self.stop_event)
        self.worker.finished.connect(self.on_scan_finished)
        self.worker.error.connect(self.on_error)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.start()

    def on_scan_finished(self, exit_code, foldername, scan_type):
        if exit_code == EXIT_STOPPED:
            self.lbl_status.setText("Status: Stopped.")
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)
            return

        self.current_tab.load_data(foldername, scan_type)

        if exit_code == EXIT_FOCUS_WARNING:
            reply = QMessageBox.question(
                self, "Focus Warning", "Focus was lost. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                self.lbl_status.setText("Status: Aborted.")
                self.btn_run.setEnabled(True)
                return

        if scan_type == 'coarse':
            self.handle_coarse_result(foldername)
        elif scan_type.startswith('fine'):
            self.handle_fine_result(foldername, scan_type)
        elif scan_type.startswith('long'):
            self.handle_long_result(foldername, scan_type)

    # -------------------------------------------------------------------------
    # COARSE
    # -------------------------------------------------------------------------
    def handle_coarse_result(self, foldername):
        try:
            classified = np.load(f'data/{foldername}/coarse/classified.npy')
            xs         = np.load(f'data/{foldername}/coarse/xs.npy')
            ys         = np.load(f'data/{foldername}/coarse/ys.npy')
        except FileNotFoundError:
            print("No coarse classified data found.")
            return

        iys, ixs = np.where(classified == 1)
        if len(ixs) == 0:
            self.lbl_status.setText("Status: Coarse complete. No emitters found.")
            QMessageBox.information(self, "Complete", "No emitters found in coarse scan.")
            self.btn_run.setEnabled(True)
            return

        self.fine_scan_queue = [(xs[ix], ys[iy]) for ix, iy in zip(ixs, iys)]
        self.lbl_status.setText(
            f"Status: Found {len(self.fine_scan_queue)} emitter(s). Starting fine scans...")
        self.trigger_next_fine_scan()

    # -------------------------------------------------------------------------
    # FINE
    # -------------------------------------------------------------------------
    def trigger_next_fine_scan(self):
        if not self.fine_scan_queue:
            self.lbl_status.setText("Status: All automation complete.")
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)
            return

        tx, ty = self.fine_scan_queue.pop(0)
        self.stop_event.clear()
        scan_type = f"fine_x{tx:.1f}_y{ty:.1f}"
        self.lbl_status.setText(f"Status: Fine scan at ({tx:.2f}, {ty:.2f})...")

        params = {
            'xdim': self.in_f_x.text(), 'ydim': self.in_f_y.text(),
            'dx':   self.in_f_dx.text(), 'dy':  self.in_f_dy.text(),
            'foldername': self.in_folder.text(),
            'scantype':   scan_type,
            'user':        self.in_user.text(),
            'center':      (tx, ty),
            'exposuretime':    1.0,
            'centerwavelength': 700,
        }
        tab_name = f"Fine ({tx:.1f}, {ty:.1f})"
        self.current_tab = ScanTab(tab_name)
        self.tabs.addTab(self.current_tab, tab_name)
        self.tabs.setCurrentWidget(self.current_tab)
        self._start_scan_worker(params)

    def handle_fine_result(self, foldername, scan_type):
        try:
            classified = np.load(f'data/{foldername}/{scan_type}/classified.npy')
            xs         = np.load(f'data/{foldername}/{scan_type}/xs.npy')
            ys         = np.load(f'data/{foldername}/{scan_type}/ys.npy')
        except FileNotFoundError:
            self.trigger_next_fine_scan()
            return

        iys, ixs = np.where(classified == 1)
        if len(ixs) == 0:
            self.trigger_next_fine_scan()
            return

        tx, ty = xs[ixs[0]], ys[iys[0]]
        self.trigger_long_scan(tx, ty)

    # -------------------------------------------------------------------------
    # LONG
    # -------------------------------------------------------------------------
    def trigger_long_scan(self, target_x, target_y):
        self.stop_event.clear()
        scan_type = f"long_x{target_x:.1f}_y{target_y:.1f}"
        self.lbl_status.setText(f"Status: Long scan at ({target_x:.2f}, {target_y:.2f})...")

        params = {
            'xdim': 0, 'ydim': 0, 'dx': 0, 'dy': 0,
            'foldername':  self.in_folder.text(),
            'scantype':    scan_type,
            'user':         self.in_user.text(),
            'center':       (target_x, target_y),
            'exposuretime':     self.in_time.text(),
            'centerwavelength': self.in_center.text(),
        }
        tab_name = f"Long ({target_x:.1f}, {target_y:.1f})"
        self.current_tab = ScanTab(tab_name)
        self.tabs.addTab(self.current_tab, tab_name)
        self.tabs.setCurrentWidget(self.current_tab)
        self._start_scan_worker(params)

    def handle_long_result(self, foldername, scan_type):
        """Long scan done — extract ZPL, set up bandpass filter, then run g2."""
        path = f'data/{foldername}/{scan_type}'
        try:
            out = np.load(f'{path}/out.npy')
            wl  = np.load(f'{path}/wl.npy')
        except FileNotFoundError:
            print("Long scan data not found — skipping filter setup.")
            self.trigger_g2()
            return

        target_wl = find_emission_fwhm_center(out[0, 0, :], wl)
        if target_wl is None:
            print("Could not find emission peak — skipping filter setup.")
            self.trigger_g2()
            return

        print(f"ZPL FWHM centre: {target_wl:.1f} nm")
        cal_folder = self.in_cal.text().strip()

        aligned = run_bandpass_setup(
            target_wl    = target_wl,
            cal_folder   = cal_folder,
            current_user = self.in_user.text(),
        )

        if aligned:
            self.trigger_g2()
        else:
            self.trigger_next_fine_scan()

    # -------------------------------------------------------------------------
    # G2
    # -------------------------------------------------------------------------
    def trigger_g2(self):
        self.lbl_status.setText("Status: Running g² acquisition...")
        try:
            target_records = int(self.in_records.text())
        except ValueError:
            target_records = 100000

        self.g2_worker = G2Worker(target_records, out_folder='g2_data')
        self.g2_worker.finished.connect(self.on_g2_finished)
        self.g2_worker.error.connect(self.on_error)
        self.g2_worker.progress_signal.connect(self.update_progress)
        self.g2_worker.start()

    def on_g2_finished(self, npz_path):
        self.lbl_status.setText("Status: Analysing g²...")
        try:
            result = g2mod.run(npz_path, out_folder='g2_data')
            if result['popt'] is not None:
                g2_0 = 1 - result['popt'][1]  # model at tau=0
                self.lbl_status.setText(f"Status: g²(0) = {g2_0:.3f}. Moving to next emitter.")
            else:
                self.lbl_status.setText("Status: g² fit did not converge. Moving to next emitter.")
        except Exception as e:
            print(f"G2 analysis error: {e}")

        fil.flip_down()   # remove filter before next scan
        self.trigger_next_fine_scan()

    # -------------------------------------------------------------------------
    # ERROR
    # -------------------------------------------------------------------------
    def on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.lbl_status.setText("Status: Error occurred.")
        self.btn_run.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
