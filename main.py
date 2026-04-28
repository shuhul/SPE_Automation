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

# --- Matplotlib Integration ---
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

# ============================================================================
# 0. HARDWARE INTERFACE
# ============================================================================
EXIT_SUCCESS = 0
EXIT_STOPPED = 1
EXIT_FOCUS_WARNING = 2

# Ensure these imports exist in your environment
import pl_init
import sgd
import pl_spec
import plotter

def init_hardware():
    pl_init.pl_init()
    sgd.sgd_init()
    # pass

def run_scan(xdim, ydim, dx, dy, foldername, current_user, center, scan_type, exposure_time, center_wavelength, stop_event, progress_signal):
    """
    Runs a scan. 
    Returns: Exit Code (0=Success, 1=Stopped, 2=Focus Warning)
    """
    print(f"HARDWARE: Starting scan {foldername} [{scan_type}] at center {center}...")
    print(f"PARAMS: Exp={exposure_time}s, WL={center_wavelength}nm")


    # return 0
    
    # if 'coarse' in scan_type:
    #     classified = np.load(f'data/{foldername}/{scan_type}/classified.npy')
    #     classified[0,2] = 1
    #     classified[2,1] = 1
    #     np.save(f'data/{foldername}/{scan_type}/classified', classified)
    #     return 0
    # elif scan_type == 'fine_x0.5_y-0.5':
    #     classified = np.load(f'data/{foldername}/{scan_type}/classified.npy')
    #     classified[0,2] = 1
    #     classified[2,1] = 1
    #     np.save(f'data/{foldername}/{scan_type}/classified', classified)
    #     return 0
    # elif scan_type == 'fine_x0.5_y-0.5':
    #     classified = np.load(f'data/{foldername}/{scan_type}/classified.npy')
    #     classified[0,2] = 1
    #     classified[2,1] = 1
    #     np.save(f'data/{foldername}/{scan_type}/classified', classified)
    #     return 0
    # else:

    
    exit_code = pl_spec.pl_spec(
        xdim=xdim, ydim=ydim, dx=dx, dy=dy, 
        foldername=foldername, current_user=current_user, 
        center=center, 
        stop_event=stop_event, progress_signal=progress_signal,
        scan_type=scan_type, exposure_time=exposure_time, 
        center_wavelength=center_wavelength
    )

    plotter.save_plot(foldername=foldername, scan_type=scan_type)

    return exit_code
        
    
    # return 0

# ============================================================================
# 1. HELPER CLASSES
# ============================================================================

class PlotPlayer:
    """Manages the interactive cursor and spectrum updates (Blitting)."""
    def __init__(self, canvas, ax_img, ax_spec, data_dict):
        self.canvas = canvas
        self.ax_img = ax_img
        self.ax_spec = ax_spec
        self.intensities = data_dict['intensities']
        self.wl = data_dict['wl']
        self.xs = data_dict['xs']
        self.ys = data_dict['ys']
        self.classified = data_dict.get('classified', None)
        
        # Handle 1x1 case (Long Scan)
        if len(self.xs) > 1:
            self.dx = self.xs[1] - self.xs[0]
        else:
            self.dx = 1.0 # Default width for single pixel

        if len(self.ys) > 1:
            self.dy = self.ys[1] - self.ys[0]
        else:
            self.dy = 1.0

        self.n_rows, self.n_cols = len(self.ys), len(self.xs)
        self.ix, self.iy = 0, 0
        self.bg = None

        # Artists
        self.cursor_rect = patches.Rectangle((0,0), self.dx, self.dy, linewidth=3, 
                                             edgecolor='red', facecolor='none', animated=True)
        self.ax_img.add_patch(self.cursor_rect)
        self.line, = self.ax_spec.plot(self.wl, self.intensities[0,0,:], lw=2, animated=True)
        self.txt_coord = self.ax_spec.text(0.5, 0.95, '', transform=self.ax_spec.transAxes, 
                                           ha='center', animated=True, bbox=dict(facecolor='white', alpha=0.9))
        
        self.cid = self.canvas.mpl_connect('draw_event', self.on_draw)

    def on_draw(self, event):
        if event is not None and event.canvas != self.canvas: return
        self.bg = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self.draw_animated_artists()

    def draw_animated_artists(self):
        if self.iy >= self.n_rows: self.iy = self.n_rows - 1
        if self.ix >= self.n_cols: self.ix = self.n_cols - 1

        spectrum = self.intensities[self.iy, self.ix, :]
        self.line.set_data(self.wl, spectrum)
        
        x_c = self.xs[self.ix] - self.dx/2
        y_c = self.ys[self.iy] - self.dy/2
        self.cursor_rect.set_xy((x_c, y_c))
        
        status = ""
        if self.classified is not None and self.iy < self.classified.shape[0] and self.ix < self.classified.shape[1]:
             if self.classified[self.iy, self.ix] == 1:
                 status = "\n[TAGGED]"
        
        self.txt_coord.set_text(f"x={self.xs[self.ix]:.2f}, y={self.ys[self.iy]:.2f}{status}")

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
            wl = np.load(f'{path}/wl.npy')
            xs = np.load(f'{path}/xs.npy')
            ys = np.load(f'{path}/ys.npy')
            classified = np.load(f'{path}/classified.npy') if os.path.exists(f'{path}/classified.npy') else None
        except Exception as e:
            print(f"Error loading data for {foldername}: {e}")
            return

        self.ax_img.clear()
        self.ax_spec.clear()
        
        summed = np.sum(intensities, axis=-1)
        
        if len(xs) > 1:
            dx = (xs[1]-xs[0])/2 
        else:
            dx = 0.5 # Default for single point

        if len(ys) > 1:
            dy = (ys[1]-ys[0])/2 
        else:
            dy = 0.5

        extent = [xs[0]-dx, xs[-1]+dx, ys[-1]+dy, ys[0]-dy] 

        self.ax_img.imshow(summed, extent=extent, origin='upper', cmap='viridis', aspect='equal')
        self.ax_img.set_title(f"{foldername} - {scan_type}")
        self.ax_img.set_xlabel("X (um)")
        self.ax_img.set_ylabel("Y (um)")
        self.ax_img.set_xlim(extent[0], extent[1])
        self.ax_img.set_ylim(extent[2], extent[3])
        
        if classified is not None:
            iys, ixs = np.where(classified == 1)
            self.ax_img.scatter(xs[ixs], ys[iys], facecolors='none', edgecolors='white', s=150, alpha=0.9, linewidths=2)

        self.ax_spec.set_ylim(np.min(intensities), np.max(intensities)*1.1 + 1)
        self.ax_spec.set_xlim(np.min(wl), np.max(wl))
        
        self.canvas.figure.tight_layout()

        data = {'intensities': intensities, 'wl': wl, 'xs': xs, 'ys': ys, 'classified': classified}
        self.player = PlotPlayer(self.canvas, self.ax_img, self.ax_spec, data)
        self.canvas.draw()

    def on_mouse_move(self, event):
        if self.player and event.inaxes == self.ax_img:
            ix = np.argmin(np.abs(self.player.xs - event.xdata))
            iy = np.argmin(np.abs(self.player.ys - event.ydata))
            self.player.set_pos(ix, iy)

# ============================================================================
# 2. THE WORKER THREAD
# ============================================================================
class ScanWorker(QThread):
    finished = pyqtSignal(int, str, str) # Exit Code, Foldername, Scan_type
    progress_signal = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, params, stop_event):
        super().__init__()
        self.params = params
        self.stop_event = stop_event

    def run(self):
        try:
            xdim = float(self.params.get('xdim', 0))
            ydim = float(self.params.get('ydim', 0))
            dx = float(self.params.get('dx', 1))
            dy = float(self.params.get('dy', 1))
            foldername = self.params['foldername']
            user = self.params['user']
            center = self.params.get('center', (0,0)) 
            scan_type = self.params.get('scantype', 'coarse')
            exposure_time = float(self.params.get('exposuretime', 1))
            center_wavelength = float(self.params.get('centerwavelength', 700))

            exit_code = run_scan(
                xdim, ydim, dx, dy, foldername, user, center, 
                scan_type, exposure_time, center_wavelength, 
                self.stop_event, self.progress_signal
            )
            
            self.finished.emit(exit_code, foldername, scan_type)

        except Exception as e:
            self.error.emit(str(e))

# ============================================================================
# 3. MAIN WINDOW
# ============================================================================
class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPE Detector")
        self.resize(1600, 900)

        self.stop_event = threading.Event()
        self.fine_scan_queue = [] 

        central = QWidget()
        self.setCentralWidget(central)
        self.layout = QHBoxLayout(central)

        self.setup_left_panel()
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs, stretch=1)
        
        init_hardware()

    def setup_left_panel(self):
        self.panel = QWidget()
        self.panel.setFixedWidth(320)
        vbox = QVBoxLayout(self.panel)

        # Settings
        grp_coarse = QGroupBox("1. Coarse Scan Settings")
        f_coarse = QFormLayout()
        self.in_c_x = QLineEdit('5')
        self.in_c_y = QLineEdit('5')
        self.in_c_dx = QLineEdit('0.5')
        self.in_c_dy = QLineEdit('0.5')
        # self.in_folder = QLineEdit(f"20260115-Scan")
        self.in_folder = QLineEdit(f"{datetime.now().strftime('%Y%m%d')}-Scan")
        self.in_user = QLineEdit('shuhul')
        
        f_coarse.addRow("Folder Prefix:", self.in_folder)
        f_coarse.addRow("User:", self.in_user)
        f_coarse.addRow("X Dim (um):", self.in_c_x)
        f_coarse.addRow("Y Dim (um):", self.in_c_y)
        f_coarse.addRow("dX (um):", self.in_c_dx)
        f_coarse.addRow("dY (um):", self.in_c_dy)
        grp_coarse.setLayout(f_coarse)
        vbox.addWidget(grp_coarse)

        grp_fine = QGroupBox("2. Fine Scan Settings")
        f_fine = QFormLayout()
        self.in_f_x = QLineEdit('0.5')
        self.in_f_y = QLineEdit('0.5')
        self.in_f_dx = QLineEdit('0.25')
        self.in_f_dy = QLineEdit('0.25')
        
        f_fine.addRow("Fine X Dim:", self.in_f_x)
        f_fine.addRow("Fine Y Dim:", self.in_f_y)
        f_fine.addRow("Fine dX:", self.in_f_dx)
        f_fine.addRow("Fine dY:", self.in_f_dy)
        grp_fine.setLayout(f_fine)
        vbox.addWidget(grp_fine)

        grp_long = QGroupBox("3. Long Scan Settings")
        f_long = QFormLayout()
        self.in_time = QLineEdit('10')
        self.in_center = QLineEdit('700')
        f_long.addRow("Long Exposure Time:", self.in_time)
        f_long.addRow("Long Center Wavelength:", self.in_center)
        grp_long.setLayout(f_long)
        vbox.addWidget(grp_long)

        # Controls
        self.btn_run = QPushButton("START FULL AUTOMATION")
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 12px;")
        self.btn_run.clicked.connect(self.start_coarse_scan)
        vbox.addWidget(self.btn_run)

        self.btn_stop = QPushButton("STOP SCAN")
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 12px;")
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

    # ========================================================================
    # LOGIC: 1. COARSE SCAN
    # ========================================================================
    def start_coarse_scan(self):
        self.tabs.clear()
        self.stop_event.clear()
        self.fine_scan_queue = []
        
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("Status: Running Coarse Scan...")

        # Coarse and Fine always use default exposure (1s) and center (700nm)
        params = {
            'xdim': self.in_c_x.text(), 'ydim': self.in_c_y.text(),
            'dx': self.in_c_dx.text(), 'dy': self.in_c_dy.text(),
            'foldername': self.in_folder.text(),
            'scantype': 'coarse',
            'user': self.in_user.text(),
            'center': (0,0),
            'exposuretime': 1.0,  # DEFAULT
            'centerwavelength': 700 # DEFAULT
        }

        self.current_tab = ScanTab("Coarse Map")
        self.tabs.addTab(self.current_tab, "Coarse Map")

        self.worker = ScanWorker(params, self.stop_event)
        self.worker.finished.connect(self.on_scan_finished) # Generic handler
        self.worker.error.connect(self.on_error)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.start()

    # ========================================================================
    # LOGIC: 2. GENERIC HANDLER (Router)
    # ========================================================================
    def on_scan_finished(self, exit_code, foldername, scan_type):
        """Called when ANY scan finishes. Decides what to do next."""
        
        if exit_code == EXIT_STOPPED:
            self.lbl_status.setText("Status: Automation Stopped.")
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)
            return
        
        # Load Data into the current tab
        self.current_tab.load_data(foldername, scan_type)
        
        if exit_code == EXIT_FOCUS_WARNING:
             reply = QMessageBox.question(self, "Focus Warning", "Focus lost. Continue?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
             if reply == QMessageBox.StandardButton.No:
                 self.lbl_status.setText("Status: Aborted.")
                 self.btn_run.setEnabled(True)
                 return

        # ROUTING LOGIC based on what just finished
        if scan_type == 'coarse':
            self.handle_coarse_result(foldername)
        elif scan_type.startswith('fine'):
            self.handle_fine_result(foldername, scan_type)
        elif scan_type.startswith('long'):
            self.handle_long_result(foldername)

    # ========================================================================
    # LOGIC: 3. ROUTING HANDLERS
    # ========================================================================
    def handle_coarse_result(self, foldername):
        """Coarse finished. Find emitters, populate queue, start Fine Scan loop."""
        try:
            path = f'data/{foldername}/coarse'
            classified = np.load(f'{path}/classified.npy')
            xs = np.load(f'{path}/xs.npy')
            ys = np.load(f'{path}/ys.npy')
            
            iys, ixs = np.where(classified == 1)
            
            if len(ixs) == 0:
                self.lbl_status.setText("Status: Coarse Scan Complete. No Emitters. Stopping.")
                QMessageBox.information(self, "Complete", "Coarse scan finished. No emitters found.")
                self.btn_run.setEnabled(True)
                return

            self.fine_scan_queue = []
            for ix, iy in zip(ixs, iys):
                self.fine_scan_queue.append((xs[ix], ys[iy]))
            
            self.lbl_status.setText(f"Status: Found {len(self.fine_scan_queue)} emitters. Starting fine scans...")
            self.trigger_next_fine_scan()

        except FileNotFoundError:
            print("No coarse data found.")

    def trigger_next_fine_scan(self):
        """Pops queue and starts a Fine Scan."""
        if not self.fine_scan_queue:
            self.lbl_status.setText("Status: All Automation Complete.")
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)
            return

        target_x, target_y = self.fine_scan_queue.pop(0)
        
        self.stop_event.clear()
        self.update_progress(0)
        self.lbl_status.setText(f"Status: Fine Scan at ({target_x:.2f}, {target_y:.2f})...")

        scan_type = f"fine_x{target_x:.1f}_y{target_y:.1f}"
        
        # FINE SCAN always uses Default params (1s, 700nm)
        params = {
            'xdim': self.in_f_x.text(), 'ydim': self.in_f_y.text(),
            'dx': self.in_f_dx.text(), 'dy': self.in_f_dy.text(),
            'foldername': self.in_folder.text(),
            'scantype': scan_type,
            'user': self.in_user.text(),
            'center': (target_x, target_y),
            'exposuretime': 1.0,  # DEFAULT
            'centerwavelength': 700 # DEFAULT
        }

        tab_name = f"Fine ({target_x:.1f}, {target_y:.1f})"
        self.current_tab = ScanTab(tab_name)
        self.tabs.addTab(self.current_tab, tab_name)
        self.tabs.setCurrentWidget(self.current_tab)

        self.worker = ScanWorker(params, self.stop_event)
        self.worker.finished.connect(self.on_scan_finished)
        self.worker.error.connect(self.on_error)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.start()

    def handle_fine_result(self, foldername, fine_scan_type):
        """Fine Scan finished. Check for emitters. If yes -> Long Scan. If no -> Next Fine."""
        try:
            path = f'data/{foldername}/{fine_scan_type}'
            classified = np.load(f'{path}/classified.npy')
            xs = np.load(f'{path}/xs.npy')
            ys = np.load(f'{path}/ys.npy')
            
            iys, ixs = np.where(classified == 1)
            
            if len(ixs) > 0:
                # Emitter Found! Pick the first one.
                ix, iy = ixs[0], iys[0]
                tx, ty = xs[ix], ys[iy]
                
                print(f"Emitter found in fine scan at {tx:.2f}, {ty:.2f}. Starting Long Scan.")
                self.trigger_long_scan(tx, ty, fine_scan_type)
            else:
                # No emitter, skip to next fine scan
                print("No emitter in fine scan. Moving to next.")
                self.trigger_next_fine_scan()

        except FileNotFoundError:
            print(f"Error reading fine scan data: {path}")
            self.trigger_next_fine_scan()

    def trigger_long_scan(self, target_x, target_y, parent_fine_scan_name):
        """Runs a Long Scan at specific coords."""
        
        self.stop_event.clear()
        self.update_progress(0)
        self.lbl_status.setText(f"Status: Long Scan at ({target_x:.2f}, {target_y:.2f})...")

        # Naming convention: long_{fine_scan_name}
        scan_type = f"long_x{target_x:.1f}_y{target_y:.1f}"
        
        # LONG SCAN uses User Inputs
        params = {
            'xdim': 0, 'ydim': 0, # Single point
            'dx': 0, 'dy': 0,
            'foldername': self.in_folder.text(),
            'scantype': scan_type,
            'user': self.in_user.text(),
            'center': (target_x, target_y),
            'exposuretime': self.in_time.text(),        # USER INPUT
            'centerwavelength': self.in_center.text()   # USER INPUT
        }

        # Reuse current tab or create new? Let's create a new tab to keep history clear
        tab_name = f"Long ({target_x:.1f}, {target_y:.1f})"
        self.current_tab = ScanTab(tab_name)
        self.tabs.addTab(self.current_tab, tab_name)
        self.tabs.setCurrentWidget(self.current_tab)

        self.worker = ScanWorker(params, self.stop_event)
        self.worker.finished.connect(self.on_scan_finished)
        self.worker.error.connect(self.on_error)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.start()

    def handle_long_result(self, foldername):
        """Long scan finished. Return to the Fine Scan loop."""
        print("Long scan finished. Returning to queue.")
        self.trigger_next_fine_scan()

    def on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.lbl_status.setText("Status: Error Occurred")
        self.btn_run.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())