"""
Persistent LightField bridge server.
Keeps a LightField Automation connection alive between Python script runs.
Listens on localhost:27028 for JSON commands.

Start automatically via lf_spec.lf_connect() — do not run manually.
"""
import sys
import os
import json
import socket
import threading
import logging
import time
import atexit
import signal
import queue
from logging.handlers import RotatingFileHandler

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lf_bridge.log')
_handler  = RotatingFileHandler(_LOG_FILE, maxBytes=1_000_000, backupCount=2)
_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
log = logging.getLogger('lf_bridge')
log.setLevel(logging.INFO)
log.addHandler(_handler)

PORT = 27028

# ── LightField imports ────────────────────────────────────────────────────────
_LF_PATH = r"C:\Program Files\Princeton Instruments\LightField"
sys.path.append(_LF_PATH)
sys.path.append(_LF_PATH + r"\AddInViews")

import clr
clr.AddReference('PrincetonInstruments.LightFieldViewV5')
clr.AddReference('PrincetonInstruments.LightField.AutomationV5')
clr.AddReference('PrincetonInstruments.LightFieldAddInSupportServices')

from PrincetonInstruments.LightField.Automation import Automation
from PrincetonInstruments.LightField.AddIns import SpectrometerSettings, CameraSettings
from System.Collections.Generic import List
from System import String

GRATING_150 = '[800nm,150][2][0]'
GRATING_600 = '[500nm,600][1][0]'

# ── State ─────────────────────────────────────────────────────────────────────
_auto              = None
_exp               = None
_connected         = False
_lock              = threading.Lock()   # serialise all LF operations
_connect_lock      = threading.Lock()   # prevent concurrent _lf_connect() calls
_acq_queue         = queue.Queue()      # acquisition requests routed to main thread
_last_acq_done     = 0.0                # time.time() when last acquisition completed
_MIN_ACQ_INTERVAL  = 1.5               # minimum seconds between acquisitions (LF save time)


# ── LightField connection ─────────────────────────────────────────────────────

def _lf_connect():
    global _auto, _exp, _connected
    try:
        _auto = Automation(True, List[String]())
        _exp  = _auto.LightFieldApplication.Experiment
        _connected = True
        log.info(f"Connected to LightField. Experiment: '{_exp.Name}'")
        return True
    except Exception as e:
        _connected = False
        log.error(f"Failed to connect to LightField: {e}")
        return False


def _cleanup():
    """Always call Dispose() on exit so LightField doesn't hang on next startup."""
    global _auto, _connected
    if _auto is not None:
        try:
            _auto.Dispose()
            log.info('LightField Automation disposed cleanly.')
        except Exception as e:
            log.error(f'Error during Dispose: {e}')
        _auto      = None
        _connected = False


atexit.register(_cleanup)


def _signal_handler(sig, frame):
    log.info(f'Bridge received signal {sig}, shutting down cleanly...')
    _cleanup()
    sys.exit(0)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT,  _signal_handler)


def _ensure_connected():
    if _connected:
        return True
    with _connect_lock:
        if _connected:   # re-check after acquiring lock
            return True
        return _lf_connect()


# ── Acquire ───────────────────────────────────────────────────────────────────

def _do_acquire_main_thread():
    """Called only from the main thread. Returns (data_list, wl_list) or raises."""
    global _exp, _last_acq_done

    # Enforce minimum interval since last acquisition so LightField can finish
    # saving the previous frame before we call Acquire() again.
    wait = _MIN_ACQ_INTERVAL - (time.time() - _last_acq_done)
    if wait > 0:
        time.sleep(wait)

    # Refresh _exp reference
    _exp = _auto.LightFieldApplication.Experiment

    result = {}
    done   = threading.Event()
    acq_ex = [None]

    def on_data(sender, args):
        try:
            frame          = args.ImageDataSet.GetFrame(0, 0)
            result['data'] = list(frame.GetData())
        except Exception as e:
            result['error'] = str(e)
        done.set()

    def _fire_acquire():
        try:
            _exp.ImageDataSetReceived += on_data
            _exp.Acquire()
        except Exception as e:
            acq_ex[0] = e
            done.set()

    threading.Thread(target=_fire_acquire, daemon=True).start()

    fired = done.wait(timeout=60)
    try:
        _exp.ImageDataSetReceived -= on_data
    except Exception:
        pass

    if acq_ex[0] is not None:
        raise acq_ex[0]
    if not fired:
        raise TimeoutError('Acquisition timed out after 60s')
    if 'error' in result:
        raise RuntimeError(result['error'])

    _last_acq_done = time.time()
    return result['data'], list(_exp.SystemColumnCalibration)


def _acquire():
    """Called from client handler threads. Delegates to main thread via queue."""
    resp_q = queue.Queue()
    _acq_queue.put(resp_q)
    result = resp_q.get(timeout=65)
    if isinstance(result, Exception):
        raise result
    return result


# ── Command dispatch ──────────────────────────────────────────────────────────

def _handle(cmd):
    global _connected
    name = cmd.get('cmd')

    if name == 'status':
        return {'status': 'ok', 'lf_connected': _connected}

    if name == 'shutdown':
        log.info('Shutdown command received.')
        threading.Thread(target=lambda: (time.sleep(0.2), _cleanup(), os._exit(0)), daemon=True).start()
        return {'status': 'ok', 'message': 'Bridge shutting down.'}

    if name == 'reconnect':
        ok = _lf_connect()
        return {'status': 'ok' if ok else 'error', 'lf_connected': ok}

    if not _ensure_connected():
        return {'status': 'error', 'message': 'LightField not connected. Reopen LightField then call lf_reconnect().'}

    try:
        if name == 'acquire':
            intensity, wl = _acquire()
            return {'status': 'ok', 'intensity': intensity, 'wl': wl}

        elif name == 'get_wavelengths':
            return {'status': 'ok', 'wl': list(_exp.SystemColumnCalibration)}

        elif name == 'set_exposure':
            _exp.SetValue(CameraSettings.ShutterTimingExposureTime, float(cmd['value']) * 1000)
            return {'status': 'ok'}

        elif name == 'set_center_wavelength':
            _exp.SetValue(SpectrometerSettings.GratingCenterWavelength, float(cmd['value']))
            return {'status': 'ok'}

        elif name == 'set_grating':
            g = cmd['value']
            if g == 150: g = GRATING_150
            elif g == 600: g = GRATING_600
            _exp.SetValue(SpectrometerSettings.GratingSelected, g)
            return {'status': 'ok'}

        elif name == 'setup':
            _exp.SetValue(CameraSettings.ShutterTimingExposureTime,
                          float(cmd.get('exposure_s', 1)) * 1000)
            _exp.SetValue(SpectrometerSettings.GratingCenterWavelength,
                          float(cmd.get('center_wavelength', 700)))
            g = cmd.get('grating', 150)
            if g == 150: g = GRATING_150
            elif g == 600: g = GRATING_600
            _exp.SetValue(SpectrometerSettings.GratingSelected, g)
            # Wait for LightField to finish any hardware moves (e.g. grating change)
            for _ in range(60):
                try:
                    if _exp.IsReadyToRun:
                        break
                except Exception:
                    pass
                time.sleep(1)
            log.info('Setup complete, LightField ready.')
            return {'status': 'ok'}

        else:
            return {'status': 'error', 'message': f'Unknown command: {name}'}

    except Exception as e:
        _connected = False
        log.error(f'Command "{name}" failed (marking LF disconnected): {e}')
        return {'status': 'error', 'message': str(e)}


# ── TCP server ────────────────────────────────────────────────────────────────

def _handle_client(conn):
    try:
        data = b''
        while b'\n' not in data:
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk
        cmd      = json.loads(data.decode())
        with _lock:
            response = _handle(cmd)
        conn.sendall((json.dumps(response) + '\n').encode())
    except Exception as e:
        log.error(f'Client handler error: {e}')
        try:
            conn.sendall((json.dumps({'status': 'error', 'message': str(e)}) + '\n').encode())
        except Exception:
            pass
    finally:
        conn.close()


def _serve():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', PORT))
    srv.listen(5)
    log.info(f'Bridge listening on localhost:{PORT}')
    while True:
        try:
            conn, _ = srv.accept()
            threading.Thread(target=_handle_client, args=(conn,), daemon=True).start()
        except Exception as e:
            log.error(f'Server accept error: {e}')


if __name__ == '__main__':
    log.info('LightField bridge starting...')
    threading.Thread(target=_serve, daemon=True).start()
    with _connect_lock:
        _lf_connect()
    # Main thread event loop — processes acquisition requests so all
    # LightField Acquire() calls happen on the thread that owns the Automation object.
    while True:
        try:
            resp_q = _acq_queue.get(timeout=0.1)
            try:
                result = _do_acquire_main_thread()
                resp_q.put(result)
            except Exception as e:
                resp_q.put(e)
        except queue.Empty:
            pass
