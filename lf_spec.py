"""
LightField client — communicates with lf_bridge.py over localhost:27028.
lf_connect() auto-starts the bridge if it isn't running.
Public API is the same as before; the rest of the codebase is unaffected.
"""
import sys
import os
import socket
import json
import subprocess
import time
import numpy as np

PORT          = 27028
_BRIDGE       = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lf_bridge.py')
_PYTHON       = sys.executable


# ── Transport ─────────────────────────────────────────────────────────────────

def _send(cmd, timeout=65):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(('127.0.0.1', PORT))
    s.sendall((json.dumps(cmd) + '\n').encode())
    data = b''
    while b'\n' not in data:
        chunk = s.recv(65536)
        if not chunk:
            break
        data += chunk
    s.close()
    resp = json.loads(data.decode())
    if resp.get('status') != 'ok':
        raise RuntimeError(f"LF bridge error: {resp.get('message', resp)}")
    return resp


def _bridge_alive():
    try:
        _send({'cmd': 'status'}, timeout=2)
        return True
    except Exception:
        return False


# ── Lifecycle ─────────────────────────────────────────────────────────────────


def lf_connect():
    """Ensure the bridge is running. If not, starts it — which also launches LightField.
    Do NOT open LightField manually; always use this function."""
    if _bridge_alive():
        print('LightField bridge already running.')
        return
    print('Starting LightField bridge (will open LightField)...')
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = subprocess.SW_HIDE
    subprocess.Popen(
        [_PYTHON, _BRIDGE],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        startupinfo=si,
    )
    for i in range(60):
        time.sleep(1)
        try:
            resp = _send({'cmd': 'status'}, timeout=2)
            if resp.get('lf_connected'):
                print('LightField bridge ready.')
                return
        except Exception:
            pass
    raise RuntimeError('LightField bridge did not start within 60s.')


def lf_disconnect():
    """No-op — bridge stays running intentionally."""
    pass


def lf_shutdown():
    """Gracefully stop the bridge (calls Dispose on LightField).
    Call this before closing LightField, never the other way around."""
    try:
        _send({'cmd': 'shutdown'}, timeout=5)
        print('LightField bridge shut down.')
    except Exception:
        pass  # bridge already gone


def lf_reconnect():
    """Call after reopening LightField to re-establish the bridge connection."""
    print('Reconnecting bridge to LightField...')
    resp = _send({'cmd': 'reconnect'})
    if resp.get('lf_connected'):
        print('Reconnected.')
    else:
        raise RuntimeError('Bridge could not reconnect to LightField — is it open?')


# ── Settings ──────────────────────────────────────────────────────────────────

def lf_set_exposure(exposure_s):
    _send({'cmd': 'set_exposure', 'value': float(exposure_s)})

def lf_set_center_wavelength(wl_nm):
    _send({'cmd': 'set_center_wavelength', 'value': float(wl_nm)})

def lf_set_grating(grating):
    _send({'cmd': 'set_grating', 'value': grating})

def lf_setup(exposure_s=1, center_wavelength=700, grating=150):
    _send({'cmd': 'setup',
           'exposure_s': exposure_s,
           'center_wavelength': center_wavelength,
           'grating': grating})

def lf_get_wavelengths():
    return np.array(_send({'cmd': 'get_wavelengths'})['wl'])


# ── Acquire ───────────────────────────────────────────────────────────────────

def lf_acquire():
    """Trigger a single acquisition. Returns (intensity, wavelength) as numpy arrays."""
    resp = _send({'cmd': 'acquire'}, timeout=65)
    return np.array(resp['intensity']), np.array(resp['wl'])
