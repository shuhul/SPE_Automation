import os
import subprocess
import matlab.engine
from time import sleep
from tqdm import tqdm
import _matlab_session

def pl_init(wait_time=10):
    # Try to attach to any already-running session before launching a new one
    sessions = matlab.engine.find_matlab()
    if sessions:
        name = 'MySharedSession' if 'MySharedSession' in sessions else sessions[0]
        try:
            eng = matlab.engine.connect_matlab(name)
            if name != 'MySharedSession':
                _matlab_session.name = name
            eng.addpath(os.path.join(os.getcwd(), 'matlab'), nargout=0)
            eng.pl_setup(nargout=0)
            print(f"Connected to existing MATLAB session ('{name}').")
            print('Done connecting!')
            return
        except Exception:
            pass

    matlab_path = r"C:\\Program Files\\MATLAB\\R2025b\\bin\\matlab.exe"

    sessions_before = set(matlab.engine.find_matlab())

    print('Launching matlab...')

    proc = subprocess.Popen([
        matlab_path,
        "-nosplash",
        "-nodesktop",
        "-r", "matlab.engine.shareEngine('MySharedSession');"
    ])

    print(f'Waiting for MATLAB session (up to {wait_time}s)...')
    session_name = None
    for _ in tqdm(range(wait_time), desc="Waiting", unit="sec"):
        sleep(1)
        rc = proc.poll()
        if rc is not None and rc != 0:
            raise RuntimeError(f"MATLAB process exited with error (exit code {rc})")
        current_sessions = set(matlab.engine.find_matlab())
        if 'MySharedSession' in current_sessions:
            session_name = 'MySharedSession'
            break
        new_sessions = current_sessions - sessions_before
        if new_sessions:
            session_name = next(iter(new_sessions))
            break
    else:
        proc.terminate()
        raise TimeoutError(f"No new MATLAB session appeared within {wait_time}s")

    print(f"Establishing shared session ('{session_name}')...")
    eng = matlab.engine.connect_matlab(session_name)

    if session_name != 'MySharedSession':
        print(f"Session registered as '{session_name}' (not 'MySharedSession'); updating shared state.")
        _matlab_session.name = session_name

    eng.addpath(os.path.join(os.getcwd(), 'matlab'), nargout=0)
    eng.pl_setup(nargout=0)

    print('Done connecting!')




