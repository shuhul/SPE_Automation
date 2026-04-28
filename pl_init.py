import subprocess
import matlab.engine
from time import sleep
from tqdm import tqdm

def pl_init(wait_time=8):
    # Path to your MATLAB executable
    matlab_path = r"C:\\Program Files\\MATLAB\\R2025b\\bin\\matlab.exe"

    print('Launching matlab...')

    # Launch MATLAB with desktop and run the shareEngine command
    subprocess.Popen([
        matlab_path,
        "-nosplash",           # Optional: skip splash screen
        "-nodesktop", 
        # "-batch"         
        # "-desktop",  # opens full MATLAB GUI
        "-r", "addpath(fullfile(pwd,'matlab')); matlab.engine.shareEngine('MySharedSession');"
    ],
        creationflags=subprocess.CREATE_NO_WINDOW
    )

    print(f'Connecting to matlab (waiting {wait_time} secs)...')
    for _ in tqdm(range(wait_time), desc="Waiting", unit="sec"):
        sleep(1)
    print("Done waiting!")
    print("Establishing shared session...")

    eng = matlab.engine.connect_matlab('MySharedSession')
    eng.pl_setup(nargout=0)

    print('Done connecting!')




