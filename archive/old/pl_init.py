import subprocess
import matlab.engine
from time import sleep

# Path to your MATLAB executable
matlab_path = r"C:\\Program Files\\MATLAB\\R2025b\\bin\\matlab.exe"

print('Launching matlab')

# Launch MATLAB with desktop and run the shareEngine command
subprocess.Popen([
    matlab_path,
    "-nosplash",           # Optional: skip splash screen
    "-nodesktop",          
    # "-desktop",  # opens full MATLAB GUI
    "-r", "matlab.engine.shareEngine('MySharedSession');"  # runs command on startup
],
    creationflags=subprocess.CREATE_NO_WINDOW
)

print('Connecting to matlab (waiting 8 secs)')

sleep(8)

eng = matlab.engine.connect_matlab('MySharedSession')
eng.pl_setup(nargout=0)

print('Done!')
