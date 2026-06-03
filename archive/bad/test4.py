import matlab.engine
import time


# Connect to the shared MATLAB session
eng = matlab.engine.connect_matlab('MySharedSession')

# Run a MATLAB script
eng.setup_pl1(nargout=0)

# MATLAB is still open and interactive
