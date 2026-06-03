import matlab.engine
from time import sleep


print('Starting matlab')

eng = matlab.engine.start_matlab('-desktop')


sleep(3)

print('Setup PL')

eng.Setup_PL(nargout=0)

sleep(3)

print('Testing')

eng.eval("disp('MATLAB is running visibly!')", nargout=0)

sleep(3)