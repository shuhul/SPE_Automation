import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
from time import sleep


print('Connecting to matlab')
eng = matlab.engine.connect_matlab('MySharedSession')

print('Getting wavelengths')

wl = np.array(eng.workspace['wl']).flatten()
np.save(f'wl', wl)


print('Starting scan...')


# print(wl)
# plt.ion()  # Turn on interactive mode

# fig, ax = plt.subplots()
# line, = ax.plot([], [])  # Initialize an empty line
# ax.set_xlabel('Wavelength (nm)')
# ax.set_ylabel('Intensity')
# ax.set_title('PL Spec')

name = 'testv2_3'
for i in range(60):
    # Get intensity from MATLAB
    intensity = np.array(eng.eval("instance1.acquire;", nargout=1)).flatten()

    np.save(f'focus/{name}_{i}', intensity)

    print(f'Num: {i}')

    # # Update the plot data
    # line.set_xdata(wl)
    # line.set_ydata(intensity)

    # # Adjust axes limits dynamically
    # ax.relim()
    # ax.autoscale_view()

    # plt.pause(0.1)  # Pause to update the figure
    sleep(0.1)

# plt.ioff()  # Turn off interactive mode if you want
# plt.show()  # Show final plot

print('Done')
