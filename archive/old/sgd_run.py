import pyvisa
import time


print('Scanning for intstruments')
resource_str = 'USB0::0xF4ED::0xEE3A::SDG10GAD1R1771::0::INSTR'
rm = pyvisa.ResourceManager()
try:
    obj1 = rm.open_resource(resource_str)
except pyvisa.VisaIOError:
    raise RuntimeError(f"Instrument {resource_str} not found.")

obj1.clear()
print("Getting ready!")
wait = 0.2
obj1.write('C1: BSWV WVTP, DC')
time.sleep(wait)
obj1.write('C1: BSWV OFST, 0')
time.sleep(wait)
obj1.write('C2: BSWV WVTP, DC')
time.sleep(wait)
obj1.write('C2: BSWV OFST, 0')
time.sleep(wait)
print('Ready for use!')

XCONV = -1.85 / 20
YCONV = 2.65 / 20
wait4action = 0.1
def sgd_on():
    print("Sgd on!")
    obj1.write('C1: OUTP ON')
    time.sleep(wait4action)
    obj1.write('C2: OUTP ON')
    time.sleep(wait4action)

def sgd_off():
    print("Sgd off!")
    obj1.write('C1: BSWV OFST, 0')
    time.sleep(wait4action)
    obj1.write('C2: BSWV OFST, 0')
    time.sleep(wait4action)
    obj1.write('C1: OUTP OFF')
    time.sleep(wait4action)
    obj1.write('C2: OUTP OFF')
    time.sleep(wait4action)

def set_position(x_um, y_um):
    print(f"Moving to ({x_um} um, {y_um} um)")
    x_volt = x_um * XCONV
    y_volt = y_um * YCONV
    if abs(x_volt) > 10 or abs(y_volt) > 10:
        raise ValueError("Requested position is out of SDG range.")
    obj1.write(f'C1: BSWV OFST, {y_volt}')
    time.sleep(wait4action)
    obj1.write(f'C2: BSWV OFST, {x_volt}')
    time.sleep(wait4action)
    print("Done moving!")

# sgd_on()
# set_position(5, 5)
# time.sleep(10)
sgd_off()
