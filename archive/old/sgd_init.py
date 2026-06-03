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
