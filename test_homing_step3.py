"""Step 3: Home (should go COUNTERCLOCKWISE from 20)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import filter as fil

fil.filter_init()
fil.filter_on()
fil.rotation_home()
fil.filter_off()

print('>>> Did it home COUNTERCLOCKWISE? Tell me yes/no.')
