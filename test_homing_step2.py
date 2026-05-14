"""Step 2: Home (should go CLOCKWISE from 320), then move to 20 deg.
Then power cycle and run step 3."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import filter as fil

fil.filter_init()
fil.filter_on()
fil.rotation_home()
print('>>> Did it home CLOCKWISE? Tell me yes/no before continuing.')
fil.rotation_move(20.0)
fil.filter_off()

print(f'\nSaved position: {fil._load_pos()} deg')
print('>>> Now POWER CYCLE the rotation stage, then run test_homing_step3.py')
