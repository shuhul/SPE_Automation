"""Step 1: Move to 320 deg. Then power cycle the rotation stage and run step 2."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import filter as fil

fil.filter_init()
fil.filter_on()
fil.rotation_move(320.0)
fil.filter_off()

print(f'\nSaved position: {fil._load_pos()} deg')
print('>>> Now POWER CYCLE the rotation stage, then run test_homing_step2.py')
