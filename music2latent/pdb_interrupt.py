import signal
import sys
import pdb

def keyboard_interrupt_handler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Entering PDB...".format(signal))
    pdb.set_trace()

# Register the keyboard interrupt handler
signal.signal(signal.SIGINT, keyboard_interrupt_handler)