# Author: Julian Hamo
# Contact mail: julian.hamo@ijclab.in2p3.fr

"""
Test command line for test of the Bokeh interface of NectaRTA.
"""

# imports
import subprocess
from pathlib import Path

# Tests
path = Path(__file__).resolve().parent.parent.parent
test_command_line = [
    "bokeh", "serve", "bokeh_app", "--show", "--dev",
    "--args", "test-interface"
]
subprocess.run(test_command_line)