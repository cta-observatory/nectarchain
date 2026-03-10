"""
Test command line for test of the Bokeh interface of NectaRTA.
"""

# imports
import subprocess
from pathlib import Path

# Tests
path = Path(__file__).resolve().parent.parent
test_command_line = [
    "bokeh",
    "serve",
    path,
    "--show",
    "--dev",
    "--args",
    "--test-interface",
]
subprocess.run(test_command_line)
