import glob
import os

import h5py
import numpy as np
import pandas as pd
from ctapipe.io import read_table
from IPython.display import display

"""
dirname = '/Users/hashkar/Desktop/ashkar_nectar/data/tests'
file_pattern = os.path.join(dirname, "DeadtimeTestTool_*.h5")
files = glob.glob(file_pattern)

for filename in files:
    print(f"Reading file: {filename}")
    with h5py.File(filename, 'r') as f:
        def print_hdf5_structure(name, obj):
            print(name)  # Prints the full path of each dataset/group
        f.visititems(print_hdf5_structure)

    toto = read_table(filename, path = 'data_1/UCTSContainer')
    display(toto)
"""


############################################################
############################################################
############################################################
############################################################


"""
dirname = '/Users/hashkar/Desktop/ashkar_nectar/data/tests'
file_pattern = os.path.join(dirname, "LinearityTestTool_*.h5")
files = glob.glob(file_pattern)

for filename in files:
    print(f"Reading file: {filename}")
    with h5py.File(filename, 'r') as f:
        def print_hdf5_structure(name, obj):
            print(name)  # Prints the full path of each dataset/group
        f.visititems(print_hdf5_structure)

    toto = read_table(filename, path = 'data_1/ChargeContainer')
    display(toto)
"""


############################################################
############################################################
############################################################
############################################################


"""
dirname = '/Users/hashkar/Desktop/ashkar_nectar/data/tests'
file_pattern = os.path.join(dirname, "PedestalTool_*.h5")
files = glob.glob(file_pattern)

for filename in files:
    print(f"Reading file: {filename}")
    with h5py.File(filename, 'r') as f:
        def print_hdf5_structure(name, obj):
            print(name)  # Prints the full path of each dataset/group
        f.visititems(print_hdf5_structure)

    toto = read_table(filename, path = 'data_1/PedestalContainer')
    display(toto)
"""


############################################################
############################################################
############################################################
############################################################

"""
dirname = '/Users/hashkar/Desktop/ashkar_nectar/data/tests'
file_pattern = os.path.join(dirname, "TimingResolutionTestTool_*.h5")
files = glob.glob(file_pattern)

for filename in files:
    print(f"Reading file: {filename}")
    with h5py.File(filename, 'r') as f:
        def print_hdf5_structure(name, obj):
            print(name)  # Prints the full path of each dataset/group
        f.visititems(print_hdf5_structure)

    toto = read_table(filename, path = 'data_1/ToMContainer')
    display(toto)

"""


############################################################
############################################################
############################################################
############################################################


"""
dirname = '/Users/hashkar/Desktop/ashkar_nectar/data/SPEfit'
file_pattern = os.path.join(dirname, "FlatFieldSPEHHVStdNectarCAM_*.h5")
files = glob.glob(file_pattern)

for filename in files:
    print(f"Reading file: {filename}")
    with h5py.File(filename, 'r') as f:
        def print_hdf5_structure(name, obj):
            print(name)  # Prints the full path of each dataset/group
        f.visititems(print_hdf5_structure)

    toto = read_table(filename, path = '/data/SPEfitContainer_0')
    display(toto)


for i in toto['high_gain']:
    print(i)
toto['high_gain'].shape
print(toto['high_gain'][0,927])


hg = np.array(toto['high_gain'][0])
print("HG", hg[24])
pixel_id = np.array(toto['pixels_id'][0])
import matplotlib.pyplot as plt
plt.hist(hg[0])

gain = [x[1] for x in hg] 
plt.hist2d(pixel_id, gain)


# initialize data of lists.
data = {'is_valid': toto['is_valid'][0],
        'high_gain_lw': [x[0] for x in toto['high_gain'][0]],
        'high_gain': [x[1] for x in toto['high_gain'][0]],
        'high_gain_up': [x[-1] for x in toto['high_gain'][0]],
        'pedestal_lw': [x[0] for x in toto['pedestal'][0]],
        'pedestal': [x[1] for x in toto['pedestal'][0]],
        'pedestal_up': [x[-1] for x in toto['pedestal'][0]],
        'pixels_id': toto['pixels_id'][0],
        # 'luminosity': toto['luminosity']
        }
# Create DataFrame
df = pd.DataFrame(data)
print(df)
"""


############################################################
############################################################
############################################################
############################################################
