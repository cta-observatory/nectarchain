import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ctapipe.io import read_table

from nectarchain.data.container import GainContainer

temperature = [5, 0, -5]
Runs = [4940, 4940, 4940]
dirname = "/Users/hashkar/Desktop/nectarchain/src/nectarchain/user_scripts/thermal_tests/output/FF/FlatFieldTests"


j = 0

for i in Runs:
    filename = (
        "/Users/hashkar/Desktop/nectarchain/src/nectarchain/user_scripts/thermal_tests/output/FF/FlatFieldTests/2FF_%s.h5"
        % str(i)
    )

    toto = read_table(filename, path="/data/FlatFieldContainer_0")
    print(toto)

    data = {
        "FF_coef": [x for x in toto["FF_coef"]],
        # 'luminosity': toto['luminosity']
    }
    print(data)
    FF_coef = np.nanmean(np.array(data["FF_coef"]))

    print(FF_coef)
    plt.plot(temperature[j], FF_coef, marker="o")
    j = j + 1

plt.xlabel("Temperature (°C)")
plt.savefig("FF_coef.png")


# print(toto)

# print(toto['high_gain'])

# for i in toto['high_gain']:
#   print(i)

# print(toto['high_gain'][0,927])
# hg = np.array(toto['high_gain'][0])
# print(len(hg))


"""

    with h5py.File(filename, 'r') as hdf:
        # List all top-level groups
        print("Groups:", list(hdf.keys()))
        group = hdf['data']
        print("Group contents:", list(group.keys()))
        dataset = hdf['data']['SPEfitContainer']
        print("Dataset shape:", dataset.shape)
        print(hdf)
        data = dataset
        print(data)
        
    spe_res = GainContainer.from_hdf5(filename)
"""
