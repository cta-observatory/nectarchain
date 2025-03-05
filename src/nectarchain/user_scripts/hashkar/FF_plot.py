import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ctapipe.io import read_table

from nectarchain.data.container import GainContainer

temperature = [5]
Runs = [4940]
dirname = "/Users/hashkar/Desktop/nectarchain/src/nectarchain/user_scripts/thermal_tests/output/FF/FlatFieldTests"


j = 0

for i in Runs:
    filename = (
        "/Users/hashkar/Desktop/nectarchain/src/nectarchain/user_scripts/thermal_tests/output/FF/FlatFieldTests/2FF_%s.h5"
        % str(i)
    )

    toto = read_table(filename, path="/data/FlatFieldContainer_0")
    print(toto.colnames)
    print(toto["event_id"], max(toto["event_id"][0]), len(toto["event_id"][0]))

    data = {
        "FF_coef": [x for x in toto["FF_coef"]],
        "bad_pixels": [x for x in toto["bad_pixels"]],
        # 'luminosity': toto['luminosity']
    }
    print(data["bad_pixels"][0][0])
    print(data["bad_pixels"][0][1])
    print(data["bad_pixels"][0][2])
    print(data["bad_pixels"][0][3])
    print(data["bad_pixels"][0][1822])
    print(data["FF_coef"][0][5206][1])
    # print(len(data['FF_coef'][0][0][1]))

    ff_co = data["FF_coef"][0][0][0]
    filtered_arr = ff_co[np.isfinite(ff_co)]
    # print(len(filtered_arr))

    FF_coef = np.nanmean(filtered_arr)

    plt.hist(filtered_arr)

    # print(FF_coef, max(ff_co), min(ff_co))
    # plt.plot(temperature[j], FF_coef, marker="o")
    j = j + 1

# plt.show()


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
