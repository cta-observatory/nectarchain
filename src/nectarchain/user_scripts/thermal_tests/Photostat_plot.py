import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ctapipe.io import read_table

from nectarchain.data.container import GainContainer

temperature = [5, 0, -5]
Runs = [3942, 3942, 3942]
dirname = "/Users/hashkar/Desktop/ashkar_nectar/data/SPEfit/"


j = 0

for i in Runs:
    filename = (
        "/Users/hashkar/Desktop/ashkar_nectar/data/SPEfit/FlatFieldSPENominalStdNectarCAM_run%s_maxevents100_LocalPeakWindowSum_window_width_16_window_shift_4.h5"
        % str(i)
    )

    filename = "/Users/hashkar/Desktop/ashkar_nectar/data/SPEfit/FlatFieldSPENominalStdNectarCAM_run3731_maxevents100_GlobalPeakWindowSum_window_width_8_window_shift_4.h5"
    with h5py.File(filename, "r") as f:

        def print_hdf5_structure(name, obj):
            print(name)  # Prints the full path of each dataset/group

        f.visititems(print_hdf5_structure)
    toto = read_table(filename, path="/data/SPEfitContainer_0")
    print(toto)

    data = {
        "is_valid": toto["is_valid"][0],
        "high_gain_lw": [x[0] for x in toto["high_gain"][0]],
        "high_gain": [x[1] for x in toto["high_gain"][0]],
        "high_gain_up": [x[-1] for x in toto["high_gain"][0]],
        "pedestal_lw": [x[0] for x in toto["pedestal"][0]],
        "pedestal": [x[1] for x in toto["pedestal"][0]],
        "pedestal_up": [x[-1] for x in toto["pedestal"][0]],
        "pixels_id": toto["pixels_id"][0],
        # 'luminosity': toto['luminosity']
    }
    # print(data)
    SPEGain = np.nanmean(data["high_gain_lw"])
    # print(np.min((data["high_gain"])))

    # averages = {key: np.mean(values) for key, values in data.items() if isinstance(values, list)}
    # print(averages["high_gain"])
    # Create DataFrame
    # df = pd.DataFrame([averages])
    # print(df)

    plt.plot(temperature[j], SPEGain, marker="o")
    j = j + 1

plt.xlabel("Temperature (°C)")
plt.savefig("mean_PhotostatGain_all.png")


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
