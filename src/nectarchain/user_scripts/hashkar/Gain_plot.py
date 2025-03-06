import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ctapipe.io import read_table

from nectarchain.data.container import GainContainer

temperature = [5, 0, -5]
Runs = [3942, 3731, 3750]

SPE = [3942]
FF = [3731, 3750]


temp_5 = [3731]
temp_0 = [3750]
temp_M5 = [3942]

GAINDATA = []
RUNNUMBER = []
TEMP = []
METHOD = []

dirname = "/Users/hashkar/Desktop/ashkar_nectar/data/SPEfit/"


j = 0

for i in Runs:
    filename = (
        "/Users/hashkar/Desktop/ashkar_nectar/data/SPEfit/FlatFieldSPENominalStdNectarCAM_run%s_maxevents100_GlobalPeakWindowSum_window_shift_4_window_width_8.h5"
        % str(i)
    )

    with h5py.File(filename, "r") as f:

        def print_hdf5_structure(name, obj):
            print(name)  # Prints the full path of each dataset/group

        f.visititems(print_hdf5_structure)

        try:
            toto = read_table(filename, path="/data/SPEfitContainer_0")
            if j == 2:
                method = "SPE"
            else:
                method = "Photostat"
        except:
            toto = read_table(filename, path="/data/PhotosatatfitContainer_0")
            method = "Photostat"

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
    Gain = np.nanmean(data["high_gain_lw"])
    GAINDATA.append(Gain)
    RUNNUMBER.append(i)
    METHOD.append(method)

    j = j + 1

df = pd.DataFrame({"Gain": GAINDATA, "Run": RUNNUMBER, "Method": METHOD})


def categorize_run1(run):
    if run in SPE:
        return "SPE"
    if run in FF:
        return "FF"


def categorize_run2(run):
    if run in temp_5:
        return 5
    if run in temp_0:
        return 0
    if run in temp_M5:
        return -5


df["Source_Label"] = df["Run"].apply(categorize_run1)
df["Temperature"] = df["Run"].apply(categorize_run2)

print(df)


plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="Temperature",
    y="Gain",
    hue="Source_Label",
    style="Method",
    marker="o",
)

# Add plot labels and title
plt.xlabel("Temperature (°C)")
plt.ylabel("Gain")
# plt.legend(title="Source")
plt.grid(True)

# Show the plot
plt.savefig("Gain_plot.png")
plt.show()
plt.close()
