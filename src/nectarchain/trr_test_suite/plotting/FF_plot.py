# import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tables

temp_25 = [7020, 7029, 7038, 7047, 7056]
temp_20 = [7077, 7086, 7095, 7104, 7113]
temp_14 = [6954, 6963, 6972, 6981, 6990]
temp_10 = [7144, 7153, 7162, 7171, 7180]
temp_5 = [6543, 6552, 6561, 6570, 6579]
temp_0 = [6672, 6681, 6690, 6699, 6708]
temp_M5 = [6729, 6738, 6747, 6756, 6765]


NSB0 = [7020, 7077, 6954, 7144, 6543, 6672, 6729]
NSB10 = [7029, 7086, 6963, 7153, 6552, 6681, 6738]
NSB20 = [7038, 7095, 6972, 7162, 6561, 6690, 6747]
NSB40 = [7047, 7104, 6981, 7171, 6570, 6699, 6756]
NSB70 = [7056, 7113, 6990, 7180, 6579, 6708, 6765]


def categorize_run1(run):
    if run in NSB0:
        return "No source"
    if run in NSB10:
        return "10.6 mA"
    if run in NSB20:
        return "20.4 mA"
    if run in NSB40:
        return "39.8 mA"
    if run in NSB70:
        return "78.8 mA"


def categorize_run2(run):
    if run in temp_25:
        return 25
    if run in temp_20:
        return 20
    if run in temp_14:
        return 14
    if run in temp_10:
        return 10
    if run in temp_5:
        return 5
    if run in temp_0:
        return 0
    if run in temp_M5:
        return -5


temp = [25, 20, 14, 10, 5, 0, -5]

Runs = [
    7056,
    7113,
    6990,
    6579,
    6708,
    6765,
    7047,
    7104,
    6981,
    7171,
    7171,
    6570,
    6699,
    6756,
    7038,
    7095,
    6972,
    7162,
    6561,
    6690,
    6747,
    7029,
    7088,
    6963,
    7153,
    6552,
    6681,
    6738,
    7020,
    7077,
    6954,
    7144,
    6543,
    6672,
    6729,
]

dirname = "/Users/hashkar/Desktop/20221108/FlatFieldTests"

j = 0
MEAN_FF_CAM = []
STD_FF_CAM = []

for i in Runs:
    filename = "/Users/hashkar/Desktop/20221108/FlatFieldTests/2FF_%s.h5" % str(i)

    h5file = tables.open_file(filename)

    # for result in h5file.root.__members__:
    #    table = h5file.root[result]["FlatFieldContainer_0"][0]
    #    ff = table["FF_coef"]

    result = h5file.root.__members__[0]
    table = h5file.root[result]["FlatFieldContainer_0"][0]
    ff = table["FF_coef"]
    bp = table["bad_pixels"]

    pix = 0
    HG = 1
    ff_pix = ff[:, HG, :]

    mean_ff_pix = np.mean(ff_pix, axis=0, where=np.isinf(ff_pix) == False)

    pixels = np.arange(1, len(mean_ff_pix) + 1)
    GDpixels = np.arange(1, len(mean_ff_pix) + 1)
    GDpixels = np.setdiff1d(GDpixels, bp)

    mean_ff_pix_filtered = np.array([mean_ff_pix[i - 1] for i in GDpixels])

    mean_ff_cam = np.mean(mean_ff_pix, axis=0, where=np.isnan(mean_ff_pix) == False)
    std_ff_cam = np.std(mean_ff_pix, axis=0, where=np.isnan(mean_ff_pix) == False)
    MEAN_FF_CAM.append(mean_ff_cam)
    STD_FF_CAM.append(std_ff_cam)

    # print(len(filtered_arr))

    fig = plt.figure(figsize=(5, 4))
    plt.scatter(GDpixels, mean_ff_pix_filtered)
    plt.ylabel("FF coefficient")
    plt.xlabel("Pixel")
    plt.savefig("./FFplots/run{}_FFpix.png".format(i))

    try:
        fig = plt.figure(figsize=(5, 4))
        plt.hist(
            mean_ff_pix,
            50,
            label=r"$\mu$=%0.3f, \nstd=%0.3f" % (mean_ff_cam, std_ff_cam),
        )

        plt.axvline(mean_ff_cam, ls="--")

        plt.xlabel("mean FF coefficient for all pixels (HG)")
        plt.legend()
        plt.savefig("./FFplots/run{}_FFcam.png".format(i))

        j = j + 1
        plt.close()
    except Exception:
        print("error")

# ----------------- Populate df ----------------- #
df = pd.DataFrame(
    {
        "Run": Runs,
        "Temperature": [categorize_run2(r) for r in Runs],
        "NSB_Label": [categorize_run1(r) for r in Runs],
        "MEAN_FF_CAM": MEAN_FF_CAM,
        "STD_FF_CAM": STD_FF_CAM,
    }
)

# ----------------- Final Temperature vs FF Plot ----------------- #

sns.set(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(12, 10))

NSB_categories = df["NSB_Label"].unique()
colors = sns.color_palette("tab10", len(NSB_categories))

for nsb, color in zip(NSB_categories, colors):
    subset = df[df["NSB_Label"] == nsb]
    ax.errorbar(
        subset["Temperature"],
        subset["MEAN_FF_CAM"],
        yerr=subset["STD_FF_CAM"],
        fmt="o",
        label=nsb,
        color=color,
        capsize=3,
    )

ax.set_xlabel("Temperature [°C]")
ax.set_ylabel("Mean FF coefficient")
ax.set_title("Camera Flat Field vs Temperature")
ax.legend(title="NSB")
plt.tight_layout()
plt.savefig("./FFplots/FFcam_vs_temp.png")
plt.close()
