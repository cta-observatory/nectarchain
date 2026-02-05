import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tables
from ff_config import Runs, categorize_run1, categorize_run2, dirname, outdir

# import all runs, NSB/temp lists, functions, and paths
sys.path.append(os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

j = 0
MEAN_FF_CAM = []
STD_FF_CAM = []
ff_filename_template = "2FF_{}.h5"

for i in Runs:
    filename = f"{dirname}/{ff_filename_template.format(i)}"

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
        plt.savefig(os.path.join(outdir, f"run{i}_FFcam.png"))

        j = j + 1
        plt.close()
    except Exception as e:
        logger.error(f"Caught error: {e}")

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
plt.savefig(os.path.join(outdir, "FFcam_vs_temp.png"), dpi=30)
plt.close()
