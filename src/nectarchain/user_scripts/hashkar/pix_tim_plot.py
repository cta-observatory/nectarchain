import ast
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils import pe2photons, photons2pe


def parse_list(value):
    return ast.literal_eval(value)


df = pd.read_csv(
    "pix_time_uncer_data.txt",
    delimiter="\t",
    converters={},  # Use the parse_list function to convert the string to list
)
print(df["Run"])


temp_5 = [
    3721,
    3722,
    3723,
    3724,
    3725,
    3727,
    3728,
    3729,
    3730,
    3714,
    3713,
    3712,
    3711,
    3710,
    3709,
    3708,
    3707,
    3706,
    3705,
]
temp_0 = [
    3741,
    3742,
    3743,
    3744,
    3745,
    3746,
    3747,
    3748,
    3749,
    3764,
    3763,
    3762,
    3761,
    3760,
    3759,
    3758,
    3757,
    3756,
    3755,
]
temp_M5 = [
    3775,
    3776,
    3777,
    3778,
    3779,
    3780,
    3781,
    3782,
    3783,
    3798,
    3797,
    3796,
    3795,
    3794,
    3793,
    3792,
    3791,
    3790,
    3789,
]

FFCLS = [
    3721,
    3722,
    3723,
    3724,
    3725,
    3727,
    3728,
    3729,
    3730,
    3741,
    3742,
    3743,
    3744,
    3745,
    3746,
    3747,
    3748,
    3749,
    3775,
    3776,
    3777,
    3778,
    3779,
    3780,
    3781,
    3782,
    3783,
]
LASER = [
    3714,
    3713,
    3712,
    3711,
    3710,
    3709,
    3708,
    3707,
    3706,
    3705,
    3764,
    3763,
    3762,
    3761,
    3760,
    3759,
    3758,
    3757,
    3756,
    3755,
    3798,
    3797,
    3796,
    3795,
    3794,
    3793,
    3792,
    3791,
    3790,
    3789,
]


def categorize_run1(run):
    if run in FFCLS:
        return "FFCLS"
    if run in LASER:
        return "Laser"


def categorize_run2(run):
    if run in temp_5:
        return 5
    if run in temp_0:
        return 0
    if run in temp_M5:
        return -5


df["Source"] = df["Run"].apply(categorize_run1)
df["Temperature"] = df["Run"].apply(categorize_run2)

print(df)


fig, axx = plt.subplots(figsize=(10, 7), constrained_layout=True)

sns.lineplot(
    data=df,
    x="photons_spline",
    y="Mean_RMS",
    hue="Temperature",
    style="Source",
    marker="o",
)
# sns.lineplot(data=df[df["Source"] == "Laser"], x="photons_spline", y="Mean_RMS", hue="Temperature", marker="v")


plt.axhline(1, ls="--", color="C4", alpha=0.6)
plt.axhline(
    1 / np.sqrt(12),
    ls="--",
    color="gray",
    alpha=0.7,
    label="Quantification rms noise",
)

plt.axvspan(20, 1000, alpha=0.1, color="C4")

axx.text(
    51.5,
    1.04,
    "CTA requirement",
    color="C4",
    fontsize=20,
    horizontalalignment="left",
    verticalalignment="center",
)
axx.annotate(
    "",
    xy=(40, 0.9),
    xytext=(40, 0.995),
    color="C4",
    alpha=0.5,
    arrowprops=dict(color="C4", alpha=0.7, lw=3, arrowstyle="->"),
)

axx.annotate(
    "",
    xy=(200, 0.9),
    xytext=(200, 0.995),
    color="C4",
    alpha=0.5,
    arrowprops=dict(color="C4", alpha=0.7, lw=3, arrowstyle="->"),
)

plt.legend(frameon=True, prop={"size": 18}, loc="upper right", handlelength=1.2)
plt.xlabel("Illumination charge [photons]")
plt.ylabel("Mean rms per pixel [ns]")
plt.xscale("log")
plt.ylim((0, 2.7))
secax = axx.secondary_xaxis("top", functions=(pe2photons, photons2pe))
secax.set_xlabel("Illumination charge [p.e.]", labelpad=7)
plt.savefig("pix_tim_uncertainty_final.png")
plt.close()
