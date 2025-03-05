import ast
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_list(value):
    return ast.literal_eval(value)


df = pd.read_csv(
    "linearity_data.txt",
    delimiter="\t",
    converters={},  # Use the parse_list function to convert the string to list
)
print(df["Run"])

temp = [-5, 0, 5]
colors = ["b", "r", "g"]

temp_5 = [3404, 3406, 3408, 3410, 3412, 3414, 3416, 3418, 3420, 3422, 3424]
temp_0 = [
    3405,
    3407,
    3413,
    3415,
    3417,
    3419,
]
temp_M5 = [3409, 3411, 3421, 3423]

Low = [i for i in range(3404, 3414)]
High = [i for i in range(3414, 3424)]


def categorize_run1(run):
    if run in Low:
        return "Low"
    if run in High:
        return "High"


def categorize_run2(run):
    if run in temp_5:
        return 5
    if run in temp_0:
        return 0
    if run in temp_M5:
        return -5


fig, axs = plt.subplots(
    3,
    1,
    sharex="col",
    sharey="row",
    figsize=(10, 11),
    gridspec_kw={"height_ratios": [4, 2, 2]},
)
axs[0].grid(True, which="both")
axs[0].set_ylabel("Estimated charge [p.e.]")
axs[0].set_yscale("log")
axs[0].set_xscale("log")
axs[0].axvspan(10, 1000, alpha=0.2, color="orange")
axs[1].grid(True, which="both")
axs[1].set_ylabel("Residuals [%]")
axs[1].set_xscale("log")
axs[1].set_ylim((-100, 100))
axs[1].axvspan(10, 1000, alpha=0.2, color="orange")
axs[2].grid(True, which="both")
axs[2].set_ylabel("HG/LG ratio")
axs[2].set_yscale("log")
axs[2].set_xscale("log")
axs[2].set_ylim((0.5, 20))
axs[2].axvspan(10, 1000, alpha=0.2, color="orange")
axs[2].set_xlabel("Illumination charge [p.e.]")


df["Gain"] = df["Run"].apply(categorize_run1)
df["Temperature"] = df["Run"].apply(categorize_run2)

print(df)


for i, color in zip(temp, colors):
    print(i)
    df1 = df[df["Temperature"] == i]

    charge_high_gain = df1["Charge_high_Gain"]
    charge_low_gain = df1["Charge_low_gain"]

    ch_charge_high_gain = df1["ch_charge_high_Gain"]
    ch_charge_err_high_gain = df1["ch_charge_err_high_Gain"]

    ch_charge_low_gain = df1["ch_charge_low_gain"]
    ch_charge_err_low_gain = df1["ch_charge_err_low_gain"]

    resid_high_gain = df1["resid_high_Gain"]
    resid_err_high_gain = df1["resid_err_high_Gain"]

    resid_low_gain = df1["resid_low_gain"]
    resid_err_low_gain = df1["resid_err_low_gain"]

    ratio = df1["ratio"]
    ratio_err = df1["ratio_std"]

    a_high = df1["a_high_Gain"].unique()[0]
    b_high = df1["b_high_Gain"].unique()[0]

    a_low = df1["a_low_gain"].unique()[0]
    b_low = df1["b_low_gain"].unique()[0]

    axs[0].errorbar(
        charge_high_gain,
        ch_charge_high_gain,
        yerr=ch_charge_err_high_gain,
        label="High",
        ls="",
        marker="o",
        color=color,
    )

    axs[0].errorbar(
        charge_low_gain,
        ch_charge_low_gain,
        yerr=ch_charge_err_low_gain,
        label="Low",
        ls="",
        marker="v",
        color=color,
    )

    y = a_high * charge_low_gain + b_high
    axs[0].plot(
        charge_low_gain,
        y,
        color=color,
    )

    y = a_low * charge_low_gain + b_low
    axs[0].plot(
        charge_low_gain,
        y,
        color=color,
    )

    axs[1].errorbar(
        charge_high_gain,
        resid_high_gain,
        yerr=resid_err_high_gain,
        label="High",
        ls="",
        marker="o",
        color=color,
    )

    axs[1].errorbar(
        charge_low_gain,
        resid_low_gain,
        yerr=resid_err_low_gain,
        label="Low",
        ls="",
        marker="v",
        color=color,
    )

    axs[2].errorbar(
        charge_low_gain,
        ratio,
        yerr=ratio_err,
        label="Low",
        ls="",
        marker="v",
        color=color,
    )


plt.show()
