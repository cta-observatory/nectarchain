import ast
import csv
import json
import os

import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Utils import pe2photons, photons2pe

plt.style.use("plot_style.mpltstyle")


# ============================================================
#                   CONSTANTS AND CONFIG
# ============================================================
def parse_list(value):
    return ast.literal_eval(value)


# ---------- Load both HG and LG data ----------

DATA_DIR = os.environ["NECTARCAMDATA"]
JSON_PATH = "./metadata/runs_metadata.json"
output_dir = os.path.join(DATA_DIR, "timing_output/plots")

df_LG = pd.read_csv(
    os.path.join(output_dir, "time_of_maximum_LG.txt"),
    delimiter="\t",
    converters={},
)

df_HG = pd.read_csv(
    os.path.join(output_dir, "time_of_maximum_HG.txt"),
    delimiter="\t",
    converters={},
)

print("LG runs:", df_LG["Run"].values)
print("HG runs:", df_HG["Run"].values)

# ---------- Metadata ----------
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# Flat dictionaries: run (str) -> value
temperature_map = data["temperature_map"]  # {"7020": 25, "7077": 20, ...}
voltage_map = data["voltage_map"]  # {"7020": 8,  "7077": 8,  ...}

# ---------- Apply temperature and voltage to both dataframes ----------
# Run column must be str to match JSON keys
for df in [df_LG, df_HG]:
    df["Run"] = df["Run"].astype(str)
    df["Temperature"] = df["Run"].map(temperature_map)
    df["Voltage"] = df["Run"].map(voltage_map)

# ============================================================
#              HG/LG RATIO (Voltage <= 12 only)
# ============================================================
# Only use runs at Voltage <= 12 to avoid saturation.
# Multiple rows per run (different illumination levels).
# Collapse to one mean photons_spline per run on each side,
# then compute the ratio cleanly — unique index, no cross join.

df_HG_low_volt = df_HG[df_HG["Voltage"] <= 12]
df_LG_low_volt = df_LG[df_LG["Voltage"] <= 12]

hg_mean = df_HG_low_volt.groupby("Run")["photons_spline"].mean().rename("photons_HG")
lg_mean = df_LG_low_volt.groupby("Run")["photons_spline"].mean().rename("photons_LG")

ratio_df = hg_mean.to_frame().join(lg_mean, how="inner")
ratio_df["ratio"] = ratio_df["photons_HG"] / ratio_df["photons_LG"]

print("\nPer-run HG/LG ratios (Voltage <= 12 only):")
print(ratio_df[["photons_HG", "photons_LG", "ratio"]])

# Use the mean ratio across all low-voltage runs as the global scaling factor
global_ratio = ratio_df["ratio"].mean()
print(f"\nGlobal HG/LG ratio (mean of Voltage <= 12 runs): {global_ratio:.4f}")

# Map the global ratio onto ALL LG rows
df_LG["ratio_HG_LG"] = global_ratio

# ============================================================
#              SPLIT BY VOLTAGE AND BUILD FINAL DF
# ============================================================
# HG: Voltage <= 12
df_HG_low = df_HG[df_HG["Voltage"] <= 12].copy()

# LG: Voltage > 12, photons scaled by global HG/LG ratio
df_LG_high = df_LG[df_LG["Voltage"] > 12].copy()
df_LG_high["photons_spline_scaled"] = (
    df_LG_high["photons_spline"] * df_LG_high["ratio_HG_LG"]
)

# ---------- Combine HG (V<=12) and LG scaled (V>12) into one "final" set ----------
df_all = pd.concat(
    [
        df_HG_low[["Temperature", "Mean_TOM"]],
        df_LG_high[["Temperature", "Mean_TOM"]],
    ],
    ignore_index=True,
)

# ---------- Group by temperature: mean and std ----------
grouped = (
    df_all.groupby("Temperature")["Mean_TOM"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
grouped.columns = ["Temperature", "TOM_mean", "TOM_std", "N"]
# Std of the mean (SEM) for the fit weights
grouped["TOM_sem"] = grouped["TOM_std"] / np.sqrt(grouped["N"])

print("\nGrouped TOM per temperature:")
print(grouped)

# ---------- Weighted linear fit (weights = 1/SEM²) ----------
temps = grouped["Temperature"].values.astype(float)
means = grouped["TOM_mean"].values
sems = grouped["TOM_sem"].values
weights = 1.0 / sems**2

# np.polyfit with weights
coeffs, cov = np.polyfit(temps, means, 1, w=np.sqrt(weights), cov=True)
slope, intercept = coeffs
slope_err, intercept_err = np.sqrt(np.diag(cov))

print(
    f"\nLinear fit: TOM = ({slope:.4f} ± {slope_err:.4f}) * T + ({intercept:.4f} ± {intercept_err:.4f})"
)

# Fit line for plotting
T_fit = np.linspace(temps.min() - 1, temps.max() + 1, 200)
TOM_fit = slope * T_fit + intercept

# ============================================================
#                        PLOTTING
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

# --- HG points (only Voltage <= 12) ---
sc_HG = ax.scatter(
    df_HG_low["Temperature"],
    df_HG_low["Mean_TOM"],
    c=df_HG_low["photons_spline"],
    cmap="viridis",
    s=100,
    alpha=0.7,
    edgecolors="k",
    linewidths=0.5,
    label="HG",
)

# --- LG points scaled by global HG/LG ratio (only Voltage > 12) ---
sc_LG = ax.scatter(
    df_LG_high["Temperature"],
    df_LG_high["Mean_TOM"],
    c=df_LG_high["photons_spline_scaled"],
    cmap="viridis",
    s=100,
    alpha=0.7,
    edgecolors="red",
    linewidths=1.2,
    marker="D",
    label="LG × (HG/LG ratio)",
)

# --- Error bars on per-temperature means ---
ax.errorbar(
    grouped["Temperature"],
    grouped["TOM_mean"],
    yerr=grouped["TOM_std"],
    fmt="o",
    color="black",
    markersize=8,
    capsize=5,
    capthick=1.5,
    zorder=5,
    label="Mean ± std per temperature",
)

# --- Linear fit ---
fit_label = (
    f"Fit: $({slope:.4f} \\pm {slope_err:.4f})\\,T$\n"
    f"$+ ({intercept:.4f} \\pm {intercept_err:.4f})$"
)
ax.plot(T_fit, TOM_fit, color="red", linewidth=2, label=fit_label)

# --- Shared colorbar (both scatters use the same viridis scale) ---
vmin = min(df_HG_low["photons_spline"].min(), df_LG_high["photons_spline_scaled"].min())
vmax = max(df_HG_low["photons_spline"].max(), df_LG_high["photons_spline_scaled"].max())
sc_HG.set_clim(vmin, vmax)
sc_LG.set_clim(vmin, vmax)

cbar = plt.colorbar(sc_HG, ax=ax)
cbar.set_label("Illumination [photons]", fontsize=14)

# --- Labels & style ---
ax.set_xlabel("Temperature [°C]", fontsize=14)
ax.set_ylabel("Time of Maximum [ns]", fontsize=14)
ax.set_title("Time of Maximum vs Temperature (HG + scaled LG)", fontsize=16)
ax.tick_params(labelsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc="best")

plt.savefig("time_of_max_scatter_HG_LG.png")
plt.close()
