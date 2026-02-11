import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nectarchain.trr_test_suite.utils import pe2photons, photons2pe

plt.style.use("../../../utils/plot_style.mpltstyle")

# ============================================================
#                   PATHS AND CONFIG
# ============================================================
DATA_DIR = os.environ["NECTARCAMDATA"]
BASEFIGURE = os.environ["NECTARCHAIN_FIGURES"]
JSON_PATH = "./metadata/runs_metadata.json"
output_dir = os.path.join(DATA_DIR, "timing_output/plots")
output_fig = os.path.join(BASEFIGURE, "timing_output/plots")

os.makedirs(output_fig, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

HG_FILE = os.path.join(DATA_DIR, "timing_output/pix_time_uncer_data_HG.txt")
LG_FILE = os.path.join(DATA_DIR, "timing_output/pix_time_uncer_data_LG.txt")

# ============================================================
#                   LOAD DATA
# ============================================================
df_HG = pd.read_csv(HG_FILE, delimiter="\t")
df_LG = pd.read_csv(LG_FILE, delimiter="\t")

# ============================================================
#                   LOAD AND MAP METADATA
# ============================================================
with open(JSON_PATH, "r") as f:
    metadata = json.load(f)

temperature_map = metadata["temperature_map"]
voltage_map = metadata["voltage_map"]

for df in [df_HG, df_LG]:
    df["Run"] = df["Run"].astype(str)
    df["Temperature"] = df["Run"].map(temperature_map)
    df["Voltage"] = df["Run"].map(voltage_map)

print("[HG]", len(df_HG), "rows |", list(df_HG.columns))
print("[LG]", len(df_LG), "rows |", list(df_LG.columns))

# ============================================================
#              HG/LG RATIO (per run, Voltage <= 12 only)
# ============================================================
# Only use runs at Voltage <= 12 to avoid saturation when computing the ratio.
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

# Map the global ratio onto ALL LG rows (both low and high voltage)
df_LG["ratio_HG_LG"] = global_ratio

# ============================================================
#              SPLIT BY VOLTAGE AND BUILD PLOT DF
# ============================================================
# HG: Voltage <= 12, photons as-is
df_HG_low = df_HG[df_HG["Voltage"] <= 12].copy()
df_HG_low["photons_final"] = df_HG_low["photons_spline"]
df_HG_low["source"] = "HG"

# LG: Voltage > 12, photons scaled by per-run HG/LG ratio
df_LG_high = df_LG[df_LG["Voltage"] > 12].copy()
df_LG_high["photons_final"] = df_LG_high["photons_spline"] * df_LG_high["ratio_HG_LG"]
df_LG_high["source"] = "LG"

# Combine — Mean_RMS is the same regardless of gain, only x-axis changes
cols = [
    "Temperature",
    "Voltage",
    "photons_final",
    "Mean_RMS",
    "rms_no_fit_weighted_err",
    "source",
]
# NSB column may or may not exist; include it only if present
if "NSB" in df_HG_low.columns:
    cols = [
        "Temperature",
        "Voltage",
        "NSB",
        "photons_final",
        "Mean_RMS",
        "rms_no_fit_weighted_err",
        "source",
    ]

df_plot = pd.concat([df_HG_low[cols], df_LG_high[cols]], ignore_index=True)

# ---------- Remove points with uncertainty > 1 ----------
before = len(df_plot)
df_plot = df_plot[df_plot["rms_no_fit_weighted_err"] <= 1].reset_index(drop=True)
print(
    f"\nFiltered out {before - len(df_plot)} points with rms_no_fit_weighted_err > 1 ({len(df_plot)} remaining)"
)

# ---------- Remove points with Mean_RMS < 1 ----------
before = len(df_plot)
df_plot = df_plot[df_plot["Mean_RMS"] <= 1].reset_index(drop=True)
print(
    f"Filtered out {before - len(df_plot)} points with Mean_RMS < 1 ({len(df_plot)} remaining)"
)

print("\ndf_plot shape:", df_plot.shape)
print(df_plot.head(10))

# ============================================================
#              WEIGHTED LINEAR FIT
# ============================================================
# Group by Temperature: mean ± std of Mean_RMS, then fit weighted by 1/SEM²
grouped = (
    df_plot.groupby("Temperature")["Mean_RMS"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
grouped.columns = ["Temperature", "RMS_mean", "RMS_std", "N"]
grouped["RMS_sem"] = grouped["RMS_std"] / np.sqrt(grouped["N"])

temps = grouped["Temperature"].values.astype(float)
means = grouped["RMS_mean"].values
sems = grouped["RMS_sem"].values
weights = 1.0 / sems**2

coeffs, cov = np.polyfit(temps, means, 1, w=np.sqrt(weights), cov=True)
slope, intercept = coeffs
slope_err, intercept_err = np.sqrt(np.diag(cov))

print(
    f"Linear fit: RMS = ({slope:.4f} ± {slope_err:.4f}) * T + ({intercept:.4f} ± {intercept_err:.4f})"
)

T_fit = np.linspace(temps.min() - 1, temps.max() + 1, 200)
RMS_fit = slope * T_fit + intercept

# ============================================================
#                        PLOTTING
# ============================================================
os.makedirs(output_dir, exist_ok=True)
fig, axx = plt.subplots(figsize=(10, 7), constrained_layout=True)

# --- Split by source for distinct markers ---
df_HG_plot = df_plot[df_plot["source"] == "HG"]
df_LG_plot = df_plot[df_plot["source"] == "LG"]

# --- HG scatter (circles, black edge) ---
sc_HG = axx.scatter(
    df_HG_plot["Temperature"],
    df_HG_plot["Mean_RMS"],
    c=df_HG_plot["photons_final"],
    cmap="viridis",
    s=60,
    alpha=0.7,
    edgecolors="k",
    linewidths=0.5,
    label="HG",
)

# --- LG scatter (diamonds, red edge) ---
sc_LG = axx.scatter(
    df_LG_plot["Temperature"],
    df_LG_plot["Mean_RMS"],
    c=df_LG_plot["photons_final"],
    cmap="viridis",
    s=60,
    alpha=0.7,
    edgecolors="red",
    linewidths=1.2,
    marker="D",
    label="LG × (HG/LG ratio)",
)

# --- Shared colorbar range ---
vmin = min(df_HG_plot["photons_final"].min(), df_LG_plot["photons_final"].min())
vmax = max(df_HG_plot["photons_final"].max(), df_LG_plot["photons_final"].max())
sc_HG.set_clim(vmin, vmax)
sc_LG.set_clim(vmin, vmax)

# --- Colorbar ---
cbar = fig.colorbar(sc_HG, ax=axx)
cbar.set_label("Illumination [photons]", fontsize=14)
cbar.ax.tick_params(labelsize=12)

# --- Error bars on per-temperature means ---
axx.errorbar(
    grouped["Temperature"],
    grouped["RMS_mean"],
    yerr=grouped["RMS_std"],
    fmt="o",
    color="black",
    markersize=8,
    capsize=5,
    capthick=1.5,
    zorder=5,
    label="Mean ± std per temperature",
)

# --- Linear fit line ---
fit_label = (
    f"Fit: $({slope:.4f} \\pm {slope_err:.4f})\\,T$\n"
    f"$+ ({intercept:.4f} \\pm {intercept_err:.4f})$"
)
axx.plot(T_fit, RMS_fit, color="red", linewidth=2, label=fit_label)

# --- Reference lines ---
axx.axhline(1, ls="--", color="C4", alpha=0.6, label="_nolegend_")
axx.axhline(1 / np.sqrt(12), ls="--", color="gray", alpha=0.7, label="_nolegend_")

# ============================================================
#                   AXES, LEGEND
# ============================================================
axx.set_xlabel("Temperature [°C]", fontsize=16)
axx.set_ylabel("Mean RMS per pixel [ns]", fontsize=16)
axx.tick_params(axis="both", which="major", labelsize=14)
axx.grid(True, alpha=0.3)
axx.legend(fontsize=12, loc="best")

# ============================================================
#                        SAVE
# ============================================================
output_path = os.path.join(output_fig, "mean_rms_vs_photons.png")
plt.savefig(output_path, dpi=300)
print(f"\nSaved to {output_path}")
plt.show()
