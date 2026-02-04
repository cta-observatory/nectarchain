#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import re
from collections import defaultdict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tables
from charge_config import (
    BAD_MODULE_IDS,
    BAD_PIXELS_GAIN,
    BAD_PIXELS_HV,
    db_data_path,
    dirname,
    ff_dir,
    ff_map,
    gain_map,
    gain_path,
    ivalnsb_map,
    outdir,
    path,
    pedestal_folder,
    temp_map,
    vvalff_map,
)
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from scipy.stats import linregress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

plt.style.use("../../../utils/plot_style.mpltstyle")


METHOD = "LocalPeakWindowSum"
WINDOW = 16
SHIFT = 4
NSIGMA_BADPIX = 5

camera_geom = CameraGeometry.from_name("NectarCam").transform_to(
    EngineeringCameraFrame()
)


"""
Charge normalization analysis script with HG/LG gain splitting:
- Uses HG data for Voltage <= 12V
- Uses LG data × mean(HG/LG ratio from V<=12) for Voltage > 12V
- Reads gain-appropriate pedestals (HG vs LG)
- Applies FF correction and SPE gain normalization (same for both)
- Produces camera-average vs temperature plots with shaded NSB spread
- Produces per-pixel slope maps per (Voltage, NSB)
"""

# ================================
# BAD MODULES DEFINITION
# ================================


def get_bad_pixels_from_modules(bad_module_ids):
    """
    Convert bad module IDs to bad pixel IDs.
    NectarCAM has 7 pixels per module:
    module M contains pixels [M*7, M*7+1, ..., M*7+6]
    """
    bad_pixels = set()
    for mod_id in bad_module_ids:
        for pixel_offset in range(7):
            bad_pixels.add(mod_id * 7 + pixel_offset)
    return bad_pixels


BAD_PIXELS_FROM_MODULES = get_bad_pixels_from_modules(BAD_MODULE_IDS)
logger.info("Total bad pixels from bad modules: %d", len(BAD_PIXELS_FROM_MODULES))
bad_ids = None


def get_bad_hv_pixels_db(
    run, path, db_data_path, hv_tolerance=4.0, telid=0, verbose=False
):
    """Identify pixels with |measured HV - target HV| > hv_tolerance."""
    bad_pixels = BAD_PIXELS_HV
    return bad_pixels


# ================================
# HELPER FUNCTIONS
# ================================
def load_charge_file(filename, gain="HG", peak_time_tolerance=6.0):
    """
    Load charge data from HDF5 file for specified gain (HG or LG).
    Applies peak time filtering: keeps only events where mean peak time per event
    is within peak_time_tolerance of the global mean, then returns mean charge per pixel.
    """
    with h5py.File(filename, "r") as f:
        logger.info("Filename: %s, Gain: %s", filename, gain)
        container = f.get("data/ChargesContainer_0")
        dataset = container.get("FLATFIELD")
        dataset0 = dataset[0]
        pixels_id = dataset0["pixels_id"]

        if gain == "HG":
            charges_all = dataset0[
                "charges_hg"
            ]  # shape: (N_events, N_pixels) or (N_pixels,)
            peaks_all = dataset0[
                "peak_hg"
            ]  # shape: (N_events, N_pixels) or (N_pixels,)
        elif gain == "LG":
            charges_all = dataset0["charges_lg"]
            peaks_all = dataset0["peak_lg"]
        else:
            raise ValueError(f"Invalid gain: {gain}")

        # Convert to numpy arrays
        charges_all = np.array(charges_all)
        peaks_all = np.array(peaks_all)

        # If 1D (single event), no filtering needed
        if charges_all.ndim == 1:
            logger.info("Single event detected, no peak time filtering applied")
            return np.array(pixels_id), charges_all

        # --- Peak time filtering for multi-event data ---
        # charges_all shape: (N_events, N_pixels)
        # peaks_all shape: (N_events, N_pixels)

        # Compute mean peak time per event (average over pixels)
        mean_peak_per_event = np.mean(peaks_all, axis=1)  # shape: (N_events,)
        global_mean_peak = np.mean(mean_peak_per_event)

        # Filter events: keep only those within tolerance
        mask_events = (
            np.abs(mean_peak_per_event - global_mean_peak) <= peak_time_tolerance
        )
        n_total = len(mean_peak_per_event)
        n_kept = np.sum(mask_events)

        logger.info(
            "Peak time filter: kept %d/%d events (tolerance=%.1f ns)",
            n_kept,
            n_total,
            peak_time_tolerance,
        )

        if n_kept == 0:
            logger.warning("All events filtered out! Returning NaN charges")
            return np.array(pixels_id), np.full(len(pixels_id), np.nan)

        # Apply mask and compute mean across filtered events
        charges_filtered = charges_all[mask_events, :]  # shape: (N_kept, N_pixels)
        charges_mean = np.mean(charges_filtered, axis=0)  # shape: (N_pixels,)

        return np.array(pixels_id), charges_mean


def read_gain_file(spe_run, window, shift, method, dirname):
    """Read SPE gain file (same for HG and LG)."""
    filename = (
        f"FlatFieldSPENominalStdNectarCAM_run{spe_run}_"
        f"maxevents5000_{method}_window_shift_{shift}_"
        f"window_width_{window}.h5"
    )
    spe_filename = os.path.join(dirname, filename)

    with h5py.File(spe_filename, "r") as f:
        data = f["/data/SPEfitContainer_0"]
        raw_high_gain = data["high_gain"][0]
        gains = [
            float(entry[0]) if hasattr(entry, "__getitem__") else float(entry)
            for entry in raw_high_gain
        ]
        pixels_id = np.array(data["pixels_id"][0], dtype=int)
    return pixels_id, np.array(gains, dtype=float)


def detect_bad_pixels(pixels_id, charges, nsigma=5):
    """Detect statistical outliers via sigma clipping."""
    vals = np.array(charges, dtype=float)
    mean_val = np.nanmean(vals)
    std_val = np.nanstd(vals)
    bad_mask = np.abs(vals - mean_val) > nsigma * std_val
    return set(np.array(pixels_id)[bad_mask])


def compute_slopes_and_stats(df_pixels, runs, temperatures):
    """Compute per-pixel slopes vs temperature."""
    pixel_gains = df_pixels.pivot_table(
        index="Pixel", columns="Run", values="Gain"
    ).reindex(columns=runs)
    pivot_pixel_ids = pixel_gains.index.values
    pixel_gains_array = pixel_gains.values
    slopes, mean_gain, std_gain = [], [], []
    for i in range(pixel_gains_array.shape[0]):
        gains = pixel_gains_array[i, :]
        mask = ~np.isnan(gains)
        x = np.array(temperatures)[mask]
        y = gains[mask]
        slope = linregress(x, y).slope if len(x) > 1 and np.ptp(x) > 0 else np.nan
        slopes.append(slope)
        mean_gain.append(np.nanmean(y))
        std_gain.append(np.nanstd(y))
    return (
        pivot_pixel_ids,
        np.array(slopes),
        np.array(mean_gain),
        np.array(std_gain),
        pixel_gains_array,
    )


def plot_camera(values, pixel_ids, camera_geom, title, fig_path, cmap="viridis"):
    """Plot camera display, masking bad pixels."""
    good_mask = np.array([pid not in bad_ids for pid in pixel_ids])
    filtered_pixel_ids = pixel_ids[good_mask]
    filtered_values = np.array(values)[good_mask]

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = CameraDisplay(
        geometry=camera_geom[filtered_pixel_ids],
        image=np.array(filtered_values, dtype=float),
        cmap=cmap,
        ax=ax,
    )
    disp.add_colorbar(label=title)
    plt.title(title)
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)


def plot_pixel_histogram(values, pixel_ids, title, fig_path, bins=50, xlabel="Value"):
    """Plot histogram of per-pixel values."""
    good_mask = np.array([pid not in bad_ids for pid in pixel_ids])
    good_values = np.array(values, dtype=float)[good_mask]
    good_values = good_values[np.isfinite(good_values)]

    mean_val = np.mean(good_values)
    std_val = np.std(good_values)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(good_values, bins=bins, histtype="stepfilled", alpha=0.7, label="Pixels")
    ax.axvline(mean_val, linestyle="--", linewidth=2, label=rf"Mean = {mean_val:.3e}")
    ax.axvline(
        mean_val + std_val, linestyle=":", linewidth=2, label=rf"+1σ = {std_val:.3e}"
    )
    ax.axvline(
        mean_val - std_val, linestyle=":", linewidth=2, label=rf"-1σ = {std_val:.3e}"
    )

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Number of pixels", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="upper right",
        fontsize=9,
        frameon=True,
        framealpha=0.85,
        borderpad=0.3,
        labelspacing=0.3,
        handlelength=1.5,
    )

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def load_ff_file(temp, ff_map, ff_dir):
    """Load flatfield coefficients."""
    ff_run = ff_map.get(temp)
    if ff_run is None:
        raise ValueError(f"No FF run defined for temp {temp}")
    ff_filename = os.path.join(ff_dir, f"FF_calibration_run{ff_run}.dat")
    df_ff = pd.read_csv(ff_filename, delim_whitespace=True)
    ff_dict = dict(zip(df_ff["pixel_id"], df_ff["FF_coef_independent_way"]))
    return ff_dict


def load_pedestal_h5(filepath, gain="HG"):
    """Load pedestal from HDF5 file for specified gain."""
    h5file = tables.open_file(filepath)
    table = h5file.root.data_combined.NectarCAMPedestalContainer_0[0]

    pixel_ids = table["pixels_id"]

    if gain == "HG":
        pedestal_wf = table["pedestal_mean_hg"]
    elif gain == "LG":
        pedestal_wf = table["pedestal_mean_lg"]
    else:
        raise ValueError(f"Invalid gain: {gain}")

    baseline = pedestal_wf.mean(axis=1)
    h5file.close()

    return pixel_ids, dict(zip(pixel_ids, baseline))


# ================================
# LOAD PEDESTAL FILE MAP
# ================================
pedestal_dir = pedestal_folder
ped_files = {}
pattern = re.compile(r"pedestal_cfilt3s_T(?P<T>-?\d+)_NSB(?P<NSB>[\d.]+)\.h5")

for fname in os.listdir(pedestal_dir):
    match = pattern.match(fname)
    if match:
        T = float(match.group("T"))
        NSB = float(match.group("NSB"))
        ped_files[(T, NSB)] = os.path.join(pedestal_dir, fname)

ped_T_vals = np.array(sorted(set(T for T, _ in ped_files)))
ped_NSB_vals = np.array(sorted(set(NSB for _, NSB in ped_files)))

# ================================
# FIRST PASS: COMPUTE HG/LG RATIO FROM V <= 12 RUNS
# ================================
logger.info("=== COMPUTING HG/LG RATIO FROM V <= 12 RUNS ===")

hg_low_volt_data = []  # (run, mean_charge_hg)
lg_low_volt_data = []  # (run, mean_charge_lg)

for temp, runs in temp_map.items():
    voltages = vvalff_map[temp]
    nsbs = ivalnsb_map[temp]

    for i, run in enumerate(runs):
        Voo = voltages[i]
        if Voo > 12:
            continue  # Skip high-voltage runs for ratio calculation

        Inn = nsbs[i]

        # Find closest pedestal
        T_sel = ped_T_vals[np.argmin(np.abs(ped_T_vals - temp))]
        NSB_sel = ped_NSB_vals[np.argmin(np.abs(ped_NSB_vals - Inn))]
        ped_file = ped_files[(T_sel, NSB_sel)]

        # Load pedestals for both gains
        ped_pixel_ids_hg, ped_dict_hg = load_pedestal_h5(ped_file, gain="HG")
        ped_pixel_ids_lg, ped_dict_lg = load_pedestal_h5(ped_file, gain="LG")

        # FF correction
        try:
            ff_dict = load_ff_file(temp, ff_map, ff_dir)
        except Exception:
            ff_dict = {pid: 1.0 for pid in ped_pixel_ids_hg}

        # SPE gain
        gain_run = gain_map[temp]
        try:
            spe_pixels, spe_gains = read_gain_file(
                gain_run, WINDOW, SHIFT, METHOD, gain_path
            )
        except Exception:
            continue
        pixel_to_gain = dict(zip(spe_pixels, spe_gains))

        # Load charge file
        charge_filename = os.path.join(
            dirname,
            f"ChargesNectarCAMCalibration_run{run}_"
            f"maxevents5000_{METHOD}_window_shift_{SHIFT}_window_width_{WINDOW}.h5",
        )
        if not os.path.exists(charge_filename):
            continue

        # Load HG and LG charges
        pixels_id_hg, charges_hg = load_charge_file(charge_filename, gain="HG")
        pixels_id_lg, charges_lg = load_charge_file(charge_filename, gain="LG")

        # Average over events if 2D
        if charges_hg.ndim > 1:
            charges_hg = np.mean(charges_hg, axis=0)
        if charges_lg.ndim > 1:
            charges_lg = np.mean(charges_lg, axis=0)

        # Process HG
        charges_hg_pedcorr = np.array(
            [
                charges_hg[i] - ped_dict_hg.get(pid, np.nan) * WINDOW
                for i, pid in enumerate(pixels_id_hg)
            ]
        )
        charges_hg_ffcorr = np.array(
            [
                charges_hg_pedcorr[i] * ff_dict.get(pid, 1.0)
                for i, pid in enumerate(pixels_id_hg)
            ]
        )
        charges_hg_normalized = np.array(
            [
                charges_hg_ffcorr[i] / pixel_to_gain.get(pid, np.nan)
                for i, pid in enumerate(pixels_id_hg)
            ]
        )

        # Process LG
        charges_lg_pedcorr = np.array(
            [
                charges_lg[i] - ped_dict_lg.get(pid, np.nan) * WINDOW
                for i, pid in enumerate(pixels_id_lg)
            ]
        )
        charges_lg_ffcorr = np.array(
            [
                charges_lg_pedcorr[i] * ff_dict.get(pid, 1.0)
                for i, pid in enumerate(pixels_id_lg)
            ]
        )
        charges_lg_normalized = np.array(
            [
                charges_lg_ffcorr[i] / pixel_to_gain.get(pid, np.nan)
                for i, pid in enumerate(pixels_id_lg)
            ]
        )

        # Camera-average charge per run (no bad pixel filtering yet)
        mean_hg = np.nanmean(charges_hg_normalized)
        mean_lg = np.nanmean(charges_lg_normalized)

        hg_low_volt_data.append((run, mean_hg))
        lg_low_volt_data.append((run, mean_lg))

        logger.info(
            "Run %s (V=%.1f): HG_mean=%.4f, LG_mean=%.4f", run, Voo, mean_hg, mean_lg
        )

# Compute per-run ratios, then take global mean
ratios = []
for (run_hg, hg_val), (run_lg, lg_val) in zip(hg_low_volt_data, lg_low_volt_data):
    if (
        run_hg == run_lg
        and not np.isnan(hg_val)
        and not np.isnan(lg_val)
        and lg_val != 0
    ):
        ratio = hg_val / lg_val
        ratios.append(ratio)
        logger.info("Run %s: HG/LG ratio = %.4f", run_hg, ratio)

global_ratio = np.mean(ratios) if ratios else 1.0
logger.info("=== GLOBAL HG/LG RATIO (V <= 12): %.4f ===", global_ratio)

# ================================
# SECOND PASS: PROCESS ALL RUNS WITH GAIN SELECTION
# ================================
logger.info("=== PROCESSING ALL RUNS WITH GAIN SELECTION ===")

avg_charges_config = defaultdict(list)
all_pixel_records = []

for temp, runs in temp_map.items():
    voltages = vvalff_map[temp]
    nsbs = ivalnsb_map[temp]

    for i, run in enumerate(runs):
        Voo = voltages[i]
        Inn = nsbs[i]

        # Select gain based on voltage
        use_gain = "HG" if Voo <= 12 else "LG"
        scaling_factor = 1.0 if use_gain == "HG" else global_ratio

        logger.info(
            "Run %s: V=%.1f → using %s (scale=%.4f)", run, Voo, use_gain, scaling_factor
        )

        # Find closest pedestal
        T_sel = ped_T_vals[np.argmin(np.abs(ped_T_vals - temp))]
        NSB_sel = ped_NSB_vals[np.argmin(np.abs(ped_NSB_vals - Inn))]
        ped_file = ped_files[(T_sel, NSB_sel)]

        # Load pedestal for selected gain
        ped_pixel_ids, ped_dict = load_pedestal_h5(ped_file, gain=use_gain)

        # FF correction
        try:
            ff_dict = load_ff_file(temp, ff_map, ff_dir)
        except Exception:
            ff_dict = {pid: 1.0 for pid in ped_pixel_ids}

        # SPE gain
        gain_run = gain_map[temp]
        try:
            spe_pixels, spe_gains = read_gain_file(
                gain_run, WINDOW, SHIFT, METHOD, gain_path
            )
        except Exception:
            continue
        pixel_to_gain = dict(zip(spe_pixels, spe_gains))

        # Load charge file
        charge_filename = os.path.join(
            dirname,
            f"ChargesNectarCAMCalibration_run{run}_"
            f"maxevents5000_{METHOD}_window_shift_{SHIFT}_window_width_{WINDOW}.h5",
        )
        if not os.path.exists(charge_filename):
            continue

        pixels_id, charges = load_charge_file(charge_filename, gain=use_gain)

        if charges.ndim > 1:
            charges = np.mean(charges, axis=0)

        logger.info("Temperature: %s, Run: %s, Gain: %s", temp, run, use_gain)
        logger.info("Mean raw charge: %.6f", np.mean(charges))

        # Pedestal subtraction + FF + SPE normalization
        charges_pedcorr = np.array(
            [
                charges[i] - ped_dict.get(pid, np.nan) * WINDOW
                for i, pid in enumerate(pixels_id)
            ]
        )
        charges_ffcorr = np.array(
            [
                charges_pedcorr[i] * ff_dict.get(pid, 1.0)
                for i, pid in enumerate(pixels_id)
            ]
        )
        charges_normalized = np.array(
            [
                charges_ffcorr[i] / pixel_to_gain.get(pid, np.nan)
                for i, pid in enumerate(pixels_id)
            ]
        )

        # Apply HG/LG scaling if using LG
        charges_final = charges_normalized * scaling_factor

        # Bad pixel identification
        hv_bad_pixels = get_bad_hv_pixels_db(
            run, path, db_data_path, hv_tolerance=4.0, telid=0, verbose=False
        )
        module_bad_pixels = set(BAD_PIXELS_FROM_MODULES)
        gain_bad_pixels = set(BAD_PIXELS_GAIN)
        non_statistical_bad = (
            set(hv_bad_pixels).union(module_bad_pixels).union(gain_bad_pixels)
        )

        remaining_pixel_ids = [p for p in pixels_id if p not in non_statistical_bad]
        remaining_charges = [
            charges_final[i]
            for i, p in enumerate(pixels_id)
            if p not in non_statistical_bad
        ]

        stat_bad_pixels = detect_bad_pixels(
            remaining_pixel_ids, remaining_charges, nsigma=NSIGMA_BADPIX
        )
        bad_ids = non_statistical_bad.union(stat_bad_pixels)

        logger.info("Total bad pixels: %d", len(bad_ids))

        # Store per-pixel records
        for pid, charge in zip(pixels_id, charges_final):
            if pid not in bad_ids:
                all_pixel_records.append(
                    {
                        "Run": run,
                        "Temperature": temp,
                        "Pixel": int(pid),
                        "V": Voo,
                        "NSB": Inn,
                        "Charge": float(charge),
                    }
                )

        # Store camera average
        good_charges = [
            g for pid, g in zip(pixels_id, charges_final) if pid not in bad_ids
        ]
        cam_avg = np.nanmean(good_charges)
        cam_std = np.nanstd(good_charges)
        avg_charges_config[(Voo, Inn)].append(
            {
                "Run": run,
                "Temperature": temp,
                "CameraAvgCharge": cam_avg,
                "CameraStd": cam_std,
            }
        )

df_all_pixels = pd.DataFrame(all_pixel_records)

# ================================
# PLOT CAMERA AVG VS TEMP ALL ON SAME FIGURE
# ================================
plt.figure(figsize=(20, 12))
for (V, I), records in avg_charges_config.items():
    df_cfg = pd.DataFrame(records).sort_values("Temperature")
    cam_avgs = df_cfg["CameraAvgCharge"].values
    n_good_pixels = len(ped_pixel_ids) - len(bad_ids)
    cam_errs = np.array(
        [g / np.sqrt(n_good_pixels) for g in df_cfg["CameraStd"].values]
    )

    plt.errorbar(
        df_cfg["Temperature"],
        cam_avgs,
        yerr=cam_errs,
        fmt="o-",
        capsize=3,
        label=f"V={V} V, NSB={I} mA",
    )

logger.info("Total configurations processed: %d", len(avg_charges_config))

plt.xlabel("Temperature (°C)", fontsize=16)
plt.ylabel("Camera Average Normalized Charge (p.e.)", fontsize=16)
plt.tick_params(axis="both", which="major", labelsize=14)
plt.tick_params(axis="both", which="minor", labelsize=12)
plt.grid(True)

handles, labels = plt.gca().get_legend_handles_labels()
if handles and labels:
    sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])
    sorted_labels, sorted_handles = zip(*sorted_handles_labels)
    plt.legend(
        sorted_handles,
        sorted_labels,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=16,
        ncol=2,
        frameon=True,
        borderaxespad=0.5,
    )
else:
    logger.warning("No data to plot - avg_charges_config is empty")

plt.tight_layout()
plt.savefig(os.path.join(outdir, "CameraAvgCharge_AllConfigs_SEM.png"), dpi=150)
plt.show()
plt.close()

# ================================
# PER-PIXEL SLOPE VS TEMP PER (V,NSB)
# ================================
for (V, I), subset in df_all_pixels.groupby(["V", "NSB"]):
    subset_runs = subset["Run"].unique()
    subset_temps = [
        next(temp for temp, runs in temp_map.items() if run in runs)
        for run in subset_runs
    ]

    pivot_pixel_ids, slopes, mean_charge, std_charge, _ = compute_slopes_and_stats(
        subset.rename(columns={"Charge": "Gain"}),
        runs=subset_runs,
        temperatures=subset_temps,
    )

    method_outdir = os.path.join(outdir, f"Slopes_V{V}_NSB{I}")
    os.makedirs(method_outdir, exist_ok=True)

    plot_camera(
        slopes,
        pivot_pixel_ids,
        camera_geom,
        title=f"Per-Pixel Slope vs Temp (V={V}V, NSB={I}) [p.e./°C]",
        fig_path=os.path.join(method_outdir, f"Slopes_V{V}_NSB{I}.png"),
        cmap="coolwarm",
    )

    plot_pixel_histogram(
        values=slopes,
        pixel_ids=pivot_pixel_ids,
        title=f"Pixel slope distribution (V={V}V, NSB={I})",
        fig_path=os.path.join(method_outdir, f"SlopeHist_V{V}_NSB{I}.png"),
        bins=60,
        xlabel="Slope [p.e./°C]",
    )

# ================================
# PLOT CAMERA AVG VS TEMP - GROUPED BY VOLTAGE WITH SHADED NSB SPREAD
# ================================
voltage_temp_data = defaultdict(lambda: defaultdict(list))
all_temp_data = defaultdict(list)

for (V, I), records in avg_charges_config.items():
    for rec in records:
        temp = rec["Temperature"]
        cam_avg = rec["CameraAvgCharge"]
        voltage_temp_data[V][temp].append(cam_avg)
        all_temp_data[temp].append(cam_avg)

# ============================================================
#              GLOBAL LINEAR FIT (all voltages + NSBs)
# ============================================================
global_temps = np.array(sorted(all_temp_data.keys()), dtype=float)
global_means = np.array([np.mean(all_temp_data[t]) for t in global_temps])
global_stds = np.array([np.std(all_temp_data[t]) for t in global_temps])
global_counts = np.array([len(all_temp_data[t]) for t in global_temps])
global_sems = global_stds / np.sqrt(global_counts)

fit_mask = global_sems > 0
weights = 1.0 / global_sems[fit_mask] ** 2

coeffs, cov = np.polyfit(
    global_temps[fit_mask], global_means[fit_mask], 1, w=np.sqrt(weights), cov=True
)
slope, intercept = coeffs
slope_err, intercept_err = np.sqrt(np.diag(cov))

logger.info(
    "Global fit: Charge = (%.4f ± %.4f) * T + (%.4f ± %.4f)",
    slope,
    slope_err,
    intercept,
    intercept_err,
)

T_fit = np.linspace(global_temps.min() - 1, global_temps.max() + 1, 200)
charge_fit = slope * T_fit + intercept

# ============================================================
#                        PLOTTING
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))

voltages = sorted(voltage_temp_data.keys())
colors = plt.cm.viridis(np.linspace(0, 1, len(voltages)))

for i, V in enumerate(voltages):
    temps = sorted(voltage_temp_data[V].keys())

    means, mins, maxs = [], [], []
    for temp in temps:
        charges_at_temp = voltage_temp_data[V][temp]
        means.append(np.mean(charges_at_temp))
        mins.append(np.min(charges_at_temp))
        maxs.append(np.max(charges_at_temp))

    temps = np.array(temps, dtype=float)
    means = np.array(means)
    mins = np.array(mins)
    maxs = np.array(maxs)

    ax.plot(
        temps,
        means,
        "o-",
        color=colors[i],
        linewidth=2,
        markersize=6,
        label=f"V={V} V",
        zorder=3,
    )
    ax.fill_between(temps, mins, maxs, color=colors[i], alpha=0.2, zorder=1)

ax.errorbar(
    global_temps,
    global_means,
    yerr=global_stds,
    fmt="o",
    color="black",
    markersize=8,
    capsize=5,
    capthick=1.5,
    zorder=5,
    label="Mean ± std (all V, NSB)",
)

fit_label = (
    f"Fit: $({slope:.4f} \\pm {slope_err:.4f})\\,T$\n"
    f"$+ ({intercept:.4f} \\pm {intercept_err:.4f})$"
)
ax.plot(
    T_fit,
    charge_fit,
    color="red",
    linewidth=2.5,
    linestyle="--",
    label=fit_label,
    zorder=4,
)

ax.set_xlabel("Temperature [°C]", fontsize=16)
ax.set_ylabel("Camera Average Normalized Charge [p.e.]", fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=14)
ax.tick_params(axis="both", which="minor", labelsize=12)
ax.grid(True, alpha=0.3)

ax.legend(loc="upper left", fontsize=12, frameon=True, framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(outdir, "CameraAvgCharge_ByVoltage_NSBSpread.png"), dpi=150)
plt.show()
plt.close()

logger.info("=== ANALYSIS COMPLETE ===")
