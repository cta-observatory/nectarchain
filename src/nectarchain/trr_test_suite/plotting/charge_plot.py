#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from collections import defaultdict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    pedestal_file,
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

plt.style.use("../../utils/plot_style.mpltstyle")

os.makedirs(outdir, exist_ok=True)

METHOD = "LocalPeakWindowSum"
WINDOW = 16
SHIFT = 4
NSIGMA_BADPIX = 5

camera_geom = CameraGeometry.from_name("NectarCam").transform_to(
    EngineeringCameraFrame()
)


"""
Charge normalization analysis script:
- Reads pedestal maps depending on (temperature, FF voltage, NSB intensity)
- Subtracts baseline dynamically for each run
- Applies FF correction, SPE gain normalization, and generates per-pixel maps
- Produces camera-average vs temperature plots
(all configurations on the same figure)
- Produces per-pixel slope maps per (Voltage, NSB)
- Excludes bad modules from calculations and plots
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


# Get all bad pixel IDs from bad modules
BAD_PIXELS_FROM_MODULES = get_bad_pixels_from_modules(BAD_MODULE_IDS)
logger.info("Total bad pixels from bad modules: %d", len(BAD_PIXELS_FROM_MODULES))
bad_ids = None


def get_bad_hv_pixels_db(
    run, path, db_data_path, hv_tolerance=4.0, telid=0, verbose=False
):
    """
    Identify pixels for which |measured HV - target HV| > hv_tolerance.

    Parameters
    ----------
    run : int
        Run number.
    path : str
        Path to rawdata (needed by DBInfos.init_from_run).
    db_data_path : str
        Path to sqlite monitoring database.
    hv_tolerance : float
        Allowed HV deviation relative to target HV (default: 4 V).
    telid : int
        Telescope ID.
    verbose : bool
        Print detailed information.

    Returns
    -------
    set
        Pixel IDs failing the HV criterion.
    """
    """
    # Load DBInfos
    dbinfos = DBInfos.init_from_run(run, path=path, dbpath=db_data_path, verbose=False)
    dbinfos.connect("monitoring_drawer_temperatures", "monitoring_channel_voltages")

    try:
        hv_measured = (
            dbinfos.tel[telid]
            .monitoring_channel_voltages
            .voltage
            .datas
        )

        hv_target = (
            dbinfos.tel[telid]
            .monitoring_channel_voltages
            .target_voltage
            .datas
        )  # input here the target voltage

    except Exception as e:
        logger.error(
        "Error retrieving HV data for run %s, telid %s: %s", run, telid, e
        )

        return set()

    # Mean values per pixel (ignore obviously wrong measurement frames)
    hv_measured_mean = np.nanmean(hv_measured, where=hv_measured > 400, axis=-1)
    hv_target_mean   = np.nanmean(hv_target,  where=hv_target > 400,   axis=-1)

    # Apply Vincent's condition: |measured - target| > tolerance
    deviation = np.abs(hv_measured_mean - hv_target_mean)
    bad_pixels = set(np.where(deviation > hv_tolerance)[0])

    if verbose:
        logger.info(
            "Run %s: %d bad pixels (|HV_meas - HV_target| > %.1f V)",
            run,
            len(bad_pixels),
            hv_tolerance,
        )
        logger.debug("Bad pixel IDs: %s", sorted(bad_pixels))

    """
    bad_pixels = BAD_PIXELS_HV
    return bad_pixels


# ================================
# HELPER FUNCTIONS
# ================================
def load_charge_file(filename, dataset_name="FLATFIELD"):
    with h5py.File(filename, "r") as f:
        logger.info("Filename: %s", filename)
        container = f.get("data/ChargesContainer_0")
        dataset = container.get(dataset_name)
        dataset0 = dataset[0]
        pixels_id = dataset0["pixels_id"]
        charges_hg = dataset0["charges_hg"].squeeze()
    return np.array(pixels_id), np.array(charges_hg)


def read_gain_file(spe_run, window, shift, method, dirname):
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
    vals = np.array(charges, dtype=float)
    mean_val = np.nanmean(vals)
    std_val = np.nanstd(vals)
    bad_mask = np.abs(vals - mean_val) > nsigma * std_val
    return set(np.array(pixels_id)[bad_mask])


def compute_slopes_and_stats(df_pixels, runs, temperatures):
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
    """Plot camera display, automatically masking bad pixels from modules"""
    # Filter out bad pixels from modules
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


def plot_pixel_histogram(
    values,
    pixel_ids,
    title,
    fig_path,
    bins=50,
    xlabel="Value",
):
    """
    Plot histogram of per-pixel values, excluding bad pixels,
    with camera average and standard deviation.
    """

    # Exclude bad pixels
    good_mask = np.array([pid not in bad_ids for pid in pixel_ids])
    good_values = np.array(values, dtype=float)[good_mask]

    # Remove NaNs / infs
    good_values = good_values[np.isfinite(good_values)]

    mean_val = np.mean(good_values)
    std_val = np.std(good_values)

    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.hist(
        good_values,
        bins=bins,
        histtype="stepfilled",
        alpha=0.7,
        label="Pixels",
    )

    # Mean and ±1σ
    ax.axvline(
        mean_val,
        linestyle="--",
        linewidth=2,
        label=rf"Mean = {mean_val:.3e}",
    )
    ax.axvline(
        mean_val + std_val,
        linestyle=":",
        linewidth=2,
        label=rf"+1σ = {std_val:.3e}",
    )
    ax.axvline(
        mean_val - std_val,
        linestyle=":",
        linewidth=2,
        label=rf"-1σ = {std_val:.3e}",
    )

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Number of pixels", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="both", which="minor", labelsize=12)
    # ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # 🔑 Compact legend INSIDE
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
    ff_run = ff_map.get(temp)
    if ff_run is None:
        raise ValueError(f"No FF run defined for temp {temp}")
    ff_filename = os.path.join(ff_dir, f"FF_calibration_run{ff_run}.dat")
    df_ff = pd.read_csv(ff_filename, delim_whitespace=True)
    ff_dict = dict(zip(df_ff["pixel_id"], df_ff["FF_coef_independent_way"]))
    return ff_dict


# ================================
# LOAD PEDESTALS
# ================================
ped_data = np.load(pedestal_file)
ped_temperature = ped_data["temperature"]
ped_baseline = ped_data["baseline"]
ped_pixel_ids = ped_data["pixel_ids"]
ped_vvalff = ped_data["vvalff"]
ped_ivalnsb = ped_data["ivalnsb"]

# ================================
# PROCESSING
# ================================
avg_charges_config = defaultdict(list)
all_pixel_records = []

for temp, runs in temp_map.items():
    voltages = vvalff_map[temp]
    nsbs = ivalnsb_map[temp]

    for i, run in enumerate(runs):
        print(run)
        Voo = voltages[i]
        Inn = nsbs[i]

        # pedestal
        t_idx = int(np.argmin(np.abs(ped_temperature - temp)))
        iff = int(np.argmin(np.abs(ped_vvalff - Voo)))
        jnsb = int(np.argmin(np.abs(ped_ivalnsb - Inn)))
        ped_vals = ped_baseline[t_idx, iff, jnsb, :]

        ped_dict = dict(zip(ped_pixel_ids, ped_vals))

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

        # load charges
        charge_filename = os.path.join(
            dirname,
            (
                f"ChargesNectarCAMCalibration_run{run}_"
                f"maxevents5000_{METHOD}_window_shift_{SHIFT}_"
                f"window_width_{WINDOW}.h5"
            ),
        )

        if not os.path.exists(charge_filename):
            continue
        pixels_id, charges_hg = load_charge_file(charge_filename)

        # Check if there are multiple events (i.e., charges_hg is 2D)
        if charges_hg.ndim > 1:
            charges_hg = np.mean(charges_hg, axis=0)
        else:
            charges_hg = charges_hg

        logger.info("Temperature: %s, Run: %s", temp, run)
        logger.info("Mean of charges_hg: %.6f", np.mean(charges_hg))

        # pedestal subtraction + FF + SPE normalization
        charges_pedcorr = np.array(
            [
                charges_hg[i] - ped_dict.get(pid, np.nan) * WINDOW
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

        # 1. Get HV-bad pixels
        hv_bad_pixels = get_bad_hv_pixels_db(
            run,
            path=path,
            db_data_path=db_data_path,
            hv_tolerance=4.0,
            telid=0,
            verbose=True,
        )

        # 2. Get module-bad pixels
        module_bad_pixels = set(BAD_PIXELS_FROM_MODULES)
        gain_bad_pixels = set(BAD_PIXELS_GAIN)

        # Combine all non-statistical bad pixels
        hv_bad_pixels = set(hv_bad_pixels)
        hv_bad_pixel = set(hv_bad_pixels)
        gain_bad_pixels = set(gain_bad_pixels)
        non_statistical_bad = hv_bad_pixels.union(module_bad_pixels).union(
            gain_bad_pixels
        )

        logger.info(
            "Fixed bad pixels (HV + modules + gain): %d", len(non_statistical_bad)
        )

        # 3. Select ONLY the remaining good pixels
        remaining_pixel_ids = [p for p in pixels_id if p not in non_statistical_bad]
        remaining_charges = [
            charges_normalized[i]
            for i, p in enumerate(pixels_id)
            if p not in non_statistical_bad
        ]

        # 4. Compute statistical bad pixels only on remaining ones
        stat_bad_relative = detect_bad_pixels(
            remaining_pixel_ids, remaining_charges, nsigma=NSIGMA_BADPIX
        )

        # Map the relative indices back to pixel IDs
        stat_bad_pixels = {remaining_pixel_ids[i] for i in stat_bad_relative}

        logger.info(
            "Statistical bad pixels (after exclusions): %d", len(stat_bad_pixels)
        )

        # 5. Final union of ALL bad pixels
        bad_ids = non_statistical_bad.union(stat_bad_pixels)

        logger.info("Total bad pixels: %d", len(bad_ids))

        # store per-pixel (excluding bad pixels)
        for pid, charge in zip(pixels_id, charges_normalized):
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

        # store camera average (excluding bad pixels)
        good_charges = [
            g for pid, g in zip(pixels_id, charges_normalized) if pid not in bad_ids
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

    # compute mean and standard error instead of std
    cam_avgs = df_cfg["CameraAvgCharge"].values
    # Count only good pixels for SEM calculation
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

# Replace the problematic section around line 880 with this:
logger.info(f"Total configurations processed: {len(avg_charges_config)}")
for (V, I), records in avg_charges_config.items():
    logger.info(f"Config V={V}, NSB={I}: {len(records)} temperature points")

plt.xlabel("Temperature (°C)", fontsize=16)
plt.ylabel("Camera Average Normalized Charge (p.e.)", fontsize=16)
plt.tick_params(axis="both", which="major", labelsize=14)
plt.tick_params(axis="both", which="minor", labelsize=12)
plt.grid(True)

# Get legend handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

# Only sort and display legend if there are items to show
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
# PLOT AVERAGE PEDESTALS PER TEMPERATURE
# ================================
avg_ped_per_temp = []
std_ped_per_temp = []
temps_sorted = sorted(temp_map.keys())

for temp in temps_sorted:
    runs = temp_map[temp]
    voltages = vvalff_map[temp]
    nsbs = ivalnsb_map[temp]

    ped_vals_all = []
    for i, run in enumerate(runs):
        Voo = voltages[i]
        Inn = nsbs[i]

        # get indices for pedestal arrays
        t_idx = int(np.argmin(np.abs(ped_temperature - temp)))
        iff = int(np.argmin(np.abs(ped_vvalff - Voo)))
        jnsb = int(np.argmin(np.abs(ped_ivalnsb - Inn)))
        ped_vals = ped_baseline[t_idx, iff, jnsb, :]

        # Filter out bad pixels from modules
        good_pixel_mask = np.array([pid not in bad_ids for pid in ped_pixel_ids])
        ped_vals_good = ped_vals[good_pixel_mask]

        ped_vals_all.append(ped_vals_good)

    # flatten all good pixel pedestals for this temperature across runs
    ped_vals_all = np.concatenate(ped_vals_all)
    avg_ped_per_temp.append(np.mean(ped_vals_all))
    std_ped_per_temp.append(np.std(ped_vals_all))

# plot
plt.figure(figsize=(8, 5))
plt.errorbar(temps_sorted, avg_ped_per_temp, yerr=std_ped_per_temp, fmt="o-", capsize=4)
plt.xlabel("Temperature (°C)")
plt.ylabel("Average Pedestal (ADC)")
plt.title(
    f"Camera Average Pedestal vs Temperature "
    f"(excluding {len(BAD_MODULE_IDS)} bad modules)"
)

plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "AvgPedestal_vs_Temperature.png"), dpi=150)
plt.show()

print("\n=== Summary ===")
print(f"Bad modules excluded: {len(BAD_MODULE_IDS)}")
print(f"Bad pixels excluded: {len(BAD_PIXELS_FROM_MODULES)}")
print(f"Total pixels analyzed: {len(ped_pixel_ids) - len(BAD_PIXELS_FROM_MODULES)}")
