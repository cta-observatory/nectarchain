import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry
from ctapipe.io import read_table
from ctapipe.visualization import CameraDisplay
from gain_config import (
    BAD_MODULE_IDS,
    BAD_PIXELS_HV,
    Photostat_runs,
    SPE_runs,
    db_data_path,
    dirname,
    dirname2,
    outdir,
    path,
    temp_map,
)
from scipy.stats import linregress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

plt.style.use("../../utils/plot_style.mpltstyle")

os.makedirs(outdir, exist_ok=True)

camera_geom = CameraGeometry.from_name("NectarCam")


# ================================
# BAD MODULE FILTERING
# ================================
def get_bad_pixels_from_modules():
    """Get pixel IDs from bad modules"""
    # Get pixel to module mapping
    try:
        from nectarchain.makers.component.core import get_pixel_to_module_mapping

        pixel_to_module = get_pixel_to_module_mapping()
        module_id = np.array(
            [pixel_to_module.get(pix, -1) for pix in camera_geom.pix_id]
        )
    except ImportError:
        # Manual mapping: each module has 7 pixels
        n_pixels = len(camera_geom.pix_id)
        module_id = np.array([pix // 7 for pix in range(n_pixels)])

    # Bad modules list
    bad_module_ids = BAD_MODULE_IDS
    # Get all pixels belonging to bad modules
    bad_pixels = set()
    for mod in bad_module_ids:
        pixel_mask = module_id == mod
        pixel_ids = camera_geom.pix_id[pixel_mask]
        bad_pixels.update(pixel_ids)

    logging.info(f"Total pixels from bad modules: {len(bad_pixels)}")
    return bad_pixels


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
        hv_measured = dbinfos.tel[telid].monitoring_channel_voltages.voltage.datas
        hv_target = (
            dbinfos.tel[telid]
            .monitoring_channel_voltages
            .target_voltage
            .datas
        )  # input here the target voltage

    except Exception as e:
        print(f"Error retrieving HV data for run {run}, telid {telid}: {e}")
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


# Get bad pixels from modules at the start
BAD_MODULE_PIXELS = get_bad_pixels_from_modules()

# ================================
# HELPERS
# ================================


def load_run(run):
    """Load per-pixel gain values from a run"""
    spe_filename = os.path.join(
        dirname,
        (
            f"FlatFieldSPENominalStdNectarCAM_run{run}_"
            f"maxevents5000_LocalPeakWindowSum_"
            f"window_shift_4_window_width_8.h5"
        ),
    )

    photostat_filename = os.path.join(
        dirname2,
        (
            f"PhotoStatisticNectarCAM_FFrun{run}_"
            f"LocalPeakWindowSum_window_shift_4_window_width_8_"
            f"Pedrun{run}_FullWaveformSum_maxevents5000.h5"
        ),
    )

    if os.path.exists(spe_filename):
        filename = spe_filename
        h5path = "/data/SPEfitContainer_0"
        label = "SPE"
    elif os.path.exists(photostat_filename):
        filename = photostat_filename
        h5path = "/data/GainContainer_0"
        label = "Photostat"
    else:
        print(f"⚠️ No file found for run {run}")
        return None, None, None

    data = read_table(filename, path=h5path)
    high_gain_lw = [x[0] for x in data["high_gain"][0]]
    pixels_id = data["pixels_id"][0]
    logging.info(f"Loaded run {run} ({label}): {len(pixels_id)} pixels")

    return pixels_id, high_gain_lw, label


def detect_bad_pixels_per_run(pixels_id, gains, nsigma=5):
    """Return a set of bad pixel IDs for a single run based on sigma clipping"""
    vals = np.array(gains, dtype=float)
    mean_val = np.nanmean(vals)
    std_val = np.nanstd(vals)
    bad_mask = np.abs(vals - mean_val) > nsigma * std_val
    return set(np.array(pixels_id)[bad_mask])


def compute_slopes_and_stats(df_pixels, runs, temperatures):
    """Compute per-pixel slopes, mean, std"""
    pixel_gains = df_pixels.pivot_table(
        index="Pixel", columns="Run", values="Gain"
    ).reindex(columns=runs)
    pivot_pixel_ids = pixel_gains.index.values
    pixel_gains_array = pixel_gains.values

    slopes = []
    for i in range(pixel_gains_array.shape[0]):
        gains = pixel_gains_array[i, :]
        mask = ~np.isnan(gains)
        if np.sum(mask) > 1:
            slope, _, _, _, _ = linregress(np.array(temperatures)[mask], gains[mask])
        else:
            slope = np.nan
        slopes.append(slope)

    slopes = np.array(slopes)
    mean_gain = np.nanmean(pixel_gains_array, axis=1)
    std_gain = np.nanstd(pixel_gains_array, axis=1)

    return pivot_pixel_ids, slopes, mean_gain, std_gain, pixel_gains_array


def plot_camera(values, pixel_ids, camera_geom, name, fig_path):
    """Plot camera with already cleaned pixels in engineering frame"""

    vals = np.asarray(values, dtype=float)

    # Slice geometry
    active_geom = camera_geom[pixel_ids]

    # 🔑 Force engineering frame
    active_geom = active_geom.transform_to(EngineeringCameraFrame())

    fig, ax = plt.subplots(figsize=(8, 6))

    disp = CameraDisplay(geometry=active_geom, image=vals, cmap=plt.cm.coolwarm, ax=ax)

    disp.add_colorbar(label=r"$\mathrm{ADC}\,\mathrm{p.e.}^{-1}/^\circ\mathrm{C}$")

    ax.set_title("Gain variation")

    plt.savefig(
        os.path.join(fig_path, f"Camera_{name.replace(' ', '_')}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_histogram(values, name, outdir, xlabel, bins=50):
    """
    Plot and save a histogram of per-pixel quantities
    with camera average and standard deviation.
    Legend kept inside but compact.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]

    mean_val = np.mean(vals)
    std_val = np.std(vals)

    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.hist(
        vals,
        bins=bins,
        histtype="stepfilled",
        alpha=0.6,
        label="Pixels",
    )

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

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of pixels")
    ax.set_title(name)
    ax.grid(True)

    # 🔑 Compact legend INSIDE the plot
    ax.legend(
        loc="upper right",
        fontsize=9,  # smaller font
        frameon=True,
        framealpha=0.85,  # slightly transparent
        borderpad=0.3,
        labelspacing=0.3,
        handlelength=1.5,
    )

    fig.savefig(
        os.path.join(outdir, f"Hist_{name.replace(' ', '_')}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


# ================================
# MAIN PROCESSING
# ================================
for run_type, run_list in zip(["SPE", "Photostat"], [SPE_runs, Photostat_runs]):
    all_pixel_records = []
    global_bad_pixels = set()  # pixels bad in any run

    module_bad_pixels = set(BAD_MODULE_PIXELS)

    for run in run_list:
        # Load run
        pixels_id, gains, label = load_run(run)
        if pixels_id is None:
            continue

        # 1. HV bad pixels for this run
        hv_bad_pixels = get_bad_hv_pixels_db(
            run,
            path=path,
            db_data_path=db_data_path,
            hv_tolerance=4.0,
            telid=0,
            verbose=True,
        )

        # 2. Combine fixed bad pixels (do NOT use them in statistical clipping)
        fixed_bad_pixels = hv_bad_pixels.union(module_bad_pixels)

        # 3. Remove fixed bad pixels before statistical clipping
        filtered_pixel_ids = []
        filtered_gains = []

        for pid, g in zip(pixels_id, gains):
            if pid not in fixed_bad_pixels:
                filtered_pixel_ids.append(pid)
                filtered_gains.append(g)

        # 4. Statistical outliers computed ONLY on the clean subset
        bad_rel = detect_bad_pixels_per_run(
            filtered_pixel_ids, filtered_gains, nsigma=5
        )

        # Map relative statistical indices → actual pixel IDs
        stat_bad_pixels = {filtered_pixel_ids[i] for i in bad_rel}

        # 5. Global set = fixed bad + statistical bad
        bad_ids_run = fixed_bad_pixels.union(stat_bad_pixels)
        global_bad_pixels.update(bad_ids_run)

        # 6. Record all pixels (for dataframe building)
        temp = temp_map.get(run, np.nan)
        for pid, g in zip(pixels_id, gains):
            all_pixel_records.append(
                {"Run": run, "Temperature": temp, "Pixel": pid, "Gain": g}
            )
        logger.info("Number of pixels in run %s: %d", run, len(pixels_id))

        logger.info(
            "bad_module_pixels=%d, bad_hv_pixels=%d, sigma_clipped=%d, "
            "total_removed=%d, global_removed=%d",
            len(BAD_MODULE_PIXELS),
            len(hv_bad_pixels),
            len(stat_bad_pixels),
            len(bad_ids_run),
            len(global_bad_pixels),
        )

    # 7. Build full dataframe
    df_pixels = pd.DataFrame(all_pixel_records)
    logging.info(f"Total pixels in dataframe: {len(df_pixels)}")

    # 8. Final mask applied to dataframe
    df_pixels_clean = df_pixels[~df_pixels["Pixel"].isin(global_bad_pixels)]

    # 9. Temperatures vector
    temperatures = [temp_map[r] for r in run_list]

    # Compute slopes and stats on cleaned pixels
    (
        pivot_pixel_ids,
        slopes,
        mean_gain,
        std_gain,
        pixel_gains_array,
    ) = compute_slopes_and_stats(df_pixels_clean, run_list, temperatures)

    # Camera plots
    plot_camera(slopes, pivot_pixel_ids, camera_geom, f"{run_type} Slopes", outdir)
    plot_camera(
        mean_gain, pivot_pixel_ids, camera_geom, f"{run_type} Mean Gain", outdir
    )
    plot_camera(std_gain, pivot_pixel_ids, camera_geom, f"{run_type} Std Gain", outdir)

    # Histograms
    plot_histogram(
        slopes,
        f"{run_type} Gain variation over temperature",
        outdir,
        xlabel=r"Gain slope (ADC p.e.$^{-1}$ C°$^{-1}$)",
    )

    plot_histogram(
        mean_gain,
        f"{run_type} Mean Gain",
        outdir,
        xlabel=r"Mean gain (ADC p.e.$^{-1}$)",
    )

    plot_histogram(
        std_gain,
        f"{run_type} Std Gain",
        outdir,
        xlabel=r"Gain standard deviation (ADC p.e.$^{-1}$)",
    )

    # Save CSV
    slopes_df = pd.DataFrame(
        {"Pixel": pivot_pixel_ids, "Slope": slopes, "Mean": mean_gain, "Std": std_gain}
    )
    slopes_df.to_csv(os.path.join(outdir, f"{run_type}_Pixel_slopes.csv"), index=False)

    # Mean ± Std vs Temperature
    grouped = df_pixels_clean.groupby("Temperature")["Gain"]
    mean_vs_temp = grouped.mean()
    std_vs_temp = grouped.std()
    count_vs_temp = grouped.count()

    temps = mean_vs_temp.index.values
    means = mean_vs_temp.values
    stds = std_vs_temp.values
    counts = count_vs_temp.values

    yerr = stds / np.sqrt(counts)
    yerr = np.nan_to_num(yerr, nan=0.0, posinf=0.0, neginf=0.0)

    plt.figure(figsize=(8, 5))
    plt.errorbar(temps, means, yerr=yerr, fmt="o", capsize=5)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Gain")
    plt.grid(True)
    plt.savefig(os.path.join(outdir, f"{run_type}_MeanStd_vs_Temperature.png"), dpi=150)
    plt.close()

    logging.info(
        f"Processed {run_type}: total pixels={len(df_pixels['Pixel'].unique())}, "
        f"global_removed={len(global_bad_pixels)}, "
        f"kept={len(pivot_pixel_ids)}"
    )


# ================================
# Combined SPE vs Photostat
# ================================
fig, ax = plt.subplots(figsize=(9, 6))

for run_type, run_list, color in zip(
    ["SPE", "Photostat"], [SPE_runs, Photostat_runs], ["tab:blue", "tab:red"]
):
    all_pixel_records = []
    global_bad_pixels = set()

    # Collect data and bad pixels per run
    for run in run_list:
        pixels_id, gains, label = load_run(run)
        if pixels_id is None:
            continue
        bad_ids_run = detect_bad_pixels_per_run(pixels_id, gains, nsigma=5)
        global_bad_pixels.update(bad_ids_run)

        temp = temp_map.get(run, np.nan)
        for pid, g in zip(pixels_id, gains):
            all_pixel_records.append(
                {"Run": run, "Temperature": temp, "Pixel": pid, "Gain": g}
            )

    df_pixels = pd.DataFrame(all_pixel_records)

    # Remove bad module pixels AND bad pixels flagged by sigma clipping
    all_bad_pixels = global_bad_pixels.union(BAD_MODULE_PIXELS)
    df_pixels_clean = df_pixels[~df_pixels["Pixel"].isin(all_bad_pixels)]

    # Group by temperature: mean, std, count
    grouped = df_pixels_clean.groupby("Temperature")["Gain"]
    mean_vs_temp = grouped.mean()
    std_vs_temp = grouped.std()
    count_vs_temp = grouped.count()

    temps = mean_vs_temp.index.values.astype(float)
    means = mean_vs_temp.values
    stds = std_vs_temp.values
    counts = count_vs_temp.values

    # SEM for error bars and fit weights
    sem = stds / np.sqrt(counts)
    sem = np.nan_to_num(sem, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Weighted linear fit (weights = 1/SEM²) ---
    # Guard: only fit where SEM > 0
    fit_mask = sem > 0
    weights = np.zeros_like(sem)
    weights[fit_mask] = 1.0 / sem[fit_mask] ** 2

    coeffs, cov = np.polyfit(
        temps[fit_mask], means[fit_mask], 1, w=np.sqrt(weights[fit_mask]), cov=True
    )
    slope, intercept = coeffs
    slope_err, intercept_err = np.sqrt(np.diag(cov))

    print(
        f"[{run_type}] Gain = ({slope:.4f} ± {slope_err:.4f}) * T "
        f"+ ({intercept:.4f} ± {intercept_err:.4f})"
    )

    # Fit line
    T_fit = np.linspace(temps.min() - 1, temps.max() + 1, 200)
    fit_line = slope * T_fit + intercept

    # --- Plot data points with error bars ---
    ax.errorbar(
        temps,
        means,
        yerr=sem,
        fmt="o",
        capsize=5,
        color=color,
        label=f"{run_type} data",
        zorder=3,
    )

    # --- Plot fit line ---
    fit_label = (
        f"{run_type} fit: $({slope:.4f} \\pm {slope_err:.4f})\\,T$\n"
        f"$+ ({intercept:.4f} \\pm {intercept_err:.4f})$"
    )
    ax.plot(T_fit, fit_line, color=color, linewidth=2, linestyle="--", label=fit_label)

ax.set_xlabel("Temperature [°C]", fontsize=14)
ax.set_ylabel("Gain [ADC p.e.$^{-1}$]", fontsize=14)
ax.tick_params(axis="both", which="major", labelsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc="best")

plt.tight_layout()
plt.savefig(
    os.path.join(outdir, "SPE_vs_Photostat_MeanStd_vs_Temperature.png"), dpi=150
)
plt.close()

logging.info("Done. Combined plot saved.")

# ================================
# Average values for camera plots
# ================================
avg_values = {}

for run_type, run_list in zip(["SPE", "Photostat"], [SPE_runs, Photostat_runs]):
    all_pixel_records = []
    global_bad_pixels = set()

    for run in run_list:
        pixels_id, gains, label = load_run(run)
        if pixels_id is None:
            continue
        bad_ids_run = detect_bad_pixels_per_run(pixels_id, gains, nsigma=5)
        global_bad_pixels.update(bad_ids_run)

        temp = temp_map.get(run, np.nan)
        for pid, g in zip(pixels_id, gains):
            all_pixel_records.append(
                {"Run": run, "Temperature": temp, "Pixel": pid, "Gain": g}
            )

    df_pixels = pd.DataFrame(all_pixel_records)

    # Remove bad module pixels AND bad pixels flagged by sigma clipping
    all_bad_pixels = global_bad_pixels.union(BAD_MODULE_PIXELS)
    df_pixels_clean = df_pixels[~df_pixels["Pixel"].isin(all_bad_pixels)]
    temperatures = [temp_map[r] for r in run_list]

    # Compute slopes and stats on cleaned pixels
    (
        pivot_pixel_ids,
        slopes,
        mean_gain,
        std_gain,
        pixel_gains_array,
    ) = compute_slopes_and_stats(df_pixels_clean, run_list, temperatures)

    # Store averages
    avg_values[f"{run_type} Slopes"] = np.nanmean(slopes)
    avg_values[f"{run_type} Mean Gain"] = np.nanmean(mean_gain)
    avg_values[f"{run_type} Std Gain"] = np.nanmean(std_gain)

# Print the averages
logging.info("Average values for camera plots:")

for name, value in avg_values.items():
    logging.info(f"{name}: {value:.4f}")

pixel_numbers = [int(p) for p in all_bad_pixels]

logging.info(f"Pixel numbers: {pixel_numbers}")
