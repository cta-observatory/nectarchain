import argparse
import copy
import json
import logging
import os
from pathlib import Path

import astropy.units as u
import numpy as np
import numpy.ma as ma
import tabulate as tab
from astropy.io import ascii
from astropy.table import Table
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.image.toymodel import Gaussian
from ctapipe.io import EventSource
from ctapipe.io.hdf5tableio import HDF5TableReader

# ctapipe modules
from ctapipe.visualization import CameraDisplay
from ctapipe_io_nectarcam.constants import N_PIXELS
from iminuit import Minuit
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize._numdiff import approx_derivative

from nectarchain.data.container import PhotostatContainer
from nectarchain.data.management import DataManagement
from nectarchain.makers.calibration import (
    FlatFieldSPENominalStdNectarCAMCalibrationTool,
    PhotoStatisticNectarCAMCalibrationTool,
)
from nectarchain.makers.extractor.utils import CtapipeExtractor
from nectarchain.utils.constants import ALLOWED_CAMERAS

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    filename=f"{os.environ.get('NECTARCHAIN_LOG', '/tmp')}/{os.getpid()}/{Path(__file__).stem}_{os.getpid()}.log",
)
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

plt.style.use("../../utils/plot_style.mpltstyle")

parser = argparse.ArgumentParser(
    description="Run NectarCAM photostatistics analysis",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--FF_run_number", required=True, help="Run number", type=int)
parser.add_argument("--SPE_run_number", required=True, help="SPE run number", type=int)
parser.add_argument(
    "-p",
    "--run-path",
    default=f'{os.environ.get("NECTARCAMDATA", "").strip()}',
    help="Path to run file",
)

parser.add_argument(
    "-c",
    "--camera",
    choices=ALLOWED_CAMERAS,
    default=[camera for camera in ALLOWED_CAMERAS if "QM" in camera][0],
    help="Process data for a specific NectarCAM camera.",
    type=str,
)

# Accept True/False as string
parser.add_argument(
    "-w",
    "--add-variance",
    action="store_true",
    help="Enable variance correction (False by default)",
)
parser.add_argument(
    "--SPE_config",
    choices=[
        "HHVfree",
        "HHVfixed",
        "nominal",
    ],
    default="nominal",
    help="SPE configuration to use, either HHVfree, HHVfixed or nominal.\
        From ICRC2025 proceedings, we recommend to use resoltion at nominal for the SPE fit.",
)
parser.add_argument(
    "--method",
    choices=[
        "FullWaveformSum",
        "FixedWindowSum",
        "GlobalPeakWindowSum",
        "LocalPeakWindowSum",
        "SlidingWindowMaxSum",
        "TwoPassWindowSum",
    ],
    default="GlobalPeakWindowSum",
    help="charge extractor method",
    type=str,
)
parser.add_argument(
    "--extractor_kwargs",
    default={"window_width": 8, "window_shift": 4},
    help="charge extractor parameters",
    type=json.loads,
)
parser.add_argument(
    "-v",
    "--verbosity",
    help="set the verbosity level of logger",
    default="INFO",
    choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
    type=str,
)

args = parser.parse_args()


def pre_process_fits(filename, camera, pdf):
    with HDF5TableReader(filename) as h5_table:
        assert h5_table._h5file.isopen == True
        for container in h5_table.read(
            "/data/PhotostatContainer_0", PhotostatContainer
        ):
            log.info(container.as_dict())
            break
    h5_table.close()

    total_pixels = N_PIXELS

    # Generate the full expected pixel ID list
    expected_pixels = np.arange(total_pixels)
    container_dict = container.as_dict()
    log.info(f"number of valid pixels : {len(container_dict['is_valid'])}")

    # Find missing pixel IDs
    existing_pixels = container_dict["pixels_id"]
    missing_pixels = np.setdiff1d(expected_pixels, existing_pixels)

    # Determine the shape of the 'high_gain' values
    hg_shape = (
        container_dict["high_gain"].shape[1]
        if len(container_dict["high_gain"].shape) > 1
        else 1
    )
    lg_shape = (
        container_dict["low_gain"].shape[1]
        if len(container_dict["low_gain"].shape) > 1
        else 1
    )
    charge_hg_std_shape = (
        container_dict["charge_hg_std"].shape[1]
        if len(container_dict["charge_hg_std"].shape) > 1
        else 1
    )

    # Create missing entries with zeros matching the correct shape
    missing_entries = {
        "pixels_id": missing_pixels,
        "high_gain": np.zeros((len(missing_pixels), hg_shape)),
        "low_gain": np.zeros((len(missing_pixels), lg_shape)),
        "charge_hg": np.zeros(len(missing_pixels)),  # Ensures same shape as 'high_gain'
        "charge_hg_std": np.zeros((len(missing_pixels), charge_hg_std_shape)),
    }

    # Merge original and missing data
    merged_pixel_ids = np.concatenate([existing_pixels, missing_entries["pixels_id"]])
    merged_hg = np.concatenate(
        [container_dict["high_gain"], missing_entries["high_gain"]], axis=0
    )
    merged_lg = np.concatenate(
        [container_dict["low_gain"], missing_entries["low_gain"]], axis=0
    )
    merged_charge_hg = np.concatenate(
        [container_dict["charge_hg"], missing_entries["charge_hg"]]
    )
    merged_charge_hg_std = np.concatenate(
        [container_dict["charge_hg"], missing_entries["charge_hg"]]
    )

    # Sort by pixel_id to maintain order
    sorted_indices = np.argsort(merged_pixel_ids)
    container_dict["pixels_id"] = merged_pixel_ids[sorted_indices]
    container_dict["high_gain"] = merged_hg[sorted_indices]
    container_dict["low_gain"] = merged_lg[sorted_indices]
    container_dict["charge_hg"] = merged_charge_hg[sorted_indices]
    container_dict["charge_hg_std"] = merged_charge_hg_std[sorted_indices]
    mask_check_hg = [a <= 0 for a in container_dict["high_gain"][:, 0]]
    masked_hg = ma.masked_array(container_dict["high_gain"][:, 0], mask=mask_check_hg)

    high_gain = container_dict["high_gain"][:, 0]
    n_pe = np.divide(
        container_dict["charge_hg"],
        high_gain,
        out=np.zeros_like(high_gain, dtype=float),
        where=high_gain > 0,
    )
    std_n_pe = np.sqrt(
        np.divide(
            container_dict["charge_hg_std"] * n_pe,
            container_dict["charge_hg"],
            out=np.zeros_like(high_gain, dtype=float),
            where=high_gain != 0,
        )
    )

    mask = [a == 0 for a in std_n_pe]

    sigma_masked = ma.masked_array(std_n_pe, mask=mask)
    n_pe = ma.masked_array(n_pe, mask=mask)

    # Perform some plots
    fig0 = plt.figure(figsize=(6, 5))
    ax = plt.subplot()
    disp = CameraDisplay(geometry=camera, show_frame=False)

    disp.image = n_pe
    disp.add_colorbar()
    # disp.set_limits_minmax(140, 165)

    cbar1 = fig0.axes[-1]
    cbar1.set_ylabel(
        r"Illumination, $n_{\rm PE}$", rotation=90, labelpad=15, fontsize=16
    )
    cbar1.tick_params(labelsize=16)  # Increase tick label size on colorbar
    # Axis labels
    ax.set_xlabel("x (m)", fontsize=16)
    ax.set_ylabel("y (m)", fontsize=16)
    # Tick label size
    ax.tick_params(axis="both", which="major", labelsize=16)
    # Title
    plt.title("Data", fontsize=18)
    pdf.savefig(fig0)

    dict_missing_pix = {
        "Missing pixels": len(missing_pixels),
        "high_gain = 0": ma.count_masked(masked_hg) - len(missing_pixels),
    }

    labels = [
        f'Missing pixels, number = {dict_missing_pix["Missing pixels"]}',
        f'high gain = 0, number = {dict_missing_pix["high_gain = 0"]}',
    ]

    return (
        n_pe,
        std_n_pe,
        sigma_masked,
        dict_missing_pix,
        container_dict["high_gain"],
        container_dict["low_gain"],
        container_dict["charge_hg"],
    )


# Fit using ctapipe
# First step of the fitting process
def Gaussian_model(camera, array=[1000.0, 0.0, 0.0, 1.5, 1.5]):
    A, x, y, std_x, std_y = array
    model = A * (
        Gaussian(x * u.m, y * u.m, std_x * u.m, std_y * u.m, psi="0d").pdf(
            camera.pix_x, camera.pix_y
        )
    )
    return model


def define_delete_out(sigma, data):
    mean = np.mean(data)
    std = np.std(data)
    outliers = [np.abs(data - mean) > 3 * std]
    sigma = ma.masked_array(sigma, mask=outliers)
    data = ma.masked_array(data, mask=outliers)
    return data, sigma, outliers


def optimize_with_outlier_rejection(sigma, data, camera, pdf):
    # least-squares score function = sum of data residuals squared
    def lsq(a0, a1, a2, a3):
        a4 = (
            a3  # This equality comes from assumption that the 2D-Gaussian is symmetric.
        )
        return np.sum(
            (n_pe - Gaussian_model(camera, [a0, a1, a2, a3, a4])) ** 2
            / (sigma_masked**2)
        )

    # Apply outlier mask based on data
    n_pe, sigma_masked, mask_upd = define_delete_out(sigma, data)

    # Fit with Minuit using previous best parameters
    minuit = Minuit(lsq, a0=1000.0, a1=0.0, a2=0.0, a3=1.5)
    minuit.migrad()

    if not minuit.fmin.is_valid:
        log.info("Warning: Fit did not converge! Stopping iteration.")
    log.info(f"covariance table: {tab.tabulate(*minuit.covariance.to_table())}")
    log.info(
        f"Fit new parameters: amplitude = {minuit.values['a0']}, "
        f"x = {minuit.values['a1']}, y = {minuit.values['a2']}, "
        f"length = {minuit.values['a3']}"
    )

    model = Gaussian_model(
        camera,
        [
            minuit.values["a0"],
            minuit.values["a1"],
            minuit.values["a2"],
            minuit.values["a3"],
            minuit.values["a3"],
        ],
    )
    residuals = (n_pe - model) / model
    max_residual = np.max(np.abs(residuals))
    log.info(f"Max residual: {max_residual*100:.2f}%")

    # Visualization
    log.info(f" number of masked elements for outliers{sigma_masked.count()}")
    fig2 = plt.figure(figsize=(12, 9))

    # --- Subplot 1 ---
    ax1 = plt.subplot(2, 2, 1)
    disp1 = CameraDisplay(geometry=camera, show_frame=False, ax=ax1)
    disp1.image = model
    disp1.add_colorbar()
    # disp1.set_limits_minmax(140, 165)

    # Set colorbar label for subplot 1
    cbar1 = fig2.axes[-1]
    cbar1.set_ylabel(r"$n_{\rm PE}$", rotation=90, labelpad=15, fontsize=14)

    ax1.set_xlabel("x (m)", fontsize=14)
    ax1.set_ylabel("y (m)", fontsize=14)

    ax1.tick_params(axis="both", which="major", labelsize=11)
    # Title
    ax1.set_title("Model", fontsize=16)

    # --- Subplot 2 ---
    ax2 = plt.subplot(2, 2, 2)
    disp2 = CameraDisplay(camera, show_frame=False, ax=ax2)
    disp2.image = residuals * 100
    disp2.cmap = plt.cm.coolwarm
    disp2.add_colorbar()

    # Set colorbar label for subplot 2
    cbar2 = fig2.axes[-1]  # Again, the last axis should be the second colorbar
    cbar2.set_ylabel(r"%", rotation=90, labelpad=15, fontsize=14)

    ax2.set_xlabel("x (m)", fontsize=14)
    ax2.set_ylabel("y (m)", fontsize=14)
    ax2.tick_params(axis="both", which="major", labelsize=11)
    # Title
    ax2.set_title("Residuals", fontsize=16)

    # Save and close
    pdf.savefig(fig2)
    plt.close(fig2)
    return n_pe, model, minuit, residuals


def propagate_scipy_compatible(model, params, cov, camera):
    """
    Computes output covariance via numerical Jacobian propagation.
    """

    def model_(parameters):
        # params = [A, mu_x, mu_y, sigma]
        print(f" PARAMETERS = {parameters}")
        sigma_y = parameters[3]  # enforce sigma_x = sigma_y
        return parameters[0] * (
            Gaussian(
                parameters[1] * u.m,
                parameters[2] * u.m,
                parameters[3] * u.m,
                sigma_y * u.m,
                psi="0d",
            ).pdf(camera.pix_x, camera.pix_y)
        )

    params = np.asarray(params)
    cov = np.asarray(cov)

    y = model_(params)
    J = approx_derivative(model, params, method="2-point")

    # Covariance propagation
    ycov = J @ cov @ J.T

    return y, ycov


def error_propagation_compute(data, minuit_resulting, camera, pdf, plot=True):
    """Compute both parameter uncertainties and per-pixel uncertainties of the model."""

    def model(params):
        # params = [A, mu_x, mu_y, sigma]
        sigma_y = params[3]  # enforce sigma_x = sigma_y
        return params[0] * (
            Gaussian(
                params[1] * u.m,
                params[2] * u.m,
                params[3] * u.m,
                sigma_y * u.m,
                psi="0d",
            ).pdf(camera.pix_x, camera.pix_y)
        )

    # --- Parameters and covariance from Minuit
    values = [minuit_resulting.values[i] for i in range(4)]
    errors = [minuit_resulting.errors[i] for i in range(4)]

    log.info("Fit results:")
    log.info(f"A = {values[0]:.2f} ± {errors[0]:.2f}")
    log.info(f"μ_x = {values[1]:.2f} ± {errors[1]:.2f}")
    log.info(f"μ_y = {values[2]:.2f} ± {errors[2]:.2f}")
    log.info(f"σ_x = {values[3]:.2f} ± {errors[3]:.2f}")
    log.info(f"σ_y = {values[3]:.2f} ± {errors[3]:.2f}")

    if len(minuit_resulting.values) == 6:
        log.info(
            f"V_int = {minuit_resulting.values[5]:.2f} "
            f"± {minuit_resulting.errors[5]:.2f}"
        )

    # --- Propagate errors through the model
    y, ycov = propagate_scipy_compatible(
        lambda p: model(p), minuit_resulting.values, minuit_resulting.covariance, camera
    )
    yerr_prop = np.sqrt(np.diag(ycov))

    theta = np.rad2deg(
        np.sqrt(
            (camera.pix_x.value - values[1]) ** 2
            + (camera.pix_y.value - values[2]) ** 2
        )
        / 12
    )
    bins = np.arange(np.min(theta), np.max(theta) + 0.1, 0.1)

    sum_y, _ = np.histogram(theta, bins=bins, weights=y)
    sum_y_err, _ = np.histogram(theta, bins=bins, weights=yerr_prop)
    count, _ = np.histogram(theta, bins=bins)

    binned_y = np.where(count > 0, sum_y / count, np.nan)
    binned_y_err = np.where(count > 0, sum_y_err / count, np.nan)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # --- Optional plotting
    if plot:
        fig_model = plt.figure(figsize=(6, 5))
        ax = plt.subplot()
        ax.scatter(theta, data, label="data", zorder=0, alpha=0.3)
        ax.fill_between(
            bin_centers,
            binned_y - binned_y_err,
            binned_y + binned_y_err,
            facecolor="C1",
            alpha=0.5,
        )
        ax.plot(bin_centers, binned_y, color="r", label="model")

        x_fov = 4.0
        idx = np.nanargmin(np.abs(bin_centers - x_fov))
        y_center = binned_y[idx]
        length = 0.02 * y_center
        cap_width = 0.25
        label = "2%"

        # compute top & bottom
        y_top = y_center + length / 2
        y_bottom = y_center - length / 2

        # vertical line
        plt.plot(
            [x_fov + cap_width, x_fov + cap_width],
            [y_bottom, y_top],
            color="black",
            lw=2,
        )

        plt.plot([x_fov, x_fov + cap_width], [y_top, y_top], color="black", lw=2)

        plt.plot([x_fov, x_fov + cap_width], [y_bottom, y_bottom], color="black", lw=2)

        plt.text(
            x_fov + cap_width + 0.05,
            y_center,
            label,
            va="center",
            ha="left",
            fontsize=14,
            color="black",
        )

        plt.axvline(
            x=x_fov, color="dimgray", linestyle="--", linewidth=1.5, label="FoV limit"
        )

        plt.ylabel("Number of photoelectrons")
        plt.xlabel("θ [deg]")
        plt.legend()
        pdf.savefig(fig_model)

    return {
        "params": values,
        "param_errors": errors,
        "model_values": y,
        "model_errors": yerr_prop,
        "rebinned": (bin_centers, binned_y, binned_y_err),
    }


def characterize_peak(minuit, camera):
    """Compute coordinated of the peak of
    the fitted Gaussian and corresponding pixel."""
    dist_squared = (camera.pix_x.value - minuit[1]) ** 2 + (
        camera.pix_y.value - minuit[2]
    ) ** 2

    # Find index of minimum distance
    closest_pixel_id = np.argmin(dist_squared)

    # Optional: get the actual closest coordinates
    closest_x = camera.pix_x.value[closest_pixel_id]
    closest_y = camera.pix_y.value[closest_pixel_id]
    distance = np.sqrt(closest_x**2 + closest_y**2)

    log.info(f"Closest pixel ID: {closest_pixel_id}")
    log.info(f"Coordinates: ({closest_x}, {closest_y})")
    log.info(
        f"The distance between the centre of the camera "
        f"and the peak of the fitted 2D gaussian: {distance:.3f} meters"
    )
    return distance


# Same fit procedure but taking into account V_int


# values for minuit are taken from the firs fit without any
def optimize_with_outlier_rejection_variance(
    sigma, data, dict_missing_pix, minuit, camera, pdf
):
    def define_delete_out(sigma, data):
        mean = np.mean(data)
        std = np.std(data)
        outliers = [np.abs(data - mean) > 3 * std]

        sigma_var = ma.masked_array(sigma, mask=outliers)
        data_var = ma.masked_array(data, mask=outliers)
        return data_var, sigma_var, outliers

    # Update data, sigma, and mask
    n_pe_var, sigma_masked_var, mask_upd = define_delete_out(sigma, data)

    # Define the least-squares function
    def lsq_wrap_var(array_parameters):
        A, x, y, std_x, v_int = array_parameters  # changed
        std_y = std_x  # changed
        return np.sum(
            (n_pe_var - Gaussian_model(camera, [A, x, y, std_x, std_y])) ** 2
            / (sigma_masked_var**2 + v_int)
            - np.log(sigma_masked_var**2 / (sigma_masked_var**2 + v_int))
        )

    # Initialize Minuit with updated function and parameters
    minuit_new = Minuit(lsq_wrap_var, minuit)
    minuit_new.limits["x0"] = (0, None)  # A>0
    minuit_new.limits["x3"] = (0, None)  # std_x > 0
    minuit_new.limits["x4"] = (0, None)  # V_int > 0
    minuit_new.migrad()

    log.info(f"covariance table: {tab.tabulate(*minuit_new.covariance.to_table())}")
    log.info(
        f"Fit new parameters: amplitude = {minuit_new.values['x0']},"
        f" x = {minuit_new.values['x1']}, y = {minuit_new.values['x2']},"
        f" length = {minuit_new.values['x3']}, "
        f" intrinsic variance = {minuit_new.values['x4']}"
    )  # changed

    model = Gaussian_model(
        camera,
        [
            minuit_new.values["x0"],
            minuit_new.values["x1"],
            minuit_new.values["x2"],
            minuit_new.values["x3"],
            minuit_new.values["x3"],
        ],
    )
    residuals = (n_pe_var - model) / model
    max_residual = np.max(np.abs(residuals))
    log.info(f"Max residual: {max_residual*100:.2f}%")

    dict_missing_pix["rejected_outliers"] = (
        N_PIXELS
        - sigma_masked_var.count()
        - dict_missing_pix["Missing pixels"]
        - dict_missing_pix["high_gain = 0"]
    )

    labels = [
        f'Missing pixels, #{dict_missing_pix["Missing pixels"]}',
        f'high gain = 0, # {dict_missing_pix["high_gain = 0"]}',
        f'rejected outliers, # {dict_missing_pix["rejected_outliers"]}',
    ]

    fig_pie_3, ax = plt.subplots()
    ax.pie(dict_missing_pix.values(), labels=labels, autopct="%.0f%%")
    fig_pie_3.suptitle("Piechart of masked pixels")
    pdf.savefig(fig_pie_3)

    # Visualization

    fig4 = plt.figure(figsize=(16, 8))
    # --- Subplot 1 ---
    ax1 = plt.subplot(2, 2, 1)
    disp1 = CameraDisplay(camera, show_frame=False, ax=ax1)
    disp1.image = model
    disp1.add_colorbar()

    # Set colorbar label for subplot 1
    cbar1 = fig4.axes[-1]
    cbar1.set_ylabel(r"$n_{\rm PE}$", rotation=90, labelpad=15, fontsize=14)
    ax1.set_xlabel("x (m)", fontsize=14)
    ax1.set_ylabel("y (m)", fontsize=14)
    ax1.tick_params(axis="both", which="major", labelsize=11)
    ax1.set_title("Model", fontsize=16)

    # --- Subplot 2 ---
    ax2 = plt.subplot(2, 2, 2)
    disp2 = CameraDisplay(camera, show_frame=False, ax=ax2)
    disp2.image = residuals * 100
    disp2.cmap = plt.cm.coolwarm
    disp2.add_colorbar()
    cbar2 = fig4.axes[-1]
    cbar2.set_ylabel(r"%", rotation=90, labelpad=15, fontsize=14)
    ax2.set_xlabel("x (m)", fontsize=14)
    ax2.set_ylabel("y (m)", fontsize=14)
    ax2.tick_params(axis="both", which="major", labelsize=11)
    ax2.set_title("Residuals", fontsize=16)
    pdf.savefig(fig4)
    plt.close(fig4)

    return n_pe_var, model, minuit_new, residuals


def compute_ff_coefs(charges, gains, pdf):
    log.info(f"SHAPE {gains.shape}")
    masked_charges = np.ma.masked_where(np.ma.getmask(charges), charges)
    masked_gains = np.ma.masked_where(np.ma.getmask(charges), gains)

    relative_signal = np.divide(
        masked_charges, masked_gains[:, 0], where=gains[:, 0] != 0
    )
    eff = relative_signal / np.mean(relative_signal)
    mean = np.mean(eff)
    std = np.std(eff)

    fig_ff_1, ax = plt.subplots(figsize=(8, 5))

    # Plot histogram
    n, bins, patches = ax.hist(
        eff,
        bins=50,
        edgecolor="black",
        alpha=0.7,
    )

    # Add vertical lines for mean and ±σ
    ax.axvline(
        mean, color="red", linestyle="solid", linewidth=2, label=f"Mean = {mean:.2f}"
    )
    ax.axvline(
        mean - std,
        color="red",
        linestyle="dashed",
        linewidth=1.5,
        label=f"±1σ = {std:.2f}",
    )
    ax.axvline(mean + std, color="red", linestyle="dashed", linewidth=1.5)
    ax.set_title(
        f"Distribution of FF coefficient, model independent",
        fontsize=16,
    )
    ax.set_xlabel("FF coefficient", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig_ff_1.tight_layout()
    pdf.savefig(fig_ff_1)

    return eff


def compute_ff_coefs_model(data, data_std, model, model_std, pdf):
    FF_coefs = np.divide(data, model, where=model != 0)
    mean = np.mean(FF_coefs)
    std = np.std(FF_coefs)
    std_FF = np.sqrt((data_std / model) ** 2 + (data * model_std / model**2) ** 2)
    fig_ff, ax = plt.subplots(figsize=(8, 5))

    # Plot histogram
    n, bins, patches = ax.hist(
        FF_coefs,
        bins=50,
        edgecolor="black",
        alpha=0.7,
    )
    # Add vertical lines for mean and ±σ
    ax.axvline(
        mean, color="red", linestyle="solid", linewidth=2, label=f"Mean = {mean:.2f}"
    )
    ax.axvline(
        mean - std,
        color="red",
        linestyle="dashed",
        linewidth=1.5,
        label=f"±1σ = {std:.2f}",
    )
    ax.axvline(mean + std, color="red", linestyle="dashed", linewidth=1.5)
    ax.set_title(f"Distribution of FF coefficient, model-based", fontsize=16)
    ax.set_xlabel("FF coefficient", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig_ff.tight_layout()
    pdf.savefig(fig_ff)
    return FF_coefs, std_FF


def main(**kwargs):
    os.makedirs(
        f"{os.environ.get('NECTARCHAIN_LOG','/tmp')}/{os.getpid()}/figures",
        exist_ok=True,
    )

    log.setLevel(args.verbosity)

    # --- Assign other variables ---
    run_number = args.FF_run_number
    run_path = args.run_path + (
        f"/runs/NectarCAM.Run" f"{str(run_number).zfill(4)}.0000.fits.fz"
    )
    spe_run_number = args.SPE_run_number
    method = args.method
    extractor = args.extractor_kwargs
    camera = args.camera
    # only one run file is loaded as it is used only to retrieve camera geometry and bad pixels

    log.info(
        f"Method is {method},  the extractor kwargs are: {extractor['window_shift']}, {extractor['window_width']}"
    )

    str_extractor_kwargs = CtapipeExtractor.get_extractor_kwargs_str(
        method=args.method, extractor_kwargs=args.extractor_kwargs
    )

    if not args.SPE_config:
        raise ValueError(
            "You must specify the SPE_config to use, either HHVfree, HHVfixed or nominal"
        )

    spe_file_found = False
    if args.SPE_config == "HHVfree":
        try:
            spe_filename = DataManagement.find_SPE_HHV(
                run_number=args.SPE_run_number,
                method=args.method,
                str_extractor_kwargs=str_extractor_kwargs,
                free_pp_n=True,
            )
            spe_file_found = True
        except FileNotFoundError:
            pass
    elif args.SPE_config == "HHVfixed":
        try:
            spe_filename = DataManagement.find_SPE_HHV(
                run_number=args.SPE_run_number,
                method=args.method,
                str_extractor_kwargs=str_extractor_kwargs,
                free_pp_n=False,
            )
            spe_file_found = True
        except FileNotFoundError:
            pass
    elif args.SPE_config == "nominal":
        try:
            spe_filename = DataManagement.find_SPE_nominal(
                run_number=args.SPE_run_number,
                method=args.method,
                str_extractor_kwargs=str_extractor_kwargs,
                free_pp_n=False,
            )
            spe_file_found = True
        except FileNotFoundError:
            pass

    if not spe_file_found:
        gain_tool = FlatFieldSPENominalStdNectarCAMCalibrationTool(
            progress_bar=True,
            run_number=spe_run_number,
            camera=camera,
            max_events=None,
            method=method,
            extractor_kwargs={
                "window_width": extractor["window_width"],
                "window_shift": extractor["window_shift"],
            },
        )
        gain_tool.setup()
        gain_tool.start()
        spe_filename = gain_tool.finish(return_output_component=True)[0]

    log.info(f"ADD_VARIANCE = {args.add_variance}")

    if args.add_variance:
        log.info("Running analysis with variance correction...")
    else:
        log.info("Running analysis without variance correction...")

    try:
        filename_ps = DataManagement.find_photostat(
            FF_run_number=run_number,
            ped_run_number=run_number,
            FF_method=method,
            ped_method="FullWaveformSum",
            str_extractor_kwargs=str_extractor_kwargs,
        )[0]
        log.info(
            f"File {filename_ps} already exists, skipping photostatistics computation."
        )
    except FileNotFoundError:
        log.info(
            f"Photostatistics results file not found, running "
            f"PhotoStatisticNectarCAMCalibrationTool..."
        )

        try:
            tool = PhotoStatisticNectarCAMCalibrationTool(
                progress_bar=True,
                run_number=run_number,
                max_events=None,
                camera=camera,
                Ped_run_number=run_number,
                SPE_result=spe_filename[0],
                **kwargs,
            )
            tool.setup()
            tool.start()
            tool.finish()
            filename_ps = DataManagement.find_photostat(
                FF_run_number=run_number,
                ped_run_number=run_number,
                FF_method=method,
                ped_method="FullWaveformSum",
                str_extractor_kwargs=str_extractor_kwargs,
            )[0]
        except Exception as e:
            log.warning(e, exc_info=True)

    log.info("SPE fit was found, begin the analysis")
    # Create PdfPages object
    pdf_file = PdfPages(f"Plots_analysis_run{run_number}.pdf")

    source = EventSource.from_url(input_url=run_path, max_events=1)
    camera_tel = source.subarray.tel[
        source.subarray.tel_ids[0]
    ].camera.geometry.transform_to(EngineeringCameraFrame())

    for event in source:
        log.info(event.index.event_id, event.trigger.event_type, event.trigger.time)

    # Looking for broken pixels
    fig00 = plt.figure(13, figsize=(5, 5))
    disp = CameraDisplay(geometry=camera_tel, show_frame=False)
    chan = 0
    disp.image = event.mon.tel[
        source.subarray.tel_ids[0]
    ].pixel_status.hardware_failing_pixels[chan]
    disp.set_limits_minmax(0, 1)
    disp.cmap = plt.cm.coolwarm
    disp.add_colorbar()
    fig00.suptitle("Broken/missing pixels")
    pdf_file.savefig(fig00)

    (
        n_pe,
        std_n_pe,
        sigma_masked,
        dict_missing_pix,
        high_gains,
        low_gains,
        charges,
    ) = pre_process_fits(filename_ps, camera_tel, pdf_file)

    # First fit no variance
    data_1, fit_1, minuit_1, residuals_1 = optimize_with_outlier_rejection(
        sigma_masked, n_pe, camera_tel, pdf_file
    )
    dict_errors = error_propagation_compute(data_1, minuit_1, camera_tel, pdf_file)
    y_1, yerr_prop_1, minuit_vals_1, minuit_vals_errors_1 = (
        dict_errors["model_values"],
        dict_errors["model_errors"],
        dict_errors["params"],
        dict_errors["param_errors"],
    )
    log.info(f"Resulting error for the model is {np.mean(yerr_prop_1/y_1)*100:.2f}%")
    characterize_peak(minuit_vals_1, camera_tel)
    log.info(minuit_1.values)

    # Visualize how many pixels were masked
    dict_missing_pix["rejected_outliers"] = (
        N_PIXELS
        - sigma_masked.count()
        - dict_missing_pix["Missing pixels"]
        - dict_missing_pix["high_gain = 0"]
    )

    labels = [
        f'Missing pixels, #{dict_missing_pix["Missing pixels"]}',
        f'high gain = 0, # {dict_missing_pix["high_gain = 0"]}',
        f'rejected outliers, # {dict_missing_pix["rejected_outliers"]}',
    ]
    fig_pie, ax = plt.subplots()
    ax.pie(dict_missing_pix.values(), labels=labels, autopct="%.0f%%")
    fig_pie.suptitle("Piechart of masked pixels")
    pdf_file.savefig(fig_pie)

    # Second fit with variance
    if args.add_variance:
        (
            data_varinace,
            fit_variance,
            minuit_variance_result,
            residuals_variance_result,
        ) = optimize_with_outlier_rejection_variance(
            sigma_masked,
            n_pe,
            dict_missing_pix,
            [
                minuit_1.values["a0"],
                minuit_1.values["a1"],
                minuit_1.values["a2"],
                minuit_1.values["a3"],
                0.0,
            ],
            camera_tel,
            pdf_file,
        )

        plt.figure()
        plt.hist(residuals_variance_result, bins=50)
        plt.title("Residuals binned", fontsize=16)
        plt.close()

        dict_error_var = error_propagation_compute(
            data_varinace, minuit_variance_result, camera_tel, pdf_file, plot=True
        )

        (
            y_variance,
            yerr_prop_variance,
            minuit_values_variance,
            minuit_values_error_variance,
        ) = (
            dict_error_var["model_values"],
            dict_error_var["model_errors"],
            dict_error_var["params"],
            dict_error_var["param_errors"],
        )

        log.info(
            f"Resulting error for the model is "
            f"{np.mean(yerr_prop_variance / y_variance) * 100:.2f}%"
        )
        log.info(np.min(camera_tel.pix_x.value))
        log.info(np.min(camera_tel.pix_y.value))
        log.info(np.mean(camera_tel.pix_x.value))
        log.info(minuit_values_variance[1])
        log.info(minuit_values_variance[2])

        characterize_peak(minuit_values_variance, fit_variance)

    # compute flat field coef
    simple_ff_coefs = compute_ff_coefs(charges, high_gains, pdf_file)
    ff_coefs_model, ff_coefs_model_err = compute_ff_coefs_model(
        n_pe, std_n_pe, y_1, yerr_prop_1, pdf_file
    )

    with open(f"Log_info_run_{run_number}_fixed.txt", "a") as f:
        # Write a header if file is empty (i.e. first time writing)
        if f.tell() == 0:
            f.write(
                "Run,Model,A,x0(rad),y0(rad),width(rad),"
                "v_int,A_err,x0_err(rad),y0_err(rad),width_err(rad),"
                "v_int_err, model_error\n"
            )

        # Convert angles
        x0_1 = np.arctan(minuit_vals_1[1] / 12)
        y0_1 = np.arctan(minuit_vals_1[2] / 12)
        width_1 = np.arctan(minuit_vals_1[3] / 12)

        x0_1_err = np.arctan(minuit_vals_errors_1[1] / 12)
        y0_1_err = np.arctan(minuit_vals_errors_1[2] / 12)
        width_1_err = np.arctan(minuit_vals_errors_1[3] / 12)

        if args.add_variance:
            x0_v = np.arctan(minuit_values_variance[1] / 12)
            y0_v = np.arctan(minuit_values_variance[2] / 12)
            width_v = np.arctan(minuit_values_variance[3] / 12)

            x0_v_err = np.arctan(minuit_values_error_variance[1] / 12)
            y0_v_err = np.arctan(minuit_values_error_variance[2] / 12)
            width_v_err = np.arctan(minuit_values_error_variance[3] / 12)

        # First model (without v_int)
        f.write(
            f"{run_number},Initial,{minuit_vals_1[0]},{x0_1},{y0_1},{width_1},,"
            f"{minuit_vals_errors_1[0]},{x0_1_err},"
            f"{y0_1_err},{width_1_err},{np.mean(yerr_prop_1/y_1)*100}\n"
        )
        if args.add_variance:
            f.write(
                f"{run_number},With_v_int,{minuit_values_variance[0]},"
                f"{x0_v},{y0_v},{width_v},"
                f"{minuit_values_error_variance[0]},{x0_v_err},"
                f"{y0_v_err},{width_v_err}, "
                f"{np.mean(yerr_prop_variance / y_variance) * 100}\n"
            )

    data = Table()
    data["pixel_id"] = camera_tel.pix_id
    data["x"] = camera_tel.pix_x
    data["y"] = camera_tel.pix_y
    data["N_photoelectrons_fited"] = y_1
    data["N_photoelectrons_std_fited"] = yerr_prop_1
    data["FF_coef_independent_way"] = simple_ff_coefs
    data["FF_coef_model_way"] = ff_coefs_model
    data["FF_coef_model_way_err"] = ff_coefs_model_err
    data["high_gain_init"] = high_gains[:, 0]
    data["low_gain_init"] = low_gains[:, 0]
    data["Charge_init"] = charges
    data["N_photoelectrons_init"] = n_pe
    data["N_photoelectrons_std_init"] = std_n_pe
    ascii.write(data, f"FF_calibration_run{run_number}.dat", overwrite=True)

    pdf_file.close()


if __name__ == "__main__":
    kwargs = copy.deepcopy(vars(args))
    kwargs.pop("camera")

    main(**kwargs)
