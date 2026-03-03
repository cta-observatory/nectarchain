import argparse
import copy
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import numpy.ma as ma
from astropy.io import ascii
from astropy.table import Table
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.image.toymodel import Gaussian
from ctapipe.io import EventSource
from ctapipe.io.hdf5tableio import HDF5TableReader
from ctapipe.visualization import CameraDisplay
from ctapipe_io_nectarcam.constants import N_PIXELS
from iminuit import Minuit
from matplotlib import pyplot as plt
from scipy.optimize._numdiff import approx_derivative

# nectarchain imports
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
    filename=f"{os.environ.get('NECTARCHAIN_LOG', '/tmp')}/{os.getpid()}/"
    f"{Path(__file__).stem}_{os.getpid()}.log",
    handlers=[logging.getLogger("__main__").handlers],
)
log = logging.getLogger(__name__)

plt.style.use(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "../utils/plot_style.mpltstyle"
    )
)


def get_args():
    """Parses command-line arguments for the deadtime test script.

    Returns
    -------
    parser : argparse.ArgumentParser
        The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Flat-field source characterization test B-TEL-1350.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--FF_run_number",
        default=6729,
        help="Flat-field run number",
        type=int,
    )
    parser.add_argument(
        "--SPE_run_number",
        default=6774,
        help="SPE run number",
        type=int,
    )
    parser.add_argument(
        "-c",
        "--camera",
        choices=ALLOWED_CAMERAS,
        default=[camera for camera in ALLOWED_CAMERAS if "QM" in camera][0],
        help="Process data for a specific NectarCAM camera.",
        type=str,
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
            From ICRC2025 proceedings, we recommend to use resoltion at nominal for "
        "the SPE fit.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory",
        default=f"{os.environ.get('NECTARCHAIN_FIGURES', f'/tmp/{os.getpid()}')}",
    )
    parser.add_argument(
        "--temp_output", help="Temporary output directory for GUI", default=None
    )
    parser.add_argument(
        "-l",
        "--log",
        help="log level",
        default="info",
        type=str,
    )

    return parser


def pre_process_fits(filename, camera, output_dir=None, temp_output=None):
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
    log.info(f"Missing pixels {missing_pixels}")

    # pixels marked as broken (0)
    is_valid_full = np.zeros(total_pixels, dtype=float)
    existing_pixels = container_dict["pixels_id"]
    is_valid_values = container_dict["is_valid"]

    # Fill the correct positions
    is_valid_full[existing_pixels] = is_valid_values

    # Bad/missing pixels plot
    fig, ax = plt.subplots()
    disp = CameraDisplay(geometry=camera, show_frame=False, ax=ax)
    disp.image = is_valid_full
    disp.set_limits_minmax(0, 1)
    disp.cmap = plt.cm.coolwarm
    disp.add_colorbar()
    fig.suptitle("Broken / Missing pixels")
    fig_name = "camera_broken_pixels"
    plot_path = os.path.join(output_dir, f"{fig_name}.png")
    plt.savefig(plot_path)

    if temp_output:
        with open(os.path.join(temp_output, f"plot_{fig_name}.pkl"), "wb") as f:
            pickle.dump(fig, f)

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
    fig, ax = plt.subplots()
    disp = CameraDisplay(geometry=camera, show_frame=False, ax=ax)

    disp.image = n_pe
    disp.add_colorbar()
    # disp.set_limits_minmax(140, 165)

    cbar = fig.axes[-1]
    cbar.set_ylabel(r"Illumination, $n_{\rm PE}$", rotation=90)
    # Axis labels
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.tick_params(axis="both", which="major")
    plt.title("Data")
    fig_name = "camera_data"
    plot_path = os.path.join(output_dir, f"{fig_name}.png")
    plt.savefig(plot_path)

    if temp_output:
        with open(os.path.join(temp_output, f"plot_{fig_name}.pkl"), "wb") as f:
            pickle.dump(fig, f)

    dict_missing_pix = {
        "Missing pixels": len(missing_pixels),
        "high_gain = 0": ma.count_masked(masked_hg) - len(missing_pixels),
    }

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


def optimize_with_outlier_rejection(
    sigma, data, camera, output_dir=None, temp_output=None
):
    # least-squares score function = sum of data residuals squared
    def lsq(a0, a1, a2, a3):
        # We assume that the 2D-Gaussian is symmetric, thus the last two arguments of
        # Gaussian_model are the same
        return np.sum(
            (n_pe - Gaussian_model(camera, [a0, a1, a2, a3, a3])) ** 2
            / (sigma_masked**2)
        )

    # Apply outlier mask based on data
    n_pe, sigma_masked, mask_upd = define_delete_out(sigma, data)

    # Fit with Minuit using previous best parameters
    minuit = Minuit(lsq, a0=1000.0, a1=0.0, a2=0.0, a3=1.5)
    minuit.migrad()

    if not minuit.fmin.is_valid:
        log.info("Warning: Fit did not converge!")
    log.info(
        f"""Covariance matrix:

{minuit.covariance}

"""
    )
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
    fig = plt.figure(figsize=(12, 9))

    # --- Subplot 1 ---
    ax1 = plt.subplot(2, 2, 1)
    disp1 = CameraDisplay(geometry=camera, show_frame=False, ax=ax1)
    disp1.image = model
    disp1.add_colorbar()
    # disp1.set_limits_minmax(140, 165)

    # Set colorbar label for subplot 1
    cbar1 = fig.axes[-1]
    cbar1.set_ylabel(r"$n_{\rm PE}$", rotation=90)

    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    ax1.tick_params(axis="both", which="major")
    # Title
    ax1.set_title("Model")

    # --- Subplot 2 ---
    ax2 = plt.subplot(2, 2, 2)
    disp2 = CameraDisplay(camera, show_frame=False, ax=ax2)
    disp2.image = residuals * 100
    disp2.cmap = plt.cm.coolwarm
    disp2.add_colorbar()

    # Set colorbar label for subplot 2
    cbar2 = fig.axes[-1]  # Again, the last axis should be the second colorbar
    cbar2.set_ylabel(r"%", rotation=90)

    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.tick_params(axis="both", which="major")
    # Title
    ax2.set_title("Residuals")

    fig_name = "camera_displays"
    plot_path = os.path.join(output_dir, f"{fig_name}.png")
    plt.savefig(plot_path)

    if temp_output:
        with open(os.path.join(temp_output, f"plot_{fig_name}.pkl"), "wb") as f:
            pickle.dump(fig, f)

    return n_pe, model, minuit, residuals


def propagate_scipy_compatible(model, params, cov, camera):
    """
    Computes output covariance via numerical Jacobian propagation.
    """

    def model_(parameters):
        # params = [A, mu_x, mu_y, sigma]
        log.info(f" PARAMETERS = {parameters}")
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


def error_propagation_compute(
    data, minuit_resulting, camera, output_dir=None, temp_output=None
):
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

    binned_y = np.divide(
        sum_y, count, out=np.full_like(sum_y, np.nan, dtype=float), where=count > 0
    )
    binned_y_err = np.divide(
        sum_y_err,
        count,
        out=np.full_like(sum_y_err, np.nan, dtype=float),
        where=count > 0,
    )
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # additional plot
    # Sort data
    sort_idx = np.argsort(theta)
    theta_sorted = theta[sort_idx]

    min_per_bin = 20
    min_bin_width = 0.1  # <- control smoothness here

    bins = [theta_sorted[0]]
    start_idx = 0

    while start_idx < len(theta_sorted):
        end_idx = start_idx + min_per_bin

        if end_idx >= len(theta_sorted):
            bins.append(theta_sorted[-1])
            break

        # ensure minimum width condition
        while (
            end_idx < len(theta_sorted)
            and theta_sorted[end_idx] - theta_sorted[start_idx] < min_bin_width
        ):
            end_idx += 1

        bins.append(theta_sorted[end_idx])
        start_idx = end_idx

    bins = np.array(bins)

    bin_indices = np.digitize(theta, bins) - 1

    binned_data = []
    binned_std = []
    bin_centers_data = []

    for i in range(len(bins) - 1):
        mask_data = bin_indices == i
        values_data = data[mask_data]

        if len(values_data) >= min_per_bin:
            binned_data.append(np.mean(values_data))
            binned_std.append(np.std(values_data, ddof=1))
            bin_centers_data.append(0.5 * (bins[i] + bins[i + 1]))

    binned_data = np.array(binned_data)
    binned_std = np.array(binned_std)
    bin_centers_data = np.array(bin_centers_data)

    # --- plotting

    fig, ax = plt.subplots()
    ax.scatter(theta, data, label="data", zorder=0, alpha=0.3)
    ax.fill_between(
        bin_centers,
        binned_y - binned_y_err,
        binned_y + binned_y_err,
        facecolor="C1",
        alpha=0.5,
        label="error on the model",
        zorder=3,
    )
    ax.plot(bin_centers, binned_y, color="r", label="model", zorder=4)

    x_fov = 4.0
    plt.axvline(
        x=x_fov, color="dimgray", linestyle="--", linewidth=1.5, label="FoV limit"
    )

    ax.fill_between(
        bin_centers,
        binned_y * 0.98,
        binned_y * 1.02,
        facecolor="blue",
        alpha=0.5,
        label="2% requirement",
        zorder=1,
    )

    plt.ylabel("Number of photoelectrons")
    plt.xlabel("θ [deg]")
    plt.legend()
    fig_name = "flatfield_source_model"
    plot_path = os.path.join(output_dir, f"{fig_name}.png")
    plt.savefig(plot_path)

    if temp_output:
        with open(os.path.join(temp_output, f"plot_{fig_name}.pkl"), "wb") as f:
            pickle.dump(fig, f)

    # plot with binned data
    fig, ax = plt.subplots()
    ax.errorbar(
        bin_centers_data,
        binned_data,
        yerr=binned_std,
        fmt="o",
        label="data",
        zorder=1,
    )
    ax.fill_between(
        bin_centers,
        binned_y - binned_y_err,
        binned_y + binned_y_err,
        facecolor="C1",
        alpha=0.5,
        label="error on the model",
        zorder=3,
    )
    ax.plot(bin_centers, binned_y, color="r", label="model", zorder=4)

    plt.axvline(
        x=x_fov, color="dimgray", linestyle="--", linewidth=1.5, label="FoV limit"
    )

    ax.fill_between(
        bin_centers,
        binned_y * 0.98,
        binned_y * 1.02,
        facecolor="blue",
        alpha=0.5,
        label="2% requirement",
        zorder=2,
    )

    plt.ylabel("Number of photoelectrons")
    plt.xlabel("θ [deg]")
    plt.legend()
    fig_name = "flatfield_source_model_binned"
    plot_path = os.path.join(output_dir, f"{fig_name}.png")
    plt.savefig(plot_path)

    if temp_output:
        with open(os.path.join(temp_output, f"plot_{fig_name}.pkl"), "wb") as f:
            pickle.dump(fig, f)

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
        "The distance between the centre of the camera "
        f"and the peak of the fitted 2D gaussian: {distance:.3f} meters"
    )
    return distance


# Same fit procedure but taking into account V_int


# values for minuit are taken from the first fit without any
def optimize_with_outlier_rejection_variance(sigma, data, minuit, camera):
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

    log.info(
        f"""Covariance matrix:

{minuit_new.covariance}

"""
    )
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

    return n_pe_var, model, minuit_new, residuals


def compute_ff_coefs(charges, gains, output_dir=None, temp_output=None):
    log.info(f"SHAPE {gains.shape}")
    masked_charges = np.ma.masked_where(np.ma.getmask(charges), charges)
    masked_gains = np.ma.masked_where(np.ma.getmask(charges), gains)

    relative_signal = np.divide(
        masked_charges, masked_gains[:, 0], where=gains[:, 0] != 0
    )
    eff = relative_signal / np.mean(relative_signal)
    mean = np.mean(eff)
    std = np.std(eff)

    fig, ax = plt.subplots()

    # Plot histogram
    ax.hist(
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
    ax.set_title("Distribution of FF coefficient, model independent")
    ax.set_xlabel("FF coefficient")
    ax.set_ylabel("Count")
    ax.tick_params(axis="both")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig_name = "flatfield_coefficients_distribution"
    plot_path = os.path.join(output_dir, f"{fig_name}.png")
    plt.savefig(plot_path)

    if temp_output:
        with open(os.path.join(temp_output, f"plot_{fig_name}.pkl"), "wb") as f:
            pickle.dump(fig, f)

    return eff


def compute_ff_coefs_model(
    data, data_std, model, model_std, output_dir=None, temp_output=None
):
    FF_coefs = np.divide(data, model, where=model != 0)
    mean = np.mean(FF_coefs)
    std = np.std(FF_coefs)
    std_FF = np.sqrt((data_std / model) ** 2 + (data * model_std / model**2) ** 2)
    fig, ax = plt.subplots()

    # Plot histogram
    ax.hist(
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
    ax.set_title("Distribution of FF coefficient, model-based")
    ax.set_xlabel("FF coefficient")
    ax.set_ylabel("Count")
    ax.tick_params(axis="both")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig_name = "flatfield_coefficients_distribution_with_model"
    plot_path = os.path.join(output_dir, f"{fig_name}.png")
    plt.savefig(plot_path)

    if temp_output:
        with open(os.path.join(temp_output, f"plot_{fig_name}.pkl"), "wb") as f:
            pickle.dump(fig, f)

    return FF_coefs, std_FF


def get_spefilenames(
    spe_config=None, spe_run_number=None, method=None, extractor_kwargs=None
):
    spe_filenames = None
    if spe_config == "HHVfree":
        try:
            spe_filenames = DataManagement.find_SPE_HHV(
                run_number=spe_run_number,
                method=method,
                str_extractor_kwargs=extractor_kwargs,
                free_pp_n=True,
            )
        except FileNotFoundError:
            pass
    elif spe_config == "HHVfixed":
        try:
            spe_filenames = DataManagement.find_SPE_HHV(
                run_number=spe_run_number,
                method=method,
                str_extractor_kwargs=extractor_kwargs,
                free_pp_n=False,
            )
        except FileNotFoundError:
            pass
    elif spe_config == "nominal":
        try:
            spe_filenames = DataManagement.find_SPE_nominal(
                run_number=spe_run_number,
                method=method,
                str_extractor_kwargs=extractor_kwargs,
                free_pp_n=False,
            )
        except FileNotFoundError:
            pass

    return spe_filenames


def main():
    """Flat-field source characterization test B-TEL-1350."""

    parser = get_args()
    args = parser.parse_args()
    log.setLevel(args.log.upper())

    os.makedirs(
        f"{os.environ.get('NECTARCHAIN_LOG','/tmp')}/{os.getpid()}/figures",
        exist_ok=True,
    )

    kwargs = copy.deepcopy(vars(args))
    kwargs.pop("camera")
    camera = args.camera

    output_dir = os.path.join(
        os.path.abspath(args.output),
        f"trr_camera_{camera}/{Path(__file__).stem}",
    )
    os.makedirs(output_dir, exist_ok=True)
    temp_output = os.path.abspath(args.temp_output) if args.temp_output else None

    # Drop arguments from the script after they are parsed, for the GUI to work properly
    sys.argv = sys.argv[:1]

    # --- Assign other variables ---
    run_number = args.FF_run_number
    run_path = os.path.join(
        os.environ.get("NECTARCAMDATA", "/tmp"),
        f"runs/NectarCAM.Run{str(run_number).zfill(4)}.0000.fits.fz",
    )
    spe_run_number = args.SPE_run_number
    method = "LocalPeakWindowSum"
    extractor_kwargs = json.loads('{"window_width": 8, "window_shift": 4}')
    add_variance = True

    log.info(
        f"Method is {method}, the extractor kwargs are: "
        f"{extractor_kwargs['window_shift']}, "
        f"{extractor_kwargs['window_width']}"
    )

    str_extractor_kwargs = CtapipeExtractor.get_extractor_kwargs_str(
        method=method, extractor_kwargs=extractor_kwargs
    )

    if not args.SPE_config:
        raise ValueError(
            "You must specify the SPE_config to use, either HHVfree, HHVfixed or "
            "nominal"
        )

    try:
        spe_filenames = get_spefilenames(
            spe_config=args.SPE_config,
            spe_run_number=spe_run_number,
            method=method,
            extractor_kwargs=str_extractor_kwargs,
        )
        log.info(f"File {spe_filenames} already exists, skipping spe fit computation.")
    except FileNotFoundError:
        log.info("SPE fit results file not found, running ")
        spe_filenames = None

    if not spe_filenames:
        gain_tool = FlatFieldSPENominalStdNectarCAMCalibrationTool(
            progress_bar=True,
            run_number=spe_run_number,
            camera=camera,
            max_events=None,
            method=method,
            extractor_kwargs={
                "window_width": extractor_kwargs["window_width"],
                "window_shift": extractor_kwargs["window_shift"],
            },
        )
        gain_tool.setup()
        gain_tool.start()
        gain_tool.finish()
        spe_filenames = get_spefilenames(
            spe_config=args.SPE_config,
            spe_run_number=spe_run_number,
            method=method,
            extractor_kwargs=str_extractor_kwargs,
        )
        log.info(f"THIS IS THE spe_filenames output {spe_filenames}")

    log.info(f"ADD_VARIANCE = {add_variance}")

    if add_variance:
        log.info("Running analysis with variance correction...")
    else:
        log.info("Running analysis without variance correction...")

    try:
        photostatistics_results_file = DataManagement.find_photostat(
            FF_run_number=run_number,
            ped_run_number=run_number,
            FF_method=method,
            ped_method="FullWaveformSum",
            str_extractor_kwargs=str_extractor_kwargs,
        )[0]
        log.info(
            f"File {photostatistics_results_file} already exists, skipping "
            "photostatistics computation."
        )
    except FileNotFoundError:
        log.info(
            "Photostatistics results file not found, running "
            "PhotoStatisticNectarCAMCalibrationTool..."
        )

        try:
            log.info(f"Using SPE results file {spe_filenames[0]}")
            tool = PhotoStatisticNectarCAMCalibrationTool(
                progress_bar=True,
                run_number=run_number,
                max_events=None,
                camera=camera,
                Ped_run_number=run_number,
                SPE_result=spe_filenames[0],
                **kwargs,
            )
            tool.setup()
            tool.start()
            tool.finish()
            photostatistics_results_file = DataManagement.find_photostat(
                FF_run_number=run_number,
                ped_run_number=run_number,
                FF_method=method,
                ped_method="FullWaveformSum",
                str_extractor_kwargs=str_extractor_kwargs,
            )[0]
        except Exception as e:
            log.critical(e, exc_info=True)
            raise e

        log.info("Photostatistics results file was found, beginning the analysis")

    log.info("SPE fit was found, begin the analysis")

    source = EventSource.from_url(input_url=run_path, max_events=1)
    camera_tel = source.subarray.tel[
        source.subarray.tel_ids[0]
    ].camera.geometry.transform_to(EngineeringCameraFrame())

    (
        n_pe,
        std_n_pe,
        sigma_masked,
        dict_missing_pix,
        high_gains,
        low_gains,
        charges,
    ) = pre_process_fits(
        photostatistics_results_file,
        camera_tel,
        output_dir=args.output,
        temp_output=temp_output,
    )

    # First fit no variance
    data_1, fit_1, minuit_1, residuals_1 = optimize_with_outlier_rejection(
        sigma_masked, n_pe, camera_tel, output_dir=args.output, temp_output=temp_output
    )
    dict_errors = error_propagation_compute(
        data_1, minuit_1, camera_tel, output_dir=args.output, temp_output=temp_output
    )
    y_1, yerr_prop_1, minuit_vals_1, minuit_vals_errors_1 = (
        dict_errors["model_values"],
        dict_errors["model_errors"],
        dict_errors["params"],
        dict_errors["param_errors"],
    )
    log.info(f"Resulting error for the model is {np.mean(yerr_prop_1/y_1)*100:.2f}%")
    characterize_peak(minuit_vals_1, camera_tel)
    log.info(minuit_1.values)

    # Second fit with variance
    if add_variance:
        (
            data_varinace,
            fit_variance,
            minuit_variance_result,
            residuals_variance_result,
        ) = optimize_with_outlier_rejection_variance(
            sigma_masked,
            n_pe,
            [
                minuit_1.values["a0"],
                minuit_1.values["a1"],
                minuit_1.values["a2"],
                minuit_1.values["a3"],
                0.0,
            ],
            camera_tel,
        )

        plt.figure()
        plt.hist(residuals_variance_result, bins=50)
        plt.title("Residuals binned")
        plt.close()

        dict_error_var = error_propagation_compute(
            data_varinace,
            minuit_variance_result,
            camera_tel,
            output_dir=args.output,
            temp_output=temp_output,
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

        characterize_peak(minuit_values_variance, camera_tel)

    # compute flat field coef
    simple_ff_coefs = compute_ff_coefs(
        charges, high_gains, output_dir=args.output, temp_output=temp_output
    )
    ff_coefs_model, ff_coefs_model_err = compute_ff_coefs_model(
        n_pe,
        std_n_pe,
        y_1,
        yerr_prop_1,
        output_dir=args.output,
        temp_output=temp_output,
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

        if add_variance:
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
        if add_variance:
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

    plt.close("all")


if __name__ == "__main__":
    main()
