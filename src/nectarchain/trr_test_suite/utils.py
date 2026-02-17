import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from lmfit.models import Model
from scipy.interpolate import interp1d
from scipy.stats import expon, poisson
from traitlets.config import Config

from nectarchain.utils.constants import GAIN_DEFAULT

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


config = Config(
    dict(
        NectarCAMEventSource=dict(
            NectarCAMR0Corrections=dict(
                calibration_path=None,
                apply_flatfield=False,
                select_gain=False,
            )
        )
    )
)

# (filter, optical density, transmission)
# configurations with the same order as in the document
filters = np.array(
    [
        [
            0.00,
            0.15,
            0.30,
            0.45,
            0.60,
            0.75,
            0.90,
            1.00,
            1.15,
            1.30,
            1.30,
            1.45,
            1.50,
            1.60,
            1.65,
            1.80,
            1.90,
            2.00,
            2.10,
            2.15,
            2.30,
            2.30,
            2.50,
            2.60,
            2.80,
            3.00,
            3.30,
            3.50,
        ],
        [
            0.000000,
            0.213853,
            0.370538,
            0.584391,
            0.739449,
            0.953302,
            1.109987,
            1.344491,
            1.558344,
            1.715029,
            1.792931,
            2.006784,
            2.089531,
            2.163469,
            2.303384,
            2.460068,
            2.532380,
            2.695312,
            2.828980,
            2.909165,
            3.065849,
            3.137422,
            3.434022,
            3.434761,
            3.882462,
            4.039803,
            4.488243,
            4.784842,
        ],
        [
            1.000000,
            0.611149,
            0.426052,
            0.260381,
            0.182201,
            0.111352,
            0.077627,
            0.045239,
            0.027647,
            0.019274,
            0.016109,
            0.009845,
            0.008137,
            0.006863,
            0.004973,
            0.003467,
            0.002935,
            0.002017,
            0.001483,
            0.001233,
            0.000859,
            0.000729,
            0.000368,
            0.000367,
            0.000131,
            0.000091,
            0.000032,
            0.000016,
        ],
    ]
).T

# the next ones are in the order of the test linearity runs by federica
# maybe more practical because the runs will be in that order again
trasmission_390ns = np.array(
    [
        0.002016919,
        0.045238588,
        0.0276475,
        0.019273981,
        0.008242516,
        0.000368111,
        9.12426e-05,
        1,
        0.611148605,
        0.260380956,
        0.111351889,
        0.004972972,
        0.001232637,
        0.009844997,
        0.426051788,
        0.077627064,
        0.006863271,
        0.003466823,
        0.000859312,
        0.016109006,
        0.182201004,
        0.002935077,
        0.001482586,
        0.000367485,
        0.008137092,
        0.00013108,
        1.64119e-05,
        3.24906e-05,
        0.000728749,
    ]
)


optical_density_390ns = np.array(
    [
        2.69531155,
        1.34449096,
        1.55834414,
        1.71502857,
        2.08394019,
        3.43402173,
        4.03980251,
        0.00000000,
        0.21385318,
        0.58439078,
        0.95330241,
        2.30338395,
        2.90916473,
        2.00678442,
        0.37053761,
        1.10998684,
        2.16346885,
        2.46006838,
        3.06584916,
        1.79293125,
        0.73944923,
        2.53238048,
        2.82898001,
        3.43476079,
        2.08953077,
        3.88246202,
        4.78484233,
        4.48824280,
        3.13742221,
    ]
)

adc_to_pe = GAIN_DEFAULT

plot_parameters = {
    "High Gain": {
        "linearity_range": [5, -5],
        "text_coords": [0.1, 600],
        "label_coords": [20, 400],
        "color": "C0",
        "initials": "HG",
    },
    "Low Gain": {
        "linearity_range": [11, -1],
        "text_coords": [0.3e2, 5],
        "label_coords": [1.5e3, 9],
        "color": "C4",
        "initials": "LG",
    },
}

intensity_percent = np.array([13.0, 15.0, 16.0, 16.5, 20.6, 22.0, 25.0, 35.0, 33])
intensity_to_charge = np.array(
    [
        1.84110824,
        2.09712394,
        2.24217532,
        2.37545181,
        10.32825426,
        28.80655155,
        102.79668643,
        191.47895686,
        188,
    ]
)


source_ids_deadtime = (
    [0 for i in range(3332, 3342)]
    + [1 for i in range(3342, 3351)]
    + [2 for i in range(3552, 3563)]
)

deadtime_labels = {
    0: {"source": "FFCLS", "color": "red"},
    1: {"source": "NSB", "color": "blue"},
    2: {"source": "Laser", "color": "purple"},
}


def get_bad_pixels_list():
    # List of modules and pixels to be rejected
    try:
        df = pd.read_json("resources/bad_pix_module.json")
        modules_list = np.array(df.bad_module[0])
        pix_list = np.array(df.bad_pixel[0])

        pix_nos = np.arange(7)

        # print("module_to_pix ", modules_list)
        module_to_pix = modules_list[:, None] * 7 + pix_nos
        combined = np.concatenate([pix_list, module_to_pix.ravel()])

        bad_pix_list = np.unique(combined)

    except (IOError, OSError):
        bad_pix_list = None

    return bad_pix_list


def get_gain_run(temperature):
    # Searches for the run number corresponding to the temperature
    # If no runs are taken at that temperature, it will return
    # the run corresponding to closest temperature
    temp = np.array([-10, -5, 0, 5, 10, 14, 20, 25])
    runs = np.array([6853, 6775, 6718, 6589, 7191, 7000, 7123, 7066])

    idx = np.argmin(np.abs(temp - temperature))
    run_no = runs[idx]

    # print(gain_file_name)

    return run_no


def pe_from_intensity_percentage(
    percent,
    percent_from_calibration=intensity_percent,
    known_charge=intensity_to_charge,
):
    """Converts a percentage of intensity to the corresponding charge value based on a
    known calibration.

    Args:
        percent (numpy.ndarray): The percentage of intensity to convert to charge.
        percent_from_calibration (numpy.ndarray, optional): The known percentages of\
            intensity used in the calibration. Defaults to `intensity_percent`.
        known_charge (numpy.ndarray, optional): The known charge values corresponding\
            to the calibration percentages. Defaults to `intensity_to_charge`.

    Returns:
        numpy.ndarray: The charge values corresponding to the input percentages.
    """
    # known values from laser calibration

    # runs are done with intensity of 15-35 percent
    f = interp1d(percent_from_calibration, known_charge)
    charge = np.zeros(len(percent))
    for i in range(len(percent)):
        charge[i] = f(percent[i])

    return charge


# functions by federica
def linear_fit_function(x, a, b):
    """Computes a linear function of the form `a*x + b`.

    Args:
        x (float): The input value.
        a (float): The slope coefficient.
        b (float): The y-intercept.

    Returns:
        float: The result of the linear function.
    """
    return a * x + b


def second_degree_fit_function(x, a, b, c):
    """Computes a quadratic function of the form `a*(x**2) + b*x + c`.

    Args:
        x (float): The input value.
        a (float): The coefficient of the squared term.
        b (float): The coefficient of the linear term.
        c (float): The constant term.

    Returns:
        float: The result of the quadratic function.
    """
    return a * (x**2) + b * x + c


def third_degree_fit_function(x, a, b, c, d):
    """Computes a function of the form `(a*x + b)/(1+c) + d`.

    Args:
        x (float): The input value.
        a (float): The coefficient of the linear term.
        b (float): The constant term in the numerator.
        c (float): The coefficient of the denominator.
        d (float): The constant term added to the result.

    Returns:
        float: The result of the function.
    """
    return (a * x + b) / (1 + c) + d


def fit_function_hv(x, a, b):
    """Computes a function of the form `a/sqrt(x) + b`.

    Args:
        x (float): The input value.
        a (float): The coefficient of the term `1/sqrt(x)`.
        b (float): The constant term.

    Returns:
        float: The result of the function.
    """
    return a / np.sqrt(x) + b


def err_ratio(nominator, denominator, err_norm, err_denom, cov_nom_den=0):
    """Computes the error ratio for a given nominator, denominator, and their respective
    errors.

    Args:
        nominator (float): The nominator value.
        denominator (float): The denominator value.
        err_norm (float): The error of the nominator.
        err_denom (float): The error of the denominator.
        cov_nom_den (float, optional): The covariance between the nominator and\
            denominator. Defaults to 0.

    Returns:
        float: The error ratio.
    """
    delta_err2 = (
        (err_norm / nominator) ** 2
        + (err_denom / denominator) ** 2
        - 2 * cov_nom_den / (nominator * denominator)
    )
    ratio = nominator / denominator
    return np.sqrt(delta_err2) * ratio


def err_sum(err_a, err_b, cov_a_b=0):
    """Computes the square root of the sum of the squares of `err_a` and `err_b`, plus
    twice the covariance `cov_a_b`.

    This function is used to calculate the combined error of two values, taking into\
        account their individual errors and the covariance between them.

    Args:
        err_a (float): The error of the first value.
        err_b (float): The error of the second value.
        cov_a_b (float, optional): The covariance between the two values. Defaults to 0.

    Returns:
        float: The combined error.
    """
    return np.sqrt(err_a**2 + err_b**2 + 2 * cov_a_b)


# from stackoverflow
def argmedian(x, axis=None):
    """Returns the index of the median element in the input array `x` along the
    specified axis.

    If `axis` is `None`, the function returns the index of the median element in\
         the flattened array.
    Otherwise, it computes the argmedian along the specified axis and returns an\
         array of indices.

    Args:
        x (numpy.ndarray): The input array.
        axis (int or None, optional): The axis along which to compute the argmedian.\
            If `None`, the argmedian is computed on the flattened array.

    Returns:
        int or numpy.ndarray: The index or indices of the median element(s) in the\
            input array.
    """
    if axis is None:
        return np.argpartition(x, len(x) // 2)[len(x) // 2]
    else:
        # Compute argmedian along specified axis
        return np.apply_along_axis(
            lambda x: np.argpartition(x, len(x) // 2)[len(x) // 2], axis=axis, arr=x
        )


def pe2photons(x):
    """Converts the input value `x` from photons to photoelectrons (PE) by multiplying
    it by 4.

    Args:
        x (float): The input value in photons.

    Returns:
        float: The input value converted to photoelectrons.
    """
    return x * 4


def photons2pe(x):
    """Converts the input value `x` from photoelectrons (PE) to photons by dividing it
    by 4.

    Args:
        x (float): The input value in photoelectrons.

    Returns:
        float: The input value converted to photons.
    """
    return x / 4


def photons2ADC(n):
    """
    converts a number of photons into ADC counts

        Args:
        n (float): The input value in photon number.

    Returns:
        float: The input value converted to a charge in ADC counts.
    """

    pe = photons2pe(n)
    charge = pe * adc_to_pe

    return charge


# from Federica's notebook
class ExponentialFitter:
    """Represents an exponential fitter class that computes the expected distribution
    and the minus 2 log likelihood for a given dataset and exponential parameters.

    Attributes:
        data (numpy.ndarray): The input data array.
        bin_edges (numpy.ndarray): The bin edges for the data.

    Methods:
        compute_expected_distribution(norm, loc, scale):
            Computes the expected distribution given the normalization, location, and\
                scale parameters.
        expected_distribution(x):
            Returns the expected distribution given the parameters in `x`.
        compute_minus2loglike(x):
            Computes the minus 2 log likelihood given the parameters in `x`.
    """

    def __init__(self, data, bin_edges):
        self.data = data.copy()
        self.bin_edges = bin_edges.copy()

    def compute_expected_distribution(self, norm, loc, scale):
        cdf_low = expon.cdf(self.bin_edges[:-1], loc=loc, scale=scale)
        cdf_up = expon.cdf(self.bin_edges[1:], loc=loc, scale=scale)
        delta_cdf = cdf_up - cdf_low
        return norm * delta_cdf

    def expected_distribution(self, x):
        return self.compute_expected_distribution(x[0], x[1], x[2])

    def compute_minus2loglike(self, x):
        norm = x[0]
        loc = x[1]
        scale = x[2]

        expected = self.compute_expected_distribution(norm, loc, scale)
        chi2_mask = expected > 0.0
        minus2loglike = -2.0 * np.sum(
            poisson.logpmf(self.data[chi2_mask], mu=expected[chi2_mask])
        )
        minus2loglike0 = -2.0 * np.sum(
            poisson.logpmf(self.data[chi2_mask], mu=self.data[chi2_mask])
        )
        return minus2loglike - minus2loglike0


def pois(x, A, R):
    """Computes the expected distribution for a Poisson process with rate parameter `R`.

    Parameters
    ----------
    x : float
        The input value.
    A : float
        The amplitude parameter.
    R : float
        The rate parameter.

    Returns
    -------
    A * np.exp(x * R) : float
        The expected distribution for the Poisson process.
    """

    # Poisson function, parameter R (rate) is the fit parameter.
    return A * np.exp(x * R)


# function by Federica
def plot_deadtime_and_expo_fit(
    total_delta_t_for_busy_time, deadtime_us, run, verbose=False, output_plot=None
):
    """Compute the deadtime and exponential fit parameters for a given dataset.

    Parameters
    ----------
    total_delta_t_for_busy_time : float
        The total time of the dataset.
    deadtime_us : numpy.ndarray
        The deadtime of the dataset in microseconds.
    run : int
        The run number.
    verbose : bool
        Whether to print the fit results or not.
    output_plot : str, optional
        The path to save the output plot.

    Returns
    -------
    deadtime : float
        The deadtime of the dataset.
    deadtime_bin : float
        The bin edge corresponding to the deadtime.
    deadtime_err : float
        The error on the deadtime.
    deadtime_bin_length : float
        The length of the deadtime bin.
    total_delta_t_for_busy_time : float
        The total time of the dataset.
    parameter_A_new : float
        The amplitude parameter of the exponential fit.
    parameter_R_new : float
        The rate parameter of the exponential fit.
    parameter_A_err_new : float
        The error on the amplitude parameter.
    parameter_R_err_new : float
        The error on the rate parameter.
    first_bin_length : float
        The length of the first bin.
    tot_nr_events_histo : int
        The total number of events in the histogram.
    """
    # Select max value for the x axis depending on what is the maximum
    # measured deadtime, to not cut the distribution
    max_x_values_for_plot = np.array([500, 1000, 1500, 2000, 2500])
    max_x_value_for_plot = max_x_values_for_plot[
        np.argmin(np.abs(max(deadtime_us) - max_x_values_for_plot))
    ]

    entries, bin_edges = np.histogram(
        deadtime_us, bins=100, range=(0, max_x_value_for_plot + 20)
    )

    # Deadtime defined as the minimum value
    # from the measured deltaT on the events timestamps
    deadtime = min(deadtime_us[~np.isnan(deadtime_us)])

    first_nonempty_bin = np.where(entries > 0)[0][0]
    deadtime_err = (
        bin_edges[first_nonempty_bin] - bin_edges[first_nonempty_bin - 1]
    ) / np.sqrt(entries[first_nonempty_bin])

    deadtime_bin = bin_edges[first_nonempty_bin]
    deadtime_bin_length = (
        bin_edges[first_nonempty_bin] - bin_edges[first_nonempty_bin - 1]
    )

    # the bins should be of integer width, because poisson is an integer distribution
    x_steps_for_bins = np.arange(0, max_x_value_for_plot + 20, step=5)
    x_steps_for_bins = x_steps_for_bins[1:] - 5

    entries, bin_edges = np.histogram(
        deadtime_us, bins=x_steps_for_bins, range=[0, total_delta_t_for_busy_time]
    )
    bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1]) - deadtime
    # bin_middles is used to fit the exponential function,
    # from https://arxiv.org/abs/2311.11631,
    # Eq. 2: f(deltaT, deadtime, R) = A x exp(-R(deltaT - deadtime))

    first_bin_length = bin_edges[1] - bin_edges[0]
    tot_nr_events_histo = np.sum(entries)

    # fit with the exponential function
    model = Model(pois)
    params = model.make_params(A=2e3, R=-0.001)
    result = model.fit(entries, params, x=bin_middles)
    if verbose:
        log.info("**** FIT RESULTS RUN {} ****".format(run))
        log.info(result.fit_report())
        log.info("chisqr = {}, redchi = {}".format(result.chisqr, result.redchi))

    parameter_A_new = result.params["A"].value
    parameter_R_new = result.params["R"].value
    parameter_A_err_new = result.params["A"].stderr
    parameter_R_err_new = result.params["R"].stderr

    if output_plot:
        _, ax = plt.subplots(1, 1, figsize=(10, 10 / 1.61), layout="constrained")
        # plot deltaT distribution
        plt.hist(
            deadtime_us,
            bins=x_steps_for_bins,
            histtype="step",
            density=0,
            lw=3,
        )

        # plot exponential function with fitted parameter
        plt.plot(
            x_steps_for_bins,
            pois(x_steps_for_bins, parameter_A_new, parameter_R_new),
            marker="",
            linestyle="-",
            lw=3,
            color="C3",
        )

        # Chi2 computation
        observed_events = entries
        expected_events = pois(x_steps_for_bins, parameter_A_new, parameter_R_new)[1:]
        chi_sqr = np.sum((observed_events - expected_events) ** 2 / expected_events)
        dof = result.summary()["nfree"]

        rate = ((-1 * parameter_R_new) * (1 / u.us)).to(u.kHz).value  # rate in kHz
        rate_stderr = ((parameter_R_err_new) * (1 / u.us)).to(u.kHz).value

        ax.text(
            max_x_value_for_plot - 50,
            np.max(entries) - 10,
            f"Run {run}"
            + "\n"
            + r"$f(\Delta t) = A \cdot$"
            + r"$\exp({-R \cdot (\delta t - \Delta_{\mathrm{min}})})$"
            + "\n"
            + r"$A=%2.2f \pm %2.2f$" % (parameter_A_new, parameter_A_err_new)
            + "\n"
            + r"$R=(%2.2f \pm %2.2f)$ kHz" % (rate, rate_stderr)
            + "\n"
            + r"$\chi^2$/dof = %2.0f/%2.0f" % (chi_sqr, dof)
            + "\n"
            + r"$\delta_{\mathrm{min}} = (%2.3f \pm %2.3f) \, \mu$s"
            % (deadtime, np.abs(deadtime_err) * 1e-3),
            backgroundcolor="white",
            bbox=dict(
                facecolor="white", edgecolor="C3", lw=2, boxstyle="round,pad=0.3"
            ),
            ha="right",
            va="top",
            fontsize=17,
            color="k",
            alpha=0.9,
        )

        ax.tick_params(axis="both", labelsize=16)
        ax.tick_params(which="major", direction="in", length=7, width=2)
        ax.tick_params(which="minor", direction="in", length=4, width=1)

        ax.set_xlim(-15, max_x_value_for_plot)

        plt.xlabel(r"$\Delta$T [$\mu$s]", fontsize=15)
        plt.ylabel("Entries", fontsize=15)
        plot_path = os.path.join(
            output_plot, "deadtime_and_expo_fit_{}.png".format(run)
        )
        plt.savefig(plot_path)
        log.info(f"Plot saved at {plot_path}")

    return (
        deadtime,
        deadtime_bin,
        deadtime_err,
        deadtime_bin_length,
        total_delta_t_for_busy_time,
        parameter_A_new,
        parameter_R_new,
        parameter_A_err_new,
        parameter_R_err_new,
        first_bin_length,
        tot_nr_events_histo,
    )
