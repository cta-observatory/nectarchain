import os

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from lmfit.models import Model
from scipy.interpolate import interp1d
from scipy.stats import expon, poisson
from traitlets.config import Config

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

adc_to_pe = 58.0

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
    + [1 for i in range(3342, 3350)]
    + [2 for i in range(3552, 3562)]
)

deadtime_labels = {
    0: {"source": "random generator", "color": "blue"},
    1: {"source": "nsb source", "color": "green"},
    2: {"source": "laser", "color": "red"},
}


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


# from federica's notebook
class ExponentialFitter:
    """Represents an exponential fitter class that computes the expected distribution
    and the minus 2 log likelihood for a given dataset and exponential parameters.

    Attributes:
        datas (numpy.ndarray): The input data array.
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

    def __init__(self, datas, bin_edges):
        self.datas = datas.copy()
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
            poisson.logpmf(self.datas[chi2_mask], mu=expected[chi2_mask])
        )
        minus2loglike0 = -2.0 * np.sum(
            poisson.logpmf(self.datas[chi2_mask], mu=self.datas[chi2_mask])
        )
        #         print(f'{minus2loglike0 = }')
        return minus2loglike - minus2loglike0


def pois(x, A, R):
    """Computes the expected distribution for a Poisson process with rate parameter `R`.

    Args:
        x (float): The input value.
        A (float): The amplitude parameter.
        R (float): The rate parameter.

    Returns:
        float: The expected distribution for the Poisson process.
    """
    """Poisson function, parameter r (rate) is the fit parameter."""
    return A * np.exp(x * R)


def deadtime_and_expo_fit(time_tot, deadtime_us, run, output_plot=None):
    """Computes the deadtime and exponential fit parameters for a given dataset.

    Args:
        time_tot (float): The total time of the dataset.
        deadtime_us (float): The deadtime of the dataset in microseconds.
        run (int): The run number.
        output_plot (str, optional): The path to save the output plot.

    Returns:
        tuple: A tuple containing the following values:
            - deadtime (float): The deadtime of the dataset.
            - deadtime_bin (float): The bin edge corresponding to the deadtime.
            - deadtime_err (float): The error on the deadtime.
            - deadtime_bin_length (float): The length of the deadtime bin.
            - total_delta_t_for_busy_time (float): The total time of the dataset.
            - parameter_A_new (float): The amplitude parameter of the exponential fit.
            - parameter_R_new (float): The rate parameter of the exponential fit.
            - parameter_A_err_new (float): The error on the amplitude parameter.
            - parameter_R_err_new (float): The error on the rate parameter.
            - first_bin_length (float): The length of the first bin.
            - tot_nr_events_histo (int): The total number of events in the histogram.
    """
    # function by federica

    total_delta_t_for_busy_time = time_tot
    data = deadtime_us
    pucts = np.histogram(data, bins=100, range=(1e-2, 50))

    deadtime = min(data[~np.isnan(data)])

    first_nonemptybin = np.where(pucts[0] > 0)[0][0]
    deadtime_err = (
        pucts[1][first_nonemptybin] - pucts[1][first_nonemptybin - 1]
    ) / np.sqrt(pucts[0][first_nonemptybin])
    deadtime_bin = pucts[1][first_nonemptybin]
    deadtime_bin_length = pucts[1][first_nonemptybin] - pucts[1][first_nonemptybin - 1]

    # the bins should be of integer width, because poisson is an integer distribution
    bins = np.arange(100000) - 0.5
    entries, bin_edges = np.histogram(
        data, bins=bins, range=[0, total_delta_t_for_busy_time]
    )
    bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    first_bin_length = bin_edges[1] - bin_edges[0]
    tot_nr_events_histo = np.sum(entries)

    # second fit
    model = Model(pois)
    params = model.make_params(A=2e3, R=-0.001)
    result = model.fit(entries, params, x=bin_middles)
    # print('**** FIT RESULTS RUN {} ****'.format(run))
    # print(result.fit_report())
    # print('chisqr = {}, redchi = {}'.format(result.chisqr, result.redchi))

    parameter_A_new = result.params["A"].value
    parameter_R_new = -1 * result.params["R"].value
    parameter_A_err_new = result.params["A"].stderr
    parameter_R_err_new = result.params["R"].stderr

    plt.close()

    if output_plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10 / 1.61))
        plt.hist(
            data,
            bins=np.arange(1000) - 0.5,
            alpha=0.8,
            histtype="step",
            density=0,
            lw=2,
            label="Run {}".format(run),
        )

        # # plot poisson-deviation with fitted parameter
        x_plot = np.arange(0, 1000)

        plt.plot(
            x_plot,
            pois(x_plot, result.params["A"].value, result.params["R"].value),
            marker="",
            linestyle="-",
            lw=3,
            color="C3",
            alpha=0.85,
            label=r"Fit result: %2.3f $\exp^{-{x}/({%2.0f}~\mu\mathrm{s})}$"
            % (result.params["A"].value, abs(1 / result.params["R"].value)),
        )

        R = ((-1 * result.params["R"].value) * (1 / u.us)).to(u.kHz).value

        R_stderr = ((result.params["R"].stderr) * (1 / u.us)).to(u.kHz).value

        ax.text(
            600,
            entries[1] / 1,
            r"$y = A \cdot \exp({-R \cdot x})$\n"
            #          + r'$A=%2.2f \pm %2.2f$'%(as_si((result.params['A'].value/1000)
            # *1e3,2), as_si((result.params['A'].stderr/1000)*1e3,2))
            + r"$A=%2.2f \pm %2.2f$"
            % (result.params["A"].value, result.params["A"].stderr)
            + "\n"
            + r"$R=(%2.2f \pm %2.2f)$ kHz" % (R, R_stderr)
            + "\n"
            + r"$\chi^2_\nu = %2.2f$" % ((result.redchi))
            + "\n"
            + r"$\delta_{\mathrm{deadtime}} = %2.3f \, \mu$s" % (deadtime),
            backgroundcolor="white",
            bbox=dict(
                facecolor="white", edgecolor="C3", lw=2, boxstyle="round,pad=0.3"
            ),
            ha="left",
            va="top",
            fontsize=17,
            color="k",
            alpha=0.9,
        )

        plt.xlabel(r"$\Delta$T [$\mu$s]")
        plt.ylabel("Entries")
        plt.title("Run {}".format(run), y=1.02)
        plt.savefig(
            os.path.join(output_plot, "deadtime_and_expo_fit_{}.png".format(run))
        )

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
