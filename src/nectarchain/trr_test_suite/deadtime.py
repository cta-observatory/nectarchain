# don't forget to set environment variable NECTARCAMDATA

import argparse
import copy
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from ctapipe.containers import EventType
from iminuit import Minuit

from nectarchain.trr_test_suite.tools_components import DeadtimeTestTool
from nectarchain.trr_test_suite.utils import ExponentialFitter
from nectarchain.trr_test_suite.utils import deadtime_labels as deadtime_labels_trr
from nectarchain.trr_test_suite.utils import (
    plot_deadtime_and_expo_fit,
    source_ids_deadtime,
)
from nectarchain.utils.constants import ALLOWED_CAMERAS

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[logging.getLogger("__main__").handlers],
)
log = logging.getLogger(__name__)

try:
    plt.style.use(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "../utils/plot_style.mpltstyle"
        )
    )
except FileNotFoundError as e:
    raise e


def get_labels():
    """Labels for source types are taken from ctapipe.containers.EventType,
       each label for the corresponding event type ID

    Returns
    -------
    dict
        Dictionary containing the source type labels
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to the JSON file
    json_path = os.path.join(
        script_dir, "../trr_test_suite/resources/source_type_labels.json"
    )

    with open(json_path, "r") as f:
        source_labels = json.load(f)

    return source_labels


def plot_deadtime_vs_collected_trigger_rate(
    ids,
    collected_trigger_rates,
    deadtime_pc,
    error_deadtime_pc,
    deadtime_pc_fit,
    show_camera_client,
    labels,
    output_dir,
    temp_output,
    test_type,
):
    """Plot the deadtime percentage vs the trigger rates

    Parameters
    ----------
    ids : np.ndarray
        Source ids for all the runs
    collected_trigger_rates : np.ndarray
        Collected trigger rate values from the camera client counters
    deadtime_pc : np.ndarray
        Deadtime percentage values computed with the camera client
    error_deadtime_pc : np.ndarray
        Error on deadtime percentage values computed with the camera client
    deadtime_pc_fit : np.ndarray
        Deadtime percentage values extracted
        from the distribution in the exponential fit
    show_camera_client : bool
        Whether to show the shadead areas for the
        percentage values extracted from the camera client counters
    labels : dict
        Labels with the source names for the plot
    output_dir : str
        Path to the output directory to save the plot
    temp_output : str
        Path to temporary output directory for the GUI
    test_type : str
        Test type to specify the source ids.
        Accepted options are 'trr' and 'av',
        for 'Test-Readiness Review' and 'Acceptance Verification'.
    """

    if test_type not in ["trr", "av"]:
        log.warning("Invalid chosen 'test_type', falling back to 'trr'.")
        test_type = "trr"
    available_ids = np.unique(ids).tolist()

    fig, ax = plt.subplots()

    for source in available_ids:
        mask_source = np.where(ids == source)[0]

        if test_type == "av":
            source = str(source)

        if show_camera_client:
            deadtime_pc_source = deadtime_pc[mask_source]
            error_deadtime_pc_source = error_deadtime_pc[mask_source]

            xx_sorted = np.sort(
                np.array(collected_trigger_rates[mask_source]) / 1e3
            )  # kHz
            argsort_xx = np.argsort(np.array(collected_trigger_rates[mask_source]))

            yy_sorted = np.array(deadtime_pc_source[argsort_xx])
            yy_err_sorted = np.array(error_deadtime_pc_source[argsort_xx])

            plt.fill_between(
                xx_sorted,  # collected trigger rate
                y1=yy_sorted - 2 * yy_err_sorted,  # deadtime % and error on deadtime %
                y2=yy_sorted + 2 * yy_err_sorted,
                alpha=0.2,
                color=labels[source]["color"],
                label=labels[source]["source"] + r"$N_{\rm busy}/N_{\rm collected}$",
            )

        xx_sorted = np.sort(np.array(collected_trigger_rates[mask_source]) / 1e3)  # kHz
        argsort_xx = np.argsort(np.array(collected_trigger_rates[mask_source]))
        deadtime_pc_source_fit = deadtime_pc_fit[mask_source]
        yy_fit_sorted = deadtime_pc_source_fit[argsort_xx]

        plt.plot(
            xx_sorted,  # fitted trigger rate
            yy_fit_sorted,
            alpha=0.6,
            ls="-",
            lw=3,
            marker="o",
            color=labels[source]["color"],
            label=labels[source]["source"] + r", $R \times \delta_{\rm min}$",
        )

    plt.xlabel("Collected Trigger Rate [kHz]", fontsize=15)
    plt.ylabel(r"Deadtime [%]", fontsize=15)

    plt.axhline(5, ls="-", lw=3, color="gray", alpha=0.4)
    plt.axvline(7, ls="-", lw=3, color="gray", alpha=0.4)

    ax.text(
        3.5,
        6.75,
        "CTAO requirement",
        color="gray",
        fontsize=20,
        alpha=0.7,
        horizontalalignment="center",
        verticalalignment="center",
    )

    plt.legend(fontsize=12)

    plt.xlim(0.0, 15)
    plt.yscale("log")
    plt.ylim(1e-2, 1e2)
    plot_path = os.path.join(output_dir, "deadtime_pc_vs_trigger_rate.png")
    plt.savefig(plot_path)

    log.info(f"Plot saved at: {plot_path}")

    if temp_output:
        with open(os.path.join(temp_output, "plot1.pkl"), "wb") as f:
            pickle.dump(fig, f)


def plot_fitted_rate_vs_collected_trigger_rate(
    ids,
    collected_trigger_rates,
    lambda_from_fit,
    lambda_from_fit_err,
    labels,
    output_dir,
    temp_output,
    test_type,
):
    """Plot the fitted vs collected trigger rates and the relative difference

    Parameters
    ----------
    ids : np.ndarray
        Source ids for all the runs.
    collected_trigger_rates : np.ndarray
        Collected trigger rate values from the camera client counters
    lambda_from_fit : np.ndarray
        Lambda parameter values from the exponential fit with `fit_rate_per_run`
    lambda_from_fit_err : np.ndarray
        Error on lambda parameter values
        from the exponential fit with `fit_rate_per_run`
    labels : dict
        Labels with the source names for the plot
    output_dir : str
        Path to the output directory to save the plot
    temp_output : str
        Path to temporary output directory for the GUI
    test_type : str
        Test type to specify the source ids.
        Accepted options are 'trr' and 'av',
        for 'Test-Readiness Review' and 'Acceptance Verification'.
    """

    if test_type not in ["trr", "av"]:
        log.warning("Invalid chosen 'test_type', falling back to 'trr'.")
        test_type = "trr"
    available_ids = np.unique(ids).tolist()

    fig, ((ax1, ax2)) = plt.subplots(
        2,
        1,
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": [5, 2]},
    )

    xx = collected_trigger_rates / 1000
    rate = np.array(lambda_from_fit)
    yy = lambda_from_fit
    rate_err = np.array(lambda_from_fit_err)
    relative = (yy - xx) / xx * 100

    x_err = 0
    err_ratio = relative * (((rate_err + x_err) / (yy - xx)) + x_err / xx)
    for source in available_ids:
        runl = np.where(ids == source)[0]

        if test_type == "av":
            source = str(source)

        ax2.errorbar(
            xx[runl],
            relative[runl],
            xerr=x_err,
            yerr=err_ratio[runl],
            alpha=0.9,
            ls=" ",
            marker="o",
            color=labels[source]["color"],
        )

    ax2.set_ylim(-15, 15)

    xx = range(0, 60)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.plot(xx, xx, color="gray", ls="--", alpha=0.5)

    ax2.plot(xx, np.zeros(len(xx)), color="gray", ls="--", alpha=0.5)
    ax2.fill_between(
        xx, np.ones(len(xx)) * (-10), np.ones(len(xx)) * (10), color="gray", alpha=0.1
    )

    ax2.set_xlabel("Collected Trigger Rate [kHz]", fontsize=15)
    ax1.set_ylabel(r"Rate from fit [kHz]", fontsize=15)
    ax2.set_ylabel(r"Relative difference [%]", fontsize=15)

    ax1.set_xlim(1e0, 60)
    ax1.set_ylim(1e0, 60)
    ax2.set_xlim(1e0, 60)

    for source in available_ids:
        runl = np.where(ids == source)[0]

        if test_type == "av":
            source = str(source)

        ax1.errorbar(
            collected_trigger_rates[runl] / 1000,
            rate[runl],
            yerr=rate_err[runl],
            alpha=0.9,
            ls=" ",
            marker="o",
            color=labels[source]["color"],
            label=labels[source]["source"],
        )

    ax1.legend(fontsize=12)

    plot_path = os.path.join(output_dir, "fitted_vs_collected_trigger_rates.png")
    plt.savefig(plot_path)

    log.info(f"Plot saved at: {plot_path}")

    if temp_output:
        with open(os.path.join(temp_output, "plot2.pkl"), "wb") as f:
            pickle.dump(fig, f)


def fit_rate_per_run(runlist: list, deadtime_us: np.ndarray):
    """Fit the exponential function over the provided run list

    Parameters
    ----------
    runlist : list
        NectarCAM run numbers
    deadtime_us : np.ndarray
        Deadtime values in mus

    histograms_deadtime, bin_edges_hist,
    results

    Returns
    -------
    parameter_A2_list : list
        Values of the norm parameter A for the exponential fit
    parameter_A2_err_list : list
        Error on values of the norm parameter A for the exponential fit
    parameter_lambda_list : list
        Values of the rate parameter lambda for the exponential fit
    parameter_lambda_err_list : list
        Error on the values of the rate parameter lambda for the exponential fit
    parameter_tau_list : list
        Values of the localization parameter tau for the exponential fit
    parameter_tau_err_list : list
        Error on the values of the localization parameter tau for the exponential fit
    parameter_R2_list : list
        Results of the r-squared test
    histograms_deadtime : list
        Histogram distributions of the deadtime values for all the runs
    bin_edges_hist : list
        Edges of the histogram distributions
    results : dict
        Results of the fitted parameters for all the runs
    """

    parameter_A2_list, parameter_A2_err_list = [], []
    parameter_lambda_list, parameter_lambda_err_list = [], []
    parameter_tau_list, parameter_tau_err_list = [], []
    parameter_R2_list = []
    histograms_deadtime, bin_edges_hist = [], []

    results = {}

    fitted_rate = []
    for ii in range(len(runlist)):
        log.info(f"Fitting rate for run {runlist[ii]}")

        deadtime_mus = deadtime_us[ii].value[deadtime_us[ii].value > 0] * 1e-6

        lim_low_mus, lim_high_mus = 0.001e-6, 120 * 1e-6

        # lower and upper limits for the deadtime binning in mus
        nr_bins = 100

        rate_initial_guess = 40000  # in Hz

        data_content, bin_edges = np.histogram(
            deadtime_mus, bins=np.linspace(lim_low_mus, lim_high_mus, nr_bins)
        )

        histograms_deadtime.append(data_content)
        bin_edges_hist.append(bin_edges)

        init_param = [np.sum(data_content), 0.6e-6, 1.0 / rate_initial_guess]

        fitter = ExponentialFitter(data_content, bin_edges=bin_edges)
        m = Minuit(
            fitter.compute_minus2loglike,
            init_param,
            name=("Norm", "Deadtime", "1/Rate"),
        )

        # Set parameter limits and tolerance
        m.errors["Norm"] = 0.3 * init_param[0]
        m.limits["Norm"] = (0.0, np.sum(data_content) * 2.0)

        m.errors["Deadtime"] = 0.1e-6
        m.limits["Deadtime"] = (0.06e-6, 2.1e-6)
        # Put some tight constraints as the fit will be
        # in trouble when it expects 0, and measures something instead...

        m.errors["1/Rate"] = init_param[2] / 2.0
        m.limits["1/Rate"] = (init_param[2] / 10.0, init_param[2] * 10.0)

        m.print_level = 0
        res = m.migrad(2000000)
        # called with the maximum number of function calls

        fitted_params = np.array([res.params[p].value for p in res.parameters])
        fitted_params_err = np.array([res.params[p].error for p in res.parameters])

        results[int(runlist[ii])] = [
            1.0e6 * fitted_params[1],
            1.0e6 * fitted_params_err[1],  # deadtime in mus
            1.0 / fitted_params[2],
            fitted_params_err[2] / (fitted_params[2] ** 2),  # Rate in Hz
            fitted_params[0] * fitted_params[2],  # Expected run duration
        ]

        fitted_rate.append(1.0 / fitted_params[2])

        y = data_content
        y_fit = fitter.expected_distribution(fitted_params)
        # residual sum of squares
        ss_res = np.sum((y - y_fit) ** 2)
        # total sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        # r-squared
        r2 = 1 - (ss_res / ss_tot)

        parameter_A2_list.append(fitted_params[0])
        parameter_A2_err_list.append(fitted_params_err[0])

        parameter_lambda_list.append(1.0 / fitted_params[2] / 1e3)  # kHz
        parameter_lambda_err_list.append(
            fitted_params_err[2] / (fitted_params[2] ** 2) / 1e3
        )

        parameter_tau_list.append(1e6 * fitted_params[1])
        parameter_tau_err_list.append(1e6 * fitted_params_err[1])

        parameter_R2_list.append(r2)

    return (
        parameter_A2_list,
        parameter_A2_err_list,
        parameter_lambda_list,
        parameter_lambda_err_list,
        parameter_tau_list,
        parameter_tau_err_list,
        parameter_R2_list,
        histograms_deadtime,
        bin_edges_hist,
        results,
    )


def run_deadtime_test_tool_process(
    runlist: list,
    camera: str,
    nevents: int,
    ids: np.ndarray,
    test_type: str = "trr",
):
    """Run `DeadtimeTestTool` from `tools_components.py` over the provided run list

    Parameters
    ----------
    runlist : list
        list containing the NectarCAM run numbers
    camera : str
        NectarCAM camera for which the test is performed
    nevents : int
        max number of events
    ids : np.ndarray
        Source ids for all the runs
    test_type : str
        Test type to specify the source ids.
        Accepted options are 'trr' and 'av',
        for 'Test-Readiness Review' and 'Acceptance Verification'.

    Returns
    -------
    ucts_timestamps : list
        All the UCTS timestamps of the extracted events
    ucts_deltat : list
        All the deltaT computed from the UCTS timestamps of the events
    event_counter : list
        The event counter. The last number corresponds to the number of events
    busy_counter : list
        The busy counter. The last number corresponds to the number of busy events
    collected_trigger_rates : list
        The values of collected trigger rates from the camera client
    time_tot : list
        The total recorded time for each run
    deadtime_us : list
        The deadtime values computed as the deltaT between recorded events
    deadtime_pc : list
        The deadtime percentage value computed with the counters for each run
    """

    if test_type not in ["trr", "av"]:
        log.warning("Invalid chosen 'test_type', falling back to 'trr'.")
        test_type = "trr"

    ucts_timestamps, ucts_deltat = [], []
    event_counter, busy_counter = [], []
    collected_trigger_rates = []
    time_tot = []
    deadtime_us, deadtime_pc = [], []

    log.info(f"Starting `DeadtimeTestTool` for test {test_type}")

    for run, id in zip(runlist, ids):
        log.info("Processing `DeadtimeTestTool` on run {}".format(run))
        tool = DeadtimeTestTool(
            progress_bar=True,
            run_number=run,
            camera=camera,
            max_events=nevents,
            events_per_slice=10000,
            log_level=20,
            method="LocalPeakWindowSum",
            extractor_kwargs={"window_width": 16, "window_shift": 6},
            overwrite=True,
        )
        tool.initialize()
        tool.setup()
        tool.start()
        output = tool.finish(id=id, test_type=test_type)

        ucts_timestamps.append(output[0])
        ucts_deltat.append(output[1])

        event_counter.append(output[2])
        busy_counter.append(output[3])

        collected_trigger_rates.append(output[4].value)

        time_tot.append(output[5].value)

        deadtime_pc.append(output[6])
        deadtime_us.append((output[1] * u.ns).to(u.us))

    return (
        ucts_timestamps,
        ucts_deltat,
        event_counter,
        busy_counter,
        collected_trigger_rates,
        time_tot,
        deadtime_us,
        deadtime_pc,
    )


def get_args():
    """Parses command-line arguments for the deadtime test script.

    Returns
    -------
    parser : argparse.ArgumentParser
        The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Deadtime tests B-TEL-1260 & B-TEL-1270. \n"
        + "According to the nectarchain component interface, you have to set a \
            NECTARCAMDATA environment variable in the folder where you have the data \
                from your runs or where you want them to be downloaded.\n"
        + "You have to provide a run number (or list of numbers), the event source, \
            a corresponding camera tag and, optionally, the number of events \
            to consider for the test and an output directory \
                to save the final plots.\n"
        + "If the data is not in NECTARCAMDATA, the files will be downloaded through \
            DIRAC.\n"
        + "You can optionally specify the number of events to be processed \
            (default 8000).\n",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-r",
        "--run_numbers",
        type=int,
        nargs="+",
        help="Run number (or list of numbers)",
        required=True,
        default=[i for i in range(3332, 3351)] + [i for i in range(3552, 3563)],
    )
    parser.add_argument(
        "--test_type",
        help="Test type to specify the source ids. "
        "Accepted options are 'trr' and 'av', "
        "for 'Test-Readiness Review' and 'Acceptance Verification'.",
        choices=["trr", "av"],
        default="trr",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--source",
        type=int,
        choices=[member.value for member in EventType],
        nargs="+",
        help="List of corresponding source for each run: "
        "- for test_type set to 'trr', 0 for random generator, 1 for NSB source, "
        "and 2 for laser if test_type is set to 'trr'; "
        "- for 'av', the available choices are among "
        "all the ids in ctapipe.containers.EventType.",
        default=[0],
    )
    # default is fixed to [0], so that the if clause in the main sets the labels
    # for 'trr' test_type, it will be source_ids_deadtime
    # for test_type 'av', following the event type ids
    # from ctapipe.containers.EventType would be:
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,32,

    parser.add_argument(
        "-e",
        "--evts",
        type=int,
        help="Number of events to process from each run.",
        required=False,
        default=8000,
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
        "-o",
        "--output",
        type=str,
        help="Output directory",
        default=f"{os.environ.get('NECTARCHAIN_FIGURES', f'/tmp/{os.getpid()}')}",
    )
    parser.add_argument(
        "--temp_output", help="Temporary output directory for GUI", default=None
    )
    parser.add_argument("--log", default="info", help="Log level", type=str)

    return parser


def main():
    """Runs the deadtime test script, which performs deadtime tests B-TEL-1260 and
    B-TEL-1270, and event rate test B-MST-1280.

    The script takes command-line arguments to specify a run number, \
        corresponding event source, the camera tag, and, optionally, \
        the number of events to consider for the test and an output directory. \
            It is also possible to choose between two test types: 'trr' and 'av', \
                for 'Test-Readiness Review' and 'Acceptance Verification'. \
            It then processes the data for each run, performs an exponential \
                fit to the deadtime distribution, and generates three plots:

    1. A plot of the exponential function fit on the deadtime\
        distribution for each run.
    2. A plot of deadtime percentage vs. collected trigger rate, with the CTAO\
        requirement indicated.
    3. A plot of the rate from the fit vs. the collected trigger rate, with the\
        relative difference shown in the bottom panel.

    The script also saves the generated plots to the specified output directory, and\
        optionally saves the last two to a temporary output directory for use\
            in a GUI.
    """

    parser = get_args()
    args = parser.parse_args()
    log.setLevel(args.log.upper())

    runlist = args.run_numbers
    ids = args.source
    # Post-processing: Apply conditional defaults
    test_type = args.test_type

    if len(runlist) != len(ids) and ids[0] == 0:
        if args.test_type == "trr":
            ids = source_ids_deadtime
        elif args.test_type == "av":
            ids = [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                32,
                32,
                32,
                32,
                32,
                32,
                32,
                32,
                32,
                32,
                32,
                32,
                32,
                32,
                32,
                32,
                32,
                32,
                32,
                32,
            ]

    assert len(runlist) == len(ids), "'runlist' and 'ids' must have the same length"

    deadtime_labels_av = get_labels()
    if test_type == "trr":
        labels = deadtime_labels_trr
    elif test_type == "av":
        labels = deadtime_labels_av

    nevents = args.evts

    kwargs = copy.deepcopy(vars(args))
    kwargs.pop("camera")
    camera = args.camera

    output_dir = os.path.join(
        os.path.abspath(args.output),
        f"{test_type}_camera_{camera}/{Path(__file__).stem}",
    )
    os.makedirs(output_dir, exist_ok=True)

    temp_output = os.path.abspath(args.temp_output) if args.temp_output else None

    # Drop arguments from the script after they are parsed, for the GUI to work properly
    sys.argv = sys.argv[:1]

    (
        _,
        _,
        event_counter,
        busy_counter,
        collected_trigger_rates,
        time_tot,
        deadtime_us,
        deadtime_pc,
    ) = run_deadtime_test_tool_process(
        runlist=runlist, camera=camera, nevents=nevents, ids=ids, test_type=test_type
    )

    results = fit_rate_per_run(runlist=runlist, deadtime_us=deadtime_us)[-1]

    log.info(f"Output directory: {output_dir}")
    log.info(f"Temporary output file: {temp_output}")
    log.info(f"N max events to be considered: {nevents}")
    log.info("-" * 40)
    for ii, (key, values) in enumerate(results.items()):
        log.info(f"For run {key}, source: {ids[ii]},")
        log.info(
            "Dead-Time extracted from the tool process: "
            f"{np.min(deadtime_us[ii]):.3f}"
        )
        log.info(f"Dead-Time from the fit: {values[0]:.3f} +- " f"{values[1]:.3f} µs")
        log.info(f"Rate from the fit: {values[2]:.2f} +- " f"{values[3]:.2f} Hz")
        log.info("Expected run duration from the fit: " f"{values[4]:.2f} s")
        log.info("-" * 40)

    ids = np.array(ids)
    runlist = np.array(runlist)

    error_deadtime_pc = []
    for run_id in range(np.array(busy_counter).shape[0]):
        error_deadtime_pc.append(
            np.sqrt(
                (busy_counter[run_id][-1] * event_counter[run_id][-1])
                / ((busy_counter[run_id][-1] + event_counter[run_id][-1]) ** 3)
            )
        )
    error_deadtime_pc = np.array(error_deadtime_pc)

    deadtime, fitted_trigger_rates, fitted_trigger_rates_err = [], [], []

    for ii, run_num in enumerate(runlist):
        results = plot_deadtime_and_expo_fit(
            total_delta_t_for_busy_time=time_tot[ii],
            deadtime_us=np.array(deadtime_us[ii].value),
            run=run_num,
            output_plot=output_dir,
            run_type=EventType(ids[ii]).name,
            temp_output=temp_output,
        )
        deadtime.append(results[0])
        fitted_trigger_rates.append(((-1 * results[6]) * (1 / u.us)).to(u.kHz).value)
        fitted_trigger_rates_err.append(((results[8]) * (1 / u.us)).to(u.kHz).value)
        plt.close()

    deadtime = np.array(deadtime)
    fitted_trigger_rates = np.array(fitted_trigger_rates)
    fitted_trigger_rates_err = np.array(fitted_trigger_rates_err)

    deadtime_pc_fit = np.array(
        [
            # the parameter_lambda is a rate value in kHz,
            # so one needs to compare the deadtime in mus with the rate in kHz
            # and finally make it a percentage value
            deadtime[ii] * rate * 1e2 * 1e-3
            for ii, rate in enumerate(fitted_trigger_rates)
        ]
    )

    if len(runlist) > 1:
        plot_deadtime_vs_collected_trigger_rate(
            np.array(ids),
            np.array(collected_trigger_rates),
            np.array(deadtime_pc),
            np.array(error_deadtime_pc),
            np.array(deadtime_pc_fit),
            False,
            labels,
            output_dir,
            temp_output,
            test_type=test_type,
        )
        plot_fitted_rate_vs_collected_trigger_rate(
            np.array(ids),
            np.array(collected_trigger_rates),
            np.array(fitted_trigger_rates),
            np.array(fitted_trigger_rates_err),
            labels,
            output_dir,
            temp_output,
            test_type=test_type,
        )

    plt.close("all")


if __name__ == "__main__":
    main()
