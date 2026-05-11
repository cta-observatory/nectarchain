import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from ctapipe.containers import EventType

from nectarchain.trr_test_suite.deadtime import (
    fit_rate_per_run,
    plot_deadtime_vs_collected_trigger_rate,
    plot_fitted_rate_vs_collected_trigger_rate,
    run_deadtime_test_tool_process,
)
from nectarchain.trr_test_suite.utils import plot_deadtime_and_expo_fit
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


def get_labels():
    with open("resources/source_type_labels.json", "r") as f:
        source_labels = json.load(f)

    return source_labels


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
    )
    parser.add_argument(
        "-s",
        "--source",
        type=int,
        nargs="+",
        choices=[member.value for member in EventType],
        # [0, 1, 2, 3, 4, 15, 16, 17, 24, 32, 255]
        help="Source number (or list of numbers)",
        required=True,
    )
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
    parser.add_argument(
        "-l",
        "--log",
        help="log level",
        default="info",
        type=str,
    )

    return parser


def main():
    """Runs the deadtime test script, which performs deadtime tests B-TEL-1260 and
    B-TEL-1270, and event rate test B-MST-1280.

    The script takes command-line arguments to specify a run number, \
        corresponding event source, the camera tag, and, optionally, \
        the number of events to consider for the test and an output directory. \
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

    assert len(runlist) == len(ids), "'runlist' and 'ids' must have the same length"

    source_labels = get_labels()

    nevents = args.evts

    kwargs = copy.deepcopy(vars(args))
    kwargs.pop("camera")
    camera = args.camera

    output_dir = os.path.join(
        os.path.abspath(args.output),
        f"av_camera_{camera}/{Path(__file__).stem}",
    )
    os.makedirs(output_dir, exist_ok=True)
    temp_output = os.path.abspath(args.temp_output) if args.temp_output else None

    log.info(f"Running the script with arguments: {kwargs}")

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
        runlist=runlist, nevents=nevents, ids=ids, test_type="av"
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
            source_labels,
            output_dir,
            temp_output,
            test_type="av",
        )
        plot_fitted_rate_vs_collected_trigger_rate(
            np.array(ids),
            np.array(collected_trigger_rates),
            np.array(fitted_trigger_rates),
            np.array(fitted_trigger_rates_err),
            source_labels,
            output_dir,
            temp_output,
            test_type="av",
        )

    plt.close("all")


if __name__ == "__main__":
    main()
