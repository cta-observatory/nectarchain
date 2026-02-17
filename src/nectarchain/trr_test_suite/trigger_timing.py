# don't forget to set environment variable NECTARCAMDATA

import argparse
import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from ctapipe.core import run_tool

from nectarchain.makers.calibration import PedestalNectarCAMCalibrationTool
from nectarchain.trr_test_suite.tools_components import TriggerTimingTestTool
from nectarchain.trr_test_suite.utils import pe2photons

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


def get_args():
    """Parses command-line arguments for the deadtime test script.

    Returns:
        argparse.ArgumentParser: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Trigger Timing Test B-TEL-1410. \n"
        + "According to the nectarchain component interface, you have to set\
            a NECTARCAMDATA environment variable in the folder where you have the data\
                from your runs or where you want them to be downloaded.\n"
        + "You have to give a list of runs (run numbers with spaces inbetween) and an\
            output directory to save the final plot.\n"
        + "If the data is not in NECTARCAMDATA, the files will be downloaded through\
            DIRAC.\n For the purposes of testing this script, default data is from the\
                runs used for this test in the TRR document.\n"
        + "You can optionally specify the number of events to be processed\
            (default 1000).\n"
    )
    parser.add_argument(
        "-r",
        "--runlist",
        type=int,
        nargs="+",
        help="List of runs (numbers separated by space)",
        required=False,
        default=[i for i in range(3259, 3263)],
    )
    parser.add_argument(
        "-e",
        "--evts",
        type=int,
        help="Number of events to process from each run. Default is 1000",
        required=False,
        default=1000,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory. If none, plot will be saved in current directory",
        required=False,
        default="./",
    )
    parser.add_argument(
        "-t",
        "--mean_charge_threshold",
        type=float,
        help="Threshold below which to select good events,"
        "in units of mean camera charge",
        required=False,
        default=10,
    )
    parser.add_argument(
        "--temp_output", help="Temporary output directory for GUI", default=None
    )

    return parser


def main():
    """Runs the deadtime test script, which performs deadtime tests B-TEL-1260 and
    B-TEL-1270.

    The script takes command-line arguments to specify the list of runs, corresponding\
        sources, number of events to process, and output directory. It then processes\
            the data for each run, performs an exponential fit to the deadtime\
                distribution, and generates two plots:

    1. A plot of deadtime percentage vs. collected trigger rate, with the CTA\
        requirement indicated.
    2. A plot of the rate from the fit vs. the collected trigger rate, with the\
        relative difference shown in the bottom panel.

    The script also saves the generated plots to the specified output directory,\
        and optionally saves them to a temporary output directory for use in a GUI.
    """

    parser = get_args()
    args = parser.parse_args()

    runlist = args.runlist

    nevents = args.evts

    output_dir = os.path.abspath(args.output)
    temp_output = os.path.abspath(args.temp_output) if args.temp_output else None

    log.debug(f"Output directory: {output_dir}")
    log.debug(f"Temporary output file: {temp_output}")

    sys.argv = sys.argv[:1]

    # ucts_timestamps = np.zeros((len(runlist),nevents))
    # delta_t = np.zeros((len(runlist),nevents-1))
    # event_counter = np.zeros((len(runlist),nevents))
    # busy_counter = np.zeros((len(runlist),nevents))
    # collected_triger_rate = np.zeros(len(runlist))
    # time_tot = np.zeros(len(runlist))
    # deadtime_us=np.zeros((len(runlist),nevents-1))
    # deadtime_pc = np.zeros(len(runlist))

    rms = []
    err = []
    charge = []

    nevents = 1000

    for i, run in enumerate(runlist):
        log.info("PROCESSING RUN {}".format(run))
        pedestal_tool = PedestalNectarCAMCalibrationTool(
            progress_bar=True,
            run_number=run,
            max_events=12000,
            events_per_slice=5000,
            log_level=20,
            overwrite=True,
            filter_method=None,
            method="FullWaveformSum",  # charges over entire window
        )
        try:
            run_tool(pedestal_tool)
        except Exception as e:
            log.warning(e)
        tool = TriggerTimingTestTool(
            progress_bar=True,
            run_number=run,
            max_events=nevents,
            events_per_slice=10000,
            log_level=20,
            method="LocalPeakWindowSum",
            extractor_kwargs={"window_width": 16, "window_shift": 6},
            overwrite=True,
            pedestal_file=pedestal_tool.output_path,
            use_default_pedestal=True,
            mean_charge_threshold=args.mean_charge_threshold,
        )
        tool.initialize()
        tool.setup()
        tool.start()
        output = tool.finish()
        # ucts_timestamps.append(output[0])
        # delta_t.append(output[1])
        rms.append(output[2])
        err.append(output[3])
        charge.append(output[4])

    rms = np.array(rms)
    err = np.array(err)
    charge = np.array(charge)
    print(rms, err, charge)

    fig, ax = plt.subplots()

    # Plot the error bars
    ax.errorbar(charge, rms, yerr=err, fmt="o", color="blue")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Illumination [p.e.]")
    ax.set_ylabel("Trigger time resolution [ns]")
    ax.set_xlim([10, 500])
    ax.set_ylim([1e-1, 10])

    # CTA requirement line
    cta_requirement_y = 5  # Y-value for the CTA requirement
    ax.axhline(y=cta_requirement_y, color="purple", linestyle="--")

    # Add the small vertical arrows starting from the CTA requirement line and pointing
    # downwards
    arrow_positions = [20, 80, 200]  # X-positions for the arrows
    for x_pos in arrow_positions:
        ax.annotate(
            "",
            xy=(x_pos, cta_requirement_y - 2),
            xytext=(x_pos, cta_requirement_y),
            arrowprops=dict(arrowstyle="->", color="purple", lw=1.5),
        )  # Arrow pointing downwards

    # Add the CTA requirement label exactly above the dashed line, centered between
    # arrows
    ax.text(
        140,
        cta_requirement_y + 0.5,
        "CTA requirement",
        color="purple",
        ha="center",
        fontsize=10,
    )

    # Create a second x-axis at the top with illumination in photons (independent scale)
    ax2 = ax.twiny()  # Create a new twin x-axis
    ax2.set_xscale("log")

    # Set the label for the top x-axis
    ax2.set_xlabel("Illumination [photons]")

    ax2.set_xlim(
        pe2photons(ax.get_xlim()[0]), pe2photons(ax.get_xlim()[1])
    )  # Match limits

    plt.savefig(os.path.join(output_dir, "trigger_timing.png"))

    if temp_output:
        with open(os.path.join(args.temp_output, "plot1.pkl"), "wb") as f:
            pickle.dump(fig, f)


if __name__ == "__main__":
    main()
