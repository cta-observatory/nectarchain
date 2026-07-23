import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ctapipe.core import run_tool
from ctapipe.utils import get_dataset_path

from nectarchain.makers.calibration import PedestalNectarCAMCalibrationTool
from nectarchain.trr_verification_package.tools_components import ToMPairsTool
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

TRANSIT_TIME_CORRECTIONS = get_dataset_path(
    filename=(
        "hv_pmt_tom_correction_laser_measurement_per_pixel_fit_sqrt_hv_newmethod" ".csv"
    ),
    url="http://cccta-dataserver.in2p3.fr/data/ctapipe-test-data/v1.1.0",
)


def get_args():
    """Parses command-line arguments for the pix_couple_tim_uncertainty_test.py script.

    Returns:
        argparse.ArgumentParser: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Time resolution (timing uncertainty between couples of pixels)\
             test B-TEL-1030.\n"
        + "According to the nectarchain component interface, you have to set a\
            NECTARCAMDATA\
             environment variable in the folder where you have the data from your runs\
                 or where you want them to be downloaded.\n"
        + "You have to give a list of runs (run numbers with spaces inbetween) and\
             an output directory to save the final plot.\n"
        + "If the data is not in NECTARCAMDATA, the files will be downloaded through\
             DIRAC.\n For the purposes of testing this script, default data is from the\
                 runs used for this test in the TRR document.\n"
        + "You can optionally specify the number of events to be processed (default\
             1000). Takes a lot of time.\n"
    )
    parser.add_argument(
        "-r",
        "--runlist",
        type=int,
        nargs="+",
        help="List of runs (numbers separated by space). You can put just one run,\
            default 3292",
        required=False,
        default=[3292],
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
        "-e",
        "--evts",
        type=int,
        help="Number of events to process from each run. Default is 100. 1000 or\
            more gives best results but takes some time",
        required=False,
        default=100,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory",
        default=f"{os.environ.get('NECTARCHAIN_FIGURES', f'/tmp/{os.getpid()}')}",
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
    """Generates a plot of the RMS of the time-of-maximum (TOM) difference for pairs of
    pixels, with a visualization of the CTA requirement.

    The script processes a list of runs, calculates the TOM difference with and without
    transit time corrections, and plots the distribution of the RMS of the corrected TOM
    differences. The CTA requirement of 2 ns RMS is visualized on the plot.

    The script takes several command-line arguments, including the list of runs to
    process, the number of events to process per run, the path to a CSV file with PMT
    transit time corrections, and the output directory for the plot.

    If a temporary output directory is specified, the plot is also saved to a pickle
    file in that directory for the gui to use.
    """

    parser = get_args()
    args = parser.parse_args()

    camera = args.camera

    runlist = args.runlist
    nevents = args.evts
    tt_path = TRANSIT_TIME_CORRECTIONS

    output_dir = os.path.join(
        os.path.abspath(args.output),
        f"trr_camera_{camera}/{Path(__file__).stem}",
    )
    os.makedirs(output_dir, exist_ok=True)
    log.debug(f"Output directory: {output_dir}")
    temp_output = os.path.abspath(args.temp_output) if args.temp_output else None
    log.debug(f"Temporary output directory: {temp_output}")

    # Drop arguments from the script after they are parsed, for the GUI to work properly
    sys.argv = sys.argv[:1]
    tom = []

    pixel_ids = []  # pixel ids for run

    tom_corrected = []

    dt_no_correction = []
    dt_corrected = []
    pixel_pairs = []

    for run in runlist:
        log.info("PROCESSING RUN {}".format(run))
        # Old runs do not have interleaved pedestals
        pedestal_tool = PedestalNectarCAMCalibrationTool(
            progress_bar=True,
            run_number=run,
            camera=camera,
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
        tool = ToMPairsTool(
            progress_bar=True,
            run_number=run,
            camera=camera,
            events_per_slice=501,
            max_events=nevents,
            log_level=20,
            method="LocalPeakWindowSum",
            extractor_kwargs={
                "window_width": 6,
                "window_shift": 3,
            },  # This width and shift works best for ToM calculation
            overwrite=True,
            pedestal_file=pedestal_tool.output_path,
            use_default_pedestal=True,  # only done if pedestal_file cannot be loaded
            mean_charge_threshold=args.mean_charge_threshold,
        )
        tool.initialize()
        tool.setup()
        tool.start()
        output = tool.finish(tt_path)
        tom.append(output[0])
        tom_corrected.append(output[1])
        pixel_ids.append(output[2])
        dt_no_correction.append(output[3])
        dt_corrected.append(output[4])
        pixel_pairs.append(output[5])

    dt_no_correction = np.array(dt_no_correction[0])
    dt_corrected = np.array(output[4])
    dt_corrected[abs(dt_corrected) > 7] = np.nan
    dt_corrected[abs(dt_no_correction) > 7] = np.nan
    std_corrected = np.nanstd(dt_corrected, axis=1)

    fig, ax = plt.subplots(figsize=(10, 10 / 1.61))
    plt.hist(std_corrected, range=(0, 5), density=True, histtype="step", lw=3, bins=200)

    plt.axvline(2, color="C4", alpha=0.8)
    ax.text(2.1, 0.5, "CTA requirement", color="C4", fontsize=20, rotation=-90)

    ax.annotate(
        "",
        xy=(1.6, 0.25),
        xytext=(2, 0.25),
        color="C4",
        alpha=0.5,
        transform=ax.transAxes,
        arrowprops=dict(color="C4", alpha=0.7, lw=3, arrowstyle="->"),
    )

    ax.annotate(
        "",
        xy=(1.6, 0.75),
        xytext=(2, 0.75),
        color="C4",
        alpha=0.5,
        transform=ax.transAxes,
        arrowprops=dict(color="C4", alpha=0.7, lw=3, arrowstyle="->"),
    )

    plt.xlabel(r"RMS of $\Delta t_{\mathrm{TOM}}$ for pairs of pixels [ns]")
    plt.ylabel("Normalized entries")

    fig_name = "pix_couple_tim_uncertainty"
    plot_path = os.path.join(output_dir, f"{fig_name}.png")
    plt.savefig(plot_path)

    if temp_output:
        with open(os.path.join(args.temp_output, f"plot_{fig_name}.pkl"), "wb") as f:
            pickle.dump(fig, f)

    plt.close("all")


if __name__ == "__main__":
    main()
