# don't forget to set environment variable NECTARCAMDATA

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ctapipe.core import run_tool

from nectarchain.makers.calibration import PedestalNectarCAMCalibrationTool
from nectarchain.trr_verification_package.tools_components import (
    TimingResolutionTestTool,
)
from nectarchain.trr_verification_package.utils import pe2photons, photons2pe
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


def get_args():
    """Parses command-line arguments for the pixel timing uncertainty test script.

    Returns:
        argparse.ArgumentParser: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Systematic pixel timing uncertainty test B-TEL-1380.\n"
        + "According to the nectarchain component interface, you have to set a\
             NECTARCAMDATA environment variable in the folder where you have the data\
                 from your runs or where you want them to be downloaded.\n"
        + "You have to give a list of runs (run numbers with spaces inbetween)\
             and an output directory to save the final plot.\n"
        + "If the data is not in NECTARCAMDATA, the files will be downloaded through\
             DIRAC.\n For the purposes of testing this script, default data is from the\
                 runs used for this test in the TRR document.\n"
        + "You can optionally specify the number of events to be processed (default\
             1200) and the number of pixels used (default 70).\n"
    )
    parser.add_argument(
        "-r",
        "--runlist",
        type=int,
        nargs="+",
        help="List of runs (numbers separated by space)",
        required=False,
        default=[i for i in range(3446, 3457)],
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
        help="Number of events to process from each run. Default is 100",
        required=False,
        default=100,
    )
    # parser.add_argument('-p','--pixels', type = int, help='Number of pixels used.
    # Default is 70', required=False, default=70)
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
    """Processes the pixel timing uncertainty test data and generates a plot.

    The function processes the data from the specified list of runs, calculates the
    weighted mean RMS and RMS error, and generates a plot of the results. The plot is
    saved to the specified output directory.

    If a temporary output directory is provided, the plot is also saved to a pickle file
    in that directory for the gui to use.
    """

    parser = get_args()
    args = parser.parse_args()

    camera = args.camera

    runlist = args.runlist
    nevents = args.evts

    output_dir = os.path.join(
        os.path.abspath(args.output),
        f"trr_camera_{camera}/{Path(__file__).stem}",
    )
    os.makedirs(output_dir, exist_ok=True)
    log.debug(f"Output directory: {output_dir}")
    temp_output = os.path.abspath(args.temp_output) if args.temp_output else None
    log.debug(f"Temporary output directory: {temp_output}")

    sys.argv = sys.argv[:1]

    # rms_mu = []
    # rms_mu_err = []
    rms_no_fit = []
    rms_no_fit_err = []
    mean_charge_pe = []

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
        tool = TimingResolutionTestTool(
            progress_bar=True,
            run_number=run,
            camera=camera,
            max_events=nevents,
            events_per_slice=9999,
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
        output = tool.finish()
        # rms_mu.append(output[0])
        # rms_mu_err.append(output[1])
        rms_no_fit.append(output[0])
        rms_no_fit_err.append(output[1])
        mean_charge_pe.append(output[2])

    log.debug(rms_no_fit)
    rms_no_fit_err = np.array(rms_no_fit_err)
    log.debug(rms_no_fit_err)
    rms_no_fit_err[rms_no_fit_err == 0] = 1e-5  # almost zero
    # rms_no_fit_err[rms_no_fit_err==np.nan]=1e-5
    log.debug(rms_no_fit_err)

    # mean_rms_mu = np.mean(rms_mu,axis=1)
    # mean_rms_no_fit = np.mean(rms_no_fit,axis=1)

    # weights_mu_pix = 1/(np.array(rms_mu_err)+1e-5)**2
    weights_no_fit_pix = 1 / (rms_no_fit_err) ** 2
    weights_no_fit_pix[weights_no_fit_pix > 1e5] = 1e5
    log.debug(weights_no_fit_pix)

    # rms_mu_weighted=[]
    # rms_mu_weighted_err=[]
    rms_no_fit_weighted = []
    rms_no_fit_weighted_err = []

    for run in range(len(runlist)):
        # rms_mu_weighted.append(np.sum(rms_mu[run]*weights_mu_pix[run])/
        # np.sum(weights_mu_pix[run]))
        # rms_mu_weighted_err.append(np.sqrt(1/np.sum(weights_mu_pix[run])))
        rms_no_fit_weighted.append(
            np.nansum(rms_no_fit[run] * weights_no_fit_pix[run])
            / np.nansum(weights_no_fit_pix[run])
        )
        rms_no_fit_weighted_err.append(np.sqrt(1 / np.nansum(weights_no_fit_pix[run])))

    log.debug(rms_no_fit_weighted)
    log.debug(rms_no_fit_weighted_err)

    # FIGURE
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    plt.errorbar(
        x=mean_charge_pe[:],
        y=np.sqrt(np.array(rms_no_fit_weighted[:]) ** 2),
        yerr=rms_no_fit_weighted_err,
        ls="",
        marker="o",
        label=r"$\mathtt{ctapipe.image.extractor}$",
    )
    # plt.errorbar(x=photons_spline[:],
    #              y=np.sqrt(np.array(rms_mu_weighted[:])**2),
    #              yerr=rms_mu_weighted_err,
    #              ls='', marker='o',
    #              label='Gaussian fit')

    plt.axhline(1, ls="--", color="C4", alpha=0.6)
    plt.axhline(
        1 / np.sqrt(12),
        ls="--",
        color="gray",
        alpha=0.7,
        label="Quantification rms noise",
    )

    plt.axvspan(photons2pe(20), photons2pe(1000), alpha=0.1, color="C4")

    ax.text(
        51.5,
        1.04,
        "CTA requirement",
        color="C4",
        fontsize=20,
        horizontalalignment="left",
        verticalalignment="center",
    )
    ax.annotate(
        "",
        xy=(40, 0.9),
        xytext=(40, 0.995),
        color="C4",
        alpha=0.5,
        arrowprops=dict(color="C4", alpha=0.7, lw=3, arrowstyle="->"),
    )

    ax.annotate(
        "",
        xy=(200, 0.9),
        xytext=(200, 0.995),
        color="C4",
        alpha=0.5,
        arrowprops=dict(color="C4", alpha=0.7, lw=3, arrowstyle="->"),
    )

    plt.legend(frameon=True, prop={"size": 18}, loc="upper right", handlelength=1.2)
    plt.xlabel("Illumination charge [p.e.]")
    plt.ylabel("Mean rms per pixel [ns]")
    plt.xscale("log")
    plt.ylim((0, 2.7))
    secax = ax.secondary_xaxis("top", functions=(pe2photons, photons2pe))
    secax.set_xlabel("Illumination charge [photons]", labelpad=7)

    fig_name = "pix_tim_uncertainty"
    plot_path = os.path.join(output_dir, f"{fig_name}.png")
    plt.savefig(plot_path)

    if temp_output:
        with open(os.path.join(args.temp_output, f"plot_{fig_name}.pkl"), "wb") as f:
            pickle.dump(fig, f)

    plt.close("all")


if __name__ == "__main__":
    main()
