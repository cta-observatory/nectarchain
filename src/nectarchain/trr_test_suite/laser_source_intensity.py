# don't forget to set environment variable NECTARCAMDATA

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ctapipe.containers import EventType
from ctapipe.core import run_tool
from ctapipe.core.traits import ComponentNameList

from nectarchain.makers import EventsLoopNectarCAMCalibrationTool
from nectarchain.makers.calibration import PedestalNectarCAMCalibrationTool
from nectarchain.makers.component import NectarCAMComponent
from nectarchain.utils.constants import ALLOWED_CAMERAS, GAIN_DEFAULT

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
    """Parses command-line arguments for the linearity test script.

    Returns:
        argparse.ArgumentParser: The parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="""Source calibration for Laser source.

According to the nectarchain component interface, you have to set a $NECTARCAMDATA
environment variable in the folder where you have the data from your runs or where you
want them to be downloaded.

You have to give a list of runs (run numbers with spaces inbetween), a corresponding
intensity list and an output directory to save the final plot.

If the data is not in NECTARCAMDATA, the files will be downloaded through DIRAC.

For the purposes of testing this script, default data is from the runs used for this
test in the TRR document.

You can optionally specify the number of events to be processed and the number of
pixels used.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-r",
        "--runlist",
        type=int,
        nargs="+",
        help="List of runs (numbers separated by space)",
        required=False,
        default=[
            6204,
            6205,
            6206,
            6207,
            6210,
            6212,
            6213,
            6214,
            6215,
            6216,
            6217,
            6218,
            6219,
            6220,
            6221,
        ],
    )

    parser.add_argument(
        "-e",
        "--evts",
        type=int,
        help="Number of events to process from each run.",
        required=False,
        default=500,
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
        "-i",
        "--intensity",
        type=float,
        nargs="+",
        help="List of corresponding voltage for each run",
        required=False,
        default=[
            24.3,
            26.0,
            28.0,
            29.5,
            31.3,
            33.2,
            34.8,
            36.6,
            38.4,
            41.5,
            43.6,
            45,
            47.5,
            52,
            56.4,
        ],
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


class LaserIntensityCalibrationTool(EventsLoopNectarCAMCalibrationTool):
    name = "LaserIntensityTestTool"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["ChargesComponent"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    def finish(self, *args, **kwargs):
        output = super().finish(return_output_component=True, *args, **kwargs)

        charge_container = output[0].containers[EventType.SUBARRAY]

        mean_charge = [0, 0]  # per channel
        std_charge = [0, 0]
        std_err = [0, 0]

        charge_hg = charge_container["charges_hg"]
        charge_lg = charge_container["charges_lg"]
        npixels = charge_container["npixels"]

        charge_pe_hg = np.array(charge_hg) / GAIN_DEFAULT
        charge_pe_lg = np.array(charge_lg) / GAIN_DEFAULT

        ratio_hglg_pix = charge_pe_hg / charge_pe_lg
        ratio_hglg_pix[np.where(charge_pe_lg == 0.0)] = np.nan
        ratio_hglg = np.nanmean(np.nanmean(ratio_hglg_pix, axis=0))

        for channel, charge in enumerate([charge_pe_hg, charge_pe_lg]):
            pix_mean_charge = np.mean(charge, axis=0)  # in pe

            pix_std_charge = np.std(charge, axis=0)

            # average of all pixels
            mean_charge[channel] = np.mean(pix_mean_charge)

            std_charge[channel] = np.mean(pix_std_charge)
            # for the charge resolution
            std_err[channel] = np.std(pix_std_charge)

        return mean_charge, std_charge, std_err, npixels, ratio_hglg


def main():
    """
    The `main()` function is the entry point of the laser calibration code. It parses
    the command-line arguments, processes the specified runs, and generates plots to
    visualize the linearity and charge resolution of the detector. The function performs
    the following key steps:

    1. Parses the command-line arguments using the `get_args()` function, which sets
    up the argument parser and handles the input parameters.
    2. Iterates through the specified run list, processing each run using the
    `LinearityTestTool` class. This tool returns the "ChargeComp" in tool_components
    and computes average charge.
    3. Plots : avg p.e. over events and over the camera as a function of laser
    intensity.

    """
    parser = get_args()
    args = parser.parse_args()

    runlist = args.runlist

    camera = args.camera
    intensity = args.intensity

    nevents = args.evts

    output_dir = os.path.join(
        os.path.abspath(args.output),
        f"trr_camera_{camera}/{Path(__file__).stem}",
    )
    os.makedirs(output_dir, exist_ok=True)

    temp_output = os.path.abspath(args.temp_output) if args.temp_output else None

    log.debug(f"Output directory: {output_dir}")
    log.debug(f"Temporary output file: {temp_output}")

    sys.argv = sys.argv[:1]

    charge = np.zeros((len(runlist), 2))
    std = np.zeros((len(runlist), 2))
    std_err = np.zeros((len(runlist), 2))
    ratio_hglg = np.zeros(len(runlist))

    log.info("runlist ==", runlist)

    for index, run in enumerate(runlist):
        log.info("PROCESSING RUN {}".format(run))
        output_file_name = Path(f"{output_dir}/NSBRateTestTool_run{str(run)}.h5")
        pedestal_tool = PedestalNectarCAMCalibrationTool(
            progress_bar=True,
            run_number=run,
            max_events=nevents,
            events_per_slice=None,
            log_level=20,
            overwrite=True,
            filter_method=None,
            method="FullWaveformSum",  # charges over entire window
        )
        run_tool(pedestal_tool)

        tool = LaserIntensityCalibrationTool(
            progress_bar=True,
            run_number=run,
            events_per_slice=999,
            max_events=nevents,
            log_level=20,
            method="LocalPeakWindowSum",
            extractor_kwargs={
                "window_width": 16,
                "window_shift": 4,
            },
            pedestal_file=pedestal_tool.output_path,
            overwrite=True,
            output_path=output_file_name,
        )
        tool.initialize()
        tool.setup()
        tool.start()
        output = tool.finish()

        charge[index], std[index], std_err[index], npixels, ratio_hglg[index] = output

    fig, ax = plt.subplots()
    ax.errorbar(
        intensity,
        charge[:, 0],
        color="green",
        yerr=np.transpose(std_err)[0],
        marker="o",
        linestyle="",
        label="HG",
    )
    ax.errorbar(
        intensity,
        charge[:, 1] * ratio_hglg[5],
        color="blue",
        yerr=np.transpose(std_err)[1],
        marker="o",
        linestyle="",
        label="LG",
    )
    ax.set_xlabel("Intensity (%)")
    ax.set_ylabel("Average charge (p.e.)")
    ax.legend()
    ax.grid()
    # plt.ylim(pow(10,-1),5.*pow(10,4))
    fig_name = f"Laser_calibration_{runlist[0]}_{runlist[len(runlist)-1]}"
    plot_path = os.path.join(output_dir, f"{fig_name}.png")
    fig.savefig(plot_path)

    if temp_output:
        with open(os.path.join(temp_output, f"plot_{fig_name}.pkl"), "wb") as f:
            pickle.dump(fig, f)

    plt.close("all")


if __name__ == "__main__":
    main()
