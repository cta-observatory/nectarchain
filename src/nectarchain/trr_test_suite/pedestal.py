# don't forget to set environment variable NECTARCAMDATA

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from ctapipe_io_nectarcam import N_PIXELS, N_SAMPLES

from nectarchain.makers.calibration import PedestalNectarCAMCalibrationTool
from nectarchain.trr_test_suite.utils import photons2ADC


def get_args():
    """Parses command-line arguments for the linearity test script.

    Returns:
        argparse.ArgumentParser: The parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Pedestal substraction test B-TEL-1370.\n"
        + "According to the nectarchain component interface, you have to set\
             a NECTARCAMDATA environment variable in the folder where you have\
                 the data from your runs or where you want them to be downloaded.\n"
        + "You have to give a list of runs (run numbers with spaces inbetwee\
            n) and an output directory to save the final plot.\n"
        + "If the data is not in NECTARCAMDATA, the files will be\
             downloaded through DIRAC.\n For the purposes of testing this script,\
                 default data is from the runs used for this test in the\
                     TRR document.\n"
        + "You can optionally specify the number of events to be processed\
             (default 1200).\n"
    )
    parser.add_argument(
        "-r",
        "--runlist",
        type=int,
        nargs="+",
        help="List of runs (numbers separated by space)",
        required=False,
        default=[3647],
    )
    parser.add_argument(
        "-e",
        "--evts",
        type=int,
        help="Number of events to process from each run. Default is 200. 4000 or more\
            gives best results but takes some time",
        required=False,
        default=200,
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
        "--temp_output", help="Temporary output directory for GUI", default=None
    )

    return parser


def main():
    """The main function that runs the pedestal subtraction test. It parses command-line
    arguments, processes the specified runs, and generates two plots:

    1. The mean baseline value for all pixels.
    2. A plot that compares the uncertainty to the limits set by the CTAO requirements.

    The function also saves the generated plots to the specified output directory\
    and optionally saves the first plot to a temporary output file.
    """

    parser = get_args()
    args = parser.parse_args()

    runlist = args.runlist
    nevents = args.evts

    # output_dir = args.output
    # temp_output_file = args.temp_output
    output_dir = os.path.abspath(args.output)
    temp_output = os.path.abspath(args.temp_output) if args.temp_output else None

    print(f"Output directory: {output_dir}")  # Debug print
    print(f"Temporary output dir: {temp_output}")  # Debug print

    sys.argv = sys.argv[:1]
    output = []

    for run in runlist:
        print("PROCESSING RUN {}".format(run))
        tool = PedestalNectarCAMCalibrationTool(
            progress_bar=True,
            run_number=run,
            max_events=nevents,
            events_per_slice=999,
            log_level=20,
            output_path=output_dir + f"pedestal_{run}.h5",
            overwrite=True,
            filter_method=None,
            method="FullWaveformSum",  # charges over entire window
        )
        tool.initialize()
        tool.setup()
        tool.start()
        output.append(tool.finish(return_output_component=True))

    # Show baseline value
    fig, ax = plt.subplots()
    for result in output:
        # mean value of pedestal per pixel
        pixels_id = result[0]["pixels_id"]
        baseline = result[0]["pedestal_charge_mean_hg"] / N_SAMPLES
        ax.plot(pixels_id, baseline, marker="o", linewidth=0, alpha=0.3)

    ax.set_title("Pedestal")
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Mean baseline (ADC)")

    plt.savefig(os.path.join(output_dir, "pedestal_baseline.png"))
    if temp_output:
        with open(os.path.join(args.temp_output, "plot1.pkl"), "wb") as f:
            pickle.dump(fig, f)

    # The next block of code produces a plot to verify requirement B-TEL-1370
    # During Observations the Camera must measure the Pedestal in each pixel
    # and the event-to-event rms of this quantity with an uncertainty
    # no greater than 20% of the event-to-event rms or 1.2 photons (if greater).
    # NB: we only test this for the HG channel because the uncertainty/RMS
    # requirement is automatically met if one uses more then 25 events for
    # pedestal estimation and the 1.2 photons limit makes no sense for LG
    fig, ax = plt.subplots()
    for result in output:
        # RMS
        pixels_id = result[0]["pixels_id"]
        ped_rms = result[0]["pedestal_charge_std_hg"]

        # uncertainty in pedestal from RMS and number of events used
        ped_unc = ped_rms / np.sqrt(result[0]["nevents"])
        ax.plot(pixels_id, ped_unc, marker="x", color="C0", linewidth=0, alpha=0.3)

        # 20% of RMS
        # sort pixel id to make plot more readable if there are missing pixels
        idx = np.unravel_index(np.argsort(pixels_id, axis=None), pixels_id.shape)
        ax.fill_between(
            pixels_id[idx],
            0,
            0.2 * ped_rms[idx],
            color="0.5",
            alpha=0.1,
        )

    # 1.2 photons
    ax.plot(
        np.arange(N_PIXELS),
        photons2ADC(1.2) * np.ones(N_PIXELS),
        linestyle="--",
        color="r",
    )

    # add annotations to explain
    ax.annotate(
        "Pedestal mean uncertainty (Full waveform sum)",
        (0.05, 0.85),
        xycoords="axes fraction",
        color="C0",
    )

    ax.annotate(
        "20% of pedestal width RMS (Full waveform sum)",
        (0.05, 0.8),
        xycoords="axes fraction",
        color="0.5",
    )

    ax.annotate(
        "1.2 photons",
        (0.05, 0.75),
        xycoords="axes fraction",
        color="r",
    )

    ax.set_title("Pedestal uncertainty")
    ax.set_xlabel("Pixel")
    ax.set_ylabel("(ADC)")

    plt.savefig(os.path.join(output_dir, "pedestal_uncertainty.png"))
    if temp_output:
        with open(os.path.join(args.temp_output, "plot1.pkl"), "wb") as f:
            pickle.dump(fig, f)

    plt.close("all")


if __name__ == "__main__":
    main()
