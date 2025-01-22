import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

from nectarchain.trr_test_suite.tools_components import ToMPairsTool


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
        "-e",
        "--evts",
        type=int,
        help="Number of events to process from each run. Default is 100. 1000 or\
            more gives best results but takes some time",
        required=False,
        default=100,
    )
    parser.add_argument(
        "-t",
        "--pmt_transit_time",
        type=str,
        help=".csv file with pmt transit time corrections",
        required=False,
        default="../transit_time/"
        "hv_pmt_tom_correction_laser_measurement_per_pixel_fit"
        "sqrt_hv_newmethod.csv",
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

    tt_path = "/Users/dm277349/nectarchain_data/transit_time/\
        hv_pmt_tom_correction_laser_measurement_per_pixel_fit_sqrt_hv_newmethod.csv"

    runlist = args.runlist
    nevents = args.evts
    tt_path = args.pmt_transit_time
    output_dir = os.path.abspath(args.output)
    temp_output = os.path.abspath(args.temp_output) if args.temp_output else None

    print(f"Output directory: {output_dir}")  # Debug print
    print(f"Temporary output file: {temp_output}")  # Debug print

    sys.argv = sys.argv[:1]
    tom = []

    pixel_ids = []  # pixel ids for run

    tom_corrected = []

    dt_no_correction = []
    dt_corrected = []
    pixel_pairs = []

    for run in runlist:
        print("PROCESSING RUN {}".format(run))
        tool = ToMPairsTool(
            progress_bar=True,
            run_number=run,
            events_per_slice=501,
            max_events=nevents,
            log_level=20,
            peak_height=10,
            window_width=16,
            overwrite=True,
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

    plt.gcf()

    plt.savefig(os.path.join(output_dir, "pix_couple_tim_uncertainty.png"))

    if temp_output:
        with open(os.path.join(args.temp_output, "plot1.pkl"), "wb") as f:
            pickle.dump(fig, f)

    plt.close("all")


if __name__ == "__main__":
    main()
