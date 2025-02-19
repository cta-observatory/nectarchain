# don't forget to set environment variable NECTARCAMDATA

import argparse
import csv
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from tools_components import PedestalTool
from utils import adc_to_pe, pe2photons


def get_args():
    """
    Parses command-line arguments for the linearity test script.

    Returns:
        argparse.ArgumentParser: The parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Pedestal substraction test B-TEL-1370.\n"
        + "According to the nectarchain component interface, you have to set a NECTARCAMDATA environment variable in the folder where you have the data from your runs or where you want them to be downloaded.\n"
        + "You have to give a list of runs (run numbers with spaces inbetween) and an output directory to save the final plot.\n"
        + "If the data is not in NECTARCAMDATA, the files will be downloaded through DIRAC.\n For the purposes of testing this script, default data is from the runs used for this test in the TRR document.\n"
        + "You can optionally specify the number of events to be processed (default 1200).\n"
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
        help="Number of events to process from each run. Default is 1200. 4000 or more gives best results but takes some time",
        required=False,
        default=10,
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
    """
    The main function that runs the pedestal subtraction test. It parses command-line arguments, processes the specified runs, and generates two plots:

    1. A 2D heatmap of the pedestal RMS for all events and pixels.
    2. A line plot of the mean pedestal RMS for each pixel, with the CTA requirement range highlighted.

    The function also saves the generated plots to the specified output directory, and optionally saves the first plot to a temporary output file.
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
    output_mean_ped_hg = []
    output_mean_ped_lg = []
    output_rms_ped_hg = []
    output_rms_ped_lg = []
    i = 0
    fig, axx = plt.subplots()

    if not output_dir.endswith("/"):
        output_dir += "/"
    filename = (
        output_dir + "pedestal_data.txt"
    )  # Concatenate the variable with the filename

    with open(filename, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["Run", "Mean_ped", "Mean_RMS", "Mean_ped_Per_Pix", "Mean_RMS_Per_Pix"]
        )

        for run in runlist:
            print("PROCESSING RUN {}".format(run))
            tool = PedestalTool(
                progress_bar=True,
                run_number=run,
                max_events=nevents,
                events_per_slice=999,
                log_level=20,
                peak_height=10,
                window_width=16,
                overwrite=True,
            )
            tool.initialize()
            print("OUTPUT_PATH", tool.output_path)
            tool.setup()
            tool.start()
            output.append(tool.finish())

            output_mean_ped_hg.append(np.array(output[0][0]))
            output_mean_ped_lg.append(np.array(output[0][1]))
            output_rms_ped_hg.append(np.array(output[0][2]))
            output_rms_ped_lg.append(np.array(output[0][3]))

            mean__mean_ped = pe2photons(np.array(output_mean_ped_hg[i]) / adc_to_pe)
            mean_ped_per_pix = np.mean(mean__mean_ped, axis=0)
            mean_ped_per_pix = mean_ped_per_pix
            mean_ped = np.mean(mean_ped_per_pix)
            print(mean_ped)

            rms_ped = pe2photons(np.array(output_rms_ped_hg[i]) / adc_to_pe)
            mean_rms_per_pix = np.mean(rms_ped, axis=0)
            mean_rms_per_pix = mean_rms_per_pix
            mean_rms = np.mean(mean_rms_per_pix)
            print(mean_rms)

            # Now append to the file and figure
            writer.writerow(
                [run, mean_ped, mean_rms, mean_ped_per_pix, mean_rms_per_pix]
            )
            axx.plot(i, mean_rms, marker="x", linestyle="")
            i = i + 1

    axx.set_title("Mean pedestal rms per run")
    axx.set_xlabel("run")
    axx.set_ylabel("p.e.")
    plt.savefig(os.path.join(output_dir, "mean_pedestal_all.png"))

    rms_ped = pe2photons(np.array(output[0]) / adc_to_pe)  # in photons
    plt.figure()
    plt.title("Pedestal rms for all events and pixels")
    plt.pcolormesh(rms_ped.T, clim=(0.8, 1.5))
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, "pedestal_rms_2d_graph.png"))

    mean_rms_per_pix = np.mean(rms_ped, axis=0)
    mean_value = np.mean(mean_rms_per_pix)

    fig, ax = plt.subplots()
    ax.set_title("Mean pedestal rms for each pixel")
    ax.set_xlabel("Pixel")
    ax.set_ylabel("p.e.")
    ax.plot(mean_rms_per_pix, marker="x", linestyle="")
    ax.axhline(1.2, color="black", alpha=0.8)
    ax.axhline(mean_value, color="red", linestyle="--", alpha=0.5)

    # Fill the region between 0.8 * mean_value and 1.2 * mean_value
    ax.fill_between(
        range(len(mean_rms_per_pix)),
        0.8 * mean_value,
        1.2 * mean_value,
        color="red",
        alpha=0.3,
    )

    right_x_position = (
        len(mean_rms_per_pix) * 0.95
    )  # Slightly left of the right edge of the plot
    ax.arrow(
        right_x_position,
        1.2 * mean_value,
        0,
        -0.4 * mean_value,
        color="grey",
        width=5,
        head_width=50,
        head_length=0.05,
        length_includes_head=True,
        zorder=3,
    )
    ax.arrow(
        right_x_position,
        0.8 * mean_value,
        0,
        0.4 * mean_value,
        color="grey",
        width=5,
        head_width=50,
        head_length=0.05,
        length_includes_head=True,
        zorder=3,
    )
    ax.text(
        right_x_position + 100,
        mean_value,
        "±20%",
        color="grey",
        fontsize=12,
        verticalalignment="center",
    )
    ax.text(10, 1.2, "CTA requirement", color="black", fontsize=12)
    ax.arrow(
        300,
        1.2,
        0,
        -0.1,
        color="black",
        width=5,
        head_width=50,
        head_length=0.05,
        length_includes_head=True,
        zorder=5,
    )
    ax.arrow(
        1500,
        1.2,
        0,
        -0.1,
        color="black",
        width=5,
        head_width=50,
        head_length=0.05,
        length_includes_head=True,
        zorder=5,
    )

    plt.savefig(os.path.join(output_dir, "mean_pedestal_rms.png"))
    if temp_output:
        with open(os.path.join(args.temp_output, "plot1.pkl"), "wb") as f:
            pickle.dump(fig, f)

    plt.close("all")


if __name__ == "__main__":
    main()
