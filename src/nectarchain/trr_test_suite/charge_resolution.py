# don't forget to set environment variable NECTARCAMDATA

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ctapipe.core import run_tool

from nectarchain.makers.calibration import (
    FlatFieldSPENominalStdNectarCAMCalibrationTool,
    PedestalNectarCAMCalibrationTool,
)
from nectarchain.trr_test_suite.tools_components import ChargeResolutionTestTool
from nectarchain.trr_test_suite.utils import err_ratio, get_gain_run, plot_parameters

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
        description="""Intensity resolution B-TEL-1010 using FF+NSB runs.

According to the nectarchain component interface, you have to set a NECTARCAMDATA
environment variable in the folder where you have the data from your runs or where
you want them to be downloaded.

You have to give a list of runs in <run_file>.json, e.g.
`charge_resolution_run_list.json` and pass it to the args corresponding value of voltage
and the NSB value of the sets and an output directory to save the final plot.

If the data are not in `$NECTARCAMDATA`, the files will be downloaded through DIRAC.

For the purposes of testing this script, default data are from the runs used for this
test in the TRR document.

You can optionally specify the number of events to be processed (default 500) and the
number of pixels used (default 1000).
"""
    )
    parser.add_argument(
        "-r",
        "--run_file",
        type=str,
        help="Run file path and name",
        required=False,
        default="resources/charge_resolution_run_list.json",
    )

    parser.add_argument(
        "-e",
        "--evts",
        type=int,
        help="Number of events to process from each run. Default is 500",
        required=False,
        default=500,
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
        "--temperature",
        type=int,
        help="Temperature of the runs",
        required=False,
        default=14,
    )
    return parser


def main():
    """
    The `main()` function is the entry point of the linearity test script. It parses \
        the command-line arguments, processes the specified runs, and generates plots\
            to visualize the linearity and charge resolution of the detector. The\
                function performs the following key steps:\
    1. Parses the command-line arguments using the `get_args()` function, which sets up\
        the argument parser and handles the input parameters.\
    2. Iterates through the specified run list, processing each run using the\
     `ChargeResolutionTestTool` class. This tool initializes, sets up, starts\
     and finishes the processing for each run, returning the relevant output data.\
    3. Generates three subplots:\
    - The first subplot shows the estimated charge vs the voltage of FF (calibration)\
            for both high and low gain channels.\
    - The charge resolution vs the charge.\
            (using low gain for points above 200 pe and high gain below that)\
    - The third subplot shows the ratio of high-gain to low-gain charge.\
    5. Saves the generated plots to the specified output directory.\
    6. Generates an additional plot to visualize the charge resolution, including the\
        statistical limit.
    """
    parser = get_args()
    args = parser.parse_args()

    if not os.path.isfile(args.run_file):
        raise FileNotFoundError(f"Run file not found: {args.run_file}")

    df = pd.read_json(args.run_file)

    NSB = df["NSB"].values
    runs_list = df["runs"].tolist()
    ff_v_list = df["ff_v"].tolist()

    color = ["black", "red", "blue", "green", "yellow"]
    log.info("NSB Run FF", NSB, runs_list, ff_v_list)

    ratio_lghg_nsb = []
    mean_resolution_nsb = []
    mean_charge = []
    mean_resolution_nsb_err = []
    mean_charge_err = []
    log.info("NSB", len(NSB), NSB)

    for iNSB in range(len(NSB)):
        runlist = runs_list[iNSB]
        ff_volt = ff_v_list[iNSB]

        temperature = args.temperature
        nevents = args.evts

        output_dir = os.path.abspath(args.output)

        log.debug(f"Output directory: {output_dir}")

        sys.argv = sys.argv[:1]

        charge = np.zeros((len(runlist), 2))
        std = np.zeros((len(runlist), 2))
        std_err = np.zeros((len(runlist), 2))
        mean_resolution = np.zeros((len(runlist), 2))
        ratio_hglg = np.zeros(len(runlist))
        index = 0
        for run in runlist:
            log.info(f"PROCESSING RUN {run}")
            pedestal_tool = PedestalNectarCAMCalibrationTool(
                progress_bar=True,
                run_number=run,
                max_events=1000,
                events_per_slice=5000,
                log_level=20,
                overwrite=True,
                filter_method=None,
                method="FullWaveformSum",  # charges over entire window
            )
            run_tool(pedestal_tool)

            window_shift = 4
            window_width = 16
            max_events = 5000
            method = "LocalPeakWindowSum"

            gain_run = int(get_gain_run(temperature))
            gain_file_name = (
                "FlatFieldSPENominalStdNectarCAM_run{}_maxevents{}_"
                "{}_window_shift_{}_window_width_{}.h5".format(
                    gain_run, max_events, method, window_shift, window_width
                )
            )

            if not os.path.exists(gain_file_name):
                gain_tool = FlatFieldSPENominalStdNectarCAMCalibrationTool(
                    progress_bar=True,
                    run_number=gain_run,
                    max_events=5000,
                    method=method,
                    output_path=gain_file_name,
                    extractor_kwargs={
                        "window_width": 16,
                        "window_shift": 4,
                    },
                )
                run_tool(gain_tool)

            log.info(f"gain_file_name: {gain_file_name}")
            tool = ChargeResolutionTestTool(
                progress_bar=True,
                run_number=run,
                max_events=nevents,
                method="LocalPeakWindowSum",
                extractor_kwargs={"window_width": 16, "window_shift": 4},
                pedestal_file=pedestal_tool.output_path,
                overwrite=True,
            )
            tool.initialize()
            tool.setup()
            tool.start()
            output = tool.finish(gain_file=gain_file_name)

            # output = read_file(run, temperature)

            (
                charge[index],
                std[index],
                std_err[index],
                npixels,
                mean_resolution[index],
                ratio_hglg[index],
            ) = output

            index += 1

            # charge with voltage
        plt.clf()
        plt.errorbar(
            ff_volt,
            np.transpose(charge)[0],
            color=plot_parameters["High Gain"]["color"],
            yerr=np.transpose(std)[0] / (np.sqrt(npixels)),
            marker="o",
            label="HG",
            linestyle="",
        )
        plt.errorbar(
            ff_volt,
            np.transpose(charge)[1],
            color=plot_parameters["Low Gain"]["color"],
            yerr=np.transpose(std_err)[1],
            marker="o",
            label="LG",
            linestyle="",
        )
        # plt.yscale('log')
        # plt.xscale('log')
        plt.xlabel("FF voltage (V)")
        plt.ylabel("Average charge (p.e.)")
        plt.grid()
        plt.legend()
        plt.ylim(-1, 600)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir,
                "Charge_FF_V_final_cuts_{}_{}.png".format(
                    runlist[0], runlist[len(runlist) - 1]
                ),
            )
        )
        ratio_lghg_nsb.append(ratio_hglg)

        ff_volt = np.array(ff_volt)

        with open(
            "{}/Charge_V_final_cuts_{}_{}.dat".format(
                output_dir, runlist[0], runlist[len(runlist) - 1]
            ),
            "w",
        ) as f:
            for a, b, c in zip(
                ff_volt, np.transpose(charge)[0], np.transpose(charge)[1]
            ):
                f.write(f"{a} {b} {c}\n")

        # charge resolution
        charge_hg = charge[:, 0]
        std_hg = std[:, 0]
        std_hg_err = std_err[:, 0]
        charge_lg = charge[:, 1]
        std_lg = std[:, 1]
        std_lg_err = std_err[:, 1]

        hg_res = std_hg / charge_hg
        hg_res_err = err_ratio(std_hg, charge_hg, std_hg_err, std_hg)

        lg_res = std_lg / charge_lg
        lg_res_err = err_ratio(std_lg, charge_lg, std_lg_err, std_lg)
        with open(
            "{}/ChargeRes_final_cuts_{}_{}.dat".format(
                output_dir, runlist[0], runlist[len(runlist) - 1]
            ),
            "w",
        ) as f:
            f.write(
                "charge_hg charge_hg_err hg_res hg_res_err \
                charge_lg charge_lg_err lg_res lg_res_err \n"
            )
            for a, ae, b, be, c, ce, d, de in zip(
                charge_hg,
                std_hg / np.sqrt(npixels),
                mean_resolution[:, 0],
                hg_res_err / np.sqrt(npixels),
                charge_lg,
                std_lg / np.sqrt(npixels),
                mean_resolution[:, 1],
                lg_res_err / np.sqrt(npixels),
            ):
                f.write(f"{a} {ae} {b} {be} {c} {ce} {d} {de}\n")
        x_i = np.where(charge_hg < 200, charge_hg, charge_lg * ratio_hglg[1])
        x_i_err = np.where(
            charge_hg < 200,
            hg_res / np.sqrt(npixels),
            lg_res * ratio_hglg[1] / np.sqrt(npixels),
        )

        y_i = np.where(charge_hg < 200, mean_resolution[:, 0], mean_resolution[:, 1])
        y_i_err = np.where(
            charge_hg < 200,
            hg_res_err / np.sqrt(npixels),
            lg_res_err * ratio_hglg[1] / np.sqrt(npixels),
        )

        mean_resolution_nsb.append(y_i)
        mean_resolution_nsb_err.append(y_i_err)
        mean_charge.append(x_i)
        mean_charge_err.append(y_i)
        del x_i, y_i, x_i_err, y_i_err

    plt.clf()
    for iNSB in range(len(NSB)):
        plt.errorbar(
            mean_charge[iNSB],
            ratio_lghg_nsb[iNSB],
            color=color[iNSB],
            marker="o",
            linestyle="",
            label="NSB={} MHz".format(NSB[iNSB]),
        )
    plt.xlabel("Charge (p.e.)")
    plt.ylabel("HG/LG ratio")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    # plt.ylim(1,600)
    plt.savefig(os.path.join(output_dir, f"HGLG_Ratio_pe_T{temperature}.png"))

    charge_plot = np.linspace(20, 1000)
    stat_limit = 1 / np.sqrt(charge_plot)  # statistical limit

    # fig = plt.figure()
    plt.clf()
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(
        charge_plot,
        stat_limit * 1.106,
        color="gray",
        ls="-",
        lw=3,
        alpha=0.8,
        label="Statistical limit x ENF ",
    )
    plt.fill_between(
        charge_plot,
        y1=stat_limit * 1.106 - 0.005,
        y2=stat_limit * 1.106 + 0.005,
        alpha=0.5,
        color="gray",
    )
    # mask = charge_hg < 2e2

    for iNSB in range(len(NSB)):
        plt.errorbar(
            mean_charge[iNSB],
            mean_resolution_nsb[iNSB],
            xerr=mean_charge_err[iNSB],
            yerr=mean_resolution_nsb_err[iNSB],
            ls="",
            marker="o",
            label="NSB = {} MHz".format(NSB[iNSB]),
            color=color[iNSB],
        )

    plt.xlabel(r"Charge $\overline{Q}$ [p.e.]")
    plt.ylabel(r"Charge resolution $\frac{\sigma_{Q}}{\overline{Q}}$")
    plt.title("T={} degrees".format(temperature))
    plt.xlim(20, 1000)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"charge_resolution_T{temperature}.png"))

    plt.clf()
    plt.close("all")


if __name__ == "__main__":
    main()
