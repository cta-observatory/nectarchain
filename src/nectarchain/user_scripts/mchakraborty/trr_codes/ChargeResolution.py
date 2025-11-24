# don't forget to set environment variable NECTARCAMDATA

import argparse
import os
import pickle
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ctapipe.coordinates import CameraFrame, EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry
from ctapipe.io import read_table
from ctapipe.visualization import CameraDisplay
from ctapipe_io_nectarcam import NectarCAMEventSource, constants
from lmfit.models import Model

# from IPython import display
from tools_components import ChargeResolutionTestTool
from utils import get_adc_to_pe, get_bad_pixels_list, get_ff_coeff

from nectarchain.trr_test_suite.utils import (
    err_ratio,
    err_sum,
    linear_fit_function,
    plot_parameters,
    trasmission_390ns,
)

plt.style.use("../plot_style.mpltstyle")


def get_args():
    """Parses command-line arguments for the linearity test script.

    Returns:
        argparse.ArgumentParser: The parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Intensity resolution B-TEL-1010 using FF+NSB runs. \n"
        + "According to the nectarchain component interface, \
            you have to set a NECTARCAMDATA environment variable\
                in the folder where you have the data from your runs\
                    or where you want them to be downloaded.\n"
        + "You have to give a list of runs in run_list.json, a\
            corresponding value of voltage and the NSB value of the sets\
            and an output directory to save the \
                final plot.\n"
        + "If the data is not in NECTARCAMDATA, the files will be downloaded through\
            DIRAC.\n For the purposes of testing this script, default data is from the\
                runs used for this test in the TRR document.\n"
        + "You can optionally specify the number of events to be processed\
            (default 500) and the number of pixels used (default 1000).\n"
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

    parser.add_argument(
        "--temp_output", help="Temporary output directory for GUI", default=None
    )

    return parser


def read_file(run, temperature):
    mean_charge = [0, 0]  # per channel
    std_charge = [0, 0]
    std_err = [0, 0]

    mean_resolution = [0, 0]

    print(np.nanmean(adc_to_pe))

    charge_hg = []
    charge_lg = []
    tom_mean = []

    filename = "FinalCuts/LinearityTestTool_run{}_maxevents1000.h5".format(run)
    # adc_to_pe = 58

    output_file = h5py.File(filename)

    for thing in output_file:
        group = output_file[thing]
        dataset = group["ChargeContainer_0"]
        data = dataset[:]
        # print("data",data)
        for tup in data:
            try:
                npixels = tup[1]
                pixel_id = tup[2]
                charge_hg.extend(tup[6])
                charge_lg.extend(tup[7])
                tom_mean.append(tup[8])

            except Exception:
                break

    output_file.close()
    # bad_pix = np.array([50,310,1841,1842, 737, 1702, 827, 385, 1556, 296, 104, 430])

    adc_to_pe = get_adc_to_pe(temperature)
    bad_pix = get_bad_pixels_list()
    ff_coeff = get_ff_coeff()

    charge_hg = np.array(charge_hg)
    charge_lg = np.array(charge_lg)
    # charge_lg[:,bad_pix] = np.nan
    # charge_hg[:,bad_pix] = np.nan

    # print("bad pix list ", bad_pix)

    charge_lg = np.array(charge_lg)
    charge_hg = np.array(charge_hg)

    # npixels = len(np.where(charge_hg[0]>-900)[0])
    # print(npixels)

    # charge_hg = np.ma.masked_invalid(charge_hg)
    # charge_lg = np.ma.masked_invalid(charge_lg)

    print("ff_coeff ", ff_coeff)

    charge_pe_hg = charge_hg / (adc_to_pe)
    charge_pe_lg = charge_lg / (adc_to_pe)

    # print(pixel_id[bad_pix],charge_pe_lg,len(charge_pe_lg),len(charge_pe_hg))
    print(
        "min ",
        np.min(np.concatenate(charge_pe_lg)),
        np.min(np.concatenate(charge_pe_hg)),
    )
    fig, axs = plt.subplots(1, 2)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("Charge LG (pe)")
    axs[0].hist(charge_pe_lg.flatten(), bins=1000, range=[-10.0, 100])
    axs[1].set_yscale("log")
    axs[1].set_xlabel("Charge HG (pe)")
    axs[1].set_ylim(0.01, pow(10, 5))
    axs[1].hist(charge_pe_hg.flatten(), bins=1000, range=[-10.0, 700])

    plt.title("Run {}".format(run))
    plt.tight_layout()
    plt.savefig("FinalCuts/Charge_distribution_cuts_{}.png".format(run))

    ratio_hglg = np.mean(np.mean(charge_pe_hg, axis=0) / np.mean(charge_pe_lg, axis=0))
    print("ratio ", ratio_hglg)

    for channel, charge in enumerate([charge_pe_hg, charge_pe_lg]):
        print(channel, charge)
        pix_mean_charge = np.nanmean(charge, axis=0)  # in pe
        print(pix_mean_charge)
        plt.clf()
        camera = CameraGeometry.from_name("NectarCam-003").transform_to(
            EngineeringCameraFrame()
        )
        # print(tom_mean, tom_mean[0], len(tom_mean[0]))
        disp = CameraDisplay(
            geometry=camera,
            image=pix_mean_charge,
            cmap="gnuplot2",
            title="Run {}, Ch {}".format(run, channel),
        )
        disp.add_colorbar()

        # disp.set_limits_minmax(zmin=18,zmax=30)
        plt.savefig("FinalCuts/Pix_ch_mean_final_cuts_{}_{}.png".format(run, channel))

        pix_std_charge = np.nanstd(charge, axis=0)

        # charge = np.ma.array(charge,mask=[charge<pix_mean_charge- 2.*pix_std_charge])

        # pix_mean_charge = np.mean(charge,axis=0)
        # pix_std_charge  = np.std(charge,axis=0)

        pix_resolution = pix_std_charge / pix_mean_charge

        # pix_resolution = np.ma.array(pix_resolution,mask=[pix_mean_charge<1e-10])
        plt.clf()
        camera = CameraGeometry.from_name("NectarCam-003").transform_to(
            EngineeringCameraFrame()
        )
        # print(tom_mean, tom_mean[0], len(tom_mean[0]))
        disp = CameraDisplay(
            geometry=camera,
            image=pix_resolution,
            cmap="gnuplot2",
            title="Run {} {}".format(run, channel),
        )
        disp.add_colorbar()

        disp.set_limits_minmax(zmin=0.055, zmax=0.062)
        plt.savefig(
            "FinalCuts/Pix_resolution_final_cuts_{}_{}.png".format(run, channel)
        )

        if run == 6959:
            df = pd.DataFrame([[pix_resolution]], columns=["pix_resolution"])
            df.to_json("Pix_resolution.json", orient="columns")

        # print(pix_mean_charge, pix_mean_charge[737],pix_mean_charge[440])
        # pix_mean_charge = np.ma.array(pix_mean_charge,mask=mask)
        # pix_resolution = np.ma.array(pix_resolution, mask=(mask))

        # print("pix ",len(pix_resolution),npixels,channel,pix_resolution,min(pix_resolution),max(pix_resolution),min(pix_mean_charge),max(pix_std_charge))

        # average of all pixels
        mean_charge[channel] = np.nanmean(pix_mean_charge)

        mean_resolution[channel] = np.nanmean(pix_resolution)

        print(
            "pix ",
            npixels,
            channel,
            pix_resolution,
            min(pix_resolution),
            max(pix_resolution),
            np.where(pix_mean_charge < 0),
            max(pix_std_charge),
        )

        # mean_res_std[channel]    = np.std(pix_resolution[pix_resolution>-500])
        std_charge[channel] = np.nanmean(pix_std_charge)
        # for the charge resolution
        std_err[channel] = np.std(pix_std_charge)

    return (
        mean_charge,
        std_charge,
        std_err,
        npixels,
        mean_resolution,
        tom_mean,
        ratio_hglg,
    )


def main():
    """
    The `main()` function is the entry point of the linearity test script. It parses \
        the command-line arguments, processes the specified runs, and generates plots\
            to visualize the linearity and charge resolution of the detector. The\
                function performs the following key steps:
    1. Parses the command-line arguments using the `get_args()` function, which sets up\
        the argument parser and handles the input parameters.
    2. Iterates through the specified run list, processing each run using the\
        `ChargeResolutionTestTool` class. This tool initializes, sets up, starts, and finishes\
            the processing for each run, returning the relevant output data.
    3. Generates three subplots:
    - The first subplot shows the estimated charge vs the voltage of FF (calibration) for both high and low gain channels.
    - The charge resolution vs the charge. (using low gain for points above 200 pe and high gain below that)
    - The third subplot shows the ratio of high-gain to low-gain charge.
    5. Saves the generated plots to the specified output directory.
    6. Generates an additional plot to visualize the charge resolution, including the\
        statistical limit.
    7. Saves the charge resolution plot to the specified output directory, and\
        optionally saves a temporary plot file for a GUI.
    """
    parser = get_args()
    args = parser.parse_args()

    df = pd.read_json("run_list.json")

    NSB = df["NSB"].values
    runs_list = df["runs"].tolist()
    ff_v_list = df["ff_v"].tolist()

    color = ["black", "red", "blue", "green", "yellow"]
    print("NSB Run FF", NSB, runs_list, ff_v_list)

    ratio_lghg_nsb = []
    mean_resolution_nsb = []
    mean_charge = []
    mean_resolution_nsb_err = []
    mean_charge_err = []
    print("NSB", len(NSB), NSB)

    for iNSB in range(len(NSB)):
        runlist = runs_list[iNSB]
        ff_volt = ff_v_list[iNSB]

        temperature = args.temperature
        nevents = args.evts

        output_dir = os.path.abspath(args.output)
        temp_output = os.path.abspath(args.temp_output) if args.temp_output else None

        print(f"Output directory: {output_dir}")  # Debug print
        print(f"Temporary output file: {temp_output}")  # Debug print

        sys.argv = sys.argv[:1]

        charge = np.zeros((len(runlist), 2))
        std = np.zeros((len(runlist), 2))
        std_err = np.zeros((len(runlist), 2))
        mean_resolution = np.zeros((len(runlist), 2))
        ratio_hglg = np.zeros(len(runlist))
        index = 0
        for run in runlist:
            window_width = 16
            print("PROCESSING RUN {}".format(run))
            tool = ChargeResolutionTestTool(
                progress_bar=True,
                run_number=run,
                events_per_slice=10000,
                max_events=nevents,
                log_level=20,
                ff_model=1,
                output_dir=output_dir,
                temperature=temperature,
                window_shift=4,
                window_width=16,
                overwrite=True,
            )
            tool.initialize()
            tool.setup()
            tool.start()

            # tool.set_thermal_params(temperature)
            output = tool.finish()

            # output = read_file(run, temperature)

            (
                charge[index],
                std[index],
                std_err[index],
                npixels,
                mean_resolution[index],
                tom_mean,
                ratio_hglg[index],
            ) = output

            index += 1
            """
            plt.clf()
            camera = CameraGeometry.from_name("NectarCam-003").transform_to(EngineeringCameraFrame())
            #print(tom_mean, tom_mean[0], len(tom_mean[0]))
            disp = CameraDisplay(geometry=camera, image=tom_mean[0], cmap='gnuplot2',title='Run {}, FF at {} V'.format(run,ff_volt[run-runlist[0]]))
            disp.add_colorbar()

            disp.set_limits_minmax(zmin=18,zmax=30)
            plt.savefig('FinalCuts/ToM_mean{}.png'.format(run))
            """

            # print("FINAL",charge,len(ff_volt),len(np.transpose(charge)[0]))

            # charge with voltage
        plt.clf()
        plt.errorbar(
            ff_volt,
            np.transpose(charge)[0],
            color=plot_parameters["High Gain"]["color"],
            yerr=np.transpose(std_err)[0],
            marker="o",
            linestyle="",
        )
        plt.errorbar(
            ff_volt,
            np.transpose(charge)[1],
            color=plot_parameters["Low Gain"]["color"],
            yerr=np.transpose(std_err)[1],
            marker="o",
            linestyle="",
        )
        # plt.yscale('log')
        # plt.xscale('log')
        plt.xlabel("FF voltage (V)")
        plt.ylabel("Average charge (p.e.)")
        plt.grid()
        plt.ylim(1, 600)
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
                "charge_hg charge_hg_err hg_res hg_res_err charge_lg charge_lg_err lg_res lg_res_err \n"
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
    # print(mean_resolution,len(mean_resolution))
    # print(hg_res,lg_res,npixels)

    plt.clf()
    for iNSB in range(len(NSB)):
        plt.errorbar(
            ff_volt,
            ratio_lghg_nsb[iNSB],
            color=color[iNSB],
            marker="o",
            linestyle="",
            label="NSB={} MHz".format(NSB[iNSB]),
        )
    plt.xlabel("FF voltage (V)")
    plt.ylabel("HG/LG ratio")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    # plt.ylim(1,600)
    plt.savefig(os.path.join(output_dir, "HGLG_Ratio_V.png"))

    charge_plot = np.linspace(20, 1000)
    stat_limit = 1 / np.sqrt(charge_plot)  # statistical limit

    fig = plt.figure()
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
    mask = charge_hg < 2e2

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
    plt.savefig(os.path.join(output_dir, "charge_resolution.png"))

    plt.clf()

    print("Std HG, LG", len(charge_hg), len(charge_lg), len(std_hg), len(std_lg))

    if temp_output:
        with open(os.path.join(args.temp_output, "plot2.pkl"), "wb") as f:
            pickle.dump(fig, f)
    plt.close("all")


if __name__ == "__main__":
    main()
