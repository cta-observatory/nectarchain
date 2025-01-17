# don't forget to set environment variable NECTARCAMDATA

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from tools_components import TimingResolutionTestTool
from utils import pe2photons, photons2pe


def get_args():
    """
    Parses command-line arguments for the pixel timing uncertainty test script.

    Returns:
        argparse.ArgumentParser: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Systematic pixel timing uncertainty test B-TEL-1380.\n"
        + "According to the nectarchain component interface, you have to set a NECTARCAMDATA environment variable in the folder where you have the data from your runs or where you want them to be downloaded.\n"
        + "You have to give a list of runs (run numbers with spaces inbetween) and an output directory to save the final plot.\n"
        + "If the data is not in NECTARCAMDATA, the files will be downloaded through DIRAC.\n For the purposes of testing this script, default data is from the runs used for this test in the TRR document.\n"
        + "You can optionally specify the number of events to be processed (default 1200) and the number of pixels used (default 70).\n"
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
        "-e",
        "--evts",
        type=int,
        help="Number of events to process from each run. Default is 100",
        required=False,
        default=100,
    )
    # parser.add_argument('-p','--pixels', type = int, help='Number of pixels used. Default is 70', required=False, default=70)
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
    Processes the pixel timing uncertainty test data and generates a plot.

    The function processes the data from the specified list of runs, calculates the weighted mean RMS and RMS error, and generates a plot of the results. The plot is saved to the specified output directory.

    If a temporary output directory is provided, the plot is also saved to a pickle file in that directory for the gui to use.
    """

    parser = get_args()
    args = parser.parse_args()

    runlist = args.runlist
    nevents = args.evts
    # npixels = args.pixels
    output_dir = os.path.abspath(args.output)
    temp_output = os.path.abspath(args.temp_output) if args.temp_output else None

    print(f"Output directory: {output_dir}")  # Debug print
    print(f"Temporary output file: {temp_output}")  # Debug print

    sys.argv = sys.argv[:1]

    # rms_mu = []
    # rms_mu_err = []
    rms_no_fit = []
    rms_no_fit_err = []
    mean_charge_pe = []

    for run in runlist:
        print("PROCESSING RUN {}".format(run))
        tool = TimingResolutionTestTool(
            progress_bar=True,
            run_number=run,
            max_events=nevents,
            events_per_slice=999,
            log_level=20,
            window_width=16,
            overwrite=True,
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

    print(rms_no_fit)
    photons_spline = np.array(mean_charge_pe) * 100 / 25
    rms_no_fit_err = np.array(rms_no_fit_err)
    print(rms_no_fit_err)
    rms_no_fit_err[rms_no_fit_err == 0] = 1e-5  # almost zero
    # rms_no_fit_err[rms_no_fit_err==np.nan]=1e-5
    print(rms_no_fit_err)

    # mean_rms_mu = np.mean(rms_mu,axis=1)
    # mean_rms_no_fit = np.mean(rms_no_fit,axis=1)

    # weights_mu_pix = 1/(np.array(rms_mu_err)+1e-5)**2
    weights_no_fit_pix = 1 / (rms_no_fit_err) ** 2
    weights_no_fit_pix[weights_no_fit_pix > 1e5] = 1e5
    print(weights_no_fit_pix)

    # rms_mu_weighted=[]
    # rms_mu_weighted_err=[]
    rms_no_fit_weighted = []
    rms_no_fit_weighted_err = []

    for run in range(len(runlist)):
        # rms_mu_weighted.append(np.sum(rms_mu[run]*weights_mu_pix[run])/np.sum(weights_mu_pix[run]))
        # rms_mu_weighted_err.append(np.sqrt(1/np.sum(weights_mu_pix[run])))
        rms_no_fit_weighted.append(
            np.nansum(rms_no_fit[run] * weights_no_fit_pix[run])
            / np.nansum(weights_no_fit_pix[run])
        )
        rms_no_fit_weighted_err.append(np.sqrt(1 / np.nansum(weights_no_fit_pix[run])))

    print(rms_no_fit_weighted)
    print(rms_no_fit_weighted_err)

    # FIGURE
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    plt.errorbar(
        x=photons_spline[:],
        y=np.sqrt(np.array(rms_no_fit_weighted[:]) ** 2),
        yerr=rms_no_fit_weighted_err,
        ls="",
        marker="o",
        label=r"$\mathtt{scipy.signal.find\_peaks}$",
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

    plt.axvspan(20, 1000, alpha=0.1, color="C4")

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
    plt.xlabel("Illumination charge [photons]")
    plt.ylabel("Mean rms per pixel [ns]")
    plt.xscale("log")
    plt.ylim((0, 2.7))
    secax = ax.secondary_xaxis("top", functions=(pe2photons, photons2pe))
    secax.set_xlabel("Illumination charge [p.e.]", labelpad=7)
    plt.savefig(os.path.join(output_dir, "pix_tim_uncertainty.png"))

    if temp_output:
        with open(os.path.join(args.temp_output, "plot1.pkl"), "wb") as f:
            pickle.dump(fig, f)

    plt.close("all")


if __name__ == "__main__":
    main()


# #don't forget to set environment variable NECTARCAMDATA

# import numpy as np
# import os
# import sys

# import matplotlib.pyplot as plt

# from utils import pe2photons,photons2pe
# from test_tools_components import TimingResolutionTestTool
# import argparse

# parser = argparse.ArgumentParser(description='Systematic pixel timing uncertainty test B-TEL-1380.\n'
#                                  +'According to the nectarchain component interface, you have to set a NECTARCAMDATA environment variable in the folder where you have the data from your runs or where you want them to be downloaded.\n'
#                                  +'You have to give a list of runs (run numbers with spaces inbetween) and an output directory to save the final plot.\n'
#                                  +'If the data is not in NECTARCAMDATA, the files will be downloaded through DIRAC.\n For the purposes of testing this script, default data is from the runs used for this test in the TRR document.\n'
#                                  +'You can optionally specify the number of events to be processed (default 1200) and the number of pixels used (default 70).\n')
# parser.add_argument('-r','--runlist', type = int, nargs='+', help='List of runs (numbers separated by space)', required=False)
# parser.add_argument('-e','--evts', type = int, help='Number of events to process from each run. Default is 1200. 4000 or more gives best results but takes some time', required=False, default=1200)
# #parser.add_argument('-p','--pixels', type = int, help='Number of pixels used. Default is 70', required=False, default=70)
# parser.add_argument('-o','--output', type=str, help='Output directory. If none, plot will be saved in current directory', required=False, default='./')

# args = parser.parse_args()

# runlist = args.runlist
# nevents = args.evts
# #npixels = args.pixels

# output_dir = args.output

# sys.argv = sys.argv[:1]

# # rms_mu = []
# # rms_mu_err = []
# rms_no_fit = []
# rms_no_fit_err = []
# mean_charge_pe = []


# for run in runlist:
#     print("PROCESSING RUN {}".format(run))
#     tool = TimingResolutionTestTool(
#         progress_bar=True, run_number=run, max_events=nevents, log_level=20, window_width=16, overwrite=True
#     )
#     tool.initialize()
#     tool.setup()
#     tool.start()
#     output = tool.finish()
#     print(output)
#     print(len(output[0]),len(output[1]))
#     # rms_mu.append(output[0])
#     # rms_mu_err.append(output[1])
#     rms_no_fit.append(output[0])
#     rms_no_fit_err.append(output[1])
#     mean_charge_pe.append(output[2])

# rms_no_fit_err=np.array(rms_no_fit_err)
# photons_spline = np.array(mean_charge_pe)*100/25
# print(photons_spline)
# # mean_rms_mu = np.mean(rms_mu,axis=1)
# # mean_rms_no_fit = np.mean(rms_no_fit,axis=1)
# rms_no_fit_err[rms_no_fit_err==0]=1e-5  #almost zero
# rms_no_fit_err[rms_no_fit_err==np.nan]=1e-5

# # print(np.isnan(rms_no_fit_err).any())
# # print(np.where(rms_no_fit_err==0))
# # weights_mu_pix = 1/(np.array(rms_mu_err)+1e-5)**2
# weights_no_fit_pix = (1/(np.array(rms_no_fit_err)))**2
# # print(np.isnan(weights_no_fit_pix))
# # print(weights_no_fit_pix.dtype)
# # print(np.isnan(weights_no_fit_pix).any())
# # print(np.max(weights_no_fit_pix))
# # print(np.isinf(weights_no_fit_pix).any())
# # print(np.where(np.isinf(weights_no_fit_pix)))
# # for i in range(len(weights_no_fit_pix[0])):
# #     print(i)
# #     print(weights_no_fit_pix[0][i])
# # print(np.nansum(weights_no_fit_pix[0]))
# # print(np.sqrt(weights_no_fit_pix[0]))

# # rms_mu_weighted=[]
# # rms_mu_weighted_err=[]
# rms_no_fit_weighted=[]
# rms_no_fit_weighted_err = []

# for run in range(len(runlist)):
#     # rms_mu_weighted.append(np.sum(rms_mu[run]*weights_mu_pix[run])/np.sum(weights_mu_pix[run]))
#     # rms_mu_weighted_err.append(np.sqrt(1/np.sum(weights_mu_pix[run])))
#     rms_no_fit_weighted.append(np.nansum(rms_no_fit[run]*weights_no_fit_pix[run])/np.nansum(weights_no_fit_pix[run]))
#     rms_no_fit_weighted_err.append(np.sqrt(1/np.nansum(weights_no_fit_pix[run])))


# print("RESULT", rms_no_fit_weighted)
# #FIGURE
# fig, ax = plt.subplots(figsize=(10,7), constrained_layout=True)


# plt.errorbar(x=photons_spline[:],
#              y=np.sqrt(np.array(rms_no_fit_weighted[:])**2),
#             yerr = rms_no_fit_weighted_err,
#              ls='', marker='o',
#              label = r'$\mathtt{scipy.signal.find\_peaks}$')
# # plt.errorbar(x=photons_spline[:],
# #              y=np.sqrt(np.array(rms_mu_weighted[:])**2),
# #              yerr=rms_mu_weighted_err,
# #              ls='', marker='o',
# #              label='Gaussian fit')


# plt.axhline(1, ls='--', color='C4', alpha=0.6)
# plt.axhline(1/np.sqrt(12), ls='--', color='gray', alpha=0.7, label='Quantification rms noise')


# plt.axvspan(20, 1000, alpha=0.1, color='C4')

# ax.text(51.5, 1.04, 'CTA requirement', color='C4',  fontsize=20,
#         horizontalalignment='left',
#         verticalalignment='center')
# ax.annotate("", xy=(40, 0.9), xytext=(40, 0.995), color='C4',  alpha=0.5,
#             arrowprops=dict(color='C4', alpha=0.7, lw=3, arrowstyle='->'))

# ax.annotate("", xy=(200, 0.9), xytext=(200, 0.995), color='C4',  alpha=0.5,
#             arrowprops=dict(color='C4', alpha=0.7, lw=3, arrowstyle='->'))

# plt.legend(frameon=True,  prop={'size':18}, loc='upper right', handlelength=1.2)
# plt.xlabel('Illumination charge [photons]' )
# plt.ylabel('Mean rms per pixel [ns]' )
# plt.xscale(u'log')
# #plt.ylim((0,2.7))
# secax = ax.secondary_xaxis('top', functions=(pe2photons, photons2pe))
# secax.set_xlabel('Illumination charge [p.e.]', labelpad=7)
# plt.savefig(os.path.join(output_dir,"pix_tim_uncertainty.png"))
