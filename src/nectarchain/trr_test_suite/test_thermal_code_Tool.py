# PENSER A TRIER LES COMMANDES

import argparse
import logging

# import math
import os

# import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# import pandas as pd
import tables

# from astropy import units as u
from ctapipe.core import run_tool
from ctapipe_io_nectarcam.constants import N_PIXELS, PIXEL_INDEX
from dateutil.parser import ParserError, parse
from iminuit import Minuit
from iminuit.cost import LeastSquares

from nectarchain.makers.calibration import (
    FlatFieldSPENominalStdNectarCAMCalibrationTool,
    PedestalNectarCAMCalibrationTool,
)
from nectarchain.trr_test_suite.dbhandler import DBInfos, to_datetime

# from nectarchain.makers.extractor.utils import CtapipeExtractor
from nectarchain.trr_test_suite.tools_components import (
    TempLongRunTestTool,
)
from nectarchain.trr_test_suite.utils import (  # pe2photons,; trasmission_390ns,
    get_bad_pixels_list,
    photons2pe,
    source_ids_deadtime,
)

# from nectarchain.utils.constants import ALLOWED_CAMERAS

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[logging.getLogger("__main__").handlers],
)
log = logging.getLogger(__name__)

#########################################################


# PEDESTAL TOOL (everything works here)
def run_ped_tool(
    run_number: list,
    max_events: int,
    events_per_slice: int,
    output_plot,
    bad_pix,
    lenpix,
):
    """This function serves almost the same purpose as the pedestal script,
    except here, the pedestal is averaged over the entire camera.
    We will be computing the baseline, and the pedestal width
    to see its temperature dependency.
    -------------------------------------------------------------
    Parameters needed:
    - the run number
    - the maximum events (usually None for the long runs)
    - the events per slice
    (We do not take combined data for a temperature gradient in time)
    - the output (optional in the arguments)
    - the bad pixels (in the .json file)
    - the number of good pixels
    -------------------------------------------------------------
    Outputs:
    - the sliced and camera averaged baseline and width
    - their "errors" (np.std()/np.sqrt(number of pixels))
    - the time period of the slices (tmin, tmax, tmean)
    -------------------------------------------------------------
    Do not forget to use different runs for the dark pedestal,
    and the pedestal with NSB, if the NSB is constant!!!
    """
    outfile = os.environ["NECTARCAMDATA"] + "/tests/pedestal_{}.h5".format(run_number)
    ped_tool = PedestalNectarCAMCalibrationTool(
        progress_bar=True,
        run_number=run_number,
        max_events=max_events,
        events_per_slice=events_per_slice,
        log_level=20,
        output_path=outfile,
        overwrite=True,
        filter_method="ChargeDistributionFilter",
        charge_sigma_low_thr=3.0,
        charge_sigma_high_thr=3.0,
        method="FullWaveformSum",
    )
    ped_tool.initialize()
    ped_tool.setup()
    ped_tool.start()
    ped_tool.finish(return_output_component=True)

    ped_file = tables.open_file(outfile)
    print(type(ped_file.root.__members__))
    print(ped_file.root.__members__)
    pedestals = np.zeros([len(ped_file.root.__members__) - 1, N_PIXELS])
    pedestals_std = np.zeros(
        [len(ped_file.root.__members__) - 1, N_PIXELS]
    )  # ne fonctionne pas pour une slice
    pedestals_w = np.zeros([len(ped_file.root.__members__) - 1, N_PIXELS])
    var_ped = np.zeros([len(ped_file.root.__members__) - 1, N_PIXELS])
    events = np.zeros([len(ped_file.root.__members__) - 1, N_PIXELS])
    print(f"{pedestals=}")
    # essayer np.array([]) et append pour une et plusieurs slices
    tmin = np.array([])
    tmax = np.array([])
    print(tmin, tmax)
    i = 0
    for result in ped_file.root.__members__:
        table = ped_file.root[result]["NectarCAMPedestalContainer_0"][0]
        wf = table["pedestal_mean_hg"]
        wf_std = table["pedestal_std_hg"]
        events_slice = table["nevents"]
        t_min = table["ucts_timestamp_min"]
        t_max = table["ucts_timestamp_max"]
        print(f"wf: {wf=}")
        print(f"{wf_std = }")
        ped_w = table["pedestal_charge_std_hg"]
        print(f"the index is: {i}")
        if result == "data_combined":
            continue
        else:
            # print(f"{pedestals.shape =}")
            # print("c'est censé fonctionner")
            pedestals[i] = np.mean(wf, axis=1)
            # print(pedestals)
            var_ped[i] = np.square(ped_w)
            pedestals_w[i] = ped_w
            pedestals_std[i] = np.mean(wf_std, axis=1) / np.sqrt(events_slice)
            events[i] = events_slice
            # print(f"{events=}")
            tmin = np.append(tmin, t_min)
            tmax = np.append(tmax, t_max)
            i += 1
            # print(f"the tmin is {t_min}, the table contains {tmin}")

            # print( f"the tmax is {t_max}, the table contains {tmax}")

    # print(f"the pedestals are {np.shape(pedestals)}, {pedestals}")
    # print(f"the pedestal_width is {np.shape(pedestals_w)}, {pedestals_w}")
    # print(f"tmin is now{tmin}, and tmax is now {tmax}")
    # print(
    # f"shape of minimum time array is:
    # {np.shape(tmin)},
    # and the values are {tmin}"
    # )
    tmin = np.unique(tmin)
    tmax = np.unique(tmax)
    tmean = (tmin + tmax) / 2
    print(f"t_mean_shape:{np.shape(tmean)}, {tmean}")
    print(f"{var_ped=}")
    ped_pix = pedestals
    ped_pix_std = pedestals_std

    print(f"{ped_pix_std=}")
    print(np.shape(ped_pix))
    print(ped_pix)
    print(np.shape(ped_pix))
    var_ped[:, bad_pix] = np.nan
    ped_pix[:, bad_pix] = np.nan
    ped_pix_std[:, bad_pix] = np.nan
    print(f"{ped_pix_std=}")
    pedestals_w[:, bad_pix] = np.nan

    ped_cam = np.nanmean(ped_pix, axis=1)
    ped_cam = np.array([x for _, x in sorted(zip(tmean, ped_cam))])
    print(ped_cam)
    ped_cam_std = np.nanstd(ped_pix, axis=1) / np.sqrt(lenpix)
    print(f"ped_cam_std before reducing:{ped_cam_std}")

    print(lenpix, np.sqrt(lenpix))
    ped_w_cam = np.nanmean(pedestals_w, axis=1)
    ped_w_cam = np.array([x for _, x in sorted(zip(tmean, ped_w_cam))])
    ped_w_cam_std = np.nanstd(pedestals_w, axis=1)
    ped_w_cam_std = ped_w_cam_std / np.sqrt(lenpix)
    ped_w_cam_std = np.array([x for _, x in sorted(zip(tmean, ped_w_cam_std))])

    print(np.shape(ped_w_cam_std), ped_w_cam_std)
    # ped_cam_std = (1 / lenpix) * np.sqrt(np.nansum(ped_pix_std**2, axis=1))
    ped_cam_std = np.array([x for _, x in sorted(zip(tmean, ped_cam_std))])
    print(f"{ped_cam_std=}")

    # HERE IS A TIME DEPENDENT PLOT, MIGHT REMOVE LATER
    plt.figure()
    plt.title(f"Camera Pedestal through time for run {run_number}")
    plt.xlabel("UCTS timestamp")
    plt.ylabel("pedestal (ADC counts)")
    plt.errorbar(
        tmean,
        ped_cam,
        xerr=[tmean - tmin, tmax - tmean],
        yerr=ped_cam_std,
        fmt="o",
        color="k",
        capsize=0.0,
    )
    plt.savefig(os.path.join(output_plot, f"avg_cam_ped_{run_number}.png"))

    plt.figure()
    plt.title(f"Camera Pedestal width through time for run {run_number}")
    plt.xlabel("UCTS timestamp")
    plt.ylabel("pedestal (ADC counts)")
    plt.errorbar(
        tmean,
        ped_w_cam,
        xerr=[tmean - tmin, tmax - tmean],
        yerr=ped_w_cam_std,
        fmt="o",
        color="k",
        capsize=0.0,
    )
    plt.savefig(os.path.join(output_plot, f"avg_cam_ped_width_{run_number}.png"))
    return (
        outfile,
        ped_cam,
        ped_cam_std,
        ped_w_cam,
        ped_w_cam_std,
        var_ped,
        tmean,
        tmin,
        tmax,
    )


#################################################################################
# THE LONG RUN TEST TOOL
def run_long_run_test_tool(
    run_number: int,
    spe_run: int,
    pedestal_file,
    nevents: int,
    events_per_slice: int,
    temp_output,
    output_dir,
    # temperature: int,
    ids: int,
    var_ped,
    # telid,
    mean_charge_ts: int,
    bad_pix,
    lenpix,
):
    """This function gathers all the other tools needed
    to fullfill the requirements of the NectarCAM pre-shipment test procedures,
    for long runs with a temperature gradient:
    - the charge/intensity resolution
    - the pixel timing resolution
    - the trigger rate
    - the deadtime rate
    - the readout rate
    It can be used for long runs without a temp gradient though.
    -------------------------------------------------------------
    Parameters needed:
    - the run number
    - the SPE run number (the SPE gain is used for ADC to p.e conversion)
    - the pedestal file (to compute the charge)
    - the number of events and the events per slice
    - the temporary output (for the gui)
    - the source used during the run (or the source of interest)
    - the mean charge threshold (for filtering (à revoir???))
    - the bad and good pixels
    -------------------------------------------------------------
    Outputs:
    - the charge properties and their errors
    - the high gain low gain ratio
    - the timing properties and their errors
    - the deadtime, trigger, and readout properties
    """

    print(spe_run, run_number)

    window_shift = 4
    window_width = 16
    max_events = 5000
    method = "LocalPeakWindowSum"
    print(f"PROCESSING RUN {run_number}")
    gain_run = spe_run
    gain_file_name = (
        os.environ["NECTARCAMDATA"]
        + "/tests/"
        + (
            "FlatFieldSPENominalStdNectarCAM_run{}_maxevents{}_"
            "{}_window_shift_{}_window_width_{}.h5".format(
                gain_run, max_events, method, window_shift, window_width
            )
        )
    )

    if not os.path.exists(gain_file_name):
        gain_tool = FlatFieldSPENominalStdNectarCAMCalibrationTool(
            progress_bar=True,
            run_number=gain_run,
            max_events=max_events,
            method=method,
            output_path=gain_file_name,
            extractor_kwargs={
                "window_width": window_width,
                "window_shift": window_shift,
            },
        )
        run_tool(gain_tool)
    print(f"{nevents}")
    print(f"gain_file_name: {gain_file_name}")
    tool = TempLongRunTestTool(
        progress_bar=True,
        run_number=run_number,
        max_events=nevents,
        events_per_slice=events_per_slice,
        log_level=20,
        method=method,
        extractor_kwargs={
            "window_width": window_width,
            "window_shift": window_shift,
        },
        pedestal_file=pedestal_file,
        use_default_pedestal=True,
        overwrite=True,
        mean_charge_threshold=mean_charge_ts,
    )
    tool.initialize()
    tool.setup()
    tool.start()
    output = tool.finish(
        gain_file=gain_file_name,
        id=ids,
        var_ped_charge=var_ped,
        bad_pix=bad_pix,
        lenpix=lenpix,
    )
    # output = read_file(run, temperature)
    (
        mean_charge_all,
        std_charge_all,
        std_err_all,
        mean_resolution_all,
        err_resolution_all,
        ratio_hglg_all,
        tom_all,
        tom_all_err,
        rms_no_fit_all,
        rms_no_fit_err_all,
        trig_rms_all,
        trig_err_all,
        ucts_deltat_all,
        deadtime_err,
        event_rate_all,
        deadtime_rate,
        deadtime_rate_err,
        collected_trigger_rate_all,
        time_tot_all,
        deadtime_pc_all,
        tmin,
        tmax,
    ) = output

    # print(f"{output=}")
    print(f"{rms_no_fit_all}")
    log.debug(rms_no_fit_all)
    rms_no_fit_err_all = np.array(rms_no_fit_err_all)
    log.debug(rms_no_fit_err_all)
    rms_no_fit_err_all[rms_no_fit_err_all == 0] = 1e-5  # almost zero

    return (
        mean_charge_all,
        std_charge_all,
        std_err_all,
        mean_resolution_all,
        err_resolution_all,
        ratio_hglg_all,
        tom_all,
        tom_all_err,
        rms_no_fit_all,
        rms_no_fit_err_all,
        trig_rms_all,
        trig_err_all,
        ucts_deltat_all,
        deadtime_err,
        event_rate_all,
        deadtime_rate,
        deadtime_rate_err,
        collected_trigger_rate_all,
        time_tot_all,
        deadtime_pc_all,
        tmin,
        tmax,
    )


###############################################
# THE ARGUMENTS


def list_of_ints(arg):
    return list(map(int, arg.split(",")))


def get_args():
    """I recycled the already available arguments,
    might remove some useless ones"""

    parser = argparse.ArgumentParser(
        description="Give a run number and the location "
        "of the runs folder (NECTARCAMDATA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-r",
        "--runlist",
        help="run number",
        type=int,
        # nargs="+",
        # ajouter des nargrs si je fais une boucle sur tous les runs
        #  et si je fais plusieurs runs
    )
    parser.add_argument(
        "-sr",
        "--spe_run",
        help="SPE run number",
        type=int,
        # nargs="+",
    )

    parser.add_argument(
        "-c",
        "--camera",
        default=0,
        help="""Process data for a specific NectarCAM camera.
        Default: NectarCAMQM (Qualification Model).""",
        type=int,
    )
    parser.add_argument(
        "-me",
        "--max_evnts",
        default=None,
        help="maximum events",
        type=int,
    )
    parser.add_argument(
        "-eps",
        "--evnts_per_slice",
        default=None,
        help="frequency of events in the run. Please put the pedestal frequency first.\
            Go through the log is unsure",
        type=int,
        # nargs="+",
        required=False,
        # allow_none=True,
    )
    parser.add_argument(
        "-s",
        "--source",
        type=int,
        choices=[0, 1, 2],
        # nargs="+",
        help="List of corresponding source for each run: 0 for random generator,\
            1 for nsb source, 2 for laser",
        required=False,
        default=source_ids_deadtime,
    )
    """parser.add_argument(
        "-tr",
        "--trans",
        type=float,
        nargs="+",
        help="List of corresponding transmission for each run",
        required=False,
        default=trasmission_390ns,
    )"""
    parser.add_argument(
        "-mct",
        "--mean_charge_threshold",
        type=float,
        help="Threshold below which to select good events,"
        "in units of mean camera charge",
        required=False,
        default=10000,
    )
    """parser.add_argument(
        "-temp",
        "--temperature",
        type=int,
        help="Temperature of the runs",
        required=False,
        default=14,
    )"""
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory, \
            If none, the plots and containers will be saved in your current directory",
        required=False,
        default="./",
    )
    parser.add_argument(
        "--temp_output",
        help="Temporary output directory for GUI",
        default=None,
        required=False,
    )
    return parser


def main():
    parser = get_args()
    args = parser.parse_args()
    run_number = args.runlist
    spe_run = args.spe_run
    nevents = args.max_evnts
    events_per_slice = args.evnts_per_slice
    telid = args.camera
    # print(f"{telid=}")
    ids = args.source
    # transmission = args.trans
    mean_charge_ts = args.mean_charge_threshold
    # temperature = args.temperature
    output_dir = os.path.abspath(args.output)
    temp_output = os.path.abspath(args.temp_output) if args.temp_output else None
    log.debug(f"Output directory: {output_dir}")
    log.debug(f"Temporary output file: {temp_output}")

    sys.argv = sys.argv[:1]

    pixel_ids = PIXEL_INDEX

    print(pixel_ids)
    bad_pix = get_bad_pixels_list()
    lenpix = N_PIXELS - len(bad_pix)

    # $NECTARCHAIN_FIGURES/trr_camera_X/script_name/figure.png
    # where script_name can be linearity
    # ^^^^^^^^^^^^^^^^IMPORTANT FOR THE FIGURES LOCATION

    # on enlève les mauvais pixels (s"il y en a, objectif: ne plus en avoir)

    outfile, ped, ped_std, ped_w, ped_w_std, var_ped, tmean, tmin, tmax = run_ped_tool(
        run_number=run_number,
        max_events=nevents,
        events_per_slice=events_per_slice,
        output_plot=output_dir,
        bad_pix=bad_pix,
        lenpix=lenpix,
    )

    (
        mean_charge_all,
        std_charge_all,
        std_err_all,
        mean_resolution_all,
        err_resolution_all,
        ratio_hglg_all,
        tom_all,
        tom_all_err,
        rms_no_fit_all,
        rms_no_fit_err_all,
        trig_rms_all,
        trig_err_all,
        ucts_deltat_all,
        deadtime_err,
        event_rate_all,
        deadtime_rate,
        deadtime_rate_err,
        collected_trigger_rate_all,
        time_tot_all,
        deadtime_pc_all,
        tmin,
        tmax,
    ) = run_long_run_test_tool(
        run_number=run_number,
        spe_run=spe_run,
        pedestal_file=outfile,
        nevents=nevents,
        events_per_slice=events_per_slice,
        temp_output=temp_output,
        output_dir=output_dir,
        # temperature=temperature,
        ids=ids,
        # telid=telid,
        var_ped=var_ped,
        mean_charge_ts=mean_charge_ts,
        bad_pix=bad_pix,
        lenpix=lenpix,
    )
    print(time_tot_all, tmin, tmax, std_charge_all)

    ####################################################
    path = Path(os.environ["NECTARCAMDATA"] + "/runs")
    db_data_path = path

    telid = telid
    run = run_number
    from datetime import datetime

    tmin_ = np.sort(tmin)
    tmax_ = np.sort(tmax)
    ts_start = tmin_[0] * 1e-9
    ts_end = tmax_[-1] * 1e-9  # C EST EN NANOSECONDES
    print(ts_end, ts_start)

    # Your Unix timestamp with microseconds
    # (we convert to be compatible with the database)
    utc_end = datetime.utcfromtimestamp(ts_end)
    utc_datetime_start = datetime.utcfromtimestamp(ts_start)

    utc_start = utc_datetime_start.strftime("%Y-%m-%d %H:%M:%S.%f")
    utc_end = utc_end.strftime("%Y-%m-%d %H:%M:%S.%f")
    print(utc_start, utc_end)
    dbinfos = DBInfos.init_from_run(
        run=run, path=path, dbpath=db_data_path, verbose=False
    )
    dbinfos.connect(
        "monitoring_drawer_temperatures", "monitoring_ib", "monitoring_ffcls"
    )

    try:
        begin_time = to_datetime(parse(f"{utc_start}"))
        end_time = to_datetime(parse(f"{utc_end}"))
    except ParserError as err:
        print(err)
    N_slices = len(tmin_)

    print(begin_time, end_time, N_slices)

    ##################################################
    """Here I exceptionally use the averaged temperature of the FPM probes
    the NMC crashed for run 7142,\
    thus the FEB/IB temperature evolution lists were incomplete.\
    This can be used as a last resort,
    otherwise I"ll only take into account the FEB/IB temperatures.\
    I interpolate the temperatures between the start and the end of the run
    to have a list matching the number of slices of the run.\
    IT SEEMS THE DATA IS NOT IN CHRONOLIGICAL ORDER.\
    AN ISSUE FOR THE INTERPOLATION (bad plots)\
    I use sorted(zip(tmean, data))  or np.sort for now\
    """

    some_times = np.linspace(begin_time, end_time, int(N_slices))
    # temperatures = dbinfos.tel[0].monitoring_drawer_temperatures.tfeb1.datas
    # temperatures_times = dbinfos.tel[0].monitoring_drawer_temperatures.tfeb1.times

    feb = dbinfos.tel[telid].monitoring_drawer_temperatures.tfeb1.at(some_times)
    temp = np.mean(feb, axis=0)
    temp = np.array(temp)
    print(f"{temp}")
    # print(interp_mean)

    ######################################################
    """I will optimize the plotting/fitting part\
    once evenything else will be plotted for the whole run\
    and everything else will be optimized on my part
    (removing bad pixels precisely)"""

    # pedestal fit
    def lin(t, a, b):
        return a * t + b

    y = (ped - ped[0]) / ped[0]
    t = temp
    sigma = np.sqrt(
        (1 / ped[0]) ** 2 * ped_std**2 + (ped / ped[0] ** 2) ** 2 * ped_std[0] ** 2
    )
    least_squares = LeastSquares(t, y, sigma, lin)
    m = Minuit(least_squares, a=y[0] - y[-1], b=y[-1])
    m.migrad()
    m.hesse()
    print("Données du fit:", m.values, m.errors)

    # Tracer les données et le modèle ajusté
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})
    ax[0].errorbar(t, y, yerr=sigma, fmt="o", ms=3, label="Data")
    ax[0].plot(t, lin(t, *m.values), label="Fitted Model", color="red")

    fit_info_simple = [
        f"$\\chi^2$/$n_\\mathrm{{dof}}$=\
            {m.fval:.1f}/{m.ndof:.0f}=\
                {m.fmin.reduced_chi2:.1f}",
    ]
    for p, v, e in zip(m.parameters, m.values, m.errors):
        fit_info_simple.append(f"{p} = ${v:.5f} \\pm {e:.5f}$")
    ax[0].legend(title="\n".join(fit_info_simple), frameon=False, fontsize="large")
    ax[0].set_title(f"Pedestal Evolution for Run {run_number}")
    ax[0].set_xlabel("T°C")
    ax[0].set_ylabel("Baseline (%)")

    residuals = y - lin(t, *m.values)
    ax[1].errorbar(
        t, residuals, yerr=sigma, fmt="o", ms=3, color="green", label="Residuals"
    )
    ax[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax[1].set_xlabel("T°C)")
    ax[1].set_ylabel("Residuals (%)")
    ax[1].set_title("Residuals of the Fit")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cam_ped_temp_run{run_number}.png"))

    # pedestal width fit

    y = (ped_w - ped_w[0]) / ped_w[0]
    t = temp
    sigma = np.sqrt(
        (1 / ped_w[0]) ** 2 * ped_w_std**2
        + (ped_w / ped_w[0] ** 2) ** 2 * ped_w_std[0] ** 2
    )
    # y=ped_w
    # t=temp
    # sigma_ped_w=ped_w_std

    least_squares = LeastSquares(t, y, sigma, lin)

    m = Minuit(least_squares, a=y[-1] - y[0], b=-y[-1])
    m.migrad()
    m.hesse()

    print("Données du fit:", m.values, m.errors)

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})
    ax[0].errorbar(t, y, yerr=sigma, fmt="o", ms=3, label="Data")
    ax[0].plot(t, lin(t, *m.values), label="Fitted Model", color="red")

    fit_info_simple = [
        f"$\\chi^2$/$n_\\mathrm{{dof}}$=\
            {m.fval:.1f}/{m.ndof:.0f}=\
                {m.fmin.reduced_chi2:.1f}",
    ]
    for p, v, e in zip(m.parameters, m.values, m.errors):
        fit_info_simple.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")
    ax[0].legend(title="\n".join(fit_info_simple), frameon=False, fontsize="large")
    ax[0].set_title(f"Pedestal width evolution for Run {run_number}")
    ax[0].set_xlabel("T°C")
    ax[0].set_ylabel("pedestal_width (%)")

    residuals = y - lin(t, *m.values)
    ax[1].errorbar(
        t, residuals, yerr=sigma, fmt="o", ms=3, color="green", label="Residuals"
    )
    ax[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax[1].set_xlabel("T°C")
    ax[1].set_ylabel("Residuals (%)")
    ax[1].set_title("Residuals of the Fit")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cam_ped_width_temp_run{run_number}.png"))
    mean_charge_hg = mean_charge_all[0]
    std_err_hg = std_err_all[0]
    mean_charge_lg = mean_charge_all[1]
    std_err_lg = std_err_all[1]
    mean_resolution_hg = mean_resolution_all[0]
    err_resolution_hg = err_resolution_all[0]

    curves = [
        (temp, np.array(mean_charge_hg), np.array(std_err_hg)),
        (temp, np.array(mean_charge_lg), np.array(std_err_lg)),
        (temp, np.array(ratio_hglg_all), None),
        (temp, np.array(mean_resolution_hg), np.array(err_resolution_hg)),
        (temp, np.array(tom_all), np.array(tom_all_err)),
        (temp, np.array(rms_no_fit_all), np.array(rms_no_fit_err_all)),
        (temp, np.array(trig_rms_all), np.array(trig_err_all)),
        (temp, np.array(ucts_deltat_all), np.array(deadtime_err)),
        (temp, np.array(deadtime_pc_all), None),
        (temp, np.array(event_rate_all), None),
        (temp, np.array(collected_trigger_rate_all), None),
    ]
    y_labels = [
        "Mean_charge_hg_(p.e)",
        "Mean_charge_lg_(p.e)",
        "Ratio_high_gain_low_gain",
        "Mean_charge_resolution_hg_(%)",
        "ToM_(ns)",
        "ToM_rms_(ns)",
        "trigger_timing_rms_(ns)",
        "Deadtime_(ns)",
        "Deadtime_percentage_(%)",
        "Event_rate",
        "Collected_trigger_rate_(%)",
    ]

    for i, (t, y, y_err) in enumerate(curves):
        t = np.asarray(t).ravel()  # Force 1D array and flatten
        y = np.asarray(y).ravel()  # Force 1D array and flatten

        if y_err is not None:
            y_err = np.asarray(y_err).ravel()  # Force 1D array and flatten
            if len(t) != len(y) or len(t) != len(y_err):
                raise ValueError(f"Length mismatch in curve {i}:\
                      t={len(t)}, y={len(y)}, y_err={len(y_err)}")
            plt.errorbar(t, y, xerr=None, yerr=y_err, fmt="o", color="k", capsize=0.0)
        else:
            if len(t) != len(y):
                raise ValueError(
                    f"Length mismatch in curve {i}: t={len(t)}, y={len(y)}"
                )
            plt.scatter(t, y, marker="o", c="k")

        plt.xlabel("Temperature (°C)")
        plt.ylabel(f"{y_labels[i]}")
        plt.savefig(
            os.path.join(output_dir, f"{y_labels[i]}_temperature_{run_number}.png")
        )
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    plt.errorbar(
        x=mean_charge_hg,
        y=np.sqrt(np.array(rms_no_fit_all) ** 2),
        yerr=rms_no_fit_err_all,
        ls="",
        marker="o",
        label=r"$\mathtt{ctapipe.image.extractor}$",
    )

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
    # Only the bottom subplot will have the x-axis label
    ax2 = ax.secondary_xaxis("top")
    display_ticks = [tmean[i] for i in range(len(tmean)) if i % 10 == 0]
    # LE PLOT MARCHE MAIS C EST MAL FAIT POUR LA TEMPERATURE

    ax2.set_xticks(display_ticks)

    # pour les ticks, penser à aligner la température avec le temps,
    #  ou d'afficher tout les 0.5°C

    # Set the labels for the displayed ticks
    displayed_labels = [f"{temp[i]:.1f}" for i in range(len(temp)) if i % 10 == 0]
    ax2.set_xticklabels(
        displayed_labels
    )  # Set the positions of the ticks to match tmean
    # displayed_labels = [f"{temp: .1f}"
    # if i % 10 == 0 else "" for i, temp in enumerate(temp)]
    # ax2.set_xticklabels(displayed_labels)  # Set the labels to your temperature values
    ax2.set_xlabel(" FPM Temperature (°C)")
    plt.savefig(os.path.join(output_dir, "pix_tim_uncertainty.png"))
    plt.close()

    #####################################################


if __name__ == "__main__":
    main()
