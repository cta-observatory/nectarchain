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

# from nectarchain.makers.extractor.utils import CtapipeExtractor
from nectarchain.trr_test_suite.tools_components import (
    TempLongRunTestTool,
)
from nectarchain.trr_test_suite.utils import (
    get_bad_pixels_list,
    source_ids_deadtime,
    trasmission_390ns,
)
from nectarchain.utils.constants import ALLOWED_CAMERAS

# import pandas as pd


# from nectarchain.trr_test_suite.test_long_run_tool
# import TempLongRunTestTool


# from matplotlib import dates
# from tqdm import tqdm


# imports from vincent database extraction noteboook
"""For these modules to be imported, we need to have the scripts available and usable.
I copy-pasted them in the trr folder, as in Vincent's folder they are not recognized.
"""

# from nectarchain.trr_test_suite.CalibrationCameraDisplay
#  import CalibrationCameraDisplay
# from nectarchain.trr_test_suite.DataUtils
# import GetFirstLastEventTime
from nectarchain.trr_test_suite.dbhandler import DBInfos, to_datetime

# from nectarchain.trr_test_suite.Utils_DB import GetCamera, GetDefaultDataPath

"""I renamed Utils.py (Vincent's script) as Utils_DB,
just to be sure Utils and utils do not talk to each other.\
Utils (Vincent)  is used in his other modules,
If it's renamed it needs to be renamed in the other modules as well.\
I am looking at your way of extracting the data from the monitoring database"""

"""THIS SCRIPT IS NOT EXACTLY SUITABLE FOR LONG RUNS
(too long, many re-reading events only for one charge container)
A TOOL IS BEING WRITTED TO OPTIMIZE EVERY TOOL WHICH INCLUDES THE CHARGES CONTAINER,
But it's still worth looking, because the computation will be the same"""

# bon courage
"""THE TEMPERATURE GRADIENT LONG RUN USED IS RUN 7142 """


"""LEFT TO DO=
-use log correctly instead of print
-adapt the script to the gui (il y a pas mal d'arguments)
-plot the event rate, charge resolution
-with respect to the internal temperature
-which is more accurate for the linearity of the components
-since it's a temperature gradient long run
"""
logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[logging.getLogger("__main__").handlers],
)
log = logging.getLogger(__name__)

#########################################################

# PEDESTAL TOOL (everything works here)
"""Here, we compute and extract the pedetal,
and the pedestal width from the pedestal container
R.A.S"""


def run_ped_tool(
    run_number: list,
    max_events: int,
    events_per_slice: int,
    output_plot,
):
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
        print(i)
        if result == "data_combined":
            continue
            # pedestal_combined = wf
            # pedestal_std_combined = wf_std
            # pedestal_width_combined = ped_w
            # events_combined = events_slice
            # PLOT for continued (? not really necessary for long runs)
        else:  # VOIR LE PEDESTAL COMBINED
            print(f"{pedestals.shape =}")
            print("c'est censé fonctionner")
            pedestals[i] = np.mean(wf, axis=1)
            print(pedestals)
            pedestals_w[i] = ped_w
            pedestals_std[i] = np.mean(wf_std, axis=1) / np.sqrt(events_slice)
            events[i] = events_slice
            print(f"{events=}")
            tmin = np.append(tmin, t_min)
            tmax = np.append(tmax, t_max)
            i += 1
            print(
                f'the tmin is {tmin}, the table contains {table["ucts_timestamp_min"]}'
            )
            print(
                f'the tmax is {tmax}, the table contains {table["ucts_timestamp_max"]}'
            )

        # IT SEEMS I IS GOING OUT OF THE RESULTS BOUNDS

        # pedestal_combined = wf
        # pedestal_std_combined = wf_std
        # pedestal_width_combined = ped_w
        # events_combined = events_slice
        # PLOT for continued (? not really necessary for long runs)

    print(f"the pedestals are {np.shape(pedestals)}, {pedestals}")
    print(f"the pedestal_width is {np.shape(pedestals_w)}, {pedestals_w}")
    print(f"tmin is now{tmin}, and tmax is now {tmax}")
    print(f"shape of minimum time array is:{np.shape(tmin)},and the values are {tmin}")
    tmin = list(dict.fromkeys(tmin))
    tmax = list(dict.fromkeys(tmax))
    tmean = 0.5 * (tmin + tmax)
    print(f"t_mean_shape:{np.shape(tmean)}, {tmean}")
    ped_pix = pedestals
    ped_pix_std = pedestals_std
    bad_pix = get_bad_pixels_list()
    lenpix = N_PIXELS - len(bad_pix)

    print(f"{ped_pix_std=}")
    print(np.shape(ped_pix))
    print(ped_pix)
    print(np.shape(ped_pix))

    ped_pix[:, bad_pix] = np.nan
    ped_pix_std[:, bad_pix] = np.nan
    print(f"{ped_pix_std=}")
    pedestals_w[:, bad_pix] = np.nan

    ped_cam = np.nanmean(ped_pix, axis=1)
    ped_cam = np.array([x for _, x in sorted(zip(tmean, ped_cam))])
    print(ped_cam)
    ped_cam_std = np.nanstd(ped_pix, axis=1)
    print(f"ped_cam_std before reducing:{ped_cam_std}")

    print(lenpix, np.sqrt(lenpix))
    ped_w_cam = np.nanmean(pedestals_w, axis=1)
    ped_w_cam = np.array([x for _, x in sorted(zip(tmean, ped_w_cam))])
    ped_w_cam_std = np.nanstd(pedestals_w, axis=1)
    ped_w_cam_std = ped_w_cam_std / np.sqrt(lenpix)
    ped_w_cam_std = np.array([x for _, x in sorted(zip(tmean, ped_w_cam_std))])

    print(np.shape(ped_w_cam_std), ped_w_cam_std)
    ped_cam_std = (1 / lenpix) * np.sqrt(np.nansum(ped_pix_std**2, axis=1))
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
    return outfile, ped_cam, ped_cam_std, ped_w_cam, ped_w_cam_std, tmean, tmin, tmax


########################################
# # PIXEL TIMING RESOLUTION TOOL

"""Here we compute the ToM from the charge container,
as well as the STD and the error.
for this snippet to be compatible with the tom uncertainty in tools_components.py,
we need to take into account the multiple slices of the run
The current config only takes one slice.
what was used in my local dev environemnt was:
        super().finish(return_output_component=True, *args, **kwargs)
        outputs = [c for c in ChargesContainers.from_hdf5(self.output_path)]
with a loop on the outputs (slices)
one might also need to implement the sources for the events
to take into account (same as deadtime):
    id=kwargs.pop("id")
    if id == 0:  # FFCLS
        event_type = EventType.FLATFIELD
    elif id == 1:  # NSB
        event_type = EventType.SUBARRAY
    elif id == 2:  # Laser
        event_type = EventType.SUBARRAY
also, in the for pix in range(npixels) loop, I changed the ToM position,
the pulse window, and the bins
(the pulses for long run 7142 peak at 15-16ns on average)
The temporary solutiion was provided by Pablo
the bad pixels haven't been used yet in this tool."""

######################################################
# DEADTIME TOOL

"""For this tool, the loop to collect all the slices has also been done,
with as many containers as there are slices.
Thanks to this tool, we can collect the deltat, to compute the deadtime rate,
the trigger rates, the deadtime percentage,
and the events counter to compute the event rate.
This was discussed with Medha
"""


#################################################
# RESOLUTION CHARGE TOOL
def run_long_run_test_tool(
    run_number: int,
    spe_run: int,
    pedestal_file,
    nevents: int,
    events_per_slice: int,
    temp_output,
    output_dir,
    temperature: int,
    ids: int,
    telid,
    mean_charge_ts: int,
):
    """
    for this tool, I used again a command that calls a container per slice.
    Since the tool.finish() takes into account high gain and low gain channels,
    the lists have to append
    in the right channel(for channel,charge in enumerate([charge_pe_hg,charge_pe_lg]):)
    The bad pixels haven't been removed from this.
    """

    """
    ratio_lghg_nsb = []
    mean_resolution_nsb = []
    mean_charge = []
    mean_resolution_nsb_err = []
    mean_charge_err = []
    """

    print(spe_run, run_number)

    window_shift = 4
    window_width = 16
    max_events = 5000
    method = "LocalPeakWindowSum"
    log.info(f"PROCESSING RUN {run_number}")
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
        # IT DOESN T WORK AT THE MOMENT
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
    log.info(f"gain_file_name: {gain_file_name}")
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
    output = tool.finish(gain_file=gain_file_name, id=ids)
    # output = read_file(run, temperature)
    (
        mean_charge_all,
        std_charge_all,
        std_err_all,
        npixels,
        mean_resolution_all,
        ratio_hglg_all,
        tom_all,
        rms_no_fit_all,
        rms_no_fit_err_all,
        trig_rms_all,
        trig_err_all,
        ucts_timestamps_all,
        ucts_deltat_all,
        event_counter_all,
        busy_counter_all,
        collected_trigger_rate_all,
        time_tot_all,
        deadtime_pc_all,
    ) = output
    # ucts_timestamps = output[0]
    """ucts_deltat = output[1]

    event_counter = output[2]
    # busy_counter = output[3]

    collected_trigger_rates = output[4]

    # time_tot = output[5]
    # print(f"{ucts_timestamps=}")
    # print(f'{ucts_deltat=}')
    print(f"{event_counter=}")
    print(f"{collected_trigger_rates=}")
    deadtime_pc = output[6]
    print(f"{deadtime_pc=}")
    # print(f"{deadtime_us=}")
    max_dt = max(len(dt) for dt in ucts_deltat)
    counter = np.array([len(ec) - 1 for ec in event_counter])
    print(f"{counter=}")
    num_events = np.array([ec[-1] for ec in event_counter])
    print(f"{num_events=}")
    event_rate = counter / num_events
    print(f"{event_rate=}")
    print(max_dt)

    padded_dt = []
    for dt in ucts_deltat:
        padded_sublist = dt + [np.nan] * (max_dt - len(dt))
        padded_dt.append(padded_sublist)

    ucts_deltat = np.array(padded_dt)
    deadtime_mean = np.nanmean(ucts_deltat, axis=1)
    deadtime_std = np.nanstd(ucts_deltat, axis=1) / np.sqrt(counter)
    print(deadtime_mean, deadtime_std)
    deadtime_mean_us = (deadtime_mean * u.ns).to(u.us)
    deadtime_mean_us = deadtime_mean_us.value
    deadtime_std_us = (deadtime_std * u.ns).to(u.us)
    deadtime_std_us = deadtime_std_us.value
    print(f"{deadtime_mean_us=}")
    print(f"{deadtime_std_us=}")
    rms_no_fit = np.array(output[0])
    rms_no_fit_err = np.array(output[1])
    mean_charge_pe = np.array(output[2])
    print("rms_no_fit:", np.shape(rms_no_fit), rms_no_fit)
    print("rms_no_fit_err:", np.shape(rms_no_fit_err), rms_no_fit_err)
    print(f"{mean_charge_pe=}")
    if max_events is None:
        filename = (
            f"{os.environ.get('NECTARCAMDATA','/tmp')}"
            "/tests/TimingResolutionTestTool_run{run_number}.h5"
        )
    else:
        filename = (
            f"{os.environ.get('NECTARCAMDATA','/tmp')}"
            "/tests/TimingResolutionTestTool_run{run_number}_"
            "maxevents{max_events}.h5"
        )

    pix_tim_file = tables.open_file(filename)
    print(filename)

    # rms_no_fit= np.array([])
    # rms_no_fit_err=np.array([]) #essayer np.array([])
    # et append pour une et plusieurs slices
    # tmin = np.array([])
    # tmax = np.array([])
    # print(tmin, tmax)
    rms_no_fit[:, bad_pix_all_flat] = np.nan
    rms_no_fit_err[:, bad_pix_all_flat] = np.nan

    rms_cam_nofit = np.nanmean(rms_no_fit, axis=1)
    rms_cam_nofit_err = (1 / lenpix) * np.sqrt(np.nansum(rms_no_fit_err**2, axis=1))
    i = -1
    tom_cam = np.zeros([math.ceil(N)])
    tom_cam_std = np.zeros([math.ceil(N)])
    print(f"{tom_cam=}")
    time_min = np.zeros([math.ceil(N)])
    time_max = np.zeros([math.ceil(N)])
    time_mean = np.zeros([math.ceil(N)])
    print("type of file:", type(pix_tim_file.root.__members__))
    print("what's in it: ", pix_tim_file.root.__members__)
    for result in pix_tim_file.root.__members__:
        table = pix_tim_file.root[result]["ChargesContainer_0"]["FLATFIELD"]
        tom = table[0][17]
        print(f"{tom=}")
        tom_pix = np.mean(tom, axis=0)
        tom_pix_std = np.std(tom, axis=0) / np.sqrt(len(tom))
        tom_pix[bad_pix_all_flat] = np.nan
        tom_pix_std[bad_pix_all_flat] = np.nan
        tom_cam_mean = np.nanmean(tom_pix)
        tom_std = (1 / lenpix) * np.sqrt(np.nansum(tom_pix_std**2))
        tom_cam[i] = tom_cam_mean
        tom_cam_std[i] = tom_std
        ucts_time = table[0][7]
        time_min[i] = np.min(ucts_time)
        time_max[i] = np.max(ucts_time)
        tmean = 0.5 * (time_min[i] + time_max[i])
        time_mean[i] = tmean
        # print ("result:", np.shape(result), result)

        print(f"{table=}")
        print(f"{ucts_time=}")
    """

    """

    a voir pour le gain photostatistique
    tool = PhotoStatisticNectarCAMCalibrationTool(
        progress_bar=True,
        run_number=run_number,
        max_events=nevents,
        events_per_slice=events_per_slice,
        camera=telid,
        Ped_run_number=run_number,
        SPE_result=gain_file_name,
        overwrite=True,
        )
    tool.setup()
    if args.reload_events and not (max_events is None):
        _figpath = f"{figpath}/ \
        {tool.name}_run{tool.run_number}_\
        maxevents{_max_events}_{tool.method}_{str_extractor_kwargs}"
    else:
        _figpath = f"{figpath}/
        {tool.name}_run{tool.run_number}_{tool.method}_{str_extractor_kwargs}"
    tool.start()
    tool.finish(figpath=_figpath)
    """
    return (
        mean_charge_all,
        std_charge_all,
        std_err_all,
        npixels,
        mean_resolution_all,
        ratio_hglg_all,
        tom_all,
        rms_no_fit_all,
        rms_no_fit_err_all,
        trig_rms_all,
        trig_err_all,
        ucts_timestamps_all,
        ucts_deltat_all,
        event_counter_all,
        busy_counter_all,
        collected_trigger_rate_all,
        time_tot_all,
        deadtime_pc_all,
    )


###############################################
"""for this tool, the weighted mean cannot be done
when the last slice contains one event or a few.
When the event is not of the desired type, it displays 0.
You have to control your slices
so that it's not a direct multiple of the max events,
nor to obtain one event in the last slice.
Might be a problem for the max_events=None (?)
implemented as well a loop and a container for each slice for now.
the bad pixels haven't been defined in this tool."""


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
        choices=ALLOWED_CAMERAS,
        default=[camera for camera in ALLOWED_CAMERAS if "QM" in camera][0],
        help="""Process data for a specific NectarCAM camera.
        Default: NectarCAMQM (Qualification Model).""",
        type=str,
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
    parser.add_argument(
        "-tr",
        "--trans",
        type=float,
        nargs="+",
        help="List of corresponding transmission for each run",
        required=False,
        default=trasmission_390ns,
    )
    parser.add_argument(
        "-mct",
        "--mean_charge_threshold",
        type=float,
        help="Threshold below which to select good events,"
        "in units of mean camera charge",
        required=False,
        default=10,
    )
    parser.add_argument(
        "-temp",
        "--temperature",
        type=int,
        help="Temperature of the runs",
        required=False,
        default=14,
    )
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
    print(f"{telid=}")
    ids = args.source
    # transmission = args.trans
    mean_charge_ts = args.mean_charge_threshold
    temperature = args.temperature
    output_dir = os.path.abspath(args.output)
    temp_output = os.path.abspath(args.temp_output) if args.temp_output else None
    log.debug(f"Output directory: {output_dir}")
    log.debug(f"Temporary output file: {temp_output}")

    sys.argv = sys.argv[:1]

    pixel_ids = PIXEL_INDEX

    print(pixel_ids)

    # $NECTARCHAIN_FIGURES/trr_camera_X/script_name/figure.png
    # where script_name can be linearity
    # ^^^^^^^^^^^^^^^^IMPORTANT FOR THE FIGURES LOCATION

    # on enlève les mauvais pixels (s'il y en a, objectif: ne plus en avoir)

    outfile, ped, ped_std, ped_w, ped_w_std, tmean, tmin, tmax = run_ped_tool(
        run_number=run_number,
        max_events=nevents,
        events_per_slice=events_per_slice,
        output_plot=output_dir,
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
        # busy_counter_all,
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
        temperature=temperature,
        ids=ids,
        telid=telid,
        mean_charge_ts=mean_charge_ts,
    )
    print(time_tot_all, tmin, tmax, std_charge_all)
    """
    print(f'{mean_charge_all=}')
    print(f'{std_charge_all=}')
    print(f'{std_err_all=}')
    print(f'{npixels=}')
    print(f'{mean_resolution_all=}')
    print(f'{ratio_hglg_all=}')
    print(f'{tom_all=}')
    print(f'{rms_no_fit_all=}')
    print(f'{rms_no_fit_err_all=}')
    print(f'{trig_rms_all=}')
    print(f'{trig_err_all=}')
    print(f'{ucts_timestamps_all=}')
    print(f'{ucts_deltat_all=}')
    print(f'{event_counter_all=}')
    print(f'{busy_counter_all=}')
    print(f'{collected_trigger_rate_all=}')
    print(f'{time_tot_all}')
    print(f'{deadtime_pc_all}')
    """

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
        "monitoring_drawer_temperatures", "monitoring_ib", "monitoring_fpm_temp"
    )

    try:
        begin_time = to_datetime(parse(f"{utc_start}"))
        end_time = to_datetime(parse(f"{utc_end}"))
    except ParserError as err:
        print(err)
    N_slices = len(tmin)

    print(begin_time, end_time, N_slices)
    ##################################################
    """Here I exceptionally use the averaged temperature of the FPM probes
    the NMC crashed for run 7142,\
    thus the FEB/IB temperature evolution lists were incomplete.\
    This can be used as a last resort,
    otherwise I'll only take into account the FEB/IB temperatures.\
    I interpolate the temperatures between the start and the end of the run
    to have a list matching the number of slices of the run.\
    IT SEEMS THE DATA IS NOT IN CHRONOLIGICAL ORDER.\
    AN ISSUE FOR THE INTERPOLATION (bad plots)\
    I use sorted(zip(tmean, data))  or np.sort for now\
    """

    some_times = np.linspace(begin_time, end_time, int(N_slices))
    # temperatures = dbinfos.tel[0].monitoring_drawer_temperatures.tfeb1.datas
    # temperatures_times = dbinfos.tel[0].monitoring_drawer_temperatures.tfeb1.times

    interp_13 = dbinfos.monitoring_fpm_temp.fpm_temp_013.at(some_times)
    interp_56 = dbinfos.monitoring_fpm_temp.fpm_temp_056.at(some_times)
    interp_62 = dbinfos.monitoring_fpm_temp.fpm_temp_062.at(some_times)
    interp_208 = dbinfos.monitoring_fpm_temp.fpm_temp_208.at(some_times)
    interp_251 = dbinfos.monitoring_fpm_temp.fpm_temp_251.at(some_times)
    interp_202 = dbinfos.monitoring_fpm_temp.fpm_temp_202.at(some_times)
    interp_196 = dbinfos.monitoring_fpm_temp.fpm_temp_196.at(some_times)
    interp_139 = dbinfos.monitoring_fpm_temp.fpm_temp_139.at(some_times)
    interp_132 = dbinfos.monitoring_fpm_temp.fpm_temp_132.at(some_times)
    interp_125 = dbinfos.monitoring_fpm_temp.fpm_temp_125.at(some_times)
    interp_219 = dbinfos.monitoring_fpm_temp.fpm_temp_219.at(some_times)
    interp_fpm = [
        interp_13,
        interp_56,
        interp_62,
        interp_125,
        interp_132,
        interp_139,
        interp_196,
        interp_202,
        interp_208,
        interp_219,
        interp_251,
    ]
    print(np.shape(interp_fpm))
    temp = np.mean(interp_fpm, axis=0)
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

    curves = [
        (temp, mean_charge_all[0], std_err_all[0]),
        (temp, mean_charge_all[1], std_err_all[1]),
        (temp, ratio_hglg_all, None),
        (temp, mean_resolution_all[0] * 100, err_resolution_all[0] * 100),
        (temp, tom_all, tom_all_err),
        (temp, rms_no_fit_all, rms_no_fit_err_all),
        (temp, trig_rms_all, trig_err_all),
        (temp, ucts_deltat_all, deadtime_err),
        (temp, deadtime_pc_all, None),
        (temp, event_rate_all, None),
        (temp, collected_trigger_rate_all, None),
    ]
    y_labels = [
        "Mean charge hg (p.e)",
        "Mean charge lg (p.e)",
        "Ratio high gain low gain",
        "Mean charge resolution hg (%)",
        "ToM (ns)",
        "ToM rms (ns)",
        "trigger timing rms (ns)",
        "Deadtime (ns)",
        "Deadtime percentage (%)",
        "Event rate",
        "Collected trigger rate (%)",
    ]

    for i, (t, y, y_err) in enumerate(curves):
        plt.figure()
        # plt.title(f"Camera Pedestal through time for run {run_number}")
        plt.xlabel("Temperature (°C)")
        plt.ylabel(f"{y_labels[i]}")
        plt.errorbar(
            t,
            y,
            xerr=None,
            yerr=y_err,
            fmt="o",
            color="k",
            capsize=0.0,
        )
        plt.savefig(os.path.join(output_dir, f"{y}_temperature_{run_number}.png"))


main()
