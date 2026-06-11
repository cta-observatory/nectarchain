import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tables
from ctapipe_io_nectarcam.constants import N_PIXELS

from nectarchain.makers.calibration import (
    PedestalNectarCAMCalibrationTool,
)
from nectarchain.trr_test_suite.utils import (
    get_bad_pixels_list,
)
from nectarchain.utils.constants import ALLOWED_CAMERAS

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[logging.getLogger("__main__").handlers],
)
log = logging.getLogger(__name__)


def run_ped_tool(
    run_number: list,
    max_events: int,
    events_per_slice: int,
    output_dir,
    bad_pix,
    lenpix,
):
    outfile = f"{output_dir}Pedestal_run_{run_number}.h5"
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
    # log.info(type(ped_file.root.__members__))
    # log.info(ped_file.root.__members__)
    pedestals = np.zeros([len(ped_file.root.__members__) - 1, N_PIXELS])
    pedestals_std = np.zeros([len(ped_file.root.__members__) - 1, N_PIXELS])
    pedestals_w = np.zeros([len(ped_file.root.__members__) - 1, N_PIXELS])
    var_ped = np.zeros([len(ped_file.root.__members__) - 1, N_PIXELS])
    events = np.zeros([len(ped_file.root.__members__) - 1, N_PIXELS])
    # log.info(f"{pedestals=}")
    # essayer np.array([]) et append pour une et plusieurs slices
    tmin = np.array([])
    tmax = np.array([])
    # log.info(tmin, tmax)
    i = 0
    for result in ped_file.root.__members__:
        table1 = ped_file.root[result]
        table = table1["NectarCAMPedestalContainer_0"][0]
        wf = table["pedestal_mean_hg"]
        wf_std = table["pedestal_std_hg"]
        events_slice = table["nevents"]
        t_min = table["ucts_timestamp_min"]
        t_max = table["ucts_timestamp_max"]
        # log.info(f"wf: {wf=}")
        # log.info(f"{wf_std = }")
        ped_w = table["pedestal_charge_std_hg"]
        # log.info(f"the index is: {i}")
        if result == "data_combined":
            continue
        else:
            # #log.info(f"{pedestals.shape =}")
            # #log.info("c'est censé fonctionner")
            pedestals[i] = np.mean(wf, axis=1)
            # #log.info(pedestals)
            var_ped[i] = np.square(ped_w)
            pedestals_w[i] = ped_w
            pedestals_std[i] = np.mean(wf_std, axis=1) / np.sqrt(events_slice)
            events[i] = events_slice
            # #log.info(f"{events=}")
            tmin = np.append(tmin, t_min)
            tmax = np.append(tmax, t_max)
            i += 1

    tmin = np.unique(tmin)
    tmax = np.unique(tmax)
    tmean = (tmin + tmax) / 2
    # log.info(f"t_mean_shape:{np.shape(tmean)}, {tmean}")
    # log.info(f"{var_ped=}")
    ped_pix = pedestals
    ped_pix_std = pedestals_std

    # log.info(f"{ped_pix_std=}")
    # log.info(np.shape(ped_pix))
    # log.info(ped_pix)
    # log.info(np.shape(ped_pix))
    var_ped[:, bad_pix] = np.nan
    ped_pix[:, bad_pix] = np.nan
    ped_pix_std[:, bad_pix] = np.nan
    # log.info(f"{ped_pix_std=}")
    pedestals_w[:, bad_pix] = np.nan

    ped_cam = np.nanmean(ped_pix, axis=1)
    ped_cam = np.array([x for _, x in sorted(zip(tmean, ped_cam))])
    # log.info(f"the camera pedestal is {ped_cam}")
    ped_cam_std = np.nanstd(ped_pix, axis=1) / np.sqrt(lenpix)
    # log.info(f"ped_cam_std before reducing:{ped_cam_std}")

    # log.info(lenpix, np.sqrt(lenpix))
    ped_w_cam = np.nanmean(pedestals_w, axis=1)
    ped_w_cam = np.array([x for _, x in sorted(zip(tmean, ped_w_cam))])
    ped_w_cam_std = np.nanstd(pedestals_w, axis=1)
    ped_w_cam_std = ped_w_cam_std / np.sqrt(lenpix)
    ped_w_cam_std = np.array([x for _, x in sorted(zip(tmean, ped_w_cam_std))])

    # log.info(
    #    f"the shape of the ped width is {np.shape(ped_w_cam_std)},"
    #    f"and the values are {ped_w_cam_std}"
    # )
    # ped_cam_std = (1 / lenpix) * np.sqrt(np.nansum(ped_pix_std**2, axis=1))
    ped_cam_std = np.array([x for _, x in sorted(zip(tmean, ped_cam_std))])
    # log.info(f"{ped_cam_std=}")

    return (
        output_dir,
        ped_cam,
        ped_cam_std,
        ped_w_cam,
        ped_w_cam_std,
        var_ped,
        tmean,
        tmin,
        tmax,
    )


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
    parser.add_argument(
        "-l",
        "--log",
        help="log level",
        default="info",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--camera",
        choices=ALLOWED_CAMERAS,
        default=[camera for camera in ALLOWED_CAMERAS if "QM" in camera][0],
        help="Process data for a specific NectarCAM camera.",
        type=str,
    )
    return parser


def main():
    parser = get_args()
    args = parser.parse_args()
    log.setLevel(args.log.upper())
    run_number = args.runlist
    nevents = args.max_evnts
    camera = args.camera
    events_per_slice = args.evnts_per_slice
    output_dir = os.path.join(
        os.path.abspath(args.output),
        f"trr_camera_{camera}/{Path(__file__)}/",
    )
    temp_output = os.path.abspath(args.temp_output) if args.temp_output else None
    log.debug(f"Output directory: {output_dir}")
    log.debug(f"Temporary output file: {temp_output}")

    sys.argv = sys.argv[:1]

    # log.info(pixel_ids)
    bad_pix = get_bad_pixels_list()
    # log.info(f"BAD PIXELS ARE {bad_pix}")
    lenpix = N_PIXELS - len(bad_pix)

    (
        output_dir,
        ped_cam,
        ped_cam_std,
        ped_w_cam,
        ped_w_cam_std,
        var_ped,
        tmean,
        tmin,
        tmax,
    ) = run_ped_tool(
        run_number=run_number,
        max_events=nevents,
        events_per_slice=events_per_slice,
        output_dir=output_dir,
        bad_pix=bad_pix,
        lenpix=lenpix,
    )
    fig, axes = plt.subplots(2, 1, figsize=(25, 17.5), sharex=True)
    ax, ax2 = axes.flatten()
    ax.set_title(
        f"Camera Pedestal and Pedestal Width through time for run {run_number}"
    )
    ax.set_ylabel("pedestal (ADC counts)")
    ax.errorbar(
        tmean,
        ped_cam,
        xerr=[tmean - tmin, tmax - tmean],
        yerr=ped_cam_std,
        fmt=" ",
        marker="o",
        color="b",
        capsize=0.0,
    )
    # plt.savefig(os.path.join(output_plot, f"avg_cam_ped_{run_number}.png"))

    ax2.set_xlabel("UCTS timestamp")
    ax2.set_ylabel("pedestal (ADC counts)")
    ax2.errorbar(
        tmean,
        ped_w_cam,
        xerr=[tmean - tmin, tmax - tmean],
        yerr=ped_w_cam_std,
        fmt=" ",
        marker="o",
        color="b",
        capsize=0.0,
    )
    fig_name = "pedestal_properties_in_time"
    plot_path = os.path.join(output_dir, f"{fig_name}.png")
    plt.savefig(plot_path)
    if temp_output:
        with open(os.path.join(args.temp_output, "plot1.pkl"), "wb") as f:
            pickle.dump(fig, f)

    plt.close("all")


if __name__ == "__main__":
    main()
