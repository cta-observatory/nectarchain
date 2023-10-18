import argparse
import glob
import json
import logging as log

import ctapipe
import matplotlib.patches as patches
import numpy as np

# from astropy import time as astropytime
from ctapipe.containers import EventType
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry

# from ctapipe.io import EventSeeker, EventSource
# from ctapipe.visualization import CameraDisplay
from ctapipe_io_nectarcam import NectarCAMEventSource, constants

# from matplotlib.path import Path
from traitlets.config import Config

from nectarchain.calibration.container import ChargeContainer, WaveformsContainer

ctapipe.__version__

parser = argparse.ArgumentParser(
    prog="split_wfs_charge_spe_wt",
    description="This program load waveforms from fits.fz run files and "
    "compute the charge for a given start and end event_id ",
)

parser.add_argument(
    "--spe_run_number",
    nargs="+",
    help="SPE run number",
    type=int,
)

parser.add_argument(
    "--nBlock", nargs="+", help="No. of block to be analysed", default="1", type=int
)

parser.add_argument(
    "--path",
    nargs="+",
    help="The path of NectarCAM raw data files",
    default="./",
    type=str,
)

parser.add_argument(
    "--extractorMethod",
    choices=[
        "FullWaveformSum",
        "FixedWindowSum",
        "GlobalPeakWindowSum",
        "LocalPeakWindowSum",
        "SlidingWindowMaxSum",
        "TwoPassWindowSum",
    ],
    default="LocalPeakWindowSum",
    help="charge extractor method",
    type=str,
)

parser.add_argument(
    "--extractor_kwargs",
    default={"window_width": 16, "window_shift": 4},
    help="charge extractor kwargs",
    type=json.loads,
)

args = vars(parser.parse_args())


def get_event_id_list():
    """
    Function to get the start event_id of each SPE-WT scan position

    :return: The list of start event_id
    """

    # npix = constants.N_PIXELS

    abspath = args["path"]
    log.info(f"{abspath[0]}/NectarCAM.*.fits.fz")
    fits_fz_list = glob.glob(f"{abspath[0]}/NectarCAM.*.fits.fz")
    fits_fz_list.sort()
    log.info(fits_fz_list)

    config = Config(
        dict(
            NectarCAMEventSource=dict(
                NectarCAMR0Corrections=dict(
                    calibration_path=None,
                    apply_flatfield=False,
                    select_gain=False,
                )
            )
        )
    )

    reader = NectarCAMEventSource(
        input_url=f"{abspath[0]}/NectarCAM.*.fits.fz", config=config
    )

    prev_ucts_timestamp = 0

    ucts_timestamp_all_events = []
    delta_t_all = []
    events_id_list = []

    n_events = 0

    for j, event in enumerate(reader):  # loop over reader
        n_events = n_events + 1

        if (
            event.trigger.event_type == EventType.SINGLE_PE
        ):  # For not mis-identifying the next block
            if len(events_id_list) == 0:
                events_id_list.append(event.nectarcam.tel[0].evt.event_id)

            ucts_timestamp = event.nectarcam.tel[0].evt.ucts_timestamp / 1e9
            if j == 0:
                prev_ucts_timestamp = ucts_timestamp

            delta_t = ucts_timestamp - prev_ucts_timestamp

            ucts_timestamp_all_events.append(ucts_timestamp)
            delta_t_all.append(delta_t)

            # waveform = event.r0.tel[0].waveform[0]
            # print(np.shape(waveform))

            if delta_t > 0.5:
                print(
                    "New postion at Event number:",
                    j,
                    "id",
                    event.nectarcam.tel[0].evt.event_id,
                )
                events_id_list.append(event.nectarcam.tel[0].evt.event_id)

        prev_ucts_timestamp = ucts_timestamp

    print("#Events:", n_events)
    print("Events id list:", events_id_list)

    return events_id_list


def pixels_under_wt(x, y):
    """
    This function finds the pixels geometrically under the white-target

    :param x: x-coordinate of white-target position
    :param y: y-coordinate of white-target position
    :return: The list of pixels
    """
    coords = 0.001 * np.array(
        [
            [17, -214],
            [-93, -214],
            [-197, -110],
            [-197, 110],
            [-93, 214],
            [17, 214],
            [197, 110],
            [197, -110],
        ]
    )
    coords[:, 0] += x
    coords[:, 1] += y

    wt = patches.Polygon(xy=coords)
    # ax = plt.gca()
    # ax.add_patch(Polygon(xy=coords, alpha=1, ec='k', facecolor='none'))

    geom = CameraGeometry.from_name("NectarCam", version=3).transform_to(
        EngineeringCameraFrame()
    )
    geomdata = geom.to_table()

    pixels_under_wt = []
    for i in range(constants.N_PIXELS):
        if wt.contains_point([geomdata["pix_x"][i], geomdata["pix_y"][i]]):
            pixels_under_wt.append(i)
            # plt.plot(geomdata["pix_x"][i], geomdata["pix_y"][i], "b.")

    # plt.show()

    return pixels_under_wt


if __name__ == "__main__":
    """
    centroids_file = ('
    /home/patel/Sonal/NectarCAM/SPE_ana/spe_scan_centroids_20220209b.dat
    ')

    centroids = pd.read_csv(
    centroids_file, sep=None, names=['x', 'y'],
    index_col=False, skiprows=1
                      )

    for i in range(40):
        # print("Bloc:", i, centroids["x"][i], centroids["y"][i])
        pixels_under_wt(centroids["x"][i], centroids["y"][i])
    """

    events_id_list = get_event_id_list()

    for i in range(args["nBlock"][0]):
        block = i
        start, end = events_id_list[i], events_id_list[i + 1]

        wfs = WaveformsContainer(args["spe_run_number"][0])
        wfs.load_wfs(start=start, end=end)
        wfs.write(args["path"][0], start, end, block, overwrite=True)

        charge = ChargeContainer.from_waveforms(wfs, method=args["extractorMethod"])
        charge.write(args["path"][0], start, end, block, overwrite=True)

        del wfs
