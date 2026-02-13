import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tables
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from ctapipe_io_nectarcam import constants

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

# Option and argument parser
parser = argparse.ArgumentParser(
    description="Give a run number and the location of the folder containing the flatfield output files",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-r",
    "--run",
    default=None,
    # action="store_true",
    help="Flatfield run number",
    type=int,
)

args = parser.parse_args()

if args.run is None:
    log.error(f"At least one run number should be provided (use -r).")
    sys.exit(1)

# Define the global environment variable NECTARCAMDATA (folder where are the hd5 files)
run_number = args.run

outputfile = os.environ["NECTARCAMDATA"] + "/FlatFieldOutput/2FF_{}.h5".format(
    run_number
)

h5file = tables.open_file(outputfile)

# for result in h5file.root.__members__:
#    table = h5file.root[result]["FlatFieldContainer_0"][0]
#    ff = table["FF_coef"]

result = h5file.root.__members__[0]
table = h5file.root[result]["FlatFieldContainer_0"][0]
eff = table["eff_coef"]
pixels_id = table["pixels_id"]
bad_pix = table["bad_pixels"]
gain_channels = ["HG", "LG"]

for G in [constants.HIGH_GAIN, constants.LOW_GAIN]:
    # Histogramm of FF coef for one pixel, per gain chanel
    pix = 0
    eff_pix = eff[:, G, pix]
    FF_pix = np.ma.array(1.0 / eff_pix, mask=eff_pix == 0)
    mean_FF_pix = np.mean(FF_pix, axis=0)
    std_FF_pix = np.std(FF_pix, axis=0)

    fig = plt.figure(figsize=(5, 4))
    plt.hist(
        FF_pix,
        20,
        label="pixel %s \n%s events \n$\mu$ = %0.3f, \nstd = %0.3f"
        % (pix, len(FF_pix), mean_FF_pix, std_FF_pix),
    )
    plt.axvline(mean_FF_pix, ls="--", color="white")
    plt.yscale("log")
    plt.xlabel(
        "FF coefficients for all events - pixel %s (%s)" % (pix, gain_channels[G])
    )
    plt.legend()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.12)
    plt.savefig(
        os.environ["NECTARCAMDATA"]
        + "/run{}_{}_FF_pix{}_hist.png".format(run_number, gain_channels[G], pix)
    )

    # Histogramm of the mean FF coef for all pixels of the camera, per gain chanel
    eff_pix = eff[:, G, :]
    mean_eff_pix = np.mean(eff_pix, axis=0, where=np.isinf(eff_pix) == False)
    mean_FF_pix = np.ma.array(1.0 / mean_eff_pix, mask=mean_eff_pix == 0)
    mean_FF_cam = np.mean(mean_FF_pix, axis=0)
    std_FF_cam = np.std(mean_FF_pix, axis=0)

    fig = plt.figure(figsize=(5, 4))
    plt.hist(
        mean_FF_pix,
        20,
        label="%s valid pixels \n%s events \n$\mu$=%0.3f, \nstd=%0.3f"
        % ((len(pixels_id) - len(bad_pix)), len(eff), mean_FF_cam, std_FF_cam),
    )
    plt.axvline(mean_FF_cam, ls="--", color="white")
    plt.yscale("log")
    plt.xlabel("mean FF coefficients for all pixels (%s)" % gain_channels[G])
    plt.legend()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.12)
    plt.savefig(
        os.environ["NECTARCAMDATA"]
        + "/run{}_{}_FFcam_hist.png".format(run_number, gain_channels[G])
    )

    # Camera visualisation
    fig = plt.figure(figsize=(5, 4))
    camgeom = CameraGeometry.from_name("NectarCam-003").transform_to(
        EngineeringCameraFrame()
    )
    disp = CameraDisplay(
        camgeom, title="Mean FF coefficients (%s)" % gain_channels[G], show_frame=False
    )
    disp.image = mean_FF_pix
    disp.set_limits_minmax(0.5, 1.5)
    disp.add_colorbar()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.12)
    plt.savefig(
        os.environ["NECTARCAMDATA"]
        + "/run{}_{}_FFcam.png".format(run_number, gain_channels[G])
    )
