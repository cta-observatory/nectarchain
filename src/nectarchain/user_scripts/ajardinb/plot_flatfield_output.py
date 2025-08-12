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

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

# Option and argument parser
parser = argparse.ArgumentParser(
    description="Give run number, run location (if no access to DIRAC) and output location",
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
parser.add_argument(
    "-p",
    "--path",
    default="./FlatFieldOutput",
    help="path to hd5 files",
    type=str,
)

args = parser.parse_args()

if args.run is None:
    log.error(f"At least one run number should be provided (use -r).")
    sys.exit(1)

# Define the global environment variable NECTARCAMDATA (folder where are the hd5 files)
run_number = args.run
os.environ["NECTARCAMDATA"] = args.path

outputfile = os.environ["NECTARCAMDATA"] + "/2FF_{}.h5".format(run_number)

h5file = tables.open_file(outputfile)

# for result in h5file.root.__members__:
#    table = h5file.root[result]["FlatFieldContainer_0"][0]
#    ff = table["FF_coef"]

result = h5file.root.__members__[0]
table = h5file.root[result]["FlatFieldContainer_0"][0]
ff = table["FF_coef"]
# bad_pix = table["bad_pixels"]
# print(bad_pix)

# Histogramm of FF coef for one pixel, one gain chanel
pix = 0
HG = 0
ff_pix = ff[:, HG, pix]
mean_ff_pix = np.mean(ff_pix, axis=0)
std_ff_pix = np.std(ff_pix, axis=0)

fig = plt.figure(figsize=(5, 4))
plt.hist(
    ff_pix,
    20,
    label="pixel %s \n$\mu$=%0.3f, \nstd=%0.3f" % (pix, mean_ff_pix, std_ff_pix),
)
plt.axvline(mean_ff_pix, ls="--", color="white")
plt.yscale("log")
plt.xlabel("FF coefficient (HG)")
plt.legend()
plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.12)
plt.savefig(
    os.environ["NECTARCAMDATA"] + "/run{}_FFpix{}_hist.png".format(run_number, pix)
)


# Histogramm of the mean FF coef for all pixels of the camera, one gain chanel
ff_pix = ff[:, HG, :]
mean_ff_pix = np.mean(ff_pix, axis=0, where=np.isinf(ff_pix) == False)
mean_ff_cam = np.mean(mean_ff_pix, axis=0, where=np.isnan(mean_ff_pix) == False)
std_ff_cam = np.std(mean_ff_pix, axis=0, where=np.isnan(mean_ff_pix) == False)

fig = plt.figure(figsize=(5, 4))
plt.hist(mean_ff_pix, 50, label="$\mu$=%0.3f, \nstd=%0.3f" % (mean_ff_cam, std_ff_cam))
plt.axvline(mean_ff_cam, ls="--", color="white")
plt.yscale("log")
plt.xlabel("mean FF coefficient for all pixels (HG)")
plt.legend()
plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.12)
plt.savefig(os.environ["NECTARCAMDATA"] + "/run{}_FFcam_hist.png".format(run_number))

# Camera visualisation
fig = plt.figure(figsize=(5, 4))
camgeom = CameraGeometry.from_name("NectarCam-003").transform_to(
    EngineeringCameraFrame()
)
disp = CameraDisplay(camgeom, title="Mean FF coefficient", show_frame=False)
disp.image = mean_ff_pix
disp.set_limits_minmax(0, 2)
disp.add_colorbar()
plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.12)
plt.savefig(os.environ["NECTARCAMDATA"] + "/run{}_FFcam.png".format(run_number))
