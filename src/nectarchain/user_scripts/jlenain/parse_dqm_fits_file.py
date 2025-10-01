"""Parse a DQM FITS file as nested dictionary

This script parses a DQM FITS output, in order to feed the ZODB locally from our Bokeh
VM, once the DQM has run on DIRAC
"""

import argparse
import logging
import os
import shutil
import sys
import tarfile
from pathlib import Path

import DIRAC
import ZEO
from astropy.io import fits
from DIRAC.Interfaces.API.Dirac import Dirac

from nectarchain.dqm.db_utils import DQMDB

logging.basicConfig(format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

prefix = "NectarCAM"
cameras = [f"{prefix}" + "QM"]
cameras.extend([f"{prefix+str(i)}" for i in range(2, 10)])

# Option and argument parser
parser = argparse.ArgumentParser(
    description="Fetch a DQM output on DIRAC, parse it and feed ZODB",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-f",
    "--force",
    default=False,
    action="store_true",
    help="if this run is already in the DB, force re-parsing its DQM output again.",
)
parser.add_argument(
    "-p",
    "--path",
    default="/ctao/user/j/jlenain/nectarcam/dqm",
    help="path on DIRAC where to grab DQM outputs (optional).",
    type=str,
)
parser.add_argument(
    "-r",
    "--runs",
    nargs="+",
    default=None,
    help="process a specific run or a list of runs.",
)
parser.add_argument(
    "-c",
    "--camera",
    choices=cameras,
    default=[camera for camera in cameras if "QM" in camera][0],
    help="Process data for a specific NectarCAM camera. Default: Qualification Model.",
    type=str,
)
args = parser.parse_args()

if args.runs is None:
    logger.critical("At least one run number should be provided.")
    sys.exit(1)

db_read = DQMDB(read_only=True)
db_read_keys = list(db_read.root.keys())
db_read.abort_and_close()

for run in args.runs:
    if not args.force and f"{args.camera}_Run{run}" in db_read_keys:
        logger.warning(
            f'The run {run} is already present in the DB for the camera {args.camera}, will not parse this DQM run, or consider forcing it with the "--force" option.'
        )
        continue

    lfn = f"{args.path}/{args.camera}/NectarCAM_DQM_Run{run}.tar.gz"

    if not os.path.exists(os.path.basename(lfn)):
        DIRAC.initialize()

        dirac = Dirac()

        dirac.getFile(
            lfn=lfn,
            destDir=f".",
            printOutput=True,
        )

    try:
        with tarfile.open(os.path.basename(lfn), "r") as tar:
            tar.extractall(".")
    except FileNotFoundError as e:
        logger.warning(
            f"Could not fetch DQM results from DIRAC for run ${args.camera} {run}, received error {e}, skipping this run..."
        )
        continue

    fits_file = (
        f"./NectarCAM_DQM_Run{run}/output/NectarCAM_Run{run}/"
        f"NectarCAM_Run{run}_calib/NectarCAM_Run{run}_Results.fits"
    )

    hdu = fits.open(fits_file)

    # Explore FITS file structure
    hdu.info()

    outdict = dict()

    for h in range(1, len(hdu)):
        extname = hdu[h].header["EXTNAME"]
        outdict[extname] = dict()
        for i in range(hdu[extname].header["TFIELDS"]):
            keyname = hdu[extname].header[f"TTYPE{i+1}"]
            outdict[extname][keyname] = hdu[extname].data[keyname]

    try:
        db = DQMDB(read_only=False)
        db.insert(f"{args.camera}_Run{run}", outdict)
        db.commit_and_close()
    except ZEO.Exceptions.ClientDisconnected as e:
        logger.critical(f"Impossible to feed the ZODB data base. Received error: {e}")

    # Remove DQM archive file and directory
    try:
        os.remove(f"NectarCAM_DQM_Run{run}.tar.gz")
    except OSError:
        logger.warning(
            f"Could not remove NectarCAM_DQM_Run{run}.tar.gz or it does not exist"
        )

    dirpath = Path(f"./NectarCAM_DQM_Run{run}")
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
