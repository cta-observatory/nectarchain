"""Parse a DQM FITS file as nested dictionary

This script parses a DQM FITS output, in order to feed the ZODB locally from our Bokeh
VM, once the DQM has run on DIRAC
"""

import argparse
import logging
import os
import sys
import tarfile

from astropy.io import fits
from DIRAC.Interfaces.API.Dirac import Dirac

from nectarchain.dqm.db_utils import DQMDB

logging.basicConfig(format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Option and argument parser
parser = argparse.ArgumentParser(
    description="Fetch a DQM output on DIRAC, parse it and feed ZODB",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-r",
    "--run",
    default=None,
    help="process a specific run.",
    type=str,
)
parser.add_argument(
    "-p",
    "--path",
    default="/vo.cta.in2p3.fr/user/j/jlenain/nectarcam/dqm",
    help="path on DIRAC where to grab DQM outputs (optional).",
    type=str,
)
args = parser.parse_args()

if args.run is None:
    logger.critical("A run number should be provided.")
    sys.exit(1)

lfn = f"{args.path}/NectarCAM_DQM_Run{args.run}.tar.gz"

if not os.path.exists(os.path.basename(lfn)):
    dirac = Dirac()

    dirac.getFile(
        lfn=lfn,
        destDir=f".",
        printOutput=True,
    )

with tarfile.open(os.path.basename(lfn), "r") as tar:
    tar.extractall(".")

fits_file = (
    f"./NectarCAM_DQM_Run"
    f"{args.run}/output/NectarCAM_Run{args.run}/NectarCAM_Run"
    f"{args.run}_calib/NectarCAM_Run{args.run}_Results.fits"
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

db = DQMDB(read_only=False)
db.insert(f"NectarCAM_Run{args.run}", outdict)
db.commit_and_close()
