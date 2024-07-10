"""Parse a DQM FITS file as nested dictionary

The idea is to parse a DQM FITS, in order to feed the ZODB locally from our Bokeh VM, once the DQM has run on DIRAC
"""

import os
import tarfile

from astropy.io import fits
from DIRAC.Interfaces.API.Dirac import Dirac
from ZODB import DB

from nectarchain.dqm.db_utils import DQMDB

# example = "/tmp/jlenain/scratch/NectarCAM_DQM_Run4971/output/NectarCAM_Run4971
# /NectarCAM_Run4971_calib/NectarCAM_Run4971_Results.fits"
run = 4971
arch = f"/vo.cta.in2p3.fr/user/j/jlenain/nectarcam/dqm/NectarCAM_DQM_Run{run}.tar.gz"

if not os.path.exists(os.path.basename(lfn)):
    dirac = Dirac()

    dirac.getFile(
        lfn=lfn,
        destDir=f".",
        printOutput=True,
    )

with tarfile.open(arch, "r") as tar:
    tar.extractall(".")

fits_file = (
    f"./NectarCAM_DQM_Run"
    f"{run}/output/NectarCAM_Run{run}/NectarCAM_Run"
    f"{run}_calib/NectarCAM_Run{run}_Results.fits"
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

db = DQMDB()
db.insert("test", outdict)
db.commit_and_close()
