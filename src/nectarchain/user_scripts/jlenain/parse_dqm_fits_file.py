"""Parse a DQM FITS file as nested dictionary

The idea is to parse a DQM FITS, in order to feed the ZODB locally from our Bokeh VM, once the DQM has run on DIRAC
"""

from astropy.io import fits
from ZODB import DB

from nectarchain.dqm.db_utils import DQMDB

example = "/tmp/jlenain/scratch/NectarCAM_DQM_Run4971/output/NectarCAM_Run4971/NectarCAM_Run4971_calib/NectarCAM_Run4971_Results.fits"

hdu = fits.open(example)

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
