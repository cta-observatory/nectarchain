#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import logging
from time import sleep

# astropy imports
from astropy import time

# nectarchain imports
from nectarchain.calibration.container.utils import DataManagement as dm

# DIRAC imports
from DIRAC.Interfaces.API.Dirac import Dirac
from DIRAC.Interfaces.API.Job import Job
from DIRAC.Resources.Catalog.FileCatalogClient import FileCatalogClient

logging.basicConfig(format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

dirac = Dirac()

# Option and argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--runs',
                    default=None,
                    help='list of runs to process',
                    nargs='+',
                    type=int)
parser.add_argument('--log',
                    default=logging.INFO,
                    help='debug output',
                    type=str)
args = parser.parse_args()

if not args.runs:
    logger.critical('A run, or list of runs, should be provided.')
    sys.exit(1)

logger.setLevel(args.log)

executable_wrapper="GainFitter.sh"
sandboxlist = [f'{executable_wrapper}']

# Get run file list from DIRAC
for run in args.runs:
    lfns = dm.get_GRID_location(run)
    for lfn in lfns:
        sandboxlist.append(f'LFN:{lfn}')

# Now, submit the DIRAC jobs:
j = Job()
j.setExecutable(f'{executable_wrapper}')
# Force job to be run from a given Computing Element:
# j.setDestination('LCG.GRIF.fr')
j.setName(f'NectarCAM Gain fitter')
# j.setNumberOfProcessors(minNumberOfProcessors=2)
j.setJobGroup('nectarchain gain')
logger.info(f'''Submitting job, with the following InputSandbox:
{sandboxlist}
''')
j.setInputSandbox(sandboxlist)

res = dirac.submitJob(j)  # , mode='local')  # for local execution, simulating a DIRAC job on the local machine, instead of submitting it to a DIRAC Computing Element
logger.info(f"Submission Result: {res['Value']}")
