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
parser.add_argument('-r', '--run',
                    default=None,
                    help='only process a specific run (optional)',
                    nargs='+',
                    type=str)
parser.add_argument('--log',
                    default=logging.INFO,
                    help='debug output',
                    type=str)
args = parser.parse_args()

logger.setLevel(args.log)

executable_wrapper="GainFitter.sh"
sandboxlist = [f'{executable_wrapper}']

# TODO: Construct sandbox list using dm.findrun

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
