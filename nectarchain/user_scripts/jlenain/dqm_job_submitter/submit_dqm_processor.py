#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Time-stamp: "2022-12-15 22:46:58 jlenain"

import argparse
import sys
import logging
from time import sleep

# astropy imports
from astropy import time

# DIRAC imports
from DIRAC.Interfaces.API.Dirac import Dirac
from DIRAC.Interfaces.API.Job import Job
from DIRAC.Resources.Catalog.FileCatalogClient import FileCatalogClient

logging.basicConfig(format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

dirac = Dirac()

# Option and argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--date',
                    default=None,
                    help='date for which NectarCAM runs should be processed',
                    type=str)
parser.add_argument('-r', '--run',
                    default=None,
                    help='only process a specific run (optional)',
                    type=str)
parser.add_argument('--dry-run',
                    action='store_true',
                    default=False,
                    help='dry run (does not actually submit jobs)')
parser.add_argument('--log',
                    default=logging.INFO,
                    help='debug output',
                    type=str)
args = parser.parse_args()

logger.setLevel(args.log)

if args.date is None:
    logger.critical('A date should be provided, in a format astropy.time.Time compliant. E.g. "2022-04-01".')
    sys.exit(1)

container="nectarchain_w_2022_50.sif"
executable_wrapper="dqm_processor.sh"

## Possible massive job processing via loop on run numbers:
# for run in ['2720', '3277', '...']:

## or from DIRAC FileCatalog directory listing:
processDate = time.Time(args.date)
dfcDir = f'/vo.cta.in2p3.fr/nectarcam/{processDate.ymdhms[0]}/{processDate.ymdhms[0]}{str(processDate.ymdhms[1]).zfill(2)}{str(processDate.ymdhms[2]).zfill(2)}'

# Sometimes, for unkown reason, the connection to the DFC can fail, try a few times:
sleep_time = 2
num_retries = 3
for x in range(0, num_retries):
    try:
        dfc = FileCatalogClient()
        str_error = None
    except Exception as e:
        str_error = str(e)

    if str_error:
        sleep(sleep_time)  # wait before trying to fetch the data again
    else:
        break
if not dfc:
    logger.fatal(f'Connection to FileCatalogClient failed, aborting...')
    sys.exit(1)

infos = dfc.listDirectory(dfcDir)
if not infos['OK'] or not infos['Value']['Successful']:
    logger.critical(f"Could not properly retrieve the file metadata for {dfcDir} ... Exiting !")
    sys.exit(1)
meta = infos['Value']['Successful'][dfcDir]

runlist = []

sqlfile = None
for f in meta['Files']:
    if f.endswith('.fits.fz'):
        run = f.split('NectarCAM.Run')[1].split('.')[0]
        if run not in runlist and run is not None:
            runlist.append(run)
    if f.endswith('.sqlite'):
        sqlfile = f
if args.run is not None:
    if args.run not in runlist:
        logger.critical(f'Your specified run {run} was not found in {dfcDir}, aborting...')
        sys.exit(1)
    runlist = [args.run]
    
logger.info(f'Found runs {runlist} in {dfcDir}')

if sqlfile is None:
    logger.critical('Could not find any SQLite file in {dfcDir}, aborting...')
    sys.exit(1)
logger.info(f'Found SQLite file {sqlfile} in {dfcDir}')

# Now, submit the DIRAC jobs:
# for run in ['2721']:
for run in runlist:
    j = Job()
    # j.setExecutable(f'{executable_wrapper}', '<SOME POSSIBLE ARGUMENTS such as run number>')
    j.setExecutable(f'{executable_wrapper}', f'-r {run}')
    # Force job to be run from a given Computing Element:
    # j.setDestination('LCG.GRIF.fr')
    j.setName(f'NectarCAM DQM run {run}')
    j.setJobGroup('NectarCAM DQM')
    sandboxlist = [f'{executable_wrapper}',
                   f'LFN:/vo.cta.in2p3.fr/user/j/jlenain/local/opt/singularity/nectarchain/{container}',
                   f'LFN:{sqlfile}']
    for f in meta['Files']:
        if f.endswith('.fits.fz') and f'NectarCAM.Run{run}' in f:
            sandboxlist.append(f'LFN:{f}')
    if len(sandboxlist) < 4:
        logger.critical(f'''Misformed sandboxlist, actual data .fits.fz files missing:
{sandboxlist}

Aborting...
''')
        sys.exit(1)
    logger.info(f'''Submitting job for run {run}, with the following InputSandbox:
{sandboxlist}
''')
    j.setInputSandbox(sandboxlist)

    if not args.dry_run:
        res = dirac.submitJob(j)  #, mode='local')  # for local execution, simulating a DIRAC job on the local machine, instead of submitting it to a DIRAC Computing Element
        logger.info(f"Submission Result: {res['Value']}")
