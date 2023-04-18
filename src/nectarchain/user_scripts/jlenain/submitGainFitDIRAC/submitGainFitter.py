#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

# DIRAC imports
from DIRAC.Interfaces.API.Dirac import Dirac
from DIRAC.Interfaces.API.Job import Job

logging.basicConfig(format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

dirac = Dirac()
logger.setLevel(logging.INFO)

executable_wrapper="GainFitter.sh"

# Now, submit the DIRAC jobs:
j = Job()
j.setExecutable(f'{executable_wrapper}')
# Force job to be run from a given Computing Element:
# j.setDestination('LCG.GRIF.fr')
j.setName(f'NectarCAM Gain fitter')
# j.setNumberOfProcessors(minNumberOfProcessors=2)
j.setJobGroup('nectarchain gain')
sandboxlist = [f'{executable_wrapper}']
logger.info(f'''Submitting job, with the following InputSandbox:
{sandboxlist}
''')
j.setInputSandbox(sandboxlist)

res = dirac.submitJob(j)  # , mode='local')  # for local execution, simulating a DIRAC job on the local machine, instead of submitting it to a DIRAC Computing Element
logger.info(f"Submission Result: {res['Value']}")
