import numpy as np
#import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import sys
import os

import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',level=logging.INFO,filename = f"{os.environ.get('NECTARCHAIN_LOG')}/{Path(__file__).stem}_{os.getpid()}_load_wfs_charge.log")
log = logging.getLogger(__name__)
##tips to add message to stdout
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)



from nectarchain.calibration.container import WaveformsContainer, WaveformsContainers
from nectarchain.calibration.container import ChargeContainer, ChargeContainers

run_number = [2633,2634]
n_events = [49227,49148]

SPE_run_number = [3784]
FF_run_number = [2608]
ped_run_number = [2630,2609]


overwrite = True

method = "LocalPeakWindowSum"
extractor_kwargs = {'window_width' : 16, 'window_shift' : 4}
path= method+'_4-12'

"""
#for charge and waveform containers
spe_run_1000V = WaveformsContainers(run_number[0],max_events = 15000)
spe_run_1000V.load_wfs()
spe_run_1000V.write(f"{os.environ['NECTARCAMDATA']}/waveforms/",overwrite = overwrite)
charge = ChargeContainers.from_waveforms(spe_run_1000V)
charge.write(f"{os.environ['NECTARCAMDATA']}/charges/std/",overwrite = overwrite)
del spe_run_1000V,charge
"""


spe_run_1000V = WaveformsContainer(run_number[0], nevents = n_events[0])
#spe_run_1000V = WaveformsContainer(run_number[0], max_events=1000)

spe_run_1000V.load_wfs()
spe_run_1000V.write(f"{os.environ['NECTARCAMDATA']}/waveforms/",overwrite = overwrite)

charge = ChargeContainer.from_waveforms(spe_run_1000V,method = method,**extractor_kwargs)
charge.write(f"{os.environ['NECTARCAMDATA']}/charges/{path}/",overwrite = overwrite)
del spe_run_1000V,charge

spe_run_1400V  =  WaveformsContainer(run_number[1],nevents = n_events[1])
#spe_run_1400V = WaveformsContainer(run_number[1], max_events=5000)

spe_run_1400V.load_wfs()
spe_run_1400V.write(f"{os.environ['NECTARCAMDATA']}/waveforms/",overwrite = overwrite)

charge = ChargeContainer.from_waveforms(spe_run_1400V,method = method,**extractor_kwargs)
charge.write(f"{os.environ['NECTARCAMDATA']}/charges/{path}/",overwrite = overwrite)
del spe_run_1400V,charge




#spe_run_1000V = WaveformsContainer(run_number[0],nevents = n_events[0])
#spe_run_1000V.load_wfs()
#charge = ChargeContainer.from_waveforms(spe_run_1000V)
#charge.write(f"/sps/hess/lpnhe/ggroller/projects/NECTARCAM/test/")
#del spe_run_1000V,charge
#
#spe_run_1400V  =  WaveformsContainer(run_number[1],nevents = n_events[1])
#spe_run_1400V.load_wfs()
#charge = ChargeContainer.from_waveforms(spe_run_1400V)
#charge.write(f"/sps/hess/lpnhe/ggroller/projects/NECTARCAM/test/")
#del spe_run_1400V,charge
#

for i in range(len(SPE_run_number)) : 
    #SPE_run = WaveformsContainer(SPE_run_number[i], nevents = 78248)
    SPE_run = WaveformsContainer(SPE_run_number[i], max_events = 60000)
    SPE_run.load_wfs()
    SPE_run.write(f"{os.environ['NECTARCAMDATA']}/waveforms/",overwrite = overwrite)
    #SPE_run = WaveformsContainer.load(f"/sps/hess/lpnhe/ggroller/projects/NECTARCAM/runs/waveforms/waveforms_run3784.fits")
    charge = ChargeContainer.from_waveforms(SPE_run,method = method,**extractor_kwargs)
    charge.write(f"{os.environ['NECTARCAMDATA']}/charges/{path}/",overwrite = overwrite)
    del SPE_run,charge



for i in range(len(FF_run_number)) : 
    FF_run = WaveformsContainer(FF_run_number[i])
    FF_run.load_wfs()
    FF_run.write(f"{os.environ['NECTARCAMDATA']}/waveforms/",overwrite = overwrite)
    charge = ChargeContainer.from_waveforms(FF_run,method = method,**extractor_kwargs)
    charge.write(f"{os.environ['NECTARCAMDATA']}/charges/{path}/",overwrite = overwrite)
    del FF_run,charge

for i in range(len(ped_run_number)) : 
    ped_run = WaveformsContainer(ped_run_number[i])
    ped_run.load_wfs()
    ped_run.write(f"{os.environ['NECTARCAMDATA']}/waveforms/",overwrite = overwrite)
    charge = ChargeContainer.from_waveforms(ped_run,method = method,**extractor_kwargs)
    charge.write(f"{os.environ['NECTARCAMDATA']}/charges/{path}/",overwrite = overwrite)
    del ped_run,charge

