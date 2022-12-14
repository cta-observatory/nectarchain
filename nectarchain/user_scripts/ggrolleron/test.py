import numpy as np
#import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import sys,os
import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',level=logging.INFO)
log = logging.getLogger(__name__)
##tips to add message to stdout
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)



from nectarchain.calibration.container import ChargeContainer,WaveformsContainer
from nectarchain.calibration.container.utils import DataManagment




run_number = [3784]
ped_run_number = [2630]
FF_run_number = [2609]

spe_run_1000V = WaveformsContainer(run_number[0],nevents = 78248)

spe_run_1000V.load_wfs()
charge = ChargeContainer.from_waveform(spe_run_1000V)
charge.write(f"{os.environ['NECTARCAMDATA']}/charges/std/",overwrite = True)


print("work completed")