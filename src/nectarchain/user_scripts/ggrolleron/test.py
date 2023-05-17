import numpy as np
#import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import sys,os
import time
import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',level=logging.INFO)
log = logging.getLogger(__name__)
##tips to add message to stdout
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

import glob



from nectarchain.calibration.container import ChargeContainer,WaveformsContainer,WaveformsContainers,ChargeContainers
from nectarchain.calibration.container.utils import DataManagement

def test_multicontainers() :
    run_number = [3731]
    waveforms = WaveformsContainers(run_number[0],max_events=1000)
    log.info("waveforms created")
    waveforms.load_wfs()
    log.info("waveforms loaded")
    
    #waveforms.write(f"{os.environ['NECTARCAMDATA']}/waveforms/",overwrite = True)
    #log.info('waveforms written')

    #waveforms = WaveformsContainers.load(f"{os.environ['NECTARCAMDATA']}/waveforms/waveforms_run{run_number[0]}")
    #log.info("waveforms loaded")

    charge = ChargeContainers.from_waveforms(waveforms,method = "LocalPeakWindowSum", window_width = 16, window_shift = 4)
    log.info("charge computed")
    
    path = "LocalPeakWindowSum_4-12"
    #charge.write(f"{os.environ['NECTARCAMDATA']}/charges/{path}/",overwrite = True)
    #log.info("charge written")

    #charge = ChargeContainers.from_file(f"{os.environ['NECTARCAMDATA']}/charges/{path}/",run_number = run_number[0])
    #log.info("charge loaded")

    charge_merged = charge.merge()
    log.info('charge merged')

def test_white_target() :
    run_number = [4129]
    waveforms = WaveformsContainers(run_number[0],max_events=10000)
    log.info("waveforms created")
    waveforms.load_wfs()
    log.info("waveforms loaded")
    
    #waveforms.write(f"{os.environ['NECTARCAMDATA']}/waveforms/",overwrite = True)
    #log.info('waveforms written')

    #waveforms = WaveformsContainers.load(f"{os.environ['NECTARCAMDATA']}/waveforms/waveforms_run{run_number[0]}")
    #log.info("waveforms loaded")

    charge = ChargeContainers.from_waveforms(waveforms,method = "LocalPeakWindowSum", window_width = 16, window_shift = 4)
    log.info("charge computed")
    
    path = "LocalPeakWindowSum_4-12"
    #charge.write(f"{os.environ['NECTARCAMDATA']}/charges/{path}/",overwrite = True)
    #log.info("charge written")

    #charge = ChargeContainers.from_file(f"{os.environ['NECTARCAMDATA']}/charges/{path}/",run_number = run_number[0])
    #log.info("charge loaded")

    #charge_merged = charge.merge()
    #log.info('charge merged')









def test_simplecontainer() :
    run_number = [2633]
    ped_run_number = [2630]
    FF_run_number = [2609]

    spe_run_1000V = WaveformsContainer(run_number[0],max_events = 1000)

    spe_run_1000V.load_wfs()

    #spe_run_1000V.write(f"{os.environ['NECTARCAMDATA']}/waveforms/",overwrite = True)

    #spe_run_1000V.load(f"{os.environ['NECTARCAMDATA']}/waveforms/waveforms_run{run_number[0]}.fits")

    charge = ChargeContainer.compute_charge(spe_run_1000V,1,method = "LocalPeakWindowSum",extractor_kwargs = {'window_width' : 16, 'window_shift' : 4})

    charge = ChargeContainer.from_waveforms(wfs, method = "SlidingWindowMaxSum", window_width = 16, window_shift = 4)
    log.info(f"SlidingWindowMaxSum duration : {time.time() - t} seconds")

    #charge = ChargeContainer.from_waveforms(wfs, method = "NeighborPeakWindowSum", window_width = 16, window_shift = 4)
    #log.info(f"NeighborPeakWindowSum duration : {time.time() - t} seconds")

    charge = ChargeContainer.from_waveforms(spe_run_1000V)

    charge = ChargeContainer.from_waveforms(wfs, method = "TwoPassWindowSum", window_width = 16, window_shift = 4)
    log.info(f"TwoPassWindowSum duration : {time.time() - t} seconds")

    #charge.write(f"{os.environ['NECTARCAMDATA']}/charges/std/",overwrite = True)



if __name__ == "__main__" : 
    #test_multicontainers()
    test_white_target()

    print("work completed")