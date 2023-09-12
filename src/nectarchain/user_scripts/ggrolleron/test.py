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



from nectarchain.data.container import ChargeContainer,WaveformsContainer,ChargeContainers,WaveformsContainers
from nectarchain.data.container.utils import DataManagement


def test_check_wfs() :
    run_number = [3731]
    files = glob.glob(f"{os.environ['NECTARCAMDATA']}/waveforms/waveforms_run{run_number[0]}_*.fits")
    if len(files) ==  0 : 
        raise FileNotFoundError(f"no splitted waveforms found")
    else :
        charge_container = ChargeContainers()
        for j,file in enumerate(files):
            log.debug(f"loading wfs from file {file}")
            wfs = WaveformsContainer.load(file)
            print(f"min : {wfs.wfs_hg.min()}")
            #fig,ax = wfs.plot_waveform_hg(0)
            #for i in range(wfs.nevents) : 
            #    wfs.plot_waveform_hg(i,figure = fig,ax = ax)

            log.debug(f"computation of charge for file {file}")
            charge = ChargeContainer.from_waveforms(wfs,
                                                    method = "LocalPeakWindowSum",
                                                    window_width = 16, 
                                                    window_shift = 4)
            hist = charge.histo_hg()
            charge_container.append(charge)
            


def test_extractor() : 
    run_number = [2633]


    wfs = WaveformsContainer(run_number[0],max_events = 200)
    #
    wfs.load_wfs()
    #
    #spe_run_1000V.write(f"{os.environ['NECTARCAMDATA']}/waveforms/",overwrite = True)

    #wfs = WaveformsContainer.load(f"{os.environ['NECTARCAMDATA']}/waveforms/waveforms_run{run_number[0]}.fits")

    #charge = ChargeContainer.compute_charge(spe_run_1000V,1,method = "gradient_extractor",)

    t = time.time()
    charge = ChargeContainer.from_waveforms(wfs, method = "LocalPeakWindowSum", window_width = 16, window_shift = 4)
    log.info(f"LocalPeakWindowSum duration : {time.time() - t} seconds")

    charge = ChargeContainer.from_waveforms(wfs, method = "GlobalPeakWindowSum", window_width = 16, window_shift = 4)
    log.info(f"GlobalPeakWindowSum duration : {time.time() - t} seconds")

    charge = ChargeContainer.from_waveforms(wfs, method = "FullWaveformSum", window_width = 16, window_shift = 4)
    log.info(f"FullWaveformSum duration : {time.time() - t} seconds")

    charge = ChargeContainer.from_waveforms(wfs, method = "FixedWindowSum", window_width = 16, window_shift = 4, peak_index = 30)
    log.info(f"FullWindowSum duration : {time.time() - t} seconds")

    charge = ChargeContainer.from_waveforms(wfs, method = "SlidingWindowMaxSum", window_width = 16, window_shift = 4)
    log.info(f"SlidingWindowMaxSum duration : {time.time() - t} seconds")

    #charge = ChargeContainer.from_waveforms(wfs, method = "NeighborPeakWindowSum", window_width = 16, window_shift = 4)
    #log.info(f"NeighborPeakWindowSum duration : {time.time() - t} seconds")

    #charge = ChargeContainer.from_waveforms(wfs, method = "BaselineSubtractedNeighborPeakWindowSum", baseline_start = 2, baseline_end = 12, window_width = 16, window_shift = 4)
    #log.info(f"BaselineSubtractedNeighborPeakWindowSum duration : {time.time() - t} seconds")

    charge = ChargeContainer.from_waveforms(wfs, method = "TwoPassWindowSum", window_width = 16, window_shift = 4)
    log.info(f"TwoPassWindowSum duration : {time.time() - t} seconds")

    #charge.write(f"{os.environ['NECTARCAMDATA']}/charges/std/",overwrite = False)




def test_simplecontainer() :
    run_number = [2633]

    spe_run_1000V = WaveformsContainer(run_number[0],max_events = 1000)

    spe_run_1000V.load_wfs()

    #spe_run_1000V.write(f"{os.environ['NECTARCAMDATA']}/waveforms/",overwrite = True)

    #spe_run_1000V.load(f"{os.environ['NECTARCAMDATA']}/waveforms/waveforms_run{run_number[0]}.fits")

    charge = ChargeContainer.compute_charge(spe_run_1000V,1,method = "LocalPeakWindowSum",extractor_kwargs = {'window_width' : 16, 'window_shift' : 4})



    charge = ChargeContainer.from_waveforms(spe_run_1000V)


    #charge.write(f"{os.environ['NECTARCAMDATA']}/charges/std/",overwrite = True)




if __name__ == "__main__" : 
    test_check_wfs()

    print("work completed")
