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


def load_wfs_compute_charge(runs_list : list,
                            reload_wfs : bool = False,
                            overwrite : bool= False,
                            charge_extraction_method : str = "FullWaveformSum",
                            **kwargs) -> None :
    """this method is used to load waveforms from zfits files and compute charge with an user specified method

    Args:
        runs_list (list): list of runs for which you want to perfrom waveforms and charge extraction
        reload_wfs (bool, optional): argument used to reload waveforms from pre-loaded waveforms (in fits format) or from zfits file. Defaults to False.
        overwrite (bool, optional): to overwrite file on disk. Defaults to False.
        charge_extraction_method (str, optional): ctapipe charge extractor. Defaults to "FullWaveformSum".

    Raises:
        e : an error occurred during zfits loading from ctapipe EventSource
    """
    
    max_events = kwargs.get("max_events",[None for i in range(len(runs_list))])
    nevents = kwargs.get("nevents",[-1 for i in range(len(runs_list))])

    charge_childpath = kwargs.get("charge_childpath",charge_extraction_method)
    extractor_kwargs = kwargs.get("extractor_kwargs",{})
    

    for i in range(len(runs_list)) : 
        log.info(f"treating run {runs_list[i]}")
        log.info("waveform computation")
        if not(reload_wfs) :
            log.info(f"trying to load waveforms from {os.environ['NECTARCAMDATA']}/waveforms/")
            try : 
                wfs = WaveformsContainer.load(f"{os.environ['NECTARCAMDATA']}/waveforms/waveforms_run{runs_list[i]}.fits")
            except FileNotFoundError as e : 
                log.warning(f"argument said to not reload waveforms from zfits files but computed waveforms not found at sps/hess/lpnhe/ggroller/projects/NECTARCAM/runs/waveforms/waveforms_run{ped_run_number[i]}.fits")
                log.warning(f"reloading from zfits files")
                wfs = WaveformsContainer(runs_list[i])
                wfs.load_wfs()
                wfs.write(f"{os.environ['NECTARCAMDATA']}/waveforms/",overwrite = overwrite)
            except Exception as e :
                log.error(e,exc_info = True)
                raise e
        else : 
            wfs = WaveformsContainer(runs_list[i])
            wfs.load_wfs()
            wfs.write(f"{os.environ['NECTARCAMDATA']}/waveforms/",overwrite = overwrite)

        log.info(f"computation of charge with {charge_extraction_method}")
        charge = ChargeContainer.from_waveforms(wfs,method = charge_extraction_method,**extractor_kwargs)
        charge.write(f"{os.environ['NECTARCAMDATA']}/charges/{path}/",overwrite = overwrite)
        del wfs,charge


def main(SPE_run_number : list = [],
        FF_run_number : list = [], 
        ped_run_number: list = [],
        **kwargs):
    
    SPE_nevents = kwargs.pop('SPE_nevents',[-1 for i in range(len(SPE_run_number))])
    FF_nevents = kwargs.pop('SPE_nevents',[-1 for i in range(len(FF_run_number))])
    ped_nevents = kwargs.pop('SPE_nevents',[-1 for i in range(len(ped_run_number))])

    SPE_max_events = kwargs.pop('SPE_max_events',[None for i in range(len(SPE_run_number))])
    FF_max_events = kwargs.pop('SPE_max_events',[None for i in range(len(FF_run_number))])
    ped_max_events = kwargs.pop('SPE_max_events',[None for i in range(len(ped_run_number))])

    runs_list = SPE_run_number + FF_run_number + ped_run_number
    nevents = SPE_nevents + FF_nevents + ped_nevents
    max_events = SPE_max_events + FF_max_events + ped_max_events

    charge_extraction_method = kwargs.get('method',"FullWaveformSum")



    load_wfs_compute_charge(runs_list = runs_list,
                            charge_extraction_method = charge_extraction_method,
                            nevents = nevents,
                            max_events = max_events,
                            **kwargs)



if __name__ == '__main__':
    SPE_run_number = [2633,2634,3784]
    FF_run_number = [2608]
    ped_run_number = [2609]

    SPE_nevents = [49227,49148,-1]

    overwrite = True
    reload_wfs = False

    method = "LocalPeakWindowSum"
    extractor_kwargs = {'window_width' : 16, 'window_shift' : 4}
    path= method+'_4-12'
    
    main(SPE_run_number, 
        FF_run_number, 
        ped_run_number, 
        SPE_nevents = SPE_nevents, 
        overwrite = overwrite, 
        reload_wfs = reload_wfs,
        method = "LocalPeakWindowSum",
        extractor_kwargs = {'window_width' : 16, 'window_shift' : 4},
        path= method+'_4-12')


