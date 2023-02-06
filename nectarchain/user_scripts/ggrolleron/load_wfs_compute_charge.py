import numpy as np
#import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import sys
import os
import argparse
import json

import logging
logging.getLogger("numba").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

from nectarchain.calibration.container import WaveformsContainer, WaveformsContainers
from nectarchain.calibration.container import ChargeContainer, ChargeContainers

parser = argparse.ArgumentParser(
                    prog = 'load_wfs_compute_charge',
                    description = 'This program load waveforms from fits.fz run files and compute charge')

#run numbers
parser.add_argument('-s', '--spe_run_number',
                    nargs="+",
                    default=[],
                    help='spe run list',
                    type=int)
parser.add_argument('-p', '--ped_run_number',
                    nargs="+",
                    default=[],
                    help='ped run list',
                    type=int)
parser.add_argument('-f', '--ff_run_number',
                    nargs="+",
                    default=[],
                    help='FF run list',
                    type=int)

#max events to be loaded
parser.add_argument('--spe_max_events',
                    nargs="+",
                    #default=[],
                    help='spe max events to be load',
                    type=int)
parser.add_argument('--ped_max_events',
                    nargs="+",
                    #default=[],
                    help='ped max events to be load',
                    type=int)
parser.add_argument('--ff_max_events',
                    nargs="+",
                    #default=[],
                    help='FF max events to be load',
                    type=list)

#n_events in runs
parser.add_argument('--spe_nevents',
                    nargs="+",
                    #default=[],
                    help='spe n events to be load',
                    type=int)
parser.add_argument('--ped_nevents',
                    nargs="+",
                    #default=[],
                    help='ped n events to be load',
                    type=int)
parser.add_argument('--ff_nevents',
                    nargs="+",
                    #default=[],
                    help='FF n events to be load',
                    type=list)

#boolean arguments
parser.add_argument('--reload_wfs',
                    action='store_true',
                    default=False,
                    help='to force re-computation of waveforms from fits.fz files'
                    )
parser.add_argument('--overwrite',
                    action='store_true',
                    default=False,
                    help='to force overwrite files on disk'
                    )

#extractor arguments
parser.add_argument('--extractorMethod',
                    choices=["FullWaveformSum","LocalPeakWindowSum"],
                    default="LocalPeakWindowSum",
                    help='charge extractor method',
                    type=str
                    )
parser.add_argument('--extractor_kwargs',
                    default={'window_width' : 16, 'window_shift' : 4},
                    help='charge extractor kwargs',
                    type=json.loads
                    )

#verbosity argument
parser.add_argument('-v',"--verbosity",
                    help='0 for FATAL, 1 for WARNING, 2 for INFO and 3 for DEBUG',
                    default=0,
                    type=int)

args = parser.parse_args()

#control shape of arguments lists
for arg in ['spe','ff','ped'] :
    run_number = eval(f"args.{arg}_run_number")
    max_events = eval(f"args.{arg}_max_events")
    nevents = eval(f"args.{arg}_nevents")

    if not(max_events is None) and len(max_events) != len(run_number) :
        e = Exception(f'{arg}_run_number and {arg}_max_events must have same length')
        log.error(e,exc_info=True)
        raise e
    if not(nevents is None) and len(nevents) != len(run_number) :
        e = Exception(f'{arg}_run_number and {arg}_nevents must have same length')
        log.error(e,exc_info=True)
        raise e

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

    #print(runs_list)
    #print(charge_extraction_method)
    #print(overwrite)
    #print(reload_wfs)
    #print(kwargs)
    
    
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
                log.warning(f"argument said to not reload waveforms from zfits files but computed waveforms not found at {os.environ['NECTARCAMDATA']}/waveforms/waveforms_run{runs_list[i]}.fits")
                log.warning(f"reloading from zfits files")
                wfs = WaveformsContainer(runs_list[i],max_events = max_events[i],nevents = nevents[i])
                wfs.load_wfs()
                wfs.write(f"{os.environ['NECTARCAMDATA']}/waveforms/",overwrite = overwrite)
            except Exception as e :
                log.error(e,exc_info = True)
                raise e
        else : 
            wfs = WaveformsContainer(runs_list[i],max_events = max_events[i],nevents = nevents[i])
            wfs.load_wfs()
            wfs.write(f"{os.environ['NECTARCAMDATA']}/waveforms/",overwrite = overwrite)

        log.info(f"computation of charge with {charge_childpath}")
        charge = ChargeContainer.from_waveforms(wfs,method = charge_childpath,**extractor_kwargs)
        del wfs
        charge.write(f"{os.environ['NECTARCAMDATA']}/charges/{path}/",overwrite = overwrite)
        del charge
    

def main(spe_run_number : list = [],
        ff_run_number : list = [], 
        ped_run_number: list = [],
        **kwargs):

    #print(kwargs)

    spe_nevents = kwargs.pop('spe_nevents',[-1 for i in range(len(spe_run_number))])
    ff_nevents = kwargs.pop('spe_nevents',[-1 for i in range(len(ff_run_number))])
    ped_nevents = kwargs.pop('spe_nevents',[-1 for i in range(len(ped_run_number))])

    spe_max_events = kwargs.pop('spe_max_events',[None for i in range(len(spe_run_number))])
    ff_max_events = kwargs.pop('spe_max_events',[None for i in range(len(ff_run_number))])
    ped_max_events = kwargs.pop('spe_max_events',[None for i in range(len(ped_run_number))])

    runs_list = spe_run_number + ff_run_number + ped_run_number
    nevents = spe_nevents + ff_nevents + ped_nevents
    max_events = spe_max_events + ff_max_events + ped_max_events

    charge_extraction_method = kwargs.get('extractorMethod',"FullWaveformSum")



    load_wfs_compute_charge(runs_list = runs_list,
                            charge_extraction_method = charge_extraction_method,
                            nevents = nevents,
                            max_events = max_events,
                            **kwargs)

if __name__ == '__main__':


    #run of interest
    #spe_run_number = [2633,2634,3784]
    #ff_run_number = [2608]
    #ped_run_number = [2609]
    #spe_nevents = [49227,49148,-1]

    args = parser.parse_args()
    logginglevel = logging.FATAL
    if args.verbosity == 1 : 
        logginglevel = logging.WARNING
    elif args.verbosity == 2 : 
        print(args)
        logginglevel = logging.INFO
    elif args.verbosity == 3 : 
        logginglevel = logging.DEBUG

    os.makedirs(f"{os.environ.get('NECTARCHAIN_LOG')}/{os.getpid()}/figures")
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',force = True, level=logginglevel,filename = f"{os.environ.get('NECTARCHAIN_LOG')}/{os.getpid()}/{Path(__file__).stem}_{os.getpid()}.log")

    log = logging.getLogger(__name__)
    log.setLevel(logginglevel)
    ##tips to add message to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logginglevel)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

    arg = vars(args)
    log.info(f"arguments are : {arg}")

    key_to_pop = []
    for key in arg.keys() :
        if arg[key] is None : 
            key_to_pop.append(key)

    for key in key_to_pop : 
        arg.pop(key)

    log.info(f"arguments passed to main are : {arg}")
 
    path= args.extractorMethod+'_4-12'
    arg['path'] = path
    
    main(**arg)