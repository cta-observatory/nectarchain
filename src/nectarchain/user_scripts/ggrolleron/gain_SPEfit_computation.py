import logging
import sys
import os
from pathlib import Path
import time

os.makedirs(os.environ.get('NECTARCHAIN_LOG'),exist_ok = True)

#to quiet numba
logging.getLogger("numba").setLevel(logging.WARNING)


import argparse

#import seaborn as sns
from nectarchain.calibration.container import ChargeContainer
from nectarchain.calibration.NectarGain import NectarGainSPESingleSignalStd,NectarGainSPESingleSignal

parser = argparse.ArgumentParser(
                    prog = 'gain_SPEfit_computation.py',
                    description = 'compute high gain with SPE fit for one run at very very high voltage (~1400V) or at nominal voltage (it can often fail)')

#run numbers
parser.add_argument('-r', '--run_number',
                    help='spe run',
                    type=int)

parser.add_argument('--overwrite',
                    action='store_true',
                    default=False,
                    help='to force overwrite files on disk'
                    )
parser.add_argument('--reduced',
                    action='store_true',
                    default=False,
                    help='to use reduced run'
                    )
parser.add_argument('--voltage_tag', 
                    type = str,
                    default='',
                    help='tag for voltage specifcication (1400V or nominal)'
                    )

#pixels selected
parser.add_argument('-p','--pixels',
                    nargs="+",
                    default=None,
                    help='pixels selected',
                    type=int)

#for let free pp and n :
parser.add_argument('--free_pp_n',
                    action='store_true',
                    default=False,
                    help='to let free pp and n'
                    )

#multiprocessing args
parser.add_argument('--multiproc',
                    action='store_true',
                    default=False,
                    help='to use multiprocessing'
                    )
parser.add_argument('--nproc',
                    help='nproc used for multiprocessing',
                    type=int)
parser.add_argument('--chunksize',
                    help='chunksize used for multiprocessing',
                    type=int)

#extractor arguments
parser.add_argument('--chargeExtractorPath',
                    help='charge extractor path where charges are saved',
                    type=str
                    )

#verbosity argument
parser.add_argument('-v',"--verbosity",
                    help='0 for FATAL, 1 for WARNING, 2 for INFO and 3 for DEBUG',
                    default=0,
                    choices=[0,1,2,3],
                    type=int)


def main(args) : 
    figpath = f"{os.environ.get('NECTARCHAIN_FIGURES')}/"

    reduced = "_reduced" if args.reduced else ""
    multipath = "MULTI-" if args.multiproc else ""
    SPEpath = "SPE" if args.free_pp_n else "SPEStd"

    charge_run_1400V = ChargeContainer.from_file(f"{os.environ.get('NECTARCAMDATA')}/charges{reduced}/{args.chargeExtractorPath}/",args.run_number)

    if args.free_pp_n :
        gain_Std = NectarGainSPESingleSignal(signal = charge_run_1400V)

    else :
        gain_Std = NectarGainSPESingleSignalStd(signal = charge_run_1400V)
    t = time.time()
    gain_Std.run(pixel = args.pixels, multiproc = args.multiproc, nproc = args.nproc, chunksize = args.chunksize, figpath = figpath+f"/{multipath}{args.voltage_tag}-{SPEpath}-{args.run_number}-{args.chargeExtractorPath}")
    log.info(f"fit time =  {time.time() - t } sec")
    gain_Std.save(f"{os.environ.get('NECTARCAMDATA')}/../SPEfit/data{reduced}/{multipath}{args.voltage_tag}-{SPEpath}-{args.run_number}-{args.chargeExtractorPath}/",overwrite = args.overwrite)
    conv_rate = len(gain_Std._output_table[gain_Std._output_table['is_valid']])/gain_Std.npixels if args.pixels is None else len(gain_Std._output_table[gain_Std._output_table['is_valid']])/len(args.pixels)
    log.info(f"convergence rate : {conv_rate}")

if __name__ == "__main__":
    args = parser.parse_args()
    logginglevel = logging.FATAL
    if args.verbosity == 1 : 
        logginglevel = logging.WARNING
    elif args.verbosity == 2 : 
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

    main(args)