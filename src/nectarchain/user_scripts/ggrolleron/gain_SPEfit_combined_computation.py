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
from nectarchain.calibration.NectarGain import NectarGainSPESingleSignalfromHHVFit

parser = argparse.ArgumentParser(
                    prog = 'gain_SPEfit_combined_computation.py',
                    description = 'compute high gain with SPE combined fit for one run at nominal voltage')

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

#pixels selected
parser.add_argument('-p','--pixels',
                    nargs="+",
                    default=None,
                    help='pixels selected',
                    type=int)


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

#for VVH combined fit
parser.add_argument('--combined',
                    action='store_true',
                    default=False,
                    help='to perform a combined fit of VVH and nominal data'
                    )
parser.add_argument('--VVH_fitted_results',
                    help='previoulsy fitted VVH data path for nominal SPE fit by fixing some shared parameters',
                    type=str
                    )
#tag for SPE fit results propagation
parser.add_argument('--SPE_fit_results_tag',
                    help='SPE fit results tag for propagate the SPE result to output',
                    type=str,
                    default=''
                    )
parser.add_argument('--same_luminosity',
                    action='store_true',
                    default=False,
                    help='if luminosity for VVH and nominal data is the same'
                    )

#verbosity argument
parser.add_argument('-v',"--verbosity",
                    help='0 for FATAL, 1 for WARNING, 2 for INFO and 3 for DEBUG',
                    default=0,
                    type=int)


def main(args) : 
    figpath = f"{os.environ.get('NECTARCHAIN_FIGURES')}/"

    reduced = "_reduced" if args.reduced else ""
    multipath = "MULTI-" if args.multiproc else ""

    charge_run = ChargeContainer.from_file(f"{os.environ.get('NECTARCAMDATA')}/charges{reduced}/{args.chargeExtractorPath}/",args.run_number)

    if args.combined : 
        raise NotImplementedError("combined fit not implemented yet")
    else : 
        gain_Std = NectarGainSPESingleSignalfromHHVFit(signal = charge_run,
                                    nectarGainSPEresult=args.VVH_fitted_results,
                                    same_luminosity=args.same_luminosity
                                    )
        t = time.time()
        gain_Std.run(pixel = args.pixels, multiproc = args.multiproc, nproc = args.nproc, chunksize = args.chunksize, figpath = figpath+f"/{multipath}nominal-prefitCombinedSPE{args.SPE_fit_results_tag}-SPEStd-{args.run_number}-{args.chargeExtractorPath}")
        log.info(f"fit time =  {time.time() - t } sec")
        gain_Std.save(f"{os.environ.get('NECTARCAMDATA')}/../SPEfit/data{reduced}/{multipath}nominal-prefitCombinedSPE{args.SPE_fit_results_tag}-SPEStd-{args.run_number}-{args.chargeExtractorPath}/",overwrite = args.overwrite)
        log.info(f"convergence rate : {len(gain_Std._output_table[gain_Std._output_table['is_valid']])/gain_Std.npixels}")

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