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
from nectarchain.calibration.makers.gain.FlatFieldSPEMakers import FlatFieldSingleNominalSPEMaker

parser = argparse.ArgumentParser(
                    prog = 'gain_SPEfit_combined_computation.py',
                    description = 'compute high gain with SPE combined fit for one run at nominal voltage.  Output data will be saved in $NECTARCAMDATA/../SPEfit/data/{multipath}nominal-prefitCombinedSPE{args.SPE_fit_results_tag}-SPEStd-{args.run_number}-{args.chargeExtractorPath}/'
                    )

#run numbers
parser.add_argument('-r', '--run_number',
                    help='spe run',
                    type=int)

parser.add_argument('--overwrite',
                    action='store_true',
                    default=False,
                    help='to force overwrite files on disk'
                    )

#output figures and path extension
parser.add_argument('--display', 
                    action='store_true',
                    default=False,
                    help='whether to save plot or not'
                    )
parser.add_argument('--output_fig_tag', 
                    type = str,
                    default='',
                    help='tag to set output figure path'
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
                    help='chunksize used for multiprocessing, with multiproccesing, create one process per pixels is not optimal, we rather prefer to group quite a few pixels in same process, chunksize is used to set the number of pixels we use for one process, for example if you want to perform the gain computation of the whole camera with 1855 pixels on 6 CPU, a chunksize of 20 seems to be quite optimal  ',
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
                    help='if True : perform a combined fit of VVH and nominal data, if False : perform a nominal fit with SPE resoltion fixed from VVH fitted data'
                    )
parser.add_argument('--VVH_fitted_results',
                    help='previoulsy fitted VVH data path for nominal SPE fit by fixing some shared parameters',
                    type=str
                    )
#tag for SPE fit results propagation
parser.add_argument('--SPE_fit_results_tag',
                    help='SPE fit results tag for propagate the SPE result to output, this tag will be used to setup the path where output data will be saved, see help for description',
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
                    help='set the verbosity level of logger',
                    default="info",
                    choices=["fatal","debug","info","warning"],
                    type=str)


def main(args) : 
    figpath = f"{os.environ.get('NECTARCHAIN_FIGURES',f'/tmp/nectarchain_log/{os.getpid()}/figure')}/"
    figpath_ext = "" if args.output_fig_tag == "" else f"-{args.output_fig_tag}"


    multipath = "MULTI-" if args.multiproc else ""

    charge_run = ChargeContainer.from_file(f"{os.environ.get('NECTARCAMDATA')}/charges/{args.chargeExtractorPath}/",args.run_number)

    if args.combined : 
        raise NotImplementedError("combined fit not implemented yet")
    else : 
        gain_Std = FlatFieldSingleNominalSPEMaker.create_from_chargeContainer(signal = charge_run,
                                    nectarGainSPEresult=args.VVH_fitted_results,
                                    same_luminosity=args.same_luminosity
                                    )
        t = time.time()
        gain_Std.make(pixels_id = args.pixels, 
                  multiproc = args.multiproc,
                  display = args.display,
                  nproc = args.nproc, 
                  chunksize = args.chunksize, 
                  figpath = figpath+f"/{multipath}nominal-prefitCombinedSPE{args.SPE_fit_results_tag}-SPEStd-{args.run_number}-{args.chargeExtractorPath}{figpath_ext}"
                  )
        
        log.info(f"fit time =  {time.time() - t } sec")
        gain_Std.save(f"{os.environ.get('NECTARCAMDATA')}/../SPEfit/data/{multipath}nominal-prefitCombinedSPE{args.SPE_fit_results_tag}-SPEStd-{args.run_number}-{args.chargeExtractorPath}/",overwrite = args.overwrite)
        log.info(f"convergence rate : {len(gain_Std._results[gain_Std._results['is_valid']])/gain_Std.npixels}")

if __name__ == "__main__":
    args = parser.parse_args()
    logginglevel = logging.FATAL
    if args.verbosity == "warning" : 
        logginglevel = logging.WARNING
    elif args.verbosity == "info" : 
        logginglevel = logging.INFO
    elif args.verbosity == "debug" : 
        logginglevel = logging.DEBUG

    os.makedirs(f"{os.environ.get('NECTARCHAIN_LOG','/tmp/nectarchain_log')}/{os.getpid()}/figures")
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