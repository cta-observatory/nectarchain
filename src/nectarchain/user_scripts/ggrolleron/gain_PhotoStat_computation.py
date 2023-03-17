import logging
import sys
import os
from pathlib import Path
import numpy as np

os.makedirs(os.environ.get('NECTARCHAIN_LOG'),exist_ok = True)

#to quiet numba
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
logging.getLogger("numba").setLevel(logging.WARNING)

import argparse

from nectarchain.calibration.NectarGain import PhotoStatGainFFandPed

parser = argparse.ArgumentParser(
                    prog = 'gain_PhotoStat_computation.py',
                    description = 'compute high gain and low gain with Photo-statistic method, it need a pedestal run and a FF run with a SPE fit results (for resolution value needed in this method). Output data will be saved in $NECTARCAMDATA/../PhotoStat/data/PhotoStat-FF{FF_run_number}-ped{ped_run_number}-SPEres{SPE_fit_results_tag}-{chargeExtractorPath}/'
                    )

#run numbers
parser.add_argument('-p', '--ped_run_number',
                    help='ped run',
                    required=True,
                    type=int)
parser.add_argument('-f', '--FF_run_number',
                    help='FF run',
                    required=True,
                    type=int)
parser.add_argument('--SPE_fit_results',
                    help='SPE fit results path for accessing SPE resolution',
                    type=str,
                    required=True
                    )

#tag for SPE fit results propagation
parser.add_argument('--SPE_fit_results_tag',
                    help='SPE fit results tag for propagate the SPE result to output, this tag will be used to setup the path where output data will be saved, see help for description',
                    type=str,
                    default=''
                    )

parser.add_argument('--overwrite',
                    action='store_true',
                    default=False,
                    help='to force overwrite files on disk'
                    )

#for plotting correlation
parser.add_argument('--correlation',
                    action='store_true',
                    default=True,
                    help='to plot correlation between SPE gain computation and Photo-statistic gain resluts'
                    )

#extractor arguments
parser.add_argument('--chargeExtractorPath',
                    help='charge extractor path where charges are saved',
                    type=str
                    )

parser.add_argument('--FFchargeExtractorWindowLength',
                    help='charge extractor window length in ns',
                    type=int
                    )

#verbosity argument
parser.add_argument('-v',"--verbosity",
                    help='set the verbosity level of logger',
                    default="info",
                    choices=["fatal","debug","info","warning"],
                    type=str)


def main(args) : 
    figpath = os.environ.get('NECTARCHAIN_FIGURES')


    photoStat_FFandPed = PhotoStatGainFFandPed(args.FF_run_number, args.ped_run_number, SPEresults = args.SPE_fit_results,method = args.chargeExtractorPath, FFchargeExtractorWindowLength = args.FFchargeExtractorWindowLength)    
    photoStat_FFandPed.run()
    photoStat_FFandPed.save(f"{os.environ.get('NECTARCAMDATA')}/../PhotoStat/data/PhotoStat-FF{args.FF_run_number}-ped{args.ped_run_number}-SPEres{args.SPE_fit_results_tag}-{args.chargeExtractorPath}/",overwrite = args.overwrite)
    log.info(f"BF^2 HG : {np.power(np.mean(photoStat_FFandPed.BHG),2)}")
    log.info(f"BF^2 LG : {np.power(np.mean(photoStat_FFandPed.BLG),2)}")

    if args.correlation : 
        fig = photoStat_FFandPed.plot_correlation()
        os.makedirs(f"{figpath}/PhotoStat-FF{args.FF_run_number}-ped{args.ped_run_number}-{args.chargeExtractorPath}/",exist_ok=True)
        fig.savefig(f"{figpath}/PhotoStat-FF{args.FF_run_number}-ped{args.ped_run_number}-{args.chargeExtractorPath}/correlation_PhotoStat_SPE{args.SPE_fit_results_tag}.pdf")

if __name__ == "__main__":
    args = parser.parse_args()
    logginglevel = logging.FATAL
    if args.verbosity == "warning" : 
        logginglevel = logging.WARNING
    elif args.verbosity == "info" : 
        logginglevel = logging.INFO
    elif args.verbosity == "debug" : 
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