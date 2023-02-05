import logging
import sys
import os
from pathlib import Path
import pandas as pd
import time

#import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import getopt
os.makedirs(os.environ.get('NECTARCHAIN_LOG'),exist_ok = True)

#to quiet numba
logging.getLogger("numba").setLevel(logging.WARNING)

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',level=logging.INFO,filename = f"{os.environ.get('NECTARCHAIN_LOG')}/{Path(__file__).stem}_{os.getpid()}.log")
log = logging.getLogger(__name__)
##tips to add message to stdout
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

import argparse
import json

#import seaborn as sns
from nectarchain.calibration.container import ChargeContainer
from nectarchain.calibration.NectarGain import PhotoStatGainFFandPed

parser = argparse.ArgumentParser(
                    prog = 'gain_PhotoStat_computation.py',
                    description = 'compute high gain and low gain with Photo-statistic method, it need a pedestal run and a FF run with a SPE fit restuls (for resolution value needed in this method)')

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
                    help='SPE fit results tag for propagate the SPE result to output',
                    type=str
                    )

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

#for plotting correlation
parser.add_argument('--correlation',
                    action='store_true',
                    default=True,
                    help='to plot correlation between SPE gain computation and Photo-statistic gain resluts'
                    )


def main(args) : 
    figpath = os.environ.get('NECTARCHAIN_FIGURES')

    reduced = "_reduced" if args.reduced else ""

    photoStat_FFandPed = PhotoStatGainFFandPed(args.FF_run_number, args.ped_run_number, SPEresults = args.SPE_fit_results)    
    photoStat_FFandPed.run()
    photoStat_FFandPed.save(f"{os.environ.get('NECTARCAMDATA')}/../data{reduced}/PhotoStat-FF{args.FF_run_number}-ped{args.ped_run_number}-SPEres{args.SPE_fit_results_tag}/",overwrite = args.overwrite)

    if args.correlation : 
        fig = photoStat_FFandPed.plot_correlation()
        os.makedirs(f"{figpath}/PhotoStat-FF{args.FF_run_number}-ped{args.ped_run_number}{reduced}/",exist_ok=True)
        fig.savefig(f"{figpath}/PhotoStat-FF{args.FF_run_number}-ped{args.ped_run_number}{reduced}/correlation_PhotoStat_SPE{args.SPE_fit_results_tag}.pdf")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)