#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# Author: Jean-Philippe Lenain <jlenain@in2p3.fr>
#
# Script as a cronjob to dynamically launch NectarCAM DQM runs on DIRAC after data transfer, to be run once a day on sedipcaa23 in CEA/Irfu.

# Log everything to $LOGFILE
LOGFILE=${0%".sh"}_$(date +%F)_$$.log
LOGFILE=$HOME/log/$(basename $LOGFILE)
exec 1>"$LOGFILE" 2>&1

source /opt/cta/mambaforge/etc/profile.d/conda.sh
conda activate nectar-dev

function usage ()
{
    echo "Usage: $(basename "$0") -c <camera>"
}

function help ()
{
    usage
    cat <<EOF

This script dynamically launch NectarCAM DQM runs on DIRAC after data transfer.

OPTIONS:
     -h                       This help message.
     -c <CAMERA>              NectarCAM camera
EOF
}

# Get options
while getopts ":hc:" option; do
   case $option in
      h) # display help
          help
	  exit;;
      c) # camera
         camera=$OPTARG;;
     \?) # Invalid option
         usage
	 exit 1;;
   esac
done
shift $((OPTIND-1))

if [ -z "$camera" ]; then
    usage
    exit 1
fi


localParentDir="/data/nvme/${camera}/ZFITS"
# remoteParentDir="/vo.cta.in2p3.fr/nectarcam"
remoteParentDir="/ctao/nectarcam/${camera}"
nectarchainScriptDir="$HOME/local/src/python/cta-observatory/nectarchain/src/nectarchain/user_scripts/jlenain/dqm_job_submitter"

cd $nectarchainScriptDir || (echo "Failed to cd into ${nectarchainScriptDir}, exiting..."; exit 1)

for run in $(find ${localParentDir} -type f -name "NectarCAM*.fits.fz" | awk -F. '{print $2}' | awk -Fn '{print $2}' | sort | uniq); do
    echo "Probing files for run ${run}"
    nbLocalFiles=$(find ${localParentDir} -type f -name "NectarCAM.Run${run}.????.fits.fz" | wc -l)
    echo "  Found $nbLocalFiles local files for run $run"
    nbRemoteFiles=$(dfind ${remoteParentDir} | grep -e "NectarCAM.Run${run}" | grep --count -e "fits.fz")
    echo "  Found $nbRemoteFiles remote files on DIRAC for run $run"
    # If number of local and remote files matching, will attempt to launch a DQM run
    if [ ${nbLocalFiles} -eq ${nbRemoteFiles} ]; then
        echo "  Run $run: number of local and remote files matching, will attempt to submit a DQM job"
        # Has this DQM run already been submitted ?
        if [ $(dstat | grep --count -e "NectarCAM DQM run ${run}") -eq 0 ]; then
            yyyymmdd=$(find ${localParentDir} -type f -name "NectarCAM.Run${run}.????.fits.fz" | head -n 1 | awk -F/ '{print $7}')
            yyyy=${yyyymmdd:0:4}
            mm=${yyyymmdd:4:2}
            dd=${yyyymmdd:6:2}
            cmd="python submit_dqm_processor.py -d "${yyyy}-${mm}-${dd}" -r $run -c $camera"
            echo "Running: $cmd"
            eval $cmd
        else
            echo "  DQM job for run $run already submitted, either ongoing or failed, skipping it."
        fi
    else
        echo "  Run $run is not yet complete on DIRAC, will wait another day before launching a DQM job on it."
    fi
done
