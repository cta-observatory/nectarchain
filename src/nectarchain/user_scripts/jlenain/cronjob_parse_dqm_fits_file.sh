#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# This script is to be used as a cronjob on the nectarcam-dqm-rw VM on the LPNHE OpenStack cloud platform, in order to feed the ZODB database from DQM run on DIRAC.

# Log everything to $LOGFILE
LOGFILE=${0%".sh"}_$(date +%F).log
LOGFILE=$HOME/log/$(basename $LOGFILE)
exec 1>"$LOGFILE" 2>&1

. "/opt/conda/etc/profile.d/conda.sh"
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

# Initialize DIRAC proxy from user certificate:
if ! dirac-proxy-init -M -g ctao_nectarcam --pwstdin < ~/.dirac.pwd; then
    echo "DIRAC proxy initialization failed..."
    exit 1
fi

remoteParentDir="/ctao/user/j/jlenain/nectarcam/dqm"
nectarchainScriptDir="/opt/cta/nectarchain/src/nectarchain/user_scripts/jlenain"

python ${nectarchainScriptDir}/parse_dqm_fits_file.py -c ${camera} -r $(dls ${remoteParentDir} | grep -ve "/ctao" | awk -F. '{print $1}' | awk -Fn '{print $2}' | tr '\n' ' ')
