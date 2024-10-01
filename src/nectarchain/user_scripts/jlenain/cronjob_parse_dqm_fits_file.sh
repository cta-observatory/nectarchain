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

remoteParentDir="/vo.cta.in2p3.fr/user/j/jlenain/nectarcam/dqm"
nectarchainScriptDir="/opt/cta/nectarchain/src/nectarchain/user_scripts/jlenain"

python ${nectarchainScriptDir}/parse_dqm_fits_file.py -r $(dls ${remoteParentDir} | grep -ve "/vo.cta" | awk -F. '{print $1}' | awk -Fn '{print $2}' | tr '\n' ' ')
