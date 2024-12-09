#!/bin/bash
export _QT_QPA_PLATFORM=$QT_QPA_PLATFORM
export QT_QPA_PLATFORM=offscreen 
for METHOD in "LocalPeakWindowSum" "GlobalPeakWindowSum"
do
    for WIDTH in 8 9 10 11 12 16
    do 
        export cmd="python gain_PhotoStat_computation.py --FF_run_number 3937 --Ped_run_number 3938 --HHV_run_number 3936 --method $METHOD  --extractor_kwargs '{\"window_width\":$WIDTH,\"window_shift\":4}' --overwrite -v INFO --figpath /home/ggroller/projects/nectarchain/src/nectarchain/user_scripts/ggrolleron/local/figures"
        echo $cmd
        eval $cmd
    done   
done 
export QT_QPA_PLATFORM=$_QT_QPA_PLATFORM
unset _QT_QPA_PLATFORM
