#!/bin/bash
export _QT_QPA_PLATFORM=$QT_QPA_PLATFORM
export QT_QPA_PLATFORM=offscreen 
for METHOD in "LocalPeakWindowSum" "GlobalPeakWindowSum"
do
    for WIDTH in 8 9 10 12
    do
        if [ $WIDTH -eq 8 ]; then
            export cmd="python gain_SPEfit_computation.py -r 3936 --multiproc --nproc 12 --method $METHOD  --extractor_kwargs '{\"window_width\":$WIDTH,\"window_shift\":4}' --overwrite -v INFO --display --figpath /home/ggroller/projects/nectarchain/src/nectarchain/user_scripts/ggrolleron/local/figures"
            echo $cmd
            eval $cmd
        else
            export cmd="python gain_SPEfit_computation.py -r 3936 --multiproc --nproc 12 --method $METHOD  --extractor_kwargs '{\"window_width\":$WIDTH,\"window_shift\":4}' --overwrite -v INFO"
            echo $cmd
            eval $cmd
        fi
    done   
done 
export QT_QPA_PLATFORM=$_QT_QPA_PLATFORM
unset _QT_QPA_PLATFORM
