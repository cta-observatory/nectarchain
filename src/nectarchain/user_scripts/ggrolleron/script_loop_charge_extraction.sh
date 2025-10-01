#!/bin/bash
# 3936 3937 3938 3942
#"FullWaveformSum" "LocalPeakWindowSum" "GlobalPeakWindowSum"
for METHOD in "FullWaveformSum" "LocalPeakWindowSum" "GlobalPeakWindowSum"
do
    if [ $METHOD = "FullWaveformSum" ];
    then
        export cmd="python load_wfs_compute_charge.py -r 3937 3938 3942 --events_per_slice 2000 --method $METHOD  --overwrite -v INFO" #--reload_wfs"
        echo $cmd
        eval $cmd
    else
        for WIDTH in 8 9 10 11 12 16
        do
            export cmd="python load_wfs_compute_charge.py -r 3937 3938 3942 --events_per_slice 2000 --method $METHOD --extractor_kwargs '{\"window_width\":$WIDTH,\"window_shift\":4}' --overwrite -v INFO"
            echo $cmd
            eval $cmd
        done
    fi
    #python load_wfs_compute_charge.py -r 3936 3942 --method LocalPeakWindowSum --extractor_kwargs "{'window_width':$WIDTH,'window_shift':4}" --overwrite -v INFO
done  
