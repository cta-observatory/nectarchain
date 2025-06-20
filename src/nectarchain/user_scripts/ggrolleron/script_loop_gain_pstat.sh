#!/bin/bash
#SBATCH --job-name=gain_pstat
#SBATCH --output=gain_pstat_%j.log
#SBATCH --mem=32000                    # Mémoire en MB par défaut
#SBATCH --time=1-00:00:00             # Délai max = 7 jours

#SBATCH --cpus-per-task=1
export _QT_QPA_PLATFORM=$QT_QPA_PLATFORM
export QT_QPA_PLATFORM=offscreen 
for METHOD in "LocalPeakWindowSum" "GlobalPeakWindowSum"
do
    for WIDTH in 8 9 10 11 12 16
    do 
        export cmd="python gain_PhotoStat_computation.py --FF_run_number 3937 --Ped_run_number 3938 --SPE_run_number 3936 --SPE_config nominal --method $METHOD  --extractor_kwargs '{\"window_width\":$WIDTH,\"window_shift\":4}' --overwrite -v INFO" #--figpath /home/ggroller/projects/nectarchain/src/nectarchain/user_scripts/ggrolleron/local/figures"
        echo $cmd
        eval $cmd
    done   
done 
export QT_QPA_PLATFORM=$_QT_QPA_PLATFORM
unset _QT_QPA_PLATFORM
