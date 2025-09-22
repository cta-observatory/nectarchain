#!/bin/bash
#SBATCH --job-name=gain_nominal
#SBATCH --output=gain_nominal_%j.log
#SBATCH --mem=32000                    # Mémoire en MB par défaut
#SBATCH --time=1-00:00:00             # Délai max = 7 jours

#SBATCH --cpus-per-task=16

export _QT_QPA_PLATFORM=$QT_QPA_PLATFORM
export QT_QPA_PLATFORM=offscreen 
for METHOD in "LocalPeakWindowSum" "GlobalPeakWindowSum"
do
    for WIDTH in 8 9 10 11 12 16
    do
        if [ $WIDTH -eq 8 ]; then
            export cmd="python gain_SPEfit_computation.py -r 3936 --multiproc --nproc 16 --method $METHOD  --extractor_kwargs '{\"window_width\":$WIDTH,\"window_shift\":4}' --overwrite -v INFO" #--display --figpath /home/ggroller/projects/nectarchain/src/nectarchain/user_scripts/ggrolleron/local/figures"
            echo $cmd
            eval $cmd
        else
            export cmd="python gain_SPEfit_computation.py -r 3936 --multiproc --nproc 16 --method $METHOD  --extractor_kwargs '{\"window_width\":$WIDTH,\"window_shift\":4}' --overwrite -v INFO"
            echo $cmd
            eval $cmd
        fi
    done   
done 
export QT_QPA_PLATFORM=$_QT_QPA_PLATFORM
unset _QT_QPA_PLATFORM
