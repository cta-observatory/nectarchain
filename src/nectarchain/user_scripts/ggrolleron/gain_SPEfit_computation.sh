#HOW TO : 

#to perform SPE fit of a HHV run 
python gain_SPEfit_computation.py -r 2634 --display --voltage_tag "1400V" --chargeExtractorPath LocalPeakWindowSum_4-12 --overwrite --multiproc --nproc 6 --chunksize 2 -p 0 1 2 3 4 5 6 7 8 9 10 11 12

#to perform SPE fit of a HHV run letting n and pp parameters free
python gain_SPEfit_computation.py -r 2634 --display --voltage_tag "1400V" --free_pp_n --chargeExtractorPath LocalPeakWindowSum_4-12 --overwrite --multiproc --nproc 6 --chunksize 2 -p 0 1 2 3 4 5 6 7 8 9 10 11 12

#to perform SPE fit of a run at nominal voltage
python gain_SPEfit_computation.py -r 3936 --display --voltage_tag "nominal" --chargeExtractorPath LocalPeakWindowSum_4-12 --overwrite --multiproc --nproc 6 --chunksize 50

#to perform SPE fit of a HHV run 
python gain_SPEfit_computation.py -r 3942 --display --output_fig_tag "testMyRes" --voltage_tag "1400V" --chargeExtractorPath LocalPeakWindowSum_4-12 --multiproc --nproc 6 --chunksize 2 -p 0 1 2 3 4 5 6 7 8 9 10 11 12 --verbosity info --overwrite