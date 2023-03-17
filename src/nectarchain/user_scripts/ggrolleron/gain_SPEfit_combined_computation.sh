#HOW TO : 
#to perform SPE fit of run 2633 (supposed to be taken at nominal voltage) from SPE fit performed on 2634 (taken at 1400V)
python gain_SPEfit_combined_computation.py -r 2633 --SPE_fit_results_tag 2634 --chargeExtractorPath LocalPeakWindowSum_4-12 --overwrite --multiproc --nproc 6 --chunksize 50 --same_luminosity --VVH_fitted_results "/data/users/ggroller/NECTARCAM/SPEfit/data/MULTI-1400V-SPEStd-2634-LocalPeakWindowSum_4-12/output_table.ecsv"

#to perform a joint SPE fit of run 2633 (supposed to be taken at nominal voltage) and run 2634 (1400V). 
python gain_SPEfit_combined_computation.py -r 2633 --SPE_fit_results_tag 2634 --chargeExtractorPath LocalPeakWindowSum_4-12 --overwrite --multiproc --nproc 6 --chunksize 50 --same_luminosity --combined

#to perform SPE fit of run 3936 (supposed to be taken at nominal voltage) from SPE fit performed on 2634 (taken at 1400V) (luminosity can not be supposed the same)
python gain_SPEfit_combined_computation.py -r 3936 --SPE_fit_results_tag 3942 --chargeExtractorPath LocalPeakWindowSum_4-12 --overwrite --multiproc --nproc 6 --chunksize 50 --VVH_fitted_results "/data/users/ggroller/NECTARCAM/SPEfit/data/MULTI-1400V-SPEStd-2634-LocalPeakWindowSum_4-12/output_table.ecsv"
