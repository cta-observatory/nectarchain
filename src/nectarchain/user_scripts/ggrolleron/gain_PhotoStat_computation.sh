#HOW TO :
#to perform photo-statistic high and low gain computation with pedestal run 2609, flat field run 2609 and SPE fit result from run 2634 (1400V run)
python gain_PhotoStat_computation.py -p 2609 -f 2608 --chargeExtractorPath LocalPeakWindowSum_4-12 --FFchargeExtractorWindowLength 16 --correlation --overwrite --SPE_fit_results "/data/users/ggroller/NECTARCAM/SPEfit/data/MULTI-1400V-SPEStd-2634-LocalPeakWindowSum_4-12/output_table.ecsv" --SPE_fit_results_tag VVH2634

python gain_PhotoStat_computation.py -p 3938 -f 3937 --chargeExtractorPath LocalPeakWindowSum_4-12 --FFchargeExtractorWindowLength 16 --correlation --overwrite --SPE_fit_results "/data/users/ggroller/NECTARCAM/SPEfit/data/MULTI-1400V-SPEStd-2634-LocalPeakWindowSum_4-12/output_table.ecsv" --SPE_fit_results_tag VVH2634
