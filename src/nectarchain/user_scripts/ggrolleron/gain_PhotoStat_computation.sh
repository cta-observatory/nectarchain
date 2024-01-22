#HOW TO :
#to perform photo-statistic high and low gain computation with pedestal run 2609, flat field run 2609 and SPE fit result from run 2634 (1400V run)
python gain_PhotoStat_computation.py --FF_run_number 3937 --Ped_run_number 3938 --HHV_run_number 3942 --max_events 100  --method LocalPeakWindowSum --extractor_kwargs '{"window_width":16,"window_shift":4}' --overwrite -v INFO --reload_events

