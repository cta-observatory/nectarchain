#HOW TO : 

#to perform SPE fit of run 3936 (supposed to be taken at nominal voltage) from SPE fit performed on 3942 (taken at 1400V) (luminosity can not be supposed the same)
python gain_SPEfit_combined_computation.py -r 3936 --HHV_run_number 3942 --multiproc --nproc 8 --chunksize 2 -p 45 56 --max_events 100  --method LocalPeakWindowSum --extractor_kwargs '{"window_width":16,"window_shift":4}' --overwrite -v DEBUG --reload_events
