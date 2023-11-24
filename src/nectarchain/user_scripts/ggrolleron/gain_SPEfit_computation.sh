#HOW TO : 

#HVV runs : 2634,3942
#nominal runs : 2633,3936

#to perform SPE fit of a run at nominal voltage
python gain_SPEfit_computation.py -r 3942 --reload_events --multiproc --nproc 8 --chunksize 2 -p 45 56 --max_events 100  --method LocalPeakWindowSum --extractor_kwargs '{"window_width":16,"window_shift":4}' --overwrite -v debug
#to perform SPE fit of a HHV run 
python gain_SPEfit_computation.py -r 3936 --reload_events --multiproc --nproc 8 --chunksize 2 -p 45 56 --max_events 100  --method LocalPeakWindowSum --extractor_kwargs '{"window_width":16,"window_shift":4}' --overwrite -v debug
#to perform SPE fit of a HHV run letting n and pp parameters free
python gain_SPEfit_computation.py -r 3942 --free_pp_n --reload_events --multiproc --nproc 8 --chunksize 2 -p 45 56 --max_events 100  --method LocalPeakWindowSum --extractor_kwargs '{"window_width":16,"window_shift":4}' --overwrite -v debug
