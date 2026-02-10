# HOW TO:
# In order to compute flat-field coefficients using the gain computed with
# photostatistincs method, one has to:
# 1) Run the SPEfit, example of bash script:
python gain_SPEfit_computation.py -r 6774 --reload_events --multiproc --nproc 8  --method LocalPeakWindowSum --extractor_kwargs '{"window_shift": 4, "window_width":8}' --overwrite -v INFO
# 2) Run the phototatistics method and the fit of the 2D Gaussian model to get
# FF-coefficients
python analysis_photostat_simple.py --FF_run_number 6729 --SPE_run_number 6774  --SPE_config nominal --method LocalPeakWindowSum --extractor_kwargs '{"window_shift": 4, "window_width":8}'