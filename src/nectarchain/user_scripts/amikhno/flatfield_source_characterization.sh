# HOW TO:
# Example to compute flat-field coefficients using the gain using the
# photostatistics method, including the fit of a 2D Gaussian model to account for a
# non-uniform response of the flat-field calibration source:
python analysis_photostat_simple.py --FF_run_number 6729 --SPE_run_number 6774  --SPE_config nominal --method LocalPeakWindowSum --extractor_kwargs '{"window_shift": 4, "window_width":8}'