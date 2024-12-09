#HOW TO : 
#FF_spe_nominal : 2633 ,3936

#FF_spe_5deg : 3731
#FF_spe_-5deg : 3784
#FF_spe_0deg : 3750

#FF_spe_HHV : 2634, 3942

#WT_spe_nominal : 4129

#ff_run_number : 2608, 3937
#ped_run_number : 2609, 3938

#FF and Ped : 3943

python load_wfs_compute_charge.py -s 2633 2634 3784 -f 2608 -p 2609 --spe_nevents 49227 49148 -1 --method LocalPeakWindowSum --extractor_kwargs '{"window_width":16,"window_shift":4}'

python load_wfs_compute_charge.py -r 3942 --max_events 100  --method GlobalPeakWindowSum --extractor_kwargs '{"window_width":16,"window_shift":4}' --overwrite -v debug

