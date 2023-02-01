#HOW TO : 
#spe_run_number = [2633,2634,3784]
#ff_run_number = [2608]
#ped_run_number = [2609]
#spe_nevents = [49227,49148,-1]

python load_wfs_compute_charge.py -s 2633 2634 3784 -f 2608 -p 2609 --spe_nevents 49227 49148 -1 --extractorMethod LocalPeakWindowSum --extractor_kwargs '{"window_width":16,"window_shift":4}'