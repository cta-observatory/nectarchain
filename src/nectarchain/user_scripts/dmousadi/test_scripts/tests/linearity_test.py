#don't forget to set environment variable NECTARCAMDATA

import numpy as np
import pathlib
import os
import sys
import matplotlib.pyplot as plt

from utils import filters, adc_to_pe, optical_density_390ns, trasmission_390ns, fit_function, err_ratio, err_sum, plot_parameters
from lmfit.models import Model
from test_tools_components import LinearityTestTool
import argparse
import pickle


def get_args():
    """
    Parses command-line arguments for the linearity test script.
    
    Returns:
        argparse.ArgumentParser: The parsed command-line arguments.
    """
        
    parser = argparse.ArgumentParser(description='Linearity test B-TEL-1390 & Intensity resolution B-TEL-1010. \n'
                                    +'According to the nectarchain component interface, you have to set a NECTARCAMDATA environment variable in the folder where you have the data from your runs or where you want them to be downloaded.\n'
                                    +'You have to give a list of runs (run numbers with spaces inbetween), a corresponding transmission list and an output directory to save the final plot.\n'
                                    +'If the data is not in NECTARCAMDATA, the files will be downloaded through DIRAC.\n For the purposes of testing this script, default data is from the runs used for this test in the TRR document.\n'
                                    +'You can optionally specify the number of events to be processed (default 500) and the number of pixels used (default 70).\n')
    parser.add_argument('-r','--runlist', type=int,nargs='+', help='List of runs (numbers separated by space)', required=False, default = [i for i in range (3404,3424)]+[i for i in range(3435,3444)])
    parser.add_argument('-t','--trans', type=float,nargs='+', help='List of corresponding transmission for each run', required=False , default = trasmission_390ns)
    parser.add_argument('-e','--evts', type = int, help='Number of events to process from each run. Default is 500', required=False, default=500)
    parser.add_argument('-o','--output', type=str, help='Output directory. If none, plot will be saved in current directory', required=False, default='./')
    parser.add_argument("--temp_output", help="Temporary output directory for GUI", default=None)
    
    return parser



def main():
    """
    The `main()` function is the entry point of the linearity test script. It parses the command-line arguments, processes the specified runs, and generates plots to visualize the linearity and charge resolution of the detector. The function performs the following key steps:

    1. Parses the command-line arguments using the `get_args()` function, which sets up the argument parser and handles the input parameters.
    2. Iterates through the specified run list, processing each run using the `LinearityTestTool` class. This tool initializes, sets up, starts, and finishes the processing for each run, returning the relevant output data.
    3. Normalizes the high-gain and low-gain charge values using the charge value at 0.01 transmission.
    4. Generates three subplots:
    - The first subplot shows the estimated charge vs. the true charge, with the fitted linear function for both high-gain and low-gain channels.
    - The second subplot shows the residuals between the estimated and true charge, as a percentage.
    - The third subplot shows the ratio of high-gain to low-gain charge, with a fitted linear function.
    5. Saves the generated plots to the specified output directory, and optionally saves temporary plot files for a GUI.
    6. Generates an additional plot to visualize the charge resolution, including the statistical limit.
    7. Saves the charge resolution plot to the specified output directory, and optionally saves a temporary plot file for a GUI.
    """
    parser = get_args()
    args = parser.parse_args()




    runlist = args.runlist
    transmission=args.trans #corresponding transmission for above data

    nevents = args.evts
    
    output_dir = os.path.abspath(args.output)
    temp_output= os.path.abspath(args.temp_output) if args.temp_output else None

    print(f"Output directory: {output_dir}")  # Debug print
    print(f"Temporary output file: {temp_output}")  # Debug print


    sys.argv = sys.argv[:1]

    #runlist = [3441]

    charge = np.zeros((len(runlist),2))
    std = np.zeros((len(runlist),2))
    std_err = np.zeros((len(runlist),2))

    index = 0
    for run in runlist:
        print("PROCESSING RUN {}".format(run))
        tool = LinearityTestTool(
            progress_bar=True, run_number=run, events_per_slice = 999, max_events=nevents,  log_level=20, window_width=14, overwrite=True
            
        )
        tool.initialize()
        tool.setup()
        tool.start()
        output = tool.finish()
        
        charge[index], std[index], std_err[index], npixels = output
        index += 1


    #print("FINAL",charge)

    #we assume that they overlap at 0.01 so they should have the same value
    #normalise high gain and low gain using charge value at 0.01
    transmission=np.array(transmission)
    norm_factor_hg = charge[np.argwhere((transmission<1.1e-2) & (transmission>9e-3)),0][0]
    #print(norm_factor_hg)
    norm_factor_lg = charge[np.argwhere((transmission<1.1e-2) & (transmission>9e-3)),1][0]
    #print(norm_factor_lg)
    charge_norm_hg = charge[:,0]/norm_factor_hg
    charge_norm_lg = charge[:,1]/norm_factor_lg
    std_norm_hg = std[:,0]/norm_factor_hg
    std_norm_lg = std[:,1]/norm_factor_lg

    #true charge is transmission as percentage of hg charge at 0.01
    true_charge = transmission/0.01  * norm_factor_hg



    fig, axs = plt.subplots(3, 1,  sharex='col', sharey='row', figsize=(10,11), gridspec_kw={'height_ratios': [4,2,2]})
    axs[0].grid(True, which="both")
    axs[0].set_ylabel("Estimated charge [p.e.]")
    axs[0].set_yscale('log')
    axs[0].set_xscale('log')
    axs[0].axvspan(10,1000,alpha=0.2,color="orange")
    axs[1].grid(True, which="both")
    axs[1].set_ylabel("Residuals [%]")
    axs[1].set_xscale('log')
    axs[1].set_ylim((-100,100))
    axs[1].axvspan(10,1000,alpha=0.2,color="orange")
    axs[2].grid(True, which="both")
    axs[2].set_ylabel("HG/LG ratio")
    axs[2].set_yscale('log')
    axs[2].set_xscale('log')
    axs[2].set_ylim((0.5,20))
    axs[2].axvspan(10,1000,alpha=0.2,color="orange")
    axs[2].set_xlabel("Illumination charge [p.e.]")






    for channel, (channel_charge, channel_std, name) in enumerate(zip([charge_norm_hg,charge_norm_lg],[std_norm_hg,std_norm_lg],["High Gain","Low Gain"])):
        yx = zip(true_charge,channel_charge,channel_std,runlist) #sort by true charge 
    
        ch_sorted = np.array(sorted(yx))
        #print(ch_sorted)
        
        #linearity
        model = Model(fit_function)
        params = model.make_params(a=100, b=0) 
        true = ch_sorted[:,0]
        
        ch_charge = ch_sorted[:,1] * norm_factor_hg[0]
        ch_std = ch_sorted[:,2] * norm_factor_hg[0]
        ch_err = ch_std/np.sqrt(npixels)

        
        ch_fit = model.fit(ch_charge[plot_parameters[name]["linearity_range"][0]:plot_parameters[name]["linearity_range"][1]], params, 
                        weights=1/ch_err[plot_parameters[name]["linearity_range"][0]:plot_parameters[name]["linearity_range"][1]], 
                        x=true[plot_parameters[name]["linearity_range"][0]:plot_parameters[name]["linearity_range"][1]])
        
        #print(ch_fit.fit_report())

    
        a = ch_fit.params['a'].value
        b = ch_fit.params['b'].value
        a_err = ch_fit.params['a'].stderr
        b_err = ch_fit.params['b'].stderr

        

        
        axs[0].errorbar(true,ch_charge,yerr = ch_err, label = name, ls='',marker='o',color = plot_parameters[name]["color"])
        axs[0].plot(true[plot_parameters[name]["linearity_range"][0]:plot_parameters[name]["linearity_range"][1]],
                    fit_function(true[plot_parameters[name]["linearity_range"][0]:plot_parameters[name]["linearity_range"][1]],a,b),color = plot_parameters[name]["color"])
        

        
        
        axs[0].text(plot_parameters[name]["label_coords"][0], plot_parameters[name]["label_coords"][0], name,
            va='top', fontsize=20, color=plot_parameters[name]["color"], alpha=0.9)
        axs[0].text(plot_parameters[name]["text_coords"][0], plot_parameters[name]["text_coords"][1], 'y = ax + b\n'
            + r'a$_{\rm %s}=%2.2f \pm %2.2f$ '%(plot_parameters[name]["initials"],round(a,2), 
                                                round(a_err,2))
            + '\n' + r'b$_{\rm %s}=(%2.2f \pm %2.2f) $ p.e.' %(plot_parameters[name]["initials"],round(b,2),
                                                        round(b_err,2)) 
            + "\n" r"$\chi_{\mathrm{%s}}^2/{\mathrm{dof}} =%2.1f/%d$"  %(plot_parameters[name]["initials"],round(ch_fit.chisqr,1), 
                                                                        int(ch_fit.chisqr/ch_fit.redchi)),
            backgroundcolor='white', bbox=dict(facecolor='white', edgecolor=plot_parameters[name]["color"], lw=3, 
                                                boxstyle='round,pad=0.5'),
            ha='left', va='top', fontsize=15, color='k', alpha=0.9)

        #residuals
        
        pred = fit_function(true,a,b) #prediction
        pred_err = a_err*true + b_err #a,b uncorrelated

        resid = (ch_charge - pred)/ch_charge
    
        numerator_err = err_sum(ch_err,pred_err)
        
        resid_err = err_ratio((ch_charge-pred),ch_charge,numerator_err,ch_std)  

        axs[1].errorbar(true,resid*100, yerr=abs(resid_err)*100, ls='', marker = 'o',color = plot_parameters[name]["color"]) #percentage
        axs[1].axhline(8, color='k', alpha=0.4, ls='--', lw=2)
        axs[1].axhline(-8, color='k', alpha=0.4, ls='--', lw=2)


    #hg/lg ratio
    ratio = charge[:,0]/charge[:,1]
    ratio_std = err_ratio(charge[:,0],charge[:,1],std[:,0],std[:,1])

    yx = zip(true_charge,ratio,ratio_std) #sort by true charge 
    ratio_sorted = np.array(sorted(yx))
    true = ratio_sorted[:,0]
    ratio = ratio_sorted[:,1]
    ratio_std = ratio_sorted[:,2]

    model = model = Model(fit_function)
    params = model.make_params(a=100, b=0) 
    ratio_fit = model.fit(ratio[10:-4], params, weights=1/ratio_std[10:-4], x=true[10:-4])


    axs[2].set_ylabel("hg/lg")
    axs[2].set_xlabel("charge(p.e.)")
    axs[2].errorbar(true,ratio,yerr=ratio_std,ls='',color='C1',marker = 'o')
    axs[2].plot(true[10:-4],fit_function(true[10:-4], ratio_fit.params['a'].value, ratio_fit.params['b'].value), ls='-',color='C1', alpha=0.9)
    axs[2].text(8, 1, 'y = ax + b\n'
            + r"$a_{\mathrm{HG/LG}}= (%2.1f\pm%2.1f)$ p.e.$^{-1}$ " %(round(ratio_fit.params['a'].value,1), 
                                                        round(ratio_fit.params['a'].stderr,1))
            + "\n" r"$b_{\mathrm{HG/LG}}=%2.1f\pm%2.1f$" %(round(ratio_fit.params['b'].value,1), 
                                                        round(ratio_fit.params['b'].stderr,1))
            + "\n" r"$\chi_{\mathrm{HG/LG}}^2/{\mathrm{dof}} =%2.1f/%d$"  %(round(ratio_fit.chisqr,1), 
                                                                        int(ratio_fit.chisqr/ratio_fit.redchi)),
            
            backgroundcolor='white', bbox=dict(facecolor='white', edgecolor='C1', lw=3, 
                                                boxstyle='round,pad=0.5'),
            ha='left', va='bottom', fontsize=11, color='k', alpha=0.9)




    plt.savefig(os.path.join(output_dir,"linearity_test.png"))
    if temp_output:
        with open(os.path.join(args.temp_output, 'plot1.pkl'), 'wb') as f:
            pickle.dump(fig, f)

    



    #charge resolution
    charge_hg = charge[:,0]
    std_hg = std[:,0]
    std_hg_err = std_err[:,0]
    charge_lg = charge[:,1]
    std_lg = std[:,1]
    std_lg_err = std_err[:,1]

    fig = plt.figure()
    charge_plot = np.linspace(3e-2,1e4)   
    stat_limit = 1/np.sqrt(charge_plot)   #statistical limit 

    hg_res = (std_hg/charge_hg)
    hg_res_err = err_ratio(std_hg,charge_hg,std_hg_err,std_hg)

    lg_res = (std_lg/charge_lg)
    lg_res_err = err_ratio(std_lg,charge_lg,std_lg_err,std_lg)
    plt.xscale(u'log')
    plt.yscale(u'log')
    plt.plot(charge_plot,stat_limit,color='gray', ls='-', lw=3,alpha=0.8,label='Statistical limit ')
    mask = charge_hg<3e2

    plt.errorbar(charge_hg[mask], hg_res[mask], 
                xerr=std_hg[mask]/np.sqrt(npixels),yerr=hg_res_err[mask]/np.sqrt(npixels),
                ls = '', marker ='o', label=None,color='C0')

    mask = np.invert(mask)
    plt.errorbar(charge_lg[mask]*ratio_fit.params['b'].value,lg_res[mask],
                xerr=std_lg[mask]/np.sqrt(npixels),yerr=lg_res_err[mask]/np.sqrt(npixels), ls = '', marker ='o', label=None,color='C0')

    plt.xlabel(r'Charge $\overline{Q}$ [p.e.]')
    plt.ylabel(r'Charge resolution $\frac{\sigma_{Q}}{\overline{Q}}$')

    plt.xlim(3e-2,4e3)
    plt.legend(frameon=False)
    plt.savefig(os.path.join(output_dir,"charge_resolution.png"))
    if temp_output:
        with open(os.path.join(args.temp_output, 'plot2.pkl'), 'wb') as f:
            pickle.dump(fig, f)
    plt.close('all')


if __name__ == "__main__":
    main()