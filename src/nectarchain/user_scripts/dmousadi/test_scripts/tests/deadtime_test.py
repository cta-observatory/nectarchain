#don't forget to set environment variable NECTARCAMDATA

import numpy as np
import pathlib
import os
import sys

import matplotlib.pyplot as plt

from utils import source_ids_deadtime,deadtime_labels, pois, deadtime_and_expo_fit, err_ratio

import argparse
from iminuit import Minuit

import matplotlib.pyplot as plt
import numpy as np
from test_tools_components import DeadtimeTestTool
from astropy import units as u
from utils import ExponentialFitter
from iminuit import Minuit
import pickle

def get_args():
    """
    Parses command-line arguments for the deadtime test script.
    
    Returns:
        argparse.ArgumentParser: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Deadtime tests B-TEL-1260 & B-TEL-1270. \n'
                                    +'According to the nectarchain component interface, you have to set a NECTARCAMDATA environment variable in the folder where you have the data from your runs or where you want them to be downloaded.\n'
                                    +'You have to give a list of runs (run numbers with spaces inbetween), a corresponding source list and an output directory to save the final plot.\n'
                                    +'If the data is not in NECTARCAMDATA, the files will be downloaded through DIRAC.\n For the purposes of testing this script, default data is from the runs used for this test in the TRR document.\n'
                                    +'You can optionally specify the number of events to be processed (default 1000).\n')
    parser.add_argument('-r','--runlist', type=int,nargs='+', help='List of runs (numbers separated by space)', required=False, default = [i for i in range(3332,3350)]+[i for i in range(3552,3562)])
    parser.add_argument('-s','--source', type=int, choices = [0,1,2], nargs='+', help='List of corresponding source for each run: 0 for random generator, 1 for nsb source, 2 for laser', required=False , default = source_ids_deadtime)
    parser.add_argument('-e','--evts', type = int, help='Number of events to process from each run. Default is 1000', required=False, default=1000)
    parser.add_argument('-o','--output', type=str, help='Output directory. If none, plot will be saved in current directory', required=False, default='./')
    parser.add_argument("--temp_output", help="Temporary output directory for GUI", default=None)

    
    return parser     





def main():
    """
    Runs the deadtime test script, which performs deadtime tests B-TEL-1260 and B-TEL-1270.

    The script takes command-line arguments to specify the list of runs, corresponding sources, number of events to process, and output directory. It then processes the data for each run, performs an exponential fit to the deadtime distribution, and generates two plots:

    1. A plot of deadtime percentage vs. collected trigger rate, with the CTA requirement indicated.
    2. A plot of the rate from the fit vs. the collected trigger rate, with the relative difference shown in the bottom panel.

    The script also saves the generated plots to the specified output directory, and optionally saves them to a temporary output directory for use in a GUI.
    """

    parser = get_args()
    args = parser.parse_args()

    runlist = args.runlist
    ids = args.source

    nevents = args.evts

    output_dir = os.path.abspath(args.output)
    temp_output = os.path.abspath(args.temp_output) if args.temp_output else None

    print(f"Output directory: {output_dir}")  # Debug print
    print(f"Temporary output file: {temp_output}")  # Debug print


    sys.argv = sys.argv[:1]

    # ucts_timestamps = np.zeros((len(runlist),nevents))
    # delta_t = np.zeros((len(runlist),nevents-1))
    # event_counter = np.zeros((len(runlist),nevents))
    # busy_counter = np.zeros((len(runlist),nevents))
    # collected_triger_rate = np.zeros(len(runlist))
    # time_tot = np.zeros(len(runlist))
    # deadtime_us=np.zeros((len(runlist),nevents-1))
    # deadtime_pc = np.zeros(len(runlist))

    ucts_timestamps = []
    delta_t = []
    event_counter = []
    busy_counter = []
    collected_trigger_rate = []
    time_tot = []
    deadtime_us = []
    deadtime_pc = []

    labels = deadtime_labels


    for i,run in enumerate(runlist):
        
        print("PROCESSING RUN {}".format(run))
        tool = DeadtimeTestTool(
            progress_bar=True, run_number=run, max_events=nevents, events_per_slice = 10000, log_level=20, peak_height=10, window_width=16, overwrite=True
        )
        tool.initialize()
        tool.setup()
        tool.start()
        output=tool.finish()
        ucts_timestamps.append(output[0])
        delta_t.append(output[1])
        event_counter.append(output[2])
        busy_counter.append(output[3])
        collected_trigger_rate.append(output[4].value)
        time_tot.append(output[5].value)
        deadtime_pc.append(output[6])

        deadtime_us.append((delta_t[i]*u.ns).to(u.us))
        

    collected_trigger_rate=np.array(collected_trigger_rate)
    deadtime_pc = np.array(deadtime_pc)


    parameter_A2_new_list = []
    parameter_lambda_new_list = []
    parameter_tau_new_list = []
    parameter_A2_err_new_list = []
    parameter_lambda_err_new_list = []
    parameter_tau_err_new_list = []
    parameter_R2_new_list = []


    fitted_rate = []
    for i in range(len(runlist)): 
        print("fitting rate for run", runlist[i])
        
        dt_mus = deadtime_us[i].value
        dt_mus = dt_mus[dt_mus>0]
        

        lim_sup_mus = 120
        lim_sup_s = lim_sup_mus*1e-6
        nr_bins=500
        
        rate_initial_guess = 40000
        
        data_content, bin_edges = np.histogram(dt_mus*1e-6,bins=np.linspace(0.001e-6,lim_sup_s,nr_bins))
        
        init_param = [np.sum(data_content),0.6e-6,1./rate_initial_guess]
        
        
        fitter = ExponentialFitter(data_content,bin_edges=bin_edges)
        m = Minuit( fitter.compute_minus2loglike, init_param, name=('Norm','deadtime','1/Rate') )
        
        # Set Parameter Limits and tolerance

        m.errors["Norm"] = 0.3*init_param[0]
        m.limits["Norm"] = (0.,None)

        m.errors["deadtime"] = 0.1e-6
        m.limits["deadtime"] = ( 0.6e-6,1.1e-6) # Put some tigh constrain as the fit will be in trouble when it expect 0. and measured something instead.

        m.print_level = 2

        
        
        res = m.migrad(2000000)
        
        
        
        fitted_params = np.array( [res.params[p].value for p in res.parameters] )
        #print(fitted_params)

        fitted_params_err = np.array( [res.params[p].error for p in res.parameters] )
        #print(fitted_params_err)

        print(f"Dead-Time is {1.e6*fitted_params[1]:.3f} +- {1.e6*fitted_params_err[1]:.3f} µs")
        print(f"Rate is {1./fitted_params[2]:.2f} +- {fitted_params_err[2]/(fitted_params[2]**2):.2f} Hz")
        print(f"Expected run duration is {fitted_params[0]*fitted_params[2]:.2f} s")

        fitted_rate.append(1./fitted_params[2])

    #     plt.savefig(figurepath + 'deadtime_exponential_fit_nsb_run{}_newfit_cutoff.png'.format(run))

        
        y = data_content
        y_fit = fitter.expected_distribution(fitted_params)
        # residual sum of squares
        ss_res = np.sum((y - y_fit) ** 2)

        # total sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        # r-squared
        r2 = 1 - (ss_res / ss_tot)
        #print(r2)

        parameter_A2_new_list.append(fitted_params[0])
        parameter_lambda_new_list.append(1./fitted_params[2]/1e3) #kHz
        parameter_tau_new_list.append(1.e6*fitted_params[1]) #musec
        parameter_A2_err_new_list.append(fitted_params_err[0])
        parameter_lambda_err_new_list.append(fitted_params_err[2]/(fitted_params[2]**2)/1e3)
        parameter_tau_err_new_list.append(1.e6*fitted_params_err[1])
        
        parameter_R2_new_list.append(r2)


        
        
    deadtime_from_fit = (parameter_tau_new_list)
    deadtime_from_fit_err = (parameter_tau_err_new_list)
    lambda_from_fit = (parameter_lambda_new_list)
    lambda_from_fit_err = (parameter_lambda_err_new_list)
    A2_from_fit = (parameter_A2_new_list)
    A2_from_fit_err = (parameter_A2_err_new_list)
    R2_from_fit = (parameter_R2_new_list)



    #######################################
    #PLOT
    # print(event_counter)
    # print(busy_counter)
    # print(collected_triger_rate)
    # print(deadtime_pc)
    # print(busy_counter[:,-1])
    # print(event_counter[:,-1])
    # print(busy_counter[:,-1]/(event_counter[:,-1]+busy_counter[:,-1]))
    plt.clf()
    fig, ax = plt.subplots(figsize=(10*1.1,10*1.1/1.61)) # constrained_layout=True)
    ids = np.array(ids)
    runlist = np.array(runlist)

    ratio_list = []
    collected_rate = []
    err = []

    for source in range(0,3):
            
            # runl = np.where(ids==source)[0]
            # for i in runl:
            #         #print(labels[ids[i]])
                    
            #         deadtime = (deadtime_from_fit[i]*1e3)*u.ns
            #         delta_deadtime = deadtime_from_fit_err[i]*1e3
            #         freq = (1/deadtime).to(u.kHz).value
            #         freq_err = delta_deadtime/deadtime.value**2
                    
            #         rate = lambda_from_fit[i]
            #         rate_err=lambda_from_fit_err[i]
            #         ratio = rate/freq
            #         ratio_list.append(ratio*100)
            #         collected_rate.append(collected_triger_rate[i])
            #         ratio_err = np.sqrt((rate_err/freq)**2 + (freq_err*rate/(freq**2)))
                    
                    
            #         err.append(ratio_err*100)

            Y = list(np.array(collected_trigger_rate[np.where(ids==source)[0]])/1000)
            X = list(np.array(deadtime_pc[np.where(ids==source)[0]]))
            #err = list(err)
            X_sorted = [x for y, x in sorted(zip(Y, X))]
            #err_sorted = [err for y,err in sorted(zip(Y,err))]
            
            plt.plot(sorted(Y), X_sorted, #yerr = err_sorted, 
                     alpha=0.6,  ls='-', marker='o',color=labels[source]['color'], label = labels[source]['source'])
            


    plt.xlabel('Collected Trigger Rate [kHz]')
    plt.ylabel(r'Deadtime [%]')


    plt.axhline(5, ls='-', color='gray', alpha=0.4)
    plt.axvline(7, ls='-', color='gray', alpha=0.4,)

    ax.text(28, 6.25, 'CTA requirement', color='gray',  fontsize=20, alpha=0.6,
            horizontalalignment='center',
            verticalalignment='center') #transform=ax.transAxes)

    plt.legend()

    plt.xlim(-0.5,16)
    # plt.grid(which='both')
    plt.yscale(u'log')
    plt.ylim(1e-2,1e2)
    plt.savefig(os.path.join(output_dir,"deadtime.png"))

    if temp_output:
        with open(os.path.join(args.temp_output, 'plot1.pkl'), 'wb') as f:
            pickle.dump(fig, f)

    





    ##################################################
    #SECOND PLOT
    plt.clf()
    plt.figure(figsize=(10,10/1.61))
    fig, ((ax1, ax2)) = plt.subplots(2, 1,  sharex='col', sharey='row',   figsize=(10,8), 
    #                                            sharex=True, sharey=True, 
                                                gridspec_kw={'height_ratios': [5,2]})



    x = collected_trigger_rate/1000
    rate=np.array(lambda_from_fit)
    y = lambda_from_fit
    rate_err=np.array(lambda_from_fit_err)
    relative = (y-x)/x * 100
    #print(np.argmin(relative))

    x_err = 0
    err_ratio = relative *  ( ((rate_err + x_err)/(y - x))  + x_err/x)
    ax2.errorbar(x, relative, 
                xerr= x_err, yerr=err_ratio, 
                alpha=0.9, ls=' ', marker='o', 
                color='C1')
    ax2.set_ylim(-25,25)

    x=range(0,60)

    ax1.set_xscale(u'log')
    ax1.set_yscale(u'log')
    ax1.plot(x, x, color='gray', ls='--', alpha=0.5)

    ax2.plot(x, np.zeros(len(x)), color='gray', ls='--', alpha=0.5)
    ax2.fill_between(x, np.ones(len(x))*(-10), np.ones(len(x))*(10), color='gray',  alpha=0.1)

    ax2.set_xlabel('Collected Trigger Rate [kHz]')
    ax1.set_ylabel(r'Rate from fit [kHz]')
    ax2.set_ylabel(r'Relative difference [%]')


    ax1.set_xlim(1,60)
    ax1.set_ylim(1,60)
    ax2.set_xlim(1,60)

    ids = np.array(ids)
    runlist = np.array(runlist)
    #print(lambda_from_fit)
    #print("coll",collected_triger_rate)
    for source in range(0,3):
        runl = np.where(ids==source)[0]
        
        
            
        # print(collected_triger_rate[runl])
        # print(rate[runl])
        ax1.errorbar(collected_trigger_rate[runl]/1000, 
                    rate[runl], 
                    #xerr=((df_mean_nsb[df_mean_nsb['Run']==run]['Collected_trigger_rate[Hz]_err']))/1000,
                    yerr=rate_err[runl],
                    alpha=0.9,
                    ls=' ', marker='o', color=labels[source]['color'], label = labels[source]['source'])
    #                  label = 'Run {} ({} V)'.format(run, df_mean_rg[df_mean_rg['Run']==run]['Voltage[V]'].values[0]))

    ax1.legend(frameon=False,  prop={'size':10},
            loc="upper left", ncol=1)


    plt.savefig(os.path.join(output_dir,"deadtime_meas.png"))

    if temp_output:
        with open(os.path.join(args.temp_output, 'plot2.pkl'), 'wb') as f:
            pickle.dump(fig, f)

    plt.close('all')




if __name__ == "__main__":
    main()






##################################PREVIOUS###############################
# collected_rate = []


# parameter_A_new_list = []
# parameter_R_new_list = []
# parameter_A_err_new_list = []
# parameter_R_err_new_list = []
# first_bin_length_list = []
# tot_nr_events_histo_list = []
# ucts_busytime = []
# ucts_event_counter = []
# total_delta_t_for_busy_time_list = []

# deadtime = list()
# deadtime_bin = list()
# deadtime_err = list()
# deadtime_bin_length = list()

# for i, run in enumerate(runlist): 
#     deadtime_run, deadtime_bin_run, deadtime_err_run, deadtime_bin_length_run, \
#     total_delta_t_for_busy_time, parameter_A_new, parameter_R_new, parameter_A_err_new, parameter_R_err_new, \
#     first_bin_length, tot_nr_events_histo = deadtime_and_expo_fit(time_tot[i],deadtime_us[i], run)
#     total_delta_t_for_busy_time_list.append(total_delta_t_for_busy_time)
#     parameter_A_new_list.append(parameter_A_new)
#     parameter_R_new_list.append(parameter_R_new)
#     parameter_A_err_new_list.append(parameter_A_err_new)
#     parameter_R_err_new_list.append(parameter_R_err_new)

#     tot_nr_events_histo_list.append(first_bin_length)
#     tot_nr_events_histo_list.append(tot_nr_events_histo)
#     deadtime.append(deadtime_run)
#     deadtime_bin.append(deadtime_bin_run)
#     deadtime_err.append(deadtime_err_run)
#     deadtime_bin_length.append(deadtime_bin_length_run/np.sqrt(12))
#     #print(run, parameter_A_new, parameter_R_new)
    

# deadtime_from_first_bin = np.array(deadtime_bin)
# deadtime_bin_length = np.array(deadtime_bin_length)
# rate = (np.array(parameter_R_new_list) * 1 / u.us).to(u.kHz).to_value()   #in kHz
# rate_err = (np.array(parameter_R_err_new_list) * 1 / u.us).to(u.kHz).to_value()
# A_from_fit = (parameter_A_new_list)
# A_from_fit_err = (parameter_A_err_new_list)
# ucts_busy_rate = (np.array(busy_counter[:,-1]) / (np.array(time_tot) * u.s).to(u.s)).to(
#     u.kHz).value
# nr_events_from_histo = (tot_nr_events_histo)
# first_bin_delta_t = first_bin_length

# deadtime_average = np.average(deadtime_bin,
#                                    weights=1 / (deadtime_bin_length ** 2))
# deadtime_average_err_nsb = np.sqrt(1 / (np.sum(1 / deadtime_bin_length ** 2)))


# #######################################################################################




# #B-TEL-1260
# plt.clf()
# fig, ax = plt.subplots(figsize=(10*1.1,10*1.1/1.61)) 
# ids = np.array(ids)
# runlist = np.array(runlist)

# ratio_list = []
# collected_rate = []
# err = []
# for source in range(0,3):
#         runl = np.where(ids==source)[0]
#         for i in runl:
#                 #print(labels[ids[i]])
                
#                 deadtime = (deadtime_bin[i]*1e3)*u.ns
#                 delta_deadtime = deadtime_bin_length[i]*1e3
#                 freq = (1/deadtime).to(u.kHz).value
#                 freq_err = delta_deadtime/deadtime.value**2
                
#                 ratio = rate[i]/freq
#                 ratio_list.append(np.array(ratio)*100)
#                 ratio_err = np.sqrt((rate_err[i]/freq)**2 + (freq_err*rate[i]/(freq**2)))
#                 err.append(ratio_err*100)


#         Y = list(np.array(collected_triger_rate[np.where(ids==source)[0]])/1000)
#         X = list(ratio_list)
#         err = list(err)
#         X_sorted = [x for y, x in sorted(zip(Y, X))]
#         err_sorted = [err for y,err in sorted(zip(Y,err))]
        
#         plt.errorbar(sorted(Y), X_sorted, yerr = err_sorted, alpha=0.6,  ls='-', marker='o',color=labels[source]['color'], label = labels[source]['source'])
        
# plt.xlabel('Collected Trigger Rate [kHz]')
# plt.ylabel(r'Deadtime [%]')


# plt.axhline(5, ls='-', color='gray', alpha=0.4)
# plt.axvline(7, ls='-', color='gray', alpha=0.4,)

# ax.text(28, 6.25, 'CTA requirement', color='gray',  fontsize=20, alpha=0.6,
#         horizontalalignment='center',
#         verticalalignment='center') 
# plt.legend()

# plt.xlim(-0.5,33)
# # plt.grid(which='both')
# plt.yscale(u'log')
# plt.ylim(1e-2,1e2)
# plt.savefig(os.path.join(output_dir,"deadtime.png"))


# #############################################################################









# #B-TEL-1670
# plt.clf()
# f, ((ax1, ax2)) = plt.subplots(2, 1,  sharex='col', sharey='row',   figsize=(10,8), 
#                                 gridspec_kw={'height_ratios': [5,2]})


# #plt.figure(figsize=(10,10/1.61))
# x = collected_triger_rate/1000
# y = rate
# relative = (y-x)/x * 100
# # print(np.argmin(relative))
# # print(collected_triger_rate[5])
# # print(rate[5])
# x_err = 0
# err_ratio = relative *  ( ((rate_err + x_err)/(y - x))  + x_err/x)
# ax2.errorbar(x, relative, 
#              xerr= x_err, yerr=err_ratio, 
#              alpha=0.9, ls=' ', marker='o', 
#              color='C1')
# ax2.set_ylim(-25,25)

# x=range(0,60)

# ax1.set_xscale(u'log')
# ax1.set_yscale(u'log')
# ax1.plot(x, x, color='gray', ls='--', alpha=0.5)

# ax2.plot(x, np.zeros(len(x)), color='gray', ls='--', alpha=0.5)
# ax2.fill_between(x, np.ones(len(x))*(-10), np.ones(len(x))*(10), color='gray',  alpha=0.1)

# ax2.set_xlabel('Collected Trigger Rate [kHz]')
# ax1.set_ylabel(r'Rate from fit [kHz]')
# ax2.set_ylabel(r'Relative difference [%]')

# ax1.set_xlim(1,60)
# ax1.set_ylim(1,60)
# ax2.set_xlim(1,60)

# ids = np.array(ids)
# runlist = np.array(runlist)
# for source in range(0,3):
#     runl = np.where(ids==source)[0]
#     #print(collected_triger_rate[runl])
#     ax1.errorbar(collected_triger_rate[runl]/1000, 
#                 rate[runl], 
#                 #xerr=((df_mean_nsb[df_mean_nsb['Run']==run]['Collected_trigger_rate[Hz]_err']))/1000,
#                 yerr=rate_err[runl],
#                 alpha=0.9,
#                 ls=' ', marker='o', color=labels[source]['color'], label = labels[source]['source'])
# #                  label = 'Run {} ({} V)'.format(run, df_mean_rg[df_mean_rg['Run']==run]['Voltage[V]'].values[0]))

# ax1.legend(frameon=False,  prop={'size':10},
#            loc="upper left", ncol=1)


# plt.savefig(os.path.join(output_dir,"deadtime_meas.png"))
