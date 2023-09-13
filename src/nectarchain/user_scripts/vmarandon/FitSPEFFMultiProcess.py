try:
    import sys
    import numpy as np
    import numpy.ma as ma
    from scipy.stats import poisson, norm
    from DataUtils import GetLongRunTimeEdges, CountEventTypes, GetTotalEventCounts
    from Utils import GetCamera, SignalIntegration, IntegrationMethod, ConvertTitleToName, CustomFormatter, GetDefaultDataPath
    from ctapipe.containers import EventType
    from FileHandler import DataReader, GetNectarCamEvents
    from Stats import CameraSampleStats, CameraStats
    from tqdm import tqdm
    from FitUtils import JPT2FitFunction
    from iminuit import Minuit
    import time
    import copy
    from matplotlib import pyplot as plt
    from ctapipe.visualization import CameraDisplay
    #from multiprocessing import Pool
    import pickle
    #from multiprocessing.dummy import Pool as ThreadPool
    from multiprocessing import Pool
    import argparse

    from IPython import embed
except ImportError as e:
    print(e)
    raise SystemExit


# %matplotlib tk
#ma.array(t0max[0],mask=css.get_lowcount_mask()[0,:,0])
class FitResult():
    def __init__(self,res,params):
        self.minuit_result = res
        self.fit_parameters = params

def split_array(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def save_data(datas, fileName):
    """Save info in a file using pickle"""
    with open(fileName,'wb') as file:
        pickle.dump(datas,file)

def FitPart(pixels,charges,sigpeds,peds):
    results = list()
    for p,c,s,ped in tqdm(zip(pixels,charges,sigpeds,peds)):
        res = FitPixel(c,s,ped)
        if res is not None:
            results.append( (p,) + res )
    return results
        

def FitPixel(hicharges,sigped=None,ped=None):
    start = time.time()
    pix_datas = hicharges
    bins = np.linspace( np.min(pix_datas)-0.5,np.max(pix_datas)+0.5,int(np.max(pix_datas)-np.min(pix_datas)+2))
    vals,x_edges = np.histogram(pix_datas,bins=bins)
    
    if len(bins)<=100:
        #print(f"Pixel: {pix} --> Too few bins... bad pixel ? --> SKIP !")
        return #continue

    jpt2 = JPT2FitFunction(x_edges,vals)
    amplitude = float(np.sum(vals))
    if ped is None:
        hiPed = x_edges[ np.argmax(vals) ] + 0.5
    else:
        hiPed = ped
    
    if sigped is None:
        sigmaHiPed = 14.
    else:
        sigmaHiPed = sigped

    ns = 1.
    meanillu = 1.
    gain = 65.
    sigmaGain = 0.45*gain
    start_parameters = [amplitude,hiPed,sigmaHiPed,ns,meanillu,gain,sigmaGain]

    m = Minuit( jpt2.Minus2LogLikelihood, start_parameters, name=("Norm","Ped","SigmaPed","Ns","Illu","Gain","SigmaGain") )
    
    m.errors["Norm"] = 0.1*amplitude
    m.errors["Ped"] = 1.* sigmaHiPed
    m.errors["SigmaPed"] = 0.2*sigmaHiPed
    m.errors["Ns"] = 0.1
    m.errors["Illu"] = 0.3
    m.errors["Gain"] = 0.3*gain
    m.errors["SigmaGain"] = 0.3*sigmaGain
    
    m.limits['Norm'] = (0.,None)
    m.limits['Illu'] = (0.,8.)
    m.limits['Gain'] = (10.,200.)
    m.limits['SigmaGain'] = (10.,None)
    m.limits['Ns'] = (0.,None)
    m.limits['SigmaPed'] = (0.,None)

    try:
        min_result = m.migrad()
        fit_params = m.params

        end = time.time()
        fitTime = end-start
        #pixelFitTime[pix] = end-start
        #pixelFitResult[pix] = FitResult( copy.deepcopy(min_result), copy.deepcopy(fit_params) )
        return FitResult( copy.deepcopy(min_result), copy.deepcopy(fit_params) ), fitTime
    
    except ValueError as err:
        #print(f"Pixel: {pix} Error: {err}")
        pass


def DoSPEFFFit(arglist):

    p = argparse.ArgumentParser(description='Perform SPE Fit for FF data',
                                epilog='examples:\n'
                                '\t python %(prog)s --run 3750  \n',
                                formatter_class=CustomFormatter)

    p.add_argument("--run", dest='run', type=int, help="Run number")
    p.add_argument("--data-path",dest='dataPath',type=str,default=GetDefaultDataPath(),help="Path to the rawdata directory. The program will recursively search in all directory for matching rawdata")
    p.add_argument("--njobs", dest='njobs', type=int, default=1, help="Number of CPU to use")

    args = p.parse_args(arglist)

    if args.run is None:
        p.print_help()
        return -1

    run = args.run
    path = args.dataPath
    
    # path = '/Users/vm273425/Programs/NectarCAM/data/'

    data = DataReader(run,path)
    ok = data.Connect("trigger","r0","mon")
    if not ok:
        data = GetNectarCamEvents(run,path,applycalib=False)

    nEvents = GetTotalEventCounts(run,path)

    css = CameraSampleStats()


    counter = 0
    nUsedEntries = 10000000
    for evt in tqdm(data,total=min(nEvents,nUsedEntries)):
        css.add( evt.r0.tel[0].waveform, validmask = ~evt.mon.tel[0].pixel_status.hardware_failing_pixels)
        counter += 1
        if counter>=nUsedEntries:
            break

    mean_waveform = css.mean

    fig1 = plt.figure()
    pixs = [777,747,320,380,123,1727,427,74]
    for p in pixs:
        plt.plot( mean_waveform[0,p,:], label=f'{p}')
    plt.grid()
    plt.legend()



    fig2 = plt.figure()
    #pixs = [777,747,320,380,123,1727,427,74]
    pixs = [923,483,1573,1751,1491,482,720]
    for p in pixs:
        plt.plot( mean_waveform[0,p,:], label=f'{p}')
    plt.grid()
    plt.legend()


    t0max = np.argmax(mean_waveform,axis=2)

    fig3 = plt.figure()
    cam_tom_hg = CameraDisplay(geometry=GetCamera(),image=ma.array(t0max[0],mask=css.get_lowcount_mask()[0,:,0]),cmap='coolwarm',title='High-Gain Mean ToM',allow_pick=True)
    cam_tom_hg.highlight_pixels(range(1855),color='grey')
    cam_tom_hg.add_colorbar()
    #cam_tom_hg.show()


    fig4 = plt.figure()
    histdata = t0max[0]
    bins = np.linspace(-0.05,50.05,502)
    good_histdata = histdata[ histdata>0. ]
    _ = plt.hist(good_histdata,bins=bins)
    plt.xlim(np.min(good_histdata)-1,np.max(good_histdata)+1)
    plt.title("ToM Average High Gain Waveform")
    plt.grid()


    data = DataReader(run,path)
    ok = data.Connect("trigger","r0","mon")
    if not ok:
        data = GetNectarCamEvents(run,path,applycalib=False)

    nEvents = GetTotalEventCounts(run,path)

    camera = GetCamera()


    ped_stats = CameraStats()
    hicharges = list()
    counter = 0

    #data.rewind()

    counter = 0
    for evt in tqdm(data,total=nEvents):
        charges, times = SignalIntegration( evt.r0.tel[0].waveform, exclusion_mask=evt.mon.tel[0].pixel_status.hardware_failing_pixels, peakpositions=t0max,method=IntegrationMethod.USERPEAK,left_bound=5,right_bound=11,camera=camera)
        
        ped = np.sum( evt.r0.tel[0].waveform[:,:,:16], axis=2 )
        ped_stats.add( ped, validmask=~evt.mon.tel[0].pixel_status.hardware_failing_pixels )

        hicharges.append( charges[0] )
        counter += 1
#        if counter >=1000:
#            break

    hicharges = np.array(hicharges)



    fig5 = plt.figure()
    cam_pedw_hg = CameraDisplay(geometry=GetCamera(),image=ma.array(ped_stats.stddev[0],mask=ped_stats.get_lowcount_mask()[0]),cmap='coolwarm',title='High-Gain Mean Ped Width',allow_pick=True)
    cam_pedw_hg.highlight_pixels(range(1855),color='grey')
    cam_pedw_hg.add_colorbar()
    #cam_pedw_hg.show()



    epedwidth   = ped_stats.stddev[0]
    eped        = ped_stats.mean[0]
    badpedwidth = ped_stats.get_lowcount_mask()[0]

    pixelFitResult = dict()
    pixelFitTime = dict()

    use_multiprocess = args.njobs>1
    simple_multiprocess = False
    # HiCharges = hicharges
    # BadPedWidth = badpedwidth
    # EPedWidth = epedwidth

    start_fit = time.time()
    print(f"Starting the fit at : {start_fit}")
    if use_multiprocess and simple_multiprocess:
        print("Using the fit by pixel method")
        pixlist = [pix for pix in range(1855)]
    
        with Pool(args.njobs) as p:
            datas = list()
            sigpeds = list()
            peds = list()
            for pix in pixlist:
                datas.append( hicharges[:,pix] )
                sigpeds.append( 14. if badpedwidth[pix] else epedwidth[pix] )
                peds.append( None if badpedwidth[pix] else eped[pix] )
            
            print(len(datas))
            results = p.starmap(FitPixel,zip(datas,sigpeds,peds))

            p.close()
            p.join()
            print(len(results))

            for pix in pixlist:
                res = results[pix]
                if res is not None:
                    pixelFitResult[pix] = res[0]
                    pixelFitTime[pix] = res[1]
    elif use_multiprocess and not simple_multiprocess:
        print("Using the fit by part method")
        nPart = args.njobs
        
        npix2use = 1855
        part_pixels = list( split_array( [pix for pix in range(npix2use)], nPart ) )
    
        part_charges = list( split_array( [hicharges[:,pix] for pix in range(npix2use) ], nPart ) )
        part_sigpeds = list( split_array( [ 14. if badpedwidth[pix] else epedwidth[pix] for pix in range(npix2use)], nPart  ) )
        part_peds    = list( split_array( [ None if badpedwidth[pix] else eped[pix] for pix in range(npix2use)], nPart  ) )

        #embed()
        #for part in part_pixels:
        #    pcharges
        #    for pix in part:
        with Pool(nPart) as p:
            results = p.starmap(FitPart,zip(part_pixels,part_charges,part_sigpeds,part_peds))
            for rs in results:
                for res in rs:
                    if res is not None:
                        cur_pixel = res[0]
                        pixelFitResult[cur_pixel] = res[1]
                        pixelFitTime[cur_pixel] = res[2]
                    


                           
    else:
        print("Using the standard approach")
        for pix in tqdm(range(1855)):
            sigped = 14. if badpedwidth[pix] else epedwidth[pix]
            ped = None if badpedwidth[pix] else eped[pix]
            res = FitPixel(hicharges[:,pix],sigped,ped )
            if res is not None:
                pixelFitResult[pix] = res[0]
                pixelFitTime[pix] = res[1]
            #if pix>=100 :
            #    break

    end_fit = time.time()

    print(f"Finishing the fit at : {end_fit}")
    print(f"Time to do it : {end_fit-start_fit} s")


    print(len(pixelFitResult))

    gains = list()
    for p, f in pixelFitResult.items():
        fit_gain = f.fit_parameters['Gain'].value
        gains.append(fit_gain)


    cam_gain_values = np.zeros(1855)
    for p, f in pixelFitResult.items():
        fit_gain = f.fit_parameters['Gain'].value
        cam_gain_values[p] = fit_gain


    fig6 = plt.figure(figsize=(8,8))
    cam_gain = CameraDisplay(geometry=GetCamera(),image=ma.array(cam_gain_values,mask=cam_gain_values<40.),cmap='turbo',title=f'High-Gain Gain (VIM Edition)\nRun: {run}',show_frame=False)
    cam_gain.highlight_pixels(range(1855),color='grey')
    cam_gain.add_colorbar()
    cam_gain.show()
    #fig.savefig(f'run_{run}_gain_camera.png')
    fig6.savefig(f'run_{run}_FittedGain_VIMEdition.png')

    fig7 = plt.figure(figsize=(8,8))
    mean_gain = np.mean(gains)
    std_gain = np.std(gains)
    _ = plt.hist(gains,bins=60,label=f'$\mu:$ {mean_gain:.2f}\n$\sigma:$ {std_gain:.2f}')
    plt.title(f"Fitted Gain Distribution (Run: {run})")
    plt.xlabel('Gain (ADC per pe)')
    plt.grid()
    plt.legend()
    #fig.savefig(f'run_{run}_gain_distribution.png')
    fig7.savefig(f'run_{run}_FittedGainDistribution_VIMEdition.png')

    cam_illu_values = np.zeros(1855)
    for p, f in pixelFitResult.items():
        fit_illu = f.fit_parameters['Illu'].value
        cam_illu_values[p] = fit_illu

    fig8 = plt.figure(figsize=(8,8))
    cam_illu = CameraDisplay(geometry=GetCamera(),image=ma.array(cam_illu_values,mask=cam_gain_values<40.),cmap='turbo',title=f'Mean Illumination (VIM Edition)\nRun: {run}',show_frame=False)
    cam_illu.highlight_pixels(range(1855),color='grey')
    cam_illu.add_colorbar()
    cam_illu.show()
    fig8.savefig(f'run_{run}_FittedIllumination_VIMEdition.png')

    cam_ns_values = np.zeros(1855)
    for p, f in pixelFitResult.items():
        fit_ns = f.fit_parameters['Ns'].value
        cam_ns_values[p] = fit_ns

    fig9 = plt.figure(figsize=(8,8))
    cam_ns = CameraDisplay(geometry=GetCamera(),image=ma.array(cam_ns_values,mask=cam_gain_values<40.),cmap='turbo',title=f'Ns Parameter (VIM Edition)\nRun: {run}',show_frame=False)
    cam_ns.highlight_pixels(range(1855),color='grey')
    cam_ns.add_colorbar()
    cam_ns.show()

    cam_pedw_values = np.zeros(1855)
    for p, f in pixelFitResult.items():
        fit_pedw = f.fit_parameters['SigmaPed'].value
        cam_pedw_values[p] = fit_pedw

    fig9 = plt.figure(figsize=(8,8))
    cam_pedw = CameraDisplay(geometry=GetCamera(),image=ma.array(cam_pedw_values,mask=cam_gain_values<40.),cmap='turbo',title=f'Pedestal Width (VIM Edition)\nRun: {run}',show_frame=False)
    cam_pedw.highlight_pixels(range(1855),color='grey')
    cam_pedw.add_colorbar()
    cam_pedw.show()
    fig9.savefig(f'run_{run}_FittedPedestalWidth_VIMEdition.png')


    ## Save results : 
    outdataname = f'run_{run}_FittedSPEResult.dat'

    with open(outdataname,'w') as outfile:
        outfile.write(f"pix Ped SigmaPed Ns Illu Gain SigmaGain\n")
        for p, f in pixelFitResult.items():
            if f.minuit_result.valid:
                outfile.write(f"{p} {f.fit_parameters['Ped'].value} {f.fit_parameters['SigmaPed'].value} {f.fit_parameters['Ns'].value} {f.fit_parameters['Illu'].value} {f.fit_parameters['Gain'].value} {f.fit_parameters['SigmaGain'].value}\n")
    
    # m.errors["Norm"] = 0.1*amplitude
    # m.errors["Ped"] = 1.* sigmaHiPed
    # m.errors["SigmaPed"] = 0.2*sigmaHiPed
    # m.errors["Ns"] = 0.1
    # m.errors["Illu"] = 0.3
    # m.errors["Gain"] = 0.3*gain
    # m.errors["SigmaGain"] = 0.3*sigmaGain

    plt.show()


if __name__ == '__main__':
    DoSPEFFFit(sys.argv[1:])

