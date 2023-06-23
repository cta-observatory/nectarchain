try:
    import sys

    import numpy as np
    import numpy.ma as ma

    import argparse

    import astropy.units as u
    from astropy.time import TimeDelta
    from ctapipe.containers import EventType
    from ctapipe.visualization import CameraDisplay

    from FileHandler import DataDumper, DataReader, GetNectarCamEvents
    from Utils import GetCamera, CustomFormatter, GetDefaultDataPath
    from CalibrationData import PedestalInfo

    from CalibrationEvoData import PedestalEvoInfo

    from tqdm import tqdm
    
    from collections import defaultdict

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import interactive
    from collections import defaultdict

    from scipy.stats import linregress
    
    from pprint import pprint

    matplotlib.use('TkAgg')
    #matplotlib.use('macosx')

except ImportError as e:
    print(e)
    raise SystemExit



def ComputePedestalEvolution(run,path):

    data = DataReader(run,path)
    if not data.Connect("pedestal"):
        print(f"Can't find the Pedestal file for the run [{run}] in directory [{path}]")
        return None

    try:
        data.ConnectDB(tables='monitoring_drawer_temperatures')
        use_db = True
    except FileNotFoundError as err:
        use_db = False

    pedestal_positions = list()
    pedestal_widths = list()
    pedestal_times = list()
    pedestal_pos_mean = list()
    pedestal_width_mean = list()

    temperatures1 = list()
    temperatures2 = list()

    

    for evt in tqdm(data):
        pedestal_times.append( evt.pedestal.tel[0].startTime.datetime )
        pedestal_positions.append( evt.pedestal.tel[0].pos.copy() )
        pedestal_widths.append( evt.pedestal.tel[0].width.copy() )
        pedestal_pos_mean.append( evt.pedestal.tel[0].pos.mean(axis=2) ) 
        pedestal_width_mean.append( evt.pedestal.tel[0].width.mean(axis=2 ) )
        if use_db:
            # assume that the shape is [channel,pixel,slices]
            temperatures1_entry = np.zeros( evt.pedestal.tel[0].width.shape[:-1])
            temperatures2_entry = np.zeros( evt.pedestal.tel[0].width.shape[:-1])
            for pix in range(evt.pedestal.tel[0].width.shape[1]):
                #print(f'pix: {pix} module: {pix//7}')
                moduleId = pix//7
                try:
                    temp1 = evt.db.tel[0]['monitoring_drawer_temperatures'][moduleId].tfeb1
                except KeyError as error:
                    temp1 = 0

                try:
                    temp2 = evt.db.tel[0]['monitoring_drawer_temperatures'][moduleId].tfeb2
                except KeyError as error:
                    temp2 = 0
                
                for chan in range(evt.pedestal.tel[0].width.shape[0]):
                    temperatures1_entry[chan,pix] = temp1
                    temperatures2_entry[chan,pix] = temp2
            
            temperatures1.append( temperatures1_entry )   
            temperatures2.append( temperatures2_entry )   
            

    pedestal_positions = ma.array( pedestal_positions )
    pedestal_widths = ma.array( pedestal_widths )
    pedestal_times = ma.array( pedestal_times )
    pedestal_pos_mean = ma.array( pedestal_pos_mean )
    pedestal_width_mean = ma.array( pedestal_width_mean )

    if use_db:
        temperatures1 = np.array(temperatures1)
        temperatures2 = np.array(temperatures2)
        temperatures1 = ma.array(temperatures1, mask = temperatures1==0.)
        temperatures2 = ma.array(temperatures2, mask = temperatures2==0.)


    ## Do the linear regression 
    slopes_t1_pedmean     = np.zeros( pedestal_positions.shape[1:3] )
    intercepts_t1_pedmean = np.zeros( pedestal_positions.shape[1:3] )
    corr_t1_pedmean       = np.zeros( pedestal_positions.shape[1:3] )
    mask_t1_pedmean       = np.full( pedestal_positions.shape[1:3], True )
    slopes_t2_pedmean     = np.zeros( pedestal_positions.shape[1:3] )
    intercepts_t2_pedmean = np.zeros( pedestal_positions.shape[1:3] )
    corr_t2_pedmean       = np.zeros( pedestal_positions.shape[1:3] )
    mask_t2_pedmean       = np.full( pedestal_positions.shape[1:3], True )

    ped_counts = pedestal_pos_mean.count(axis=0)

    
    for chan in range(pedestal_positions.shape[1]):
        for pix in range(pedestal_positions.shape[2]):
            if ped_counts[chan,pix] == len(temperatures1[:,chan,pix]):
                lr1 = linregress( temperatures1[:,chan,pix], pedestal_pos_mean[:,chan,pix] )
                slopes_t1_pedmean[chan,pix] = lr1.slope
                intercepts_t1_pedmean[chan,pix] = lr1.intercept
                corr_t1_pedmean[chan,pix] = lr1.rvalue
                lr2 = linregress( temperatures2[:,chan,pix], pedestal_pos_mean[:,chan,pix] )
                slopes_t2_pedmean[chan,pix] = lr2.slope
                intercepts_t2_pedmean[chan,pix] = lr2.intercept
                corr_t2_pedmean[chan,pix] = lr2.rvalue
                mask_t1_pedmean[chan,pix] = False
                mask_t2_pedmean[chan,pix] = False

    
    print("Filling PedestalEvoInfo structure")
    evo_info = PedestalEvoInfo()
    evo_info.run = run
    evo_info.times = pedestal_times
    evo_info.positions = pedestal_positions
    evo_info.widths = pedestal_widths
    evo_info.meanpositions = pedestal_pos_mean
    evo_info.meanwidths = pedestal_width_mean


    if use_db:
        evo_info.temperatures1 = temperatures1
        evo_info.temperatures2 = temperatures2

            # 2D : channel, pixel
        evo_info.slopes_t1_pedmean     = ma.array( slopes_t1_pedmean, mask=mask_t1_pedmean)
        evo_info.intercepts_t1_pedmean = ma.array( intercepts_t1_pedmean, mask=mask_t1_pedmean) 
        evo_info.corr_t1_pedmean       = ma.array( corr_t1_pedmean, mask=mask_t1_pedmean)

        evo_info.slopes_t2_pedmean     =  ma.array( slopes_t2_pedmean, mask=mask_t2_pedmean)
        evo_info.intercepts_t2_pedmean =  ma.array( intercepts_t2_pedmean, mask=mask_t2_pedmean)
        evo_info.corr_t2_pedmean       =  ma.array( corr_t2_pedmean, mask=mask_t2_pedmean)

    return evo_info

def ShowPedestalEvolutionData(pedevo):
    #interactive(True)
    #plt.ion()

    #print("ShowPedestalEvolutionData")
    pedevo.ShowPedestalTemperature1Correlation()
    pedevo.ShowTemperatureEvolution()
    pedevo.ShowPedestalEvolution()
    
    #interactive(False)
    #input()
    plt.show(block=False)
    input("Press Enter to quit")
    

def ShowPedesatalEvolution(arglist):

    p = argparse.ArgumentParser(description='Show the Pedestal Evolution for a given run',
                                epilog='examples:\n'
                                '\t python %(prog)s --run 123456  \n',
                                formatter_class=CustomFormatter)

    p.add_argument("--run", dest='run', type=int, help="Run number to be converted")
    p.add_argument("--data-path",dest='dataPath',type=str,default=None,help="Path to the rawdata directory. The program will recursively search in all directory for matching rawdata")


    args = p.parse_args(arglist)

    if args.run is None:
        p.print_help()
        return -1

    dataPath = args.dataPath if args.dataPath is not None else GetDefaultDataPath()

    ped_evo = ComputePedestalEvolution(args.run,dataPath)
    if ped_evo is None:
        return -1
    ShowPedestalEvolutionData(ped_evo)

if __name__ == "__main__":
    ShowPedesatalEvolution(sys.argv[1:])
