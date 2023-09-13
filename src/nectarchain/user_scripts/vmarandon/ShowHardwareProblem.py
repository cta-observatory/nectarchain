try:
    import sys

    import numpy as np
    import numpy.ma as ma

    import argparse

    import matplotlib
    from matplotlib import pyplot as plt

    from FileHandler import DataReader, GetNectarCamEvents
    from DataUtils import  GetTotalEventCounts
    from Utils import GetEventTypeFromString, GetCamera, CustomFormatter, GetDefaultDataPath
    from FitUtils import GaussHistoFitFunction
    from Stats import CameraSampleStats, CameraStats
    from CalibrationData import CalibrationCameraDisplay
    
    from scipy.optimize import curve_fit
    from scipy.stats import norm
    from iminuit import Minuit

    from ctapipe.visualization import CameraDisplay

    from tqdm import tqdm

    from IPython import embed

except ImportError as e:
    print(e)
    raise SystemExit

class HardwareProblemStatsMaker:
    '''
    Class to compute stats on possible hardware problems
    '''
    def __init__(self,telid = 0):
        self.telid = telid
        self.missing = None

def ShowHardwareProblem(arglist):

    p = argparse.ArgumentParser(description='Show stats on the hardware_failing_pixels flag',
                                epilog='examples:\n'
                                '\t python %(prog)s --run 123456  \n',
                                formatter_class=CustomFormatter)

    p.add_argument("--run", dest='run', type=int, help="Run number to be converted")
    p.add_argument("--data-path",dest='dataPath',type=str,default=None,help="Path to the rawdata directory. The program will recursively search in all directory for matching rawdata")
    p.add_argument("--telid",dest="telId",type=int,default=0,help="Telescope id for which we show the data distributions")
    p.add_argument("--list",dest="listEvent",action='store_true',help="List pixels for which there is hardware problems")
    p.add_argument("--savefig",dest="savefig",action='store_true',help='Save figure')
    p.add_argument("--debug",dest="debug",action='store_true',help='Debug mode (only 10000 events processed)')

    args = p.parse_args(arglist)

    if args.run is None:
        p.print_help()
        return -1

    if args.dataPath is None:
        path = GetDefaultDataPath()
    else:
        path = args.dataPath
    
    data = DataReader(args.run,path)
    ok = data.Connect( "mon","trigger" )
    #evt.mon.tel[self.telid].pixel_status.hardware_failing_pixels
    if not ok:
        data = GetNectarCamEvents(args.run,path,applycalib=False)

    dataevents = GetTotalEventCounts(args.run,path)

    camstats = CameraStats()

    sum_failing = None
    counter = 0
    try:
        for evt in tqdm(data,total=dataevents):
            hardware_failing = evt.mon.tel[args.telId].pixel_status.hardware_failing_pixels
            camstats.add( hardware_failing )
            
            if sum_failing is not None:
                sum_failing += hardware_failing
            else:
                sum_failing = hardware_failing.copy().astype(int)


            if args.debug:
                counter += 1
                if counter > 10000:
                    break
    except Exception as e:
        print(e)
    
    lw = 0.2
    
    fig_counts, axs_counts = plt.subplots(nrows=1,ncols=2, figsize=(12,6))

    cam_count_hg = CameraDisplay(geometry=GetCamera(), cmap='turbo', image=ma.array(camstats.mean[0]*camstats.count[0],mask=camstats.mean[0]==0.), title=f'HG Hardware Failing Counts\nrun {args.run}',ax=axs_counts[0],show_frame=False,allow_pick=True)
    cam_count_hg.highlight_pixels(range(1855),linewidth=lw,color='grey')
    cam_count_hg.add_colorbar()
    
    cam_count_lg = CameraDisplay(geometry=GetCamera(), cmap='turbo', image=ma.array(camstats.mean[1]*camstats.count[1],mask=camstats.mean[1]==0.), title=f'LG Hardware Failing Counts\nrun {args.run}',ax=axs_counts[1],show_frame=False,allow_pick=True)
    cam_count_lg.highlight_pixels(range(1855),linewidth=lw,color='grey')
    cam_count_lg.add_colorbar()

    if args.savefig:
        figname = f'run_{args.run}_Hardware_Failing_Pixels_Counter.png'

    fig_counts.show()
    

    fig_counts2, axs_counts2 = plt.subplots(nrows=1,ncols=2, figsize=(12,6))

    cam_count_hg2 = CameraDisplay(geometry=GetCamera(), cmap='turbo', image=ma.array(sum_failing[0],mask=sum_failing[0]==0.), title=f'HG Hardware Failing Counts 2\nrun {args.run}',ax=axs_counts2[0],show_frame=False,allow_pick=True)
    cam_count_hg2.highlight_pixels(range(1855),linewidth=lw,color='grey')
    cam_count_hg2.add_colorbar()
    
    cam_count_lg2 = CameraDisplay(geometry=GetCamera(), cmap='turbo', image=ma.array(sum_failing[1],mask=sum_failing[1]==0.), title=f'LG Hardware Failing Counts 2\nrun {args.run}',ax=axs_counts2[1],show_frame=False,allow_pick=True)
    cam_count_lg2.highlight_pixels(range(1855),linewidth=lw,color='grey')
    cam_count_lg2.add_colorbar()

    fig_counts2.show()



    fig_frac, axs_frac = plt.subplots(nrows=1,ncols=2, figsize=(12,6))

    cam_frac_hg = CameraDisplay(geometry=GetCamera(), cmap='turbo', image=ma.array(camstats.mean[0]*100,mask=camstats.mean[0]==0.), title=f'HG Hardware Failing Fraction (%)\nrun {args.run}',ax=axs_frac[0],show_frame=False,allow_pick=True)
    cam_frac_hg.add_colorbar()
    cam_frac_hg.highlight_pixels(range(1855),linewidth=lw,color='grey')
    cam_frac_hg.colorbar.set_label('%')

    cam_frac_lg = CameraDisplay(geometry=GetCamera(), cmap='turbo', image=ma.array(camstats.mean[1]*100,mask=camstats.mean[1]==0.), title=f'LG Hardware Failing Fraction (%)\nrun {args.run}',ax=axs_frac[1],show_frame=False,allow_pick=True)
    cam_frac_lg.add_colorbar()
    cam_frac_lg.highlight_pixels(range(1855),linewidth=lw,color='grey')
    cam_frac_lg.colorbar.set_label('%')

    if args.savefig:
        figname = f'run_{args.run}_Hardware_Failing_Pixels_Fraction.png'

    fig_frac.show()
    
    
    #embed()

    ## To go further one could use this information : 
    #module_status[ evt.nectarcam.tel[0].svc.module_ids ] = evt.nectarcam.tel[0].evt.module_status
    # to know the module that was expected to be there during acquisition in order to avoid false positive.
    # There is a similar info for pixels
    #pixel_status = np.zeros(N_PIXELS)
    #pixel_status[self.camera_config.expected_pixels_id] = event.pixel_status
    #status_container.hardware_failing_pixels[:] = pixel_status == 0



    #plt.show()
    input("Press Enter to quit\n")

    embed()



if __name__ == "__main__":
    ShowHardwareProblem(sys.argv[1:])
