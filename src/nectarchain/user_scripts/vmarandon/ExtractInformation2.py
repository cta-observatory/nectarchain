try:
    import os
    import argparse
    import sys
    import time
    import random
    import copy

    from glob import glob

    from ctapipe.io import EventSource, EventSeeker
    from ctapipe.instrument import CameraGeometry
    from ctapipe.visualization import CameraDisplay
    from ctapipe.coordinates import EngineeringCameraFrame
    from ctapipe_io_nectarcam import NectarCAMEventSource
    from ctapipe.containers import EventType

    from multiprocessing.dummy import Pool as ThreadPool
    #from multiprocessing import Pool as ThreadPool



    from tqdm import tqdm

    from Utils import NeighborPeakIntegration, GetCamera, CustomFormatter, GetNumberOfDataBlocks, FindFiles, FindDataPath
    from FileHandler import DataDumper, GetNectarCamEvents
    from Utils import GetDefaultDataPath

except ImportError as e:
    print(e)
    raise SystemExit



#        data_path = FindDataPath(run,args.dataPath)

def ExtractInformationSingleRun(run,data_path,dest_path,data_block,applycalib=True,keepR1=True,nnint=False,onlytrigger=False):

    sleep_time = random.uniform(0.,1.) 
    #print(sleep_time)
    time.sleep( sleep_time )

    #print(f"Extract {data_block = }")
    camera = GetCamera()

    # Create new type of rawdata
    # same information related to trigger
    # integrated rawdata

    #  'simulation': Simulated Event Information,
    #  'trigger': central trigger information,
    #  'count': number of events processed,
    #  'pointing': Array and telescope pointing positions,
    #  'calibration': Container for calibration coefficients for the current event,
    #  'mon': container for event-wise monitoring data (MON),
    #  'nectarcam': NectarCAM specific Information}

    with DataDumper(run=run,path=dest_path,data_block=data_block) as dd:

        #counter = 0
        #print(f"{run = } {data_path = } {data_block = } {applycalib = }")
        events = GetNectarCamEvents(run=run,path=data_path,data_block=data_block,applycalib=applycalib)
        nEvents = events.get_entries()
        #print(nEvents)
        # max_events = 1000
        # if max_events < 0  or max_events > nEvents:
        #     max_events = nEvents
        #for i,evt in tqdm(enumerate(events)):

        doIntegration = nnint

        if onlytrigger:
            doIntegration = False

        for evt in tqdm(events):
            #if data_block != 42 and data_block!=8:
            #    break
            # if i  >= max_events:
            #     break
            #counter += 1
            #if counter > 1000:
            #    break
            event_time = evt.trigger.time  #.to_datetime()

            if doIntegration:
                waveform = evt.r0.tel[0].waveform
                bad_pix = evt.mon.tel[0].pixel_status.hardware_failing_pixels
                charges, times = NeighborPeakIntegration(waveform=waveform.copy(),camera=camera,bad_pix=bad_pix) # copy as it can touch waveform
                charges[ bad_pix ] = 0
                times[ bad_pix ] = 0

                # touch the R0 data to remove the annoying 65535 as bad waveform value.
                # Should we do it or delegate that to later ?
                waveform[ bad_pix ] = 0
                #dd['waveform'].dump( waveform )
                dd['charge'].dump( copy.deepcopy(charges), time=event_time )
                dd['T0'].dump( copy.deepcopy(times), time=event_time )

            # dd['trigger'].dump( evt.trigger )
            # dd['mon'].dump( evt.mon )
            # dd['simulation'].dump( evt.simulation )
            # dd['count'].dump( evt.count )
            # dd['pointing'].dump( evt.pointing )
            # dd['calibration'].dump( evt.calibration )
            # dd['nectarcam'].dump( evt.nectarcam )
            # dd['index'].dump( evt.index )
            # dd['r0'].dump( evt.r0 )
            # dd['r1'].dump( evt.r1 )

            # Split the whold content of the rawdata file

            # Don't transfer r1 as it is too big and poorly filled anyway at the moment.        
            if keepR1:
                exclusion = {""}
            else:
                exclusion = {"r1"}
                
            for k in evt.keys():
                if k not in exclusion:
                    if onlytrigger and k!="trigger":
                        continue
                    else:
                        dd[k].dump( copy.deepcopy(getattr(evt,k)), time=event_time )

def TrueOrFalse(arg):
    ua = str(arg).upper()
    if ua == "TRUE" or ua == "1":
       return True
    elif ua == "FALSE" or ua == "0":
       return False
    else:
       raise ValueError("Given argument should be either True or False")


def ExtractInformation(arglist):
    ## To adapt to loop on telescopes

    p = argparse.ArgumentParser(description='Extract R0 information from a run and split it in multiple sub-files to speed up exploration',
                                epilog='examples:\n'
                                '\t python %(prog)s --run 123456 --data-path /Users/vm273425/Programs/NectarCAM/data/ --dest-path /Users/vm273425/Programs/NectarCAM/scripts/ \n',
                                formatter_class=CustomFormatter)

    p.add_argument("--run", dest='runs', type=int , nargs='+', help="Run numbers to be converted. Can be one or more.")
    p.add_argument("--data-path",dest='dataPath',type=str,default=None,help="Path to the rawdata directory. The program will recursively search in all directory for matching rawdata")
    p.add_argument("--dest-path",dest='destPath',type=str,default="None",help="Path to the destination directory where the extracted data will be written")
    p.add_argument("--apply-calib",dest="applyCalib",type=str,default="True",help="Don't apply the calibration coefficient. Useful when the channel selection has been done")
    p.add_argument("--keep-r1",dest='keepR1',type=str,default="True",help="Save the R1 data if True")
    p.add_argument("--split",dest='split',action='store_true',help='Split the files per groups. 0-1 together in a file, 2-3 in another, etc... Need the ctapipe_io_nectarcam version compatible with ctapipe 0.18')
    p.add_argument("--nnint",dest='nnint',action='store_true',help='Do an integration of the data using Next Neighbor Peak Search. At the moment hard coded to be 10 ns -4 and +6 ns after the max. Will create charge and TO data set')
    p.add_argument("--only-trigger",dest='onlytrig',action='store_true',help='Extract only the trigger information to a file. Useful for big runs')

    args = p.parse_args(arglist)

    if args.runs is None:
        p.print_help()
        return -1

    dataPath = args.dataPath if args.dataPath is not None else GetDefaultDataPath()

    applyCalib = TrueOrFalse(args.applyCalib)
    keepR1 = TrueOrFalse(args.keepR1)



    for run in args.runs:


        data_path = FindDataPath(run,dataPath)
        if args.destPath == "None":
            dest_path = data_path
        else:
            dest_path = args.destPath


        if False and args.split:

            runs = list()
            paths = list()
            dest_paths = list()
            blocks = list()
            calib = list()
            keepR1 = list()
            nnints = list()
            trigonly = list()

            for block in range(GetNumberOfDataBlocks(run,data_path)):
            #for block in range(8):
                #print(block)
                runs.append(run)
                paths.append(data_path)
                dest_paths.append(dest_path)
                blocks.append(block)
                calib.append(applyCalib)
                keepR1.append(  keepR1)
                nnints.append(args.nnint)
                trigonly.append(args.onlytrig)

            # Make the Pool of workers
            pool = ThreadPool(4)

            # Open the URLs in their own threads
            # and return the results
            results = pool.starmap(ExtractInformationSingleRun, zip(runs,paths,dest_paths,blocks,calib,keepR1,nnints,trigonly) )

            # Close the pool and wait for the work to finish
            pool.close()
            pool.join()
            #ExtractInformationSingleRun(run,args.dataPath,args.destPath,data_block=block)

        else:
            if args.split:
                nBlocks = GetNumberOfDataBlocks(run,data_path)
                for block in range(nBlocks):
                    print(f'block: {block+1}/{nBlocks}')
                    ExtractInformationSingleRun(run=run,data_path=data_path,dest_path=dest_path,data_block=block,applycalib=applyCalib,keepR1=keepR1,nnint=args.nnint,onlytrigger=args.onlytrig)
            else:
                ExtractInformationSingleRun(run=run,data_path=data_path,dest_path=dest_path,data_block=-1,applycalib=applyCalib,keepR1=keepR1,nnint=args.nnint,onlytrigger=args.onlytrig)
                
            #def ExtractInformationSingleRun(run,data_path,dest_path,data_block,applycalib=True,keepR1=True):


if __name__ == "__main__":
    #run = 3830
    #data_path='/Users/vm273425/Programs/NectarCAM/data/'
    ExtractInformation(sys.argv[1:])

    #tels=[0]
    #cameras=[cameras]
