try:
    import time 
    import argparse
    import sys
    
    import numpy as np
    import numpy.ma as ma

    import astropy.units as u
    from astropy.time import TimeDelta
    from ctapipe.containers import EventType
    from ctapipe.visualization import CameraDisplay

    from FileHandler import DataDumper, DataReader, GetNectarCamEvents 
    from Utils import GetCamera, FindDataPath, CustomFormatter, GetEventTypeFromString, GetDefaultDataPath, NeighborPeakIntegration, IntegrationMethod, SignalIntegration
    from DataUtils import CountEventTypes, GetTotalEventCounts, GetLongRunTimeEdges
    from CalibrationData import FlatFieldInfo, CalibInfo

    from tqdm import tqdm
    
    from collections import defaultdict

    import matplotlib.pyplot as plt

    from collections import defaultdict

    from pprint import pprint

    from IPython import embed

except ImportError as e:
    print(e)
    raise SystemExit


class FlatFieldEstimator():
    def __init__(self,tel_id=0,valid_camera_fraction=0.9,min_charge=10.):
        self.tel_id = tel_id
        self.firstEventTime = None
        self.currentEventTime = None
        self.validFraction = valid_camera_fraction
        self.minCharge = min_charge
        self.pedestal = None
        self.gain = None
        self.hilo = None
        self.camera = GetCamera()
        self.Updated = False

        self.listFFEvent = list()
        self.listAmplitude = list()
        self.listFFBadMask = list()


    def SetPedestal(self,pedestal):
        self.pedestal = pedestal
    
    def SetGain(self,gain):
        self.gain = gain
    
    def SetHiLo(self,hilo):
        self.hilo = hilo


    def ProcessEvent(self,evt):
        
        if self.firstEventTime is None:
            self.firstEventTime = evt.trigger.time

        self.currentEventTime = evt.trigger.time
        mask_sample = np.ones_like( evt.r0.tel[self.tel_id].waveform ) & evt.mon.tel[self.tel_id].pixel_status.hardware_failing_pixels[:,:,None]

        wvf_adc = evt.r0.tel[self.tel_id].waveform - self.pedestal
        wvf_pe = wvf_adc/self.gain 
        wvf_pe[1] = wvf_adc[1]/self.hilo

        sum_pe, time_evt = SignalIntegration(wvf_pe, exclusion_mask=mask_sample, method = IntegrationMethod.PEAKSEARCH,left_bound=5,right_bound=7,camera=self.camera)

        
        badsum_mask = sum_pe == 0.
        sum_pe[badsum_mask] = np.nan

        nPixel_HG = np.count_nonzero( sum_pe[0] > self.minCharge )
        nPixel_LG = np.count_nonzero( sum_pe[1] > self.minCharge )
        nMinPixel = self.camera.n_pixels * self.validFraction
        validFF = (nPixel_HG > nMinPixel ) and (nPixel_LG > nMinPixel)

        if validFF:
            self.listAmplitude.append( sum_pe )
            ## compute FF:
            mean_charge = np.nanmean( sum_pe, axis=1)
            FFevent = mean_charge/sum_pe
            self.listFFEvent.append(FFevent)
            self.listFFBadMask.append(badsum_mask)

            if self._FFReady:
                self._DoEstimation()
                self.Updated = True
                self.firstEventTime = self.currentEventTime
                self.listFFEvent.clear()
                self.listFFBadMask.clear()


    def _FFReady(self,lastEstimate):
        nEntries = len(self.listFFEvent)
        if lastEstimate:
            nEntriesRequired = self.nMinimumEvents
        else:
            nEntriesRequired = self.nEventsPerSlice
        
        return nEntries >= nEntriesRequired

    def _DoEstimation(self):


        selected_pedestals = self._GetPedestalEvents()

        pedInfo = PedestalInfo()
        pedInfo.startTime = self.firstEventTime
        pedInfo.endTime = self.currentEventTime
        pedInfo.pos = ma.mean( selected_pedestals, axis=0 )
        pedInfo.width = ma.std( selected_pedestals, axis=0 )
        pedInfo.min = ma.min( selected_pedestals, axis=0 )
        pedInfo.max = ma.max( selected_pedestals, axis=0 )
        #pedInfo.median = ma.median( selected_pedestals, axis=0 )
        pedInfo.nEvents = ma.count( selected_pedestals, axis = 0)

        self.Pedestal = pedInfo

        pedInfoIntegrated = PedestalInfo()
        sum_pedestals = selected_pedestals.sum(axis=3)
        pedInfoIntegrated.startTime = self.firstEventTime
        pedInfoIntegrated.endTime = self.currentEventTime
        pedInfoIntegrated.pos = ma.mean( sum_pedestals, axis=0 )
        pedInfoIntegrated.width = ma.std( sum_pedestals, axis=0 )
        pedInfoIntegrated.min = ma.min( sum_pedestals, axis=0 )
        pedInfoIntegrated.max = ma.max( sum_pedestals, axis=0 )
        pedInfoIntegrated.nEvents = ma.count( sum_pedestals, axis = 0)

        self.IntegratedPedestal = pedInfoIntegrated





class PedestalEstimator():
    def __init__(self,tel_id=0,min_events=500,events_per_slice=1000,cleaning=True):
        self.tel_id = tel_id
        self.firstEventTime = None
        self.currentEventTime = None

        self.nMinimumEvents = min_events        
        self.nEventsPerSlice = events_per_slice
        self.firstEstimateDone = False if cleaning else True
        self.relativeWidthFilter = 3. # sigma to remove event

        ## Need the previous pedestal and integrated
        self.Pedestal = None
        self.IntegratedPedestal = None
        
        self.listPedestalEvent = list()

        self.doCleaning = cleaning
        self.Updated = False
        #print(f"min_events: {self.nMinimumEvents}")
        #print(f"events_per_slice: {self.nEventsPerSlice}")
        
    def ProcessEvent(self,evt):
        return self.AddEvent(evt)

    def AddEvent(self,evt):

        if self.firstEventTime is None:
            self.firstEventTime = evt.trigger.time

        self.currentEventTime = evt.trigger.time
        mask_sample = np.ones_like( evt.r0.tel[self.tel_id].waveform ) & evt.mon.tel[0].pixel_status.hardware_failing_pixels[:,:,None]
        wvf = ma.masked_array( evt.r0.tel[self.tel_id].waveform, mask = mask_sample )
        self.Updated = False
        
        if self._PedestalReady(False):
            self._DoEstimation()
            self.Updated = True
            self.firstEstimateDone = True
            self.firstEventTime = self.currentEventTime
            self.listPedestalEvent.clear()
    

        if self.firstEstimateDone and self.doCleaning:
            filter_mask = self._ContaminatedMask(wvf,evt.nectarcam.tel[0].evt.event_id)
            wvf.mask = wvf.mask | filter_mask[:,:,None]
        
        self.listPedestalEvent.append( wvf )

    def Finish(self):
        if self._PedestalReady(True):
            self._DoEstimation()
        self.Pedestal.endTime = self.currentEventTime



    def _ContaminatedMask(self,wvf,event_id=-1):   
        # sum the trace to get the total charge     
        wvf_sum = wvf.sum(axis=2) 
        wvf_charge_sigma = np.abs( (wvf_sum - self.IntegratedPedestal.pos)/self.IntegratedPedestal.width )
        contaminated_mask = wvf_charge_sigma > self.relativeWidthFilter        
        #if contaminated_mask[0,1724]:
        #    print(f"Pixel 1724> Event: {event_id} charge hg: {wvf_sum[0,1724] - self.IntegratedPedestal.pos[0,1724]} charge lg; {wvf_sum[1,1724] - self.IntegratedPedestal.pos[1,1724]} charge hg (sigma): {wvf_charge_sigma[0,1724]} charge lg (sigma): {wvf_charge_sigma[1,1724]}")

        # if len(contaminated_mask)>0:
        #     print("cleaned mask result : ")
        #     wvf_charge = (wvf_sum - self.IntegratedPedestal.pos)
        #     print( "charge : ", wvf_charge[contaminated_mask] )
        #     wvf_charge[0] = wvf_charge[0]/58.
        #     wvf_charge[1] = wvf_charge[1]/(58./15.)

        #     print( "charge pe : ", wvf_charge[contaminated_mask] )
        #     print( "charge (sigma): ", wvf_charge_sigma[contaminated_mask] )
        #     print( "pos and wifth: ", self.IntegratedPedestal.pos[0,1724], self.IntegratedPedestal.width[0,1570] )
        #     print( "calib pos:",self.IntegratedPedestal.pos[ contaminated_mask ],"width:",self.IntegratedPedestal.width[ contaminated_mask ])
        # Make it so that the two channel have the same mask
        contaminated_mask[0] = contaminated_mask[0] | contaminated_mask[1]
        contaminated_mask[1] = contaminated_mask[0] | contaminated_mask[1]
        return contaminated_mask


    def _GetPedestalEvents(self):
        selected_pedestals = None
        if self.firstEstimateDone and self.doCleaning:
            selected_pedestals = ma.masked_array( self.listPedestalEvent )
        else:
            selected_pedestals = ma.masked_array( self.listPedestalEvent )
            # First remove the outliers by searching them on the integrated trace
            ped_sum = selected_pedestals.sum( axis = 3, keepdims=True )
            ped_sum_mean = ped_sum.mean( axis = 0, keepdims=True)
            ped_sum_std  = ped_sum.std( axis = 0, keepdims=True) 
            ped_sum = ma.masked_greater(ped_sum, ped_sum_mean + 4*ped_sum_std, copy=False)
            ped_sum = ma.masked_less(ped_sum, ped_sum_mean - 4*ped_sum_std, copy=False)

            selected_pedestals.mask = ma.mask_or( selected_pedestals.mask,  ped_sum.mask, copy=True, shrink=False)

            entries = selected_pedestals.count(axis=0,keepdims=True)
            entry_mask = entries < self.nMinimumEvents
            selected_pedestals.mask = ma.mask_or( selected_pedestals.mask, entry_mask, copy=True, shrink=False)

            #embed()
        return selected_pedestals


# Other method : filter out a fix fraction of events
#            pedestals = ma.masked_array(self.listPedestalEvent)
#            nEventsToRemove = int( pedestals.shape[0]*0.005 )
#            #for i in range(nEventsToRemove):
#
#            sorted_pedestals = ma.sort(pedestals,axis=0)
#            selected_pedestals = sorted_pedestals[nEventsToRemove:-nEventsToRemove]


    def _DoEstimation(self):
        
        selected_pedestals = self._GetPedestalEvents()

        pedInfo = PedestalInfo()
        pedInfo.startTime = self.firstEventTime
        pedInfo.endTime = self.currentEventTime
        pedInfo.pos = ma.mean( selected_pedestals, axis=0 )
        pedInfo.width = ma.std( selected_pedestals, axis=0 )
        pedInfo.min = ma.min( selected_pedestals, axis=0 )
        pedInfo.max = ma.max( selected_pedestals, axis=0 )
        #pedInfo.median = ma.median( selected_pedestals, axis=0 )
        pedInfo.nEvents = ma.count( selected_pedestals, axis = 0)

        self.Pedestal = pedInfo

        pedInfoIntegrated = PedestalInfo()
        sum_pedestals = selected_pedestals.sum(axis=3)
        pedInfoIntegrated.startTime = self.firstEventTime
        pedInfoIntegrated.endTime = self.currentEventTime
        pedInfoIntegrated.pos = ma.mean( sum_pedestals, axis=0 )
        pedInfoIntegrated.width = ma.std( sum_pedestals, axis=0 )
        pedInfoIntegrated.min = ma.min( sum_pedestals, axis=0 )
        pedInfoIntegrated.max = ma.max( sum_pedestals, axis=0 )
        pedInfoIntegrated.nEvents = ma.count( sum_pedestals, axis = 0)

        self.IntegratedPedestal = pedInfoIntegrated

    def _PedestalReady(self,lastEstimate):
        nEntries = len(self.listPedestalEvent)
        if lastEstimate or not self.firstEstimateDone:
            nEntriesRequired = self.nMinimumEvents
        else:
            nEntriesRequired = self.nEventsPerSlice
        
        return nEntries >= nEntriesRequired

 
class PedestalEstimatorTimeInterval(PedestalEstimator):
    def __init__(self, time_edges, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.time_edges = time_edges
        self.current_time_edge = 0
        
    def _PedestalReady(self, lastEstimate):
        time_min = self.time_edges[self.current_time_edge][0]
        time_max = self.time_edges[self.current_time_edge][1]
        ready = False
        if time_max < self.currentEventTime:
            ready = True
            self.current_time_edge += 1
        #return super()._PedestalReady(lastEstimate)
        return ready
        

def EstimateFlatField(arglist):
    ## To adapt to loop on telescopes

    p = argparse.ArgumentParser(description='Estimate the FlatFielding for a given run',
                                epilog='examples:\n'
                                '\t python %(prog)s --run 123456  \n',
                                formatter_class=CustomFormatter)

    p.add_argument("--run", dest='run', type=int, help="Run number")
    p.add_argument("--data-path",dest='dataPath',type=str,default=None,help="Path to the rawdata directory. The program will recursively search in all directory for matching rawdata")
    p.add_argument("--event-type",dest='eventType',type=str,default="FLATFIELD",help='Event type to be used to be computed for the FlatField')
    p.add_argument("--use-time-interval",dest='useTimeInterval',action='store_true',help='Option for the runs that are done by bunch (like 5 seconds every 10 minutes). This will compute the time interval and use those to define slices')
    p.add_argument("--events",dest='nEvents',type=int,default=1000,help='Number of events')
    p.add_argument("--min-events",dest='minEvents',type=int,default=500,help='Minimum number of events to consider the FlatField to be valid')
    p.add_argument("--verbose",dest='verbose',action='store_true',help='print some information')

    print("WARNING> The code is not ready to deal with multiple telescopes... But will we ever be needed ?")

    args = p.parse_args(arglist)

    if args.run is None:
        p.print_help()
        return -1
    

    dataPath = args.dataPath if args.dataPath is not None else GetDefaultDataPath()

    event_type = GetEventTypeFromString(args.eventType)

    if args.useTimeInterval:
        times_edges = GetLongRunTimeEdges(run=args.run,path=dataPath,event_type=event_type)
        print(f"There is {len(times_edges)} time intervals")
        if args.verbose:
            for times in times_edges:
                print(f'[{times[0].datetime},{times[1].datetime}]')

    data = DataReader(args.run,path=dataPath)
    data.Connect("trigger","r0","mon","nectarcam")

    counter = 0
    current_time_edge = 0

    dump_path = FindDataPath(args.run,dataPath)

    pedestal_infos = CalibInfo()

    ped_estimators = dict()
    
    with DataDumper(run=args.run,path=dump_path,data_block=0) as dd:
        pixel_ref = 1724
        counter = 0
        total_events = GetTotalEventCounts(args.run,dataPath)
        for evt in tqdm(data,total=total_events):
            if evt.trigger.event_type == event_type:
                for tel_id in evt.trigger.tels_with_trigger:
                    if tel_id not in ped_estimators:
                        if args.useTimeInterval:
                            ped_estimators[tel_id] = PedestalEstimatorTimeInterval(times_edges,tel_id=tel_id,min_events=args.minEvents,events_per_slice=args.nEvents,cleaning=False)
                        else:
                            ped_estimators[tel_id] = PedestalEstimator(tel_id=tel_id,min_events=args.minEvents,events_per_slice=args.nEvents,cleaning=False)
                    
                    ped_estimator = ped_estimators[tel_id]
                    ped_estimator.AddEvent(evt)

                    if ped_estimator.Updated:
                        if args.verbose:
                            pos = ped_estimator.Pedestal.pos[0,pixel_ref,0]
                            width = ped_estimator.Pedestal.width[0,pixel_ref,0]
                            nEvents = ped_estimator.Pedestal.nEvents[0,pixel_ref,0]
                            err = width/np.sqrt(nEvents)
                            print(f"NewPedestal> {counter} pedestal events, [{ped_estimator.Pedestal.startTime.datetime},{ped_estimator.Pedestal.endTime.datetime}] ped: {pos} width: {width} count: {nEvents} err: {err}")
                        pedestal_infos.tel[tel_id] = ped_estimator.Pedestal
                        dd["pedestal"].dump(pedestal_infos,ped_estimator.Pedestal.startTime,ped_estimator.Pedestal.endTime)
                        #dd["pedestal"].dump(ped_estimator.Pedestal,ped_estimator.Pedestal.startTime,ped_estimator.Pedestal.endTime)
                counter += 1
                if len(ped_estimators)>1:
                    print("WARNING> I CAN'T STORE PROPERLY INFORMATION IF THERE IS MORE THAN ONE TELESCOPE !!!!")

#            if counter >= 10000:
#                break


        for tel_id, ped_estimator in ped_estimators.items():
            ped_estimator.Finish()
            if ped_estimator.Updated:
    #            dd["pedestal"].dump(ped_estimator.Pedestal,ped_estimator.Pedestal.startTime,ped_estimator.Pedestal.endTime)
                pedestal_infos.tel[tel_id] = ped_estimator.Pedestal
                dd["pedestal"].dump(pedestal_infos,ped_estimator.Pedestal.startTime,ped_estimator.Pedestal.endTime)



if __name__ == "__main__":
    EstimatePedestal(sys.argv[1:])


