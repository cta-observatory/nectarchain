try:
    import argparse
    import sys
    import time
    import warnings
    from collections import defaultdict
    from pprint import pprint

    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy.ma as ma
    from astropy.time import TimeDelta
    from CalibrationData import CalibInfo, PedestalInfo
    from ctapipe.containers import EventType
    from ctapipe.visualization import CameraDisplay
    from DataUtils import CountEventTypes, GetLongRunTimeEdges, GetTotalEventCounts
    from FileHandler import DataDumper, DataReader, GetNectarCamEvents
    from IPython import embed
    from tqdm import tqdm
    from Utils import (
        CustomFormatter,
        FindDataPath,
        GetCamera,
        GetDefaultDataPath,
        GetEventTypeFromString,
    )

except ImportError as e:
    print(e)
    raise SystemExit


class PedestalEstimator:
    def __init__(
        self,
        tel_id=0,
        min_events=500,
        events_per_slice=1000,
        cleaning=False,
        widthFilter=3.0,
        verbose=False,
    ):
        ## Cleaning mean clean the shower... It's not really working yet

        self.tel_id = tel_id
        self.firstEventTime = None
        self.currentEventTime = None

        self.nMinimumEvents = min_events
        self.nEventsPerSlice = events_per_slice
        self.firstEstimateDone = False if cleaning else True
        ## no need to distinguish between the first estimate and the rest if there is no cleaning to be done
        self.relativeWidthFilter = widthFilter  # sigma to remove event

        ## Need the previous pedestal and integrated
        self.Pedestal = None
        self.IntegratedPedestal = None

        self.listPedestalEvent = list()
        self.listBadMask = list()

        self.doCleaning = cleaning
        self.Updated = False

        self.verbose = verbose

    def ProcessEvent(self, evt):
        return self.AddEvent(evt)

    def AddEvent(self, evt):
        if self.firstEventTime is None:
            self.firstEventTime = evt.trigger.time

        self.currentEventTime = evt.trigger.time
        wvf_mask = (
            np.ones_like(evt.r0.tel[self.tel_id].waveform)
            & evt.mon.tel[self.tel_id].pixel_status.hardware_failing_pixels[:, :, None]
        )
        wvf = evt.r0.tel[self.tel_id].waveform.astype(float)
        self.Updated = False

        if self._PedestalReady(False):
            self._DoEstimation()
            self.Updated = True
            self.firstEstimateDone = True
            self.firstEventTime = self.currentEventTime
            self.listPedestalEvent.clear()
            self.listBadMask.clear()

        if self.firstEstimateDone and self.doCleaning:
            filter_mask = self._ContaminatedMask(
                wvf, evt.nectarcam.tel[self.tel_id].evt.event_id
            )
            wvf_mask = wvf_mask | filter_mask[:, :, None]

        self.listPedestalEvent.append(wvf)
        self.listBadMask.append(wvf_mask)

    def Finish(self):
        if self._PedestalReady(True):
            self._DoEstimation()
        self.Pedestal.endTime = self.currentEventTime

    def _ContaminatedMask(self, wvf, event_id=-1):
        # TO BE IMPLEMENTED

        # sum the trace to get the total charge

        # wvf_sum = wvf.sum(axis=2)
        # wvf_charge_sigma = np.abs( (wvf_sum - self.IntegratedPedestal.pos)/self.IntegratedPedestal.width )
        # contaminated_mask = wvf_charge_sigma > self.relativeWidthFilter
        # if contaminated_mask[0,1724]:
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
        # contaminated_mask[0] = contaminated_mask[0] | contaminated_mask[1]
        # contaminated_mask[1] = contaminated_mask[0] | contaminated_mask[1]
        return None

    def _GetPedestalEvents(self):
        # selected_pedestals = None

        ## A little bit weird
        selected_pedestals = np.array(self.listPedestalEvent)
        selected_badmask = np.array(self.listBadMask, dtype=bool)
        # selected_pedestals[ selected_badmask ] = np.nan

        if not (self.firstEstimateDone and self.doCleaning):
            # First remove the outliers by searching them on the integrated trace
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # ped_sum = np.nansum( selected_pedestals, axis = 3, keepdims=True ) # perhaps here we can remove nan ?
                # ped_sum_mean = np.nanmean(ped_sum, axis = 0, keepdims=True)
                # ped_sum_std  = np.nanstd(ped_sum, axis = 0, keepdims=True)
                ped_sum = np.sum(
                    selected_pedestals, axis=3, where=~selected_badmask, keepdims=True
                )  # perhaps here we can remove nan ?
                ped_sum_nanmask = np.isnan(ped_sum)
                ped_sum_goodmask = ~ped_sum_nanmask
                ped_sum_mean = np.mean(
                    ped_sum, axis=0, where=ped_sum_goodmask, keepdims=True
                )
                ped_sum_std = np.std(
                    ped_sum, axis=0, where=ped_sum_goodmask, keepdims=True, ddof=1
                )

                # too_big_mask = selected_pedestals > (ped_sum_mean + self.relativeWidthFilter*ped_sum_std)
                # too_low_mask = selected_pedestals < (ped_sum_mean - self.relativeWidthFilter*ped_sum_std)
                too_big_mask = ped_sum > (
                    ped_sum_mean + self.relativeWidthFilter * ped_sum_std
                )
                too_low_mask = ped_sum < (
                    ped_sum_mean - self.relativeWidthFilter * ped_sum_std
                )

                selected_badmask = (
                    selected_badmask | too_big_mask | too_low_mask | ped_sum_nanmask
                )

                # Check the number of entries... If too low --> remove pixel...
                # But what to do ? replace by the previous one ?
                entries = np.count_nonzero(
                    ~np.isnan(selected_pedestals), axis=0, keepdims=True
                )
                entries_badmask = entries < self.nMinimumEvents

                selected_badmask = selected_badmask | entries_badmask

                # selected_pedestals[ selected_badmask ] = np.nan

        return selected_pedestals, selected_badmask

    # Other method : filter out a fix fraction of events
    #            pedestals = ma.masked_array(self.listPedestalEvent)
    #            nEventsToRemove = int( pedestals.shape[0]*0.005 )
    #            #for i in range(nEventsToRemove):
    #
    #            sorted_pedestals = ma.sort(pedestals,axis=0)
    #            selected_pedestals = sorted_pedestals[nEventsToRemove:-nEventsToRemove]

    def _DoEstimation(self):
        tstart = time.perf_counter()
        selected_pedestals, selected_badmask = self._GetPedestalEvents()
        selected_validmask = ~selected_badmask
        tstop = time.perf_counter()
        dtime = tstop - tstart
        if self.verbose:
            print(f"time to select pedestals: {dtime} s")

        tstart = time.perf_counter()
        pedInfo = PedestalInfo()
        pedInfo.startTime = self.firstEventTime
        pedInfo.endTime = self.currentEventTime
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # print(f'{selected_validmask.shape = }')
            # print(f'{selected_pedestals.shape = }')
            pedInfo.pos = np.mean(selected_pedestals, where=selected_validmask, axis=0)
            pedInfo.width = np.std(
                selected_pedestals, where=selected_validmask, axis=0, ddof=1
            )
            # print(pedInfo.width)
            # embed()
            pedInfo.min = np.min(
                selected_pedestals, where=selected_validmask, initial=np.inf, axis=0
            )
            pedInfo.max = np.max(
                selected_pedestals, where=selected_validmask, initial=-np.inf, axis=0
            )
            pedInfo.nEvents = np.count_nonzero(~np.isnan(selected_pedestals), axis=0)

            bad_mask = np.isnan(pedInfo.width)
            pedInfo.pos = ma.array(pedInfo.pos, mask=bad_mask)
            pedInfo.width = ma.array(pedInfo.width, mask=bad_mask)
            pedInfo.min = ma.array(pedInfo.min, mask=bad_mask)
            pedInfo.max = ma.array(pedInfo.max, mask=bad_mask)
            pedInfo.nEvents = ma.array(pedInfo.nEvents, mask=bad_mask)

            # embed()
            # pedInfo.median = ma.median( selected_pedestals, axis=0 )
        self.Pedestal = pedInfo

        pedInfoIntegrated = PedestalInfo()
        sum_pedestals = np.sum(selected_pedestals, where=selected_validmask, axis=3)
        pedInfoIntegrated.startTime = self.firstEventTime
        pedInfoIntegrated.endTime = self.currentEventTime
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sum_pedestals_valid = ~np.isnan(sum_pedestals)
            pedInfoIntegrated.pos = np.mean(
                sum_pedestals, where=sum_pedestals_valid, axis=0
            )
            pedInfoIntegrated.width = np.std(
                sum_pedestals, where=sum_pedestals_valid, axis=0, ddof=1
            )
            pedInfoIntegrated.min = np.min(
                sum_pedestals, where=sum_pedestals_valid, initial=np.inf, axis=0
            )
            pedInfoIntegrated.max = np.max(
                sum_pedestals, where=sum_pedestals_valid, initial=-np.inf, axis=0
            )
            pedInfoIntegrated.nEvents = np.count_nonzero(
                ~np.isnan(sum_pedestals), axis=0
            )

            bad_mask = np.isnan(pedInfoIntegrated.width)
            pedInfoIntegrated.pos = ma.array(pedInfoIntegrated.pos, mask=bad_mask)
            pedInfoIntegrated.width = ma.array(pedInfoIntegrated.width, mask=bad_mask)
            pedInfoIntegrated.min = ma.array(pedInfoIntegrated.min, mask=bad_mask)
            pedInfoIntegrated.max = ma.array(pedInfoIntegrated.max, mask=bad_mask)
            pedInfoIntegrated.nEvents = ma.array(
                pedInfoIntegrated.nEvents, mask=bad_mask
            )

        self.IntegratedPedestal = pedInfoIntegrated

        tstop = time.perf_counter()
        dtime = tstop - tstart
        if self.verbose:
            print(f"time to compute pedestals: {dtime} s")

    def _PedestalReady(self, lastEstimate):
        nEntries = len(self.listPedestalEvent)
        if lastEstimate or not self.firstEstimateDone:
            nEntriesRequired = self.nMinimumEvents
        else:
            nEntriesRequired = self.nEventsPerSlice

        return nEntries >= nEntriesRequired


class PedestalEstimatorTimeInterval(PedestalEstimator):
    def __init__(self, time_edges, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_edges = time_edges
        self.current_time_edge = 0

    def _PedestalReady(self, lastEstimate):
        time_min = self.time_edges[self.current_time_edge][0]
        time_max = self.time_edges[self.current_time_edge][1]
        ready = False
        if time_max < self.currentEventTime:
            ready = True
            self.current_time_edge += 1
        # return super()._PedestalReady(lastEstimate)
        return ready


def EstimatePedestal(arglist):
    ## To adapt to loop on telescopes

    p = argparse.ArgumentParser(
        description="Estimate the Pedestal for a given run",
        epilog="examples:\n" "\t python %(prog)s --run 123456  \n",
        formatter_class=CustomFormatter,
    )

    p.add_argument("--run", dest="run", type=int, help="Run number to be converted")
    p.add_argument(
        "--data-path",
        dest="dataPath",
        type=str,
        default=None,
        help="Path to the rawdata directory. The program will recursively search in all directory for matching rawdata",
    )
    p.add_argument(
        "--event-type",
        dest="eventType",
        type=str,
        default="SKY_PEDESTAL",
        help="Event type to be used to be computed for the pedestal",
    )
    p.add_argument(
        "--use-time-interval",
        dest="useTimeInterval",
        action="store_true",
        help="Option for the runs that are done by bunch (like 5 seconds every 10 minutes). This will compute the time interval and use those to define slices",
    )
    p.add_argument(
        "--events", dest="nEvents", type=int, default=2000, help="Number of events"
    )
    p.add_argument(
        "--min-events",
        dest="minEvents",
        type=int,
        default=500,
        help="Minimum number of events to consider the pedestal valid",
    )
    p.add_argument(
        "--verbose", dest="verbose", action="store_true", help="print some information"
    )
    p.add_argument(
        "--widthcut",
        dest="widthCut",
        type=float,
        default=4.0,
        help="how many standard deviation of the integrated charge to consider a trace contaminated by a signal",
    )

    print(
        "WARNING> The code is not ready to deal with multiple telescopes... But will we ever be needed ?"
    )

    args = p.parse_args(arglist)

    if args.run is None:
        p.print_help()
        return -1

    print(f"Estimating Pedestal for run {args.run}")

    dataPath = args.dataPath if args.dataPath is not None else GetDefaultDataPath()

    # print(f'EventType: {args.eventType}')
    event_type = GetEventTypeFromString(args.eventType)
    # print(f'Event Type: [{event_type}]')

    # print(f"DataPath: {dataPath}")
    # evt.pedestal.tel[0]

    # if use_time_interval:
    #    ped_estimator = PedestalEstimatorTimeInterval(times_edges,min_events=min_events,events_per_slice=events_per_slice,cleaning=False)
    # else:
    #    ped_estimator = PedestalEstimator(min_events=min_events,events_per_slice=events_per_slice,cleaning=False)

    if args.useTimeInterval:
        times_edges = GetLongRunTimeEdges(
            run=args.run, path=dataPath, event_type=event_type
        )
        print(f"There is {len(times_edges)} time intervals")
        if args.verbose:
            for times in times_edges:
                print(f"[{times[0].datetime},{times[1].datetime}]")

    data = DataReader(args.run, path=dataPath)
    data.Connect("trigger", "r0", "mon", "nectarcam")

    counter = 0
    current_time_edge = 0

    dump_path = FindDataPath(args.run, dataPath)

    pedestal_infos = CalibInfo()

    ped_estimators = dict()

    with DataDumper(run=args.run, path=dump_path, data_block=0) as dd:
        pixel_ref = 1724
        counter = 0
        total_events = GetTotalEventCounts(args.run, dataPath)
        for evt in tqdm(data, total=total_events):
            if evt.trigger.event_type == event_type:
                for tel_id in evt.trigger.tels_with_trigger:
                    if tel_id not in ped_estimators:
                        if args.useTimeInterval:
                            ped_estimators[tel_id] = PedestalEstimatorTimeInterval(
                                times_edges,
                                tel_id=tel_id,
                                min_events=args.minEvents,
                                events_per_slice=args.nEvents,
                                cleaning=False,
                                widthFilter=args.widthCut,
                                verbose=args.verbose,
                            )
                        else:
                            ped_estimators[tel_id] = PedestalEstimator(
                                tel_id=tel_id,
                                min_events=args.minEvents,
                                events_per_slice=args.nEvents,
                                cleaning=False,
                                widthFilter=args.widthCut,
                                verbose=args.verbose,
                            )

                    ped_estimator = ped_estimators[tel_id]
                    ped_estimator.AddEvent(evt)

                    if ped_estimator.Updated:
                        if args.verbose:
                            pos = ped_estimator.Pedestal.pos[0, pixel_ref, 0]
                            width = ped_estimator.Pedestal.width[0, pixel_ref, 0]
                            nEvents = ped_estimator.Pedestal.nEvents[0, pixel_ref, 0]
                            err = width / np.sqrt(nEvents)
                            print(
                                f"NewPedestal> {counter} pedestal events, [{ped_estimator.Pedestal.startTime.datetime},{ped_estimator.Pedestal.endTime.datetime}] ped: {pos} width: {width} count: {nEvents} err: {err}"
                            )
                        pedestal_infos.tel[tel_id] = ped_estimator.Pedestal
                        dd["pedestal"].dump(
                            pedestal_infos,
                            ped_estimator.Pedestal.startTime,
                            ped_estimator.Pedestal.endTime,
                        )
                        # dd["pedestal"].dump(ped_estimator.Pedestal,ped_estimator.Pedestal.startTime,ped_estimator.Pedestal.endTime)
                counter += 1
                if len(ped_estimators) > 1:
                    print(
                        "WARNING> I CAN'T STORE PROPERLY INFORMATION IF THERE IS MORE THAN ONE TELESCOPE !!!!"
                    )

        #            if counter >= 10000:
        #                break

        for tel_id, ped_estimator in ped_estimators.items():
            ped_estimator.Finish()
            if ped_estimator.Updated:
                #            dd["pedestal"].dump(ped_estimator.Pedestal,ped_estimator.Pedestal.startTime,ped_estimator.Pedestal.endTime)
                pedestal_infos.tel[tel_id] = ped_estimator.Pedestal
                dd["pedestal"].dump(
                    pedestal_infos,
                    ped_estimator.Pedestal.startTime,
                    ped_estimator.Pedestal.endTime,
                )


if __name__ == "__main__":
    EstimatePedestal(sys.argv[1:])
