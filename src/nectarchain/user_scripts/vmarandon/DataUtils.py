try:
    import sys
    import os
    import lz4.frame
    import bz2
    import lzma
    import pickle
    from glob import glob

    from enum import Enum

    import astropy.units as u

    from ctapipe_io_nectarcam import NectarCAMEventSource, EventSource
    from ctapipe.containers import EventType

    #from traitlets.config import Config
    from astropy.time import TimeDelta


    from DBHandler import DB
    from FileHandler import GetNectarCamEvents, DataReader
    #from Utils import GetRunURL
    from Utils import GetDefaultDataPath
    from tqdm import tqdm

    from collections import defaultdict

except ImportError as e:
    print(e)
    raise SystemExit



def GetTotalEventCounts(run,path=None):

    if path is None:
        path = GetDefaultDataPath()

    ## in case one does not have the "latest" version of the ctapipe_io_nectarcam (for ctapipe 0.19)
    ## otherwise the len(data) would work !
    data = GetNectarCamEvents(run,path,applycalib=False)
    
    tot_events = 0
    
    if hasattr(data, '__len__') and callable( getattr(data, '__len__') ):
        tot_events = len(data)
    else:
        # This line is what the len function is doing ig it exists. Nevertheless, I prefer to be consistent when it's possible.
        for v in data.multi_file._events_table.values():
            tot_events += len(v)

    return tot_events


def CountEventTypes(run,path=None):
    if path is None:
        path = GetDefaultDataPath()

    trig_types = defaultdict(int)

    data = DataReader(run,path)
    ok = data.Connect("trigger")
    if not ok:
        data = GetNectarCamEvents(run,path,applycalib=False)

    nEvents = GetTotalEventCounts(run,path)

    for evt in tqdm(data,total=nEvents):
        trig_types[ evt.trigger.event_type ] += 1
    return trig_types
    



def GetRunTimeBoundaries(run,path=None):

    if path is None:
        path = GetDefaultDataPath()

    start_time = None
    end_time = None
    data = DataReader(run,path)
    ok = data.Connect("trigger")
    if not ok:
        data = GetNectarCamEvents(run,path,applycalib=False)

    for evt in tqdm(data):
        if start_time is None:
            start_time = data.trigger.time
        end_time = data.trigger.time

    return start_time,end_time



def GetLongRunTimeEdges(run,path=None,event_type=None,delta_t_second=10.):

    #print(path)
    if path is None:
        path = GetDefaultDataPath()

    if event_type is None:
        event_type = EventType.SKY_PEDESTAL
    
    nEvents = GetTotalEventCounts(run,path)

    #print(nEvents)

    data = DataReader(run,path=path)
    if not data.Connect("trigger"):
        data = GetNectarCamEvents(run=run,path=path,applycalib=False)

    times_edges = list()

    time_start = None
    time_end = None

    previous_time = None

    # if there is a time gap of more than delta_t seconds, then we consider that this is the end of a data block
    delta_t = TimeDelta(delta_t_second,format="sec")
    try:
        for evt in tqdm(data,total=nEvents):
            current_time = evt.trigger.time
            
            if time_start is None:
                time_start = current_time
            
            if evt.trigger.event_type == event_type:

                if previous_time is None:
                    previous_time = current_time
                
                if current_time - previous_time > delta_t:
                    #print(f"time: {time} previous time: {previous_time} delta: {(time - previous_time).to_value('s')}")
                    #if (previous_time - time) > delta_t:
                    times_edges.append( (time_start,previous_time) )
                    time_start = current_time

                previous_time = current_time
    except Exception as err:
        print(f"Error while reading file: [{err}]")
    times_edges.append( (time_start,current_time) )
    # write the last time
    #print(f"There is : {len(times_edges)} intervals")
    return times_edges


if __name__ == "__main__":
    print("DataUtils is not meant to be run ==> You have likely done something wrong !")
