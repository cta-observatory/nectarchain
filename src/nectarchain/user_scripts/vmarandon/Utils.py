try:
    import os
    import numpy as np
    from glob import glob
    import argparse
    import datetime
    import re
    import enum

    from ctapipe.io import EventSource
    from ctapipe.instrument import CameraGeometry
    from ctapipe.coordinates import EngineeringCameraFrame
    from ctapipe.containers import EventType


    from tqdm import tqdm

    from numba import njit, prange
    
    import scipy as sp
    #from scipy import interpolate
    #from scipy.interpolate import splrep, BSpline


    #from FileHandler import GetNectarCamEvents #, DataReader

except ImportError as e:
    print(e)
    raise SystemExit

#Let's do a multi-inheritence for the fun, since argp-arse does not
#provide it, maybe because it was too easy....
# thanks : http://stackoverflow.com/questions/18462610
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def GetDefaultDataPath():
    return os.environ.get( 'NECTARCAMDATA' , '/Users/vm273425/Programs/NectarCAM/data')

def GetDAQDateFromTime(t):
     # A datetime is expected

    if t.hour>=12:
        str_time = t.strftime('%Y-%m-%d')
    else:
        t_past = t - datetime.timedelta(seconds=86400)
        str_time = t_past.strftime('%Y-%m-%d')
    print(str_time)
    return str_time

def GetDBNameFromTime(t):
     return "nectarcam_monitoring_db_" + GetDAQDateFromTime(t) + ".sqlite"

def NaNFiltered(x):
    return x[~np.isnan(x)]

def ReplaceNaN(x,val=0):
    x[ np.isnan(x) ] = val

#def RemoveVal(x,val=0):
#    x[ x == val ] = np.nan
def FindFile(filename,path):
    #print(f"FindFile> {filename = } {path = }")
    for (dirpath, _ , filenames) in os.walk(path):
        if filename in filenames:
            return os.path.join(dirpath,filename)

def FindFiles(filename,path):
    # As it is regular expression, you should not use * but .* , etc...
    filename = filename.replace('.*','*').replace('*','.*') # dirty trick to have the wild card * working as one can use in a command line
    files = list()
    for (dirpath, _ , filenames) in os.walk(path):
        for f in filenames:
            #print(f"{filename = } {f = }")
            if re.match(filename,f):
                files.append( os.path.join(dirpath,f))
    return files

def FindDataPath(run,dataPath):
    files = FindFiles(f".*Run{run}.*.fits.fz",dataPath)
    ## now guess the path based on the most numerous entries
    paths = [ os.path.dirname(f) for f in files ]
    likely_path = max(set(paths), key=paths.count)
    return likely_path


def GetRunURL(run,path):
    pattern = f'NectarCAM.Run{run}.'
    runpath = ''
    for (dirpath, dirnames, filenames) in os.walk(path):
        for f in filenames:
            if f.startswith(pattern):
                runpath = dirpath
                break
    return runpath + '/' + pattern + '*.fits.fz'

def GetNumberOfDataBlocks(run,path):
    run_url = GetRunURL(run,path)
    #print(f'{run_url = }')
    files = glob(run_url)
    #print(f'{files = }')
    nFiles = len(files)
    #print(f'{nFiles = }')
    nBlocks = nFiles//2
    #print(f'{nBlocks = }')
    return nBlocks    

def GetBlockListFromURL(run_url,data_block):
    files = glob(run_url)
    files.sort()
    #print(f"GetBlockListFromURL> {files[ 2*data_block : 2*(data_block+1) ]}")
    return files[ 2*data_block : 2*(data_block+1) ]



def GetCamera():
    return CameraGeometry.from_name("NectarCam-003").transform_to(EngineeringCameraFrame())

def clean_waveform_with_badpix(wvf, bad_pix, use_nan=True):
    bad_value = 0
    if use_nan:
        wvf = np.array(wvf,dtype='float')
        bad_value = np.nan

    if bad_pix is not None:
        wvf[ bad_pix ] = bad_value  
    return wvf      

    

    #bad_pix = evt.mon.tel[0].pixel_status.hardware_failing_pixels
def clean_waveform(wvf, use_nan = True):
    waveform_fl = np.array(wvf,dtype='float')
    for chan in np.arange( waveform_fl.shape[0] ):
        for pix in np.arange( waveform_fl.shape[1] ):
            if np.count_nonzero( waveform_fl[chan,pix,:] == 65535 ) == waveform_fl.shape[2]:
                waveform_fl[chan,pix,:] = np.nan if use_nan else 0.
    return waveform_fl


class IntegrationMethod(enum.Enum):
    PEAKSEARCH = enum.auto()
    NNSEARCH = enum.auto()




def SignalIntegration(waveform,exclusion_mask=None, method = IntegrationMethod.PEAKSEARCH,left_bound=5,right_bound=7,camera=None):
    wvf = waveform.astype(float)
    if exclusion_mask is not None:
        wvf[ exclusion_mask ] = 0.

    if method == IntegrationMethod.PEAKSEARCH:
        charge, time = PeakIntegration(waveform, left_bound=left_bound, right_bound=right_bound)
    elif method == IntegrationMethod.NNSEARCH:
        if camera is None:
            camera = GetCamera()
        charge, time = NeighborPeakIntegration(waveform, camera.neighbor_matrix, left_bound=5, right_bound=7)
    else:
        print("I Don't know about this method !")

    return charge, time 

@njit(parallel=True)
def PeakIntegration(waveform, left_bound=5, right_bound=7):

    wvf_shape = waveform.shape
    n_channel = wvf_shape[0]
    n_pixels = wvf_shape[1]
    n_samples = wvf_shape[2]

    integrated_signal = np.zeros( (n_channel,n_pixels) )
    signal_timeslice = np.zeros( (n_channel,n_pixels) )

    integ_window = left_bound+right_bound

    for chan in prange(2):
        chan_wvf = waveform[chan,:,:]
        for pix in range(n_pixels):
            trace = chan_wvf[pix,:]
            peak_pos = np.argmax(trace)
            if (peak_pos-left_bound) < 0:
                lo_bound = 0
                up_bound = integ_window
            elif (peak_pos + right_bound) > n_samples:
                lo_bound = n_samples-integ_window
                up_bound = n_samples
            else:
                lo_bound = peak_pos - left_bound
                up_bound = peak_pos + right_bound
            integrated_signal[chan,pix] = np.sum( trace[lo_bound:up_bound]  )
            signal_timeslice[chan,pix] = peak_pos
            
    return integrated_signal, signal_timeslice

@njit(parallel=True)
def NeighborPeakIntegration(waveform, neighbor_matrix, left_bound=5, right_bound=7):

    wvf_shape = waveform.shape
    n_channel = wvf_shape[0]
    n_pixels = wvf_shape[1]
    n_samples = wvf_shape[2]

    integrated_signal = np.zeros( (n_channel,n_pixels) )
    signal_timeslice = np.zeros( (n_channel,n_pixels) )

    integ_window = left_bound+right_bound

    for chan in prange(2):
        chan_wvf = waveform[chan,:,:]
        for pix in range(n_pixels):
            neighbor_trace = np.sum(chan_wvf[ neighbor_matrix[pix] ],axis=0)
            peak_pos = np.argmax(neighbor_trace)
            if (peak_pos-left_bound) < 0:
                lo_bound = 0
                up_bound = integ_window
            elif (peak_pos + right_bound) > n_samples:
                lo_bound = n_samples-integ_window
                up_bound = n_samples
            else:
                lo_bound = peak_pos - left_bound
                up_bound = peak_pos + right_bound
            integrated_signal[chan,pix] = np.sum( chan_wvf[pix,lo_bound:up_bound]  )
            signal_timeslice[chan,pix] = peak_pos
            
    return integrated_signal, signal_timeslice



def GetPixelMaxRiseTime(waveform):
    nsamples = len(waveform)
    times = np.arange(0,nsamples)
    
    times_oversample =  np.linspace(0,nsamples-1,2*nsamples-1)
    f = sp.interpolate.interp1d(times, waveform,fill_value="extrapolate",kind="linear")
    wvf_oversample = f(times_oversample)

    ydiff = wvf_oversample[3:] + wvf_oversample[2:-1] - wvf_oversample[1:-2] - wvf_oversample[:-3]
    xdiff = times_oversample[2:-1]
    tmax_pos =  np.argmax(ydiff)
    tmax = xdiff[ tmax_pos ]
    #return tmax

    trange = 6
    nbins = 2*trange*50 + 1
    pos_min = tmax_pos-trange
    pos_max = tmax_pos+trange
    pos_min = pos_min if pos_min>=0 else 0
    pos_max = pos_max if pos_max<len(times_oversample) else len(times_oversample)-1
    #print(f"tmax_pos: {tmax_pos} pos_min: {pos_min} pos_max: {pos_max}")
    tck = sp.interpolate.splrep( times_oversample[pos_min:pos_max+1],wvf_oversample[pos_min:pos_max+1])
    newtimes = np.linspace(times_oversample[pos_min],times_oversample[pos_max],nbins)
    newtrace = sp.interpolate.BSpline(*tck)(newtimes)
    return newtimes[ np.argmax(newtrace) ]



def GetPixelT0Spline(waveform):
    times = np.arange(0,len(waveform))
    tmax_pos = np.argmax(waveform)
    #tmax = times[ tmax_pos ]

    trange = 3
    nbins = 2*trange*100 + 1 #to get to 0.01s
    pos_min = tmax_pos-trange
    pos_max = tmax_pos+trange
    pos_min = pos_min if pos_min>=0 else 0
    pos_max = pos_max if pos_max<len(times) else len(times)-1
    #print(f"pos_min: {pos_min} pos_max: {pos_max}")
    tck = sp.interpolate.splrep( times[pos_min:pos_max+1],waveform[pos_min:pos_max+1])
    newtimes = np.linspace(times[pos_min],times[pos_max],nbins)
    newtrace = sp.interpolate.BSpline(*tck)(newtimes)
    return newtimes[ np.argmax(newtrace) ]

# @njit(parallel=True)
# def NeighborPeakIntegrationCameraAgnostic(waveform, n_pixels=None, neighbor_matrix=None, nchannel=None, left_bound=5, right_bound=7):

#     integrated_signal = np.zeros( (nchannel,n_pixels) )
#     signal_timeslice = np.zeros( (nchannel,n_pixels) )

#     n_samples = waveform.shape[2]

#     integ_window = left_bound+right_bound


#     for chan in prange(2):
#         chan_wvf = waveform[chan,:,:]
#         for pix in range(n_pixels):
#             neighbor_trace = np.sum(chan_wvf[ neighbor_matrix[pix] ],axis=0)
#             peak_pos = np.argmax(neighbor_trace)
#             if (peak_pos-left_bound) < 0:
#                 lo_bound = 0
#                 up_bound = integ_window
#             elif (peak_pos + right_bound) > n_samples:
#                 lo_bound = n_samples-integ_window
#                 up_bound = n_samples
#             else:
#                 lo_bound = peak_pos - left_bound
#                 up_bound = peak_pos + right_bound
#             integrated_signal[chan,pix] = np.sum( chan_wvf[pix,lo_bound:up_bound]  )
#             signal_timeslice[chan,pix] = peak_pos
            
#     return integrated_signal, signal_timeslice



def GetEventTypeFromString(event_str):
    
    if event_str == "FLATFIELD":
        evt_type = EventType.FLATFIELD
    elif event_str == "SINGLE_PE":
        evt_type = EventType.SINGLE_PE
    elif event_str == "SKY_PEDESTAL":
        evt_type = EventType.SKY_PEDESTAL
    elif event_str == "DARK_PEDESTAL":
        evt_type = EventType.DARK_PEDESTAL
    elif event_str == "ELECTRONIC_PEDESTAL":
        evt_type = EventType.ELECTRONIC_PEDESTAL
    elif event_str == "OTHER_CALIBRATION":
        evt_type = EventType.OTHER_CALIBRATION
    elif event_str == "MUON":
        evt_type = EventType.MUON
    elif event_str == "HARDWARE_STEREO":
        evt_type = EventType.HARDWARE_STEREO
    elif event_str == "DAQ":
        evt_type = EventType.DAQ
    elif event_str == "SUBARRAY":
        evt_type = EventType.SUBARRAY
    else:
        print(f"WARNING> Don't know about the event type [{event_str}]")
        evt_type = EventType.UNKNOWN
    return evt_type
