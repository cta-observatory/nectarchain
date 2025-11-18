try:
    import argparse
    import datetime
    import enum
    import os
    import pickle
    import re
    from glob import glob

    import astropy
    import lz4.frame
    import numpy as np
    import scipy as sp
    from astropy.time import Time
    from ctapipe.containers import EventType
    from ctapipe.coordinates import EngineeringCameraFrame
    from ctapipe.instrument import CameraGeometry

    # from datetime import datetime, timedelta
    from ctapipe.io import EventSource
    from numba import njit, prange
    from scipy.interpolate import InterpolatedUnivariateSpline
    from tqdm import tqdm

    # from scipy import interpolate
    # from scipy.interpolate import splrep, BSpline
    # from FileHandler import GetNectarCamEvents #, DataReer

except ImportError as e:
    print(e)
    raise SystemExit


# Let's do a multi-inheritence for the fun, since argp-arse does not
# provide it, maybe because it was too easy....
# thanks : http://stackoverflow.com/questions/18462610
class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    pass


def ConvertTitleToName(title: str):
    title = title.replace(" ", "_")
    title = title.replace("\n", "_")
    title = title.replace("(", "_")
    title = title.replace(")", "_")
    title = title.replace(":", "_")
    title = title.replace(",", "_")
    title = title.replace("=", "_")
    title = title.replace("_-_", "_")
    title = title.replace(">", "Above")
    title = title.replace("<", "Below")
    # title = title.replace("0.","0d")
    title = title.replace(".", "d")

    ## do a bit of cleanup of the _
    while title.find("__") != -1:
        title = title.replace("__", "_")
    return title


def GetDefaultDataPath():
    return os.environ.get("NECTARCAMDATA", "./")


def GetDefaultDBPath():
    return os.environ.get("NECTARCAMDB", "./")


def GetDAQTimeFromTime(t):
    if isinstance(t, astropy.time.core.Time):
        print("GetDAQTimeFromTime> converting to datetime")
        t = t.to_datetime()
    if t.hour >= 12:
        daq_time = datetime.datetime(year=t.year, month=t.month, day=t.day, hour=12)
    else:
        t_past = t - datetime.timedelta(seconds=86400)
        daq_time = datetime.datetime(
            year=t_past.year, month=t_past.month, day=t_past.day, hour=12
        )
    return daq_time


def GetDAQDateFromTime(t):
    # A datetime is expected

    if t.hour >= 12:
        str_time = t.strftime("%Y-%m-%d")
    else:
        t_past = t - datetime.timedelta(seconds=86400)
        str_time = t_past.strftime("%Y-%m-%d")
    print(str_time)
    return str_time


def GetDBNameFromTime(t):
    return "nectarcam_monitoring_db_" + GetDAQDateFromTime(t) + ".sqlite"


def ReformatTime(t):
    return Time(t.value, format=t.format, precision=9)


def NaNFiltered(x):
    return x[~np.isnan(x)]


def ReplaceNaN(x, val=0):
    x[np.isnan(x)] = val


# def RemoveVal(x,val=0):
#    x[ x == val ] = np.nan
def FindFile(filename, path):
    # print(f"FindFile> {filename = } {path = }")
    for dirpath, _, filenames in os.walk(path):
        if filename in filenames:
            return os.path.join(dirpath, filename)


def FindFiles(filename, path, recursive=True, remove_hidden_files=True):
    # As it is regular expression, you should not use * but .* , etc...
    filename = filename.replace(".*", "*").replace(
        "*", ".*"
    )  # dirty trick to have the wild card * working as one can use in a command line
    files = list()
    for dirpath, _, filenames in os.walk(path):
        ## Go for a pedestrian way as list comprehension is a bit unreadable in this case:
        for name in filenames:
            matchPattern = re.match(filename, name)
            hidden_file = name.startswith(".")
            if matchPattern and not (hidden_file and remove_hidden_files):
                files.append(os.path.abspath(os.path.join(dirpath, name)))
        if not recursive:
            break
        # for f in filenames:
        #     #print(f"{filename = } {f = }")
        #     if re.match(filename,f):
        #         files.append( os.path.join(dirpath,f))
    return files


def FindDataPath(run, dataPath):
    files = FindFiles(f".*Run{run}.*.fits.fz", dataPath)
    ## now guess the path based on the most numerous entries
    paths = [os.path.dirname(f) for f in files]
    likely_path = max(set(paths), key=paths.count)
    return likely_path


def GetRunURL(run, path):
    pattern = f"NectarCAM.Run{run:04}."
    runpath = ""
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            if f.startswith(pattern):
                runpath = dirpath
                break
    return runpath + "/" + pattern + "*.fits.fz"


def GetNumberOfDataBlocks(run, path):
    run_url = GetRunURL(run, path)
    # print(f'{run_url = }')
    files = glob(run_url)
    # print(f'{files = }')
    nFiles = len(files)
    # print(f'{nFiles = }')
    nBlocks = nFiles // 2
    # print(f'{nBlocks = }')
    return nBlocks


def GetBlockListFromURL(run_url, data_block):
    files = glob(run_url)
    files.sort()
    # print(f"GetBlockListFromURL> {files[ 2*data_block : 2*(data_block+1) ]}")
    return files[2 * data_block : 2 * (data_block + 1)]


def GetCamera():
    return CameraGeometry.from_name("NectarCam-003").transform_to(
        EngineeringCameraFrame()
    )


def clean_waveform_with_badpix(wvf, bad_pix, use_nan=True):
    bad_value = 0
    if use_nan:
        wvf = np.array(wvf, dtype="float")
        bad_value = np.nan

    if bad_pix is not None:
        wvf[bad_pix] = bad_value
    return wvf

    # bad_pix = evt.mon.tel[0].pixel_status.hardware_failing_pixels


def clean_waveform(wvf, use_nan=True):
    waveform_fl = np.array(wvf, dtype="float")
    for chan in np.arange(waveform_fl.shape[0]):
        for pix in np.arange(waveform_fl.shape[1]):
            if (
                np.count_nonzero(waveform_fl[chan, pix, :] == 65535)
                == waveform_fl.shape[2]
            ):
                waveform_fl[chan, pix, :] = np.nan if use_nan else 0.0
    return waveform_fl


class IntegrationMethod(enum.Enum):
    PEAKSEARCH = enum.auto()
    NNSEARCH = enum.auto()
    USERPEAK = enum.auto()


def SignalIntegration(
    waveform,
    exclusion_mask=None,
    method=IntegrationMethod.PEAKSEARCH,
    left_bound=5,
    right_bound=7,
    camera=None,
    peakpositions=None,
):
    wvf = waveform.astype(float)
    if exclusion_mask is not None:
        wvf[exclusion_mask] = 0.0

    if method == IntegrationMethod.PEAKSEARCH:
        charge, time = PeakIntegration(
            wvf, left_bound=left_bound, right_bound=right_bound
        )
    elif method == IntegrationMethod.NNSEARCH:
        if camera is None:
            camera = GetCamera()
        if len(wvf.shape) == 3:
            charge, time = NeighborPeakIntegration2Gain(
                wvf,
                camera.neighbor_matrix,
                left_bound=left_bound,
                right_bound=right_bound,
            )
        else:
            charge, time = NeighborPeakIntegrationGainSelected(
                wvf,
                camera.neighbor_matrix,
                left_bound=left_bound,
                right_bound=right_bound,
            )
    elif method == IntegrationMethod.USERPEAK:
        # print("HERE")
        charge, time = UserPeakIntegration(
            wvf,
            peakpositions=peakpositions,
            left_bound=left_bound,
            right_bound=right_bound,
        )
    else:
        print("I Don't know about this method !")

    return charge, time


@njit(parallel=True)
def PeakIntegration(waveform, left_bound=5, right_bound=7):
    wvf_shape = waveform.shape
    n_channel = wvf_shape[0]
    n_pixels = wvf_shape[1]
    n_samples = wvf_shape[2]

    integrated_signal = np.zeros((n_channel, n_pixels))
    signal_timeslice = np.zeros((n_channel, n_pixels))

    integ_window = left_bound + right_bound

    for chan in prange(2):
        chan_wvf = waveform[chan, :, :]
        for pix in range(n_pixels):
            trace = chan_wvf[pix, :]
            peak_pos = np.argmax(trace)
            if (peak_pos - left_bound) < 0:
                lo_bound = 0
                up_bound = integ_window
            elif (peak_pos + right_bound) > n_samples:
                lo_bound = n_samples - integ_window
                up_bound = n_samples
            else:
                lo_bound = peak_pos - left_bound
                up_bound = peak_pos + right_bound
            integrated_signal[chan, pix] = np.sum(trace[lo_bound:up_bound])
            signal_timeslice[chan, pix] = peak_pos

    return integrated_signal, signal_timeslice


@njit(parallel=True)
def NeighborPeakIntegration2Gain(
    waveform, neighbor_matrix, left_bound=5, right_bound=7
):
    wvf_shape = waveform.shape
    n_channel = wvf_shape[0]
    n_pixels = wvf_shape[1]
    n_samples = wvf_shape[2]

    integrated_signal = np.zeros((n_channel, n_pixels))
    signal_timeslice = np.zeros((n_channel, n_pixels))

    integ_window = left_bound + right_bound

    for chan in prange(2):
        chan_wvf = waveform[chan, :, :]
        for pix in range(n_pixels):
            neighbor_trace = np.sum(chan_wvf[neighbor_matrix[pix]], axis=0)
            peak_pos = np.argmax(neighbor_trace)
            if (peak_pos - left_bound) < 0:
                lo_bound = 0
                up_bound = integ_window
            elif (peak_pos + right_bound) > n_samples:
                lo_bound = n_samples - integ_window
                up_bound = n_samples
            else:
                lo_bound = peak_pos - left_bound
                up_bound = peak_pos + right_bound
            integrated_signal[chan, pix] = np.sum(chan_wvf[pix, lo_bound:up_bound])
            signal_timeslice[chan, pix] = peak_pos

    return integrated_signal, signal_timeslice


@njit(parallel=True)
def NeighborPeakIntegrationGainSelected(
    waveform, neighbor_matrix, left_bound=5, right_bound=7
):
    wvf_shape = waveform.shape
    n_pixels = wvf_shape[0]
    n_samples = wvf_shape[1]

    integrated_signal = np.zeros((n_pixels,))
    signal_timeslice = np.zeros((n_pixels,))

    integ_window = left_bound + right_bound

    for pix in prange(n_pixels):
        neighbor_trace = np.sum(waveform[neighbor_matrix[pix]], axis=0)
        peak_pos = np.argmax(neighbor_trace)
        if (peak_pos - left_bound) < 0:
            lo_bound = 0
            up_bound = integ_window
        elif (peak_pos + right_bound) > n_samples:
            lo_bound = n_samples - integ_window
            up_bound = n_samples
        else:
            lo_bound = peak_pos - left_bound
            up_bound = peak_pos + right_bound
        integrated_signal[pix] = np.sum(waveform[pix, lo_bound:up_bound])
        signal_timeslice[pix] = peak_pos

    return integrated_signal, signal_timeslice


@njit(parallel=True)
def UserPeakIntegration(waveform, peakpositions, left_bound=5, right_bound=7):
    wvf_shape = waveform.shape
    n_channel = wvf_shape[0]
    n_pixels = wvf_shape[1]
    n_samples = wvf_shape[2]

    integrated_signal = np.zeros((n_channel, n_pixels))
    signal_timeslice = np.zeros((n_channel, n_pixels))

    integ_window = left_bound + right_bound

    for chan in prange(2):
        chan_wvf = waveform[chan, :, :]
        for pix in range(n_pixels):
            trace = chan_wvf[pix, :]
            peak_pos = peakpositions[chan, pix]
            if (peak_pos - left_bound) < 0:
                lo_bound = 0
                up_bound = integ_window
            elif (peak_pos + right_bound) > n_samples:
                lo_bound = n_samples - integ_window
                up_bound = n_samples
            else:
                lo_bound = peak_pos - left_bound
                up_bound = peak_pos + right_bound
            integrated_signal[chan, pix] = np.sum(trace[lo_bound:up_bound])
            signal_timeslice[chan, pix] = peak_pos

    return integrated_signal, signal_timeslice


def getPixelT0Spline(waveform):
    times = np.arange(0, len(waveform))
    pmax = np.argmax(waveform)
    # tmax = times[ pmax ]

    ts = times[pmax - 5 : pmax + 6]
    ws = waveform[pmax - 5 : pmax + 6]
    interp = InterpolatedUnivariateSpline(ts, ws)

    ts = times[pmax - 1 : pmax + 2]
    it = np.linspace(ts[0], ts[-1], (len(ts) - 1) * 100 + 1)
    iw = interp(it)

    ipmax = np.argmax(iw)
    itmax = it[ipmax]


def getPixelRiseTime(waveform):
    times = np.arange(0, len(waveform))
    pmax = np.argmax(waveform)

    ts = times[pmax - 5 : pmax + 6]
    ws = waveform[pmax - 5 : pmax + 6]
    interp = InterpolatedUnivariateSpline(ts, ws).derivative()

    ts = times[pmax - 4 : pmax + 1]
    it = np.linspace(ts[0], ts[-1], (len(ts) - 1) * 100 + 1)
    iw = interp(it)

    ipmax = np.argmax(iw)
    itmax = it[ipmax]
    return itmax


def GetMirrorModules(camera=None):
    mod_pos = list()

    cam = GetCamera() if camera is None else camera

    pix_x = cam.pix_x.to_value("m")
    pix_y = cam.pix_y.to_value("m")

    for m in range(265):
        pos_xs = pix_x[7 * m : 7 * (m + 1)]
        pos_ys = pix_y[7 * m : 7 * (m + 1)]
        mean_xs = np.mean(pos_xs)
        mean_ys = np.mean(pos_ys)
        mod_pos.append([mean_xs, mean_ys])

    mod_pos = np.array(mod_pos)

    m = 0

    mod_2_mirror = 265 * [0]
    mirror_2_mod = 265 * [0]

    # for m in tqdm(range(265)):
    for m in tqdm(range(265)):
        mean_xs = mod_pos[m, 0]
        mean_ys = mod_pos[m, 1]

        mirror_xs = -mean_xs
        mirror_ys = mean_ys

        dists = np.empty(265)
        for mm in range(265):
            mm_xs = mod_pos[mm, 0]
            mm_ys = mod_pos[mm, 1]

            dist = (mm_xs - mirror_xs) ** 2.0 + (mm_ys - mirror_ys) ** 2.0
            dists[mm] = dist
        mirror_module = np.argmin(dists)
        mod_2_mirror[m] = mirror_module
        mirror_2_mod[mirror_module] = m

    return np.array(mod_2_mirror)

    # pix_2_mirror = list()
    # mirror_2_pix = list()
    # for m in range(265):
    #     mm = mod_2_mirror[m]
    #     for p in range(7):
    #         pix = 7*mm+p
    #         pix_2_mirror.append(pix)
    # for mm in range(265):
    #     m = mod_2_mirror[mm]
    #     for p in range(7):
    #         pix = 7*m+p
    #         mirror_2_pix.append(pix)
    # print(pix_2_mirror)
    # print(mirror_2_pix)


# def GetPixelMaxRiseTime(waveform):
#     nsamples = len(waveform)
#     times = np.arange(0,nsamples)

#     times_oversample =  np.linspace(0,nsamples-1,2*nsamples-1)
#     f = sp.interpolate.interp1d(times, waveform,fill_value="extrapolate",kind="linear")
#     wvf_oversample = f(times_oversample)

#     ydiff = wvf_oversample[3:] + wvf_oversample[2:-1] - wvf_oversample[1:-2] - wvf_oversample[:-3]
#     xdiff = times_oversample[2:-1]
#     tmax_pos =  np.argmax(ydiff)
#     tmax = xdiff[ tmax_pos ]
#     #return tmax

#     trange = 6
#     nbins = 2*trange*50 + 1
#     pos_min = tmax_pos-trange
#     pos_max = tmax_pos+trange
#     pos_min = pos_min if pos_min>=0 else 0
#     pos_max = pos_max if pos_max<len(times_oversample) else len(times_oversample)-1
#     #print(f"tmax_pos: {tmax_pos} pos_min: {pos_min} pos_max: {pos_max}")
#     tck = sp.interpolate.splrep( times_oversample[pos_min:pos_max+1],wvf_oversample[pos_min:pos_max+1])
#     newtimes = np.linspace(times_oversample[pos_min],times_oversample[pos_max],nbins)
#     newtrace = sp.interpolate.BSpline(*tck)(newtimes)
#     return newtimes[ np.argmax(newtrace) ]


# def GetPixelT0Spline(waveform):
#     times = np.arange(0,len(waveform))
#     tmax_pos = np.argmax(waveform)
#     #tmax = times[ tmax_pos ]

#     trange = 3
#     nbins = 2*trange*100 + 1 #to get to 0.01s
#     pos_min = tmax_pos-trange
#     pos_max = tmax_pos+trange
#     pos_min = pos_min if pos_min>=0 else 0
#     pos_max = pos_max if pos_max<len(times) else len(times)-1
#     #print(f"pos_min: {pos_min} pos_max: {pos_max}")
#     tck = sp.interpolate.splrep( times[pos_min:pos_max+1],waveform[pos_min:pos_max+1])
#     newtimes = np.linspace(times[pos_min],times[pos_max],nbins)
#     newtrace = sp.interpolate.BSpline(*tck)(newtimes)
#     return newtimes[ np.argmax(newtrace) ]

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
    # print(f"event_str: [{event_str}]")

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


def save_simple_data(datas, fileName):
    """Save info in a file using pickle"""
    with lz4.frame.open(fileName, "wb") as f:
        pickle.dump(datas, f)

    # with open(fileName,'wb') as file:
    #    pickle.dump(datas,file)


def read_simple_data(fileName):
    """Read info from a file using pickle"""
    with lz4.frame.open(fileName, "rb") as f:
        return pickle.load(f)


class TriggerInfos:
    def __init__(
        self,
        trigs=None,
        event_id=None,
        event_types=None,
        event_times=None,
        busy_counts=None,
        trig_pats=None,
        feb_abs_evt_id=None,
        feb_evt_id=None,
        feb_pps_cnt=None,
        feb_ts1=None,
        feb_ts2_trig=None,
        feb_ts2_pps=None,
        ext_device=None,
        ucts_stereo_pattern=None,
        ucts_trigger_type=None,
        missing_module_info=None,
        ucts_timestamp=None,
    ):
        self.trigs = trigs
        self.event_id = event_id
        self.event_types = event_types
        self.event_times = event_times
        self.busy_counts = busy_counts
        self.trig_pats = trig_pats
        self.feb_abs_evt_id = feb_abs_evt_id
        self.feb_evt_id = feb_evt_id
        self.feb_pps_cnt = feb_pps_cnt
        self.feb_ts1 = feb_ts1
        self.feb_ts2_trig = feb_ts2_trig
        self.feb_ts2_pps = feb_ts2_pps
        self.ext_device = ext_device
        self.ucts_stereo_pattern = ucts_stereo_pattern
        self.ucts_trigger_type = ucts_trigger_type
        self.missing_module_info = missing_module_info
        self.ucts_timestamp = ucts_timestamp

    def get_event_deltat(self, unit="µs"):
        dt = np.array(
            [
                0.0,
            ]
            + [dt.to_value(unit) for dt in self.event_times[1:] - self.event_times[:-1]]
        )
        return dt

    def get_feb_time_ns(self):
        rough_ts = self.feb_pps_cnt * 1.0e9 + self.feb_ts1 * 8.0
        prec_ts = rough_ts + self.feb_ts2_trig - self.feb_ts2_pps
        return prec_ts

    def get_feb_deltat_ns(self):
        feb_times = self.get_feb_time_ns()
        dt_feb_ns = np.zeros_like(feb_times)
        dt_feb_ns[1:] = feb_times[1:] - feb_times[:-1]
        return dt_feb_ns

    def get_trigger_time_datetime(self):
        return np.array([t.to_datetime() for t in tqdm(self.event_times)])

    def get_delta_busy(self):
        db = np.zeros_like(self.busy_counts)
        db[1:] = self.busy_counts[1:] - self.busy_counts[:-1]
        return db


class LightEvent:
    class Charge:
        def __init__(self):
            self.tel = dict()
            self.method = None

    class Saturated:
        def __init__(self):
            self.tel = dict()

    class Timing:
        def __init__(self):
            self.tel = dict()

    class Pedestal:
        def __init__(self):
            self.tel = dict()

    class FWHM:
        def __init__(self):
            self.tel = dict()

    class Stat:
        def __init__(self):
            self.tel = dict()

    class WaveformStat:
        def __init__(self, wvf):
            self.min = wvf.min(axis=-1)
            self.max = wvf.max(axis=-1)
            self.mean = wvf.mean(axis=-1).astype(np.float16)
            self.std = wvf.std(axis=-1, ddof=1).astype(np.float16)

        @property
        def ptp(self):
            return self.max - self.min

    def __init__(self, r0=None, trigger=None, nectarcam=None, mon=None):
        self.r0 = r0
        self.trigger = trigger
        self.nectarcam = nectarcam
        self.mon = mon
        self.charge = LightEvent.Charge()
        self.timing = LightEvent.Timing()
        self.pedestal = LightEvent.Pedestal()
        self.saturated = LightEvent.Saturated()
        self.fwhm = LightEvent.FWHM()
        self.stat = LightEvent.Stat()
