try:
    import argparse
    import copy
    import sys
    from collections import defaultdict

    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy.ma as ma
    from astropy.time import TimeDelta
    from ctapipe.containers import EventType
    from ctapipe.visualization import CameraDisplay
    from FileHandler import GetNectarCamEvents
    from IPython import embed
    from numba import jit, njit, prange
    from PickleDataManager import PickleDataWriter
    from Stats import CameraSampleStats, CameraStats, Stats
    from tqdm import tqdm
    from Utils import (
        CustomFormatter,
        GetCamera,
        GetDefaultDataPath,
        IntegrationMethod,
        LightEvent,
        SignalIntegration,
        save_simple_data,
    )

except ImportError as e:
    print(e)
    raise SystemExit


def IsFFEventCandidate(trig_mask, npix_thresh=1835):
    return np.count_nonzero(trig_mask) > npix_thresh


### Tools to compute the FWHM
def lin_interp(x, y, i, half):
    return x[i] + (x[i + 1] - x[i]) * ((half - y[i]) / (y[i + 1] - y[i]))


def half_max_x(x, y, maxfrac=0.5):
    half = maxfrac * max(y)
    signs = np.sign(np.add(y, -half))
    for i in np.where(signs == 0.0)[0]:
        signs[i] = signs[i - 1] if i > 0 else signs[i + 1]
    print(signs)
    zero_crossings = signs[0:-2] != signs[1:-1]
    # print(zero_crossings)
    zero_crossings_i = np.where(zero_crossings)[0]
    # print(zero_crossings_i)
    if len(zero_crossings_i) > 0:
        try:
            delta = [
                lin_interp(x, y, zero_crossings_i[0], half),
                lin_interp(x, y, zero_crossings_i[1], half),
            ]
        except Exception:
            delta = [0.0, 0.0]
            # return lin_interp(x, y, zero_crossings_i[1], half) - lin_interp(x, y, zero_crossings_i[0], half)
    else:
        delta = [0.0, 0.0]
    return delta


def half_max_dist(x, y, maxfrac=0.5):
    coords = half_max_x(x, y, maxfrac=maxfrac)
    try:
        dist = coords[1] - coords[0]
    except Exception:
        dist = 0.0
    return dist


def get_fwhms(excess, nSamples=60, maxfrac=0.5):
    slice_times = np.linspace(0, nSamples - 1, nSamples)
    fwhm = np.empty(shape=excess.shape[0:-1])
    for i in range(fwhm.shape[0]):
        for j in range(fwhm.shape[1]):
            fwhm[i, j] = half_max_dist(slice_times, excess[i, j], maxfrac=maxfrac)
    return fwhm


@njit(debug=False)
def lin_interp_fast(x, y, i, half):
    res = x[i] + (x[i + 1] - x[i]) * ((half - y[i]) / (y[i + 1] - y[i]))
    return res


@njit(debug=False)
def half_max_x_fast(x, y, maxfrac=0.5):
    half = maxfrac * max(y)
    signs = np.sign(np.add(y, -half))
    try:
        crossings = np.where(signs == 0.0)[0]
        for i in np.where(signs == 0.0)[0]:
            signs[i] = signs[i - 1] if i > 0 else signs[i + 1]
    except Exception:
        pass
    zero_crossings = signs[0:-2] != signs[1:-1]
    zero_crossings_i = np.where(zero_crossings)[0]
    if len(zero_crossings_i) >= 2:
        try:
            delta = [
                lin_interp_fast(x, y, zero_crossings_i[0], half),
                lin_interp_fast(x, y, zero_crossings_i[1], half),
            ]
        except Exception:
            delta = [0.0, 0.0]
    else:
        delta = [0.0, 0.0]
    return delta


@njit(debug=False)
def half_max_dist_fast(x, y, maxfrac=0.5):
    coords = half_max_x_fast(x, y, maxfrac=maxfrac)
    try:
        dist = coords[1] - coords[0]
    except Exception:
        dist = 0.0
    return dist


@njit(parallel=True)
def get_fwhms_fast(excess, nSamples=60, maxfrac=0.5):
    slice_times = np.linspace(0, nSamples - 1, nSamples)
    fwhm = np.empty(shape=excess.shape[0:-1])
    for i in range(fwhm.shape[0]):
        for j in prange(fwhm.shape[1]):
            fw = half_max_dist_fast(slice_times, excess[i, j], maxfrac)
            fwhm[i, j] = fw
    return fwhm


def ExtractLightEvent(arglist):
    p = argparse.ArgumentParser(
        description="Extract events according to criteria",
        epilog="examples:\n"
        "\t python %(prog)s --run 123456  \nWarning: code not adapted for the true R1 data with gain selection",
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
        "--nevents",
        dest="nEvents",
        type=int,
        default=-1,
        help="Number of event to be analysed",
    )
    p.add_argument(
        "--ff-no-internal",
        dest="ffNoInternal",
        action="store_true",
        help="Tell the code that we are dealing with a FF run, having real illumination but no internal trigger",
    )
    p.add_argument(
        "--internal-ff",
        dest="internalFF",
        action="store_true",
        help="Tell the code that the FF event are only in internal mode so it has to guess if the event is a FF",
    )
    p.add_argument(
        "--ff-npixel-threshold",
        dest="ffNPixThreshold",
        type=int,
        default=1800,
        help="Number of pixel to declare a ff valid",
    )
    p.add_argument(
        "--r0", dest="saveR0", action="store_true", help="Save the r0 for all events"
    )
    p.add_argument(
        "--telid",
        dest="telId",
        type=int,
        default=0,
        help="Telescope id for which we show the data distributions",
    )
    p.add_argument(
        "--ped-length",
        dest="pedLength",
        type=int,
        default=16,
        help="Number of slice to use for pedestal estimation at the beginning of each event",
    )
    p.add_argument(
        "--noise-thresh",
        dest="noiseThresh",
        type=float,
        default=1e6,
        help="Threshold to filter out noise in waveform (only useful if no HV)",
    )

    args = p.parse_args(arglist)

    if args.run is None:
        p.print_help()
        return -1

    if args.dataPath is None:
        path = GetDefaultDataPath()
    else:
        path = args.dataPath

    run = args.run
    return ExtractLightEventForRun(
        run=run,
        path=path,
        nUserEvents=args.nEvents,
        ffNPixThreshold=args.ffNPixThreshold,
        ffNoInternal=args.ffNoInternal,
        internalFF=args.internalFF,
        saveR0=args.saveR0,
        telId=args.telId,
        noiseThresh=args.noiseThresh,
        pedLength=args.pedLength,
    )


def ExtractLightEventForRun(
    run,
    path,
    nUserEvents=-1,
    ffNPixThreshold=1800,
    ffNoInternal=False,
    internalFF=False,
    saveR0=False,
    telId=0,
    noiseThresh=1e6,
    pedLength=16,
):
    data = GetNectarCamEvents(run=run, path=path, applycalib=False, load_feb_info=True)
    if data is None:
        return 2
    camera = GetCamera()

    nEvents = len(data)
    nMax = np.minimum(nUserEvents, nEvents) if nUserEvents > 0 else nEvents
    default_mask = None

    ped_mask = None

    with PickleDataWriter(f"run_{run}_lightevents.pickle.lz4") as outFile:
        for i, evt in enumerate(tqdm(data, total=nMax)):
            if i == nMax:
                break

            R1Data = evt.r0.tel[telId].waveform is None
            waveform = (
                evt.r1.tel[telId].waveform if R1Data else evt.r0.tel[telId].waveform
            )

            trigger = copy.deepcopy(evt.trigger)

            nectarcam = copy.deepcopy(evt.nectarcam)
            nectarcam.tel[telId].evt.trigger_pattern = np.any(
                evt.nectarcam.tel[telId].evt.trigger_pattern, axis=0
            )
            ffcandidate = internalFF and IsFFEventCandidate(
                nectarcam.tel[telId].evt.trigger_pattern, ffNPixThreshold
            )
            nectarcam.tel[telId].dst = None
            nectarcam.tel[telId].svc = None

            r0 = None
            mon = None
            evt_charge = None
            evt_timing = None

            npix = np.count_nonzero(
                np.any(evt.nectarcam.tel[telId].evt.trigger_pattern, axis=0)
            )
            bad_hdw = evt.mon.tel[telId].pixel_status.hardware_failing_pixels

            if R1Data:
                bad_hdw = bad_hdw[telId]

            if ped_mask is None:
                ped_mask = (
                    np.tile(np.arange(waveform.shape[-1]), waveform.shape[:-1] + (1,))
                    < pedLength
                )

            # embed()

            mon = copy.deepcopy(evt.mon)

            # if evt.trigger.event_type == EventType.SUBARRAY and not ffcandidate:
            #     if npix>40 and npix<=1855:
            #         r0 = copy.deepcopy( evt.r1 if R1Data else evt.r0 )
            #         mon = copy.deepcopy( evt.mon )
            # if evt.trigger.event_type == EventType.FLATFIELD or ffcandidate:
            #     if  npix < ffNPixThreshold and not ffNoInternal:
            #         r0 = copy.deepcopy( evt.r0 )
            #         mon = copy.deepcopy( evt.mon )

            if saveR0:
                r0 = copy.deepcopy(evt.r0)
                r0.tel[telId].waveform = evt.r0.tel[telId].waveform.astype(np.int16)

            if default_mask is None or default_mask.shape != waveform.shape:
                default_mask = np.zeros_like(waveform, dtype=bool)

            le = LightEvent(r0=r0, trigger=trigger, nectarcam=nectarcam, mon=mon)

            ## do the integration only for the Flatfield at the moment (because of disk size)
            le.saturated.tel[telId] = np.any(
                waveform > 4000, axis=-1
            )  # In principle 4095 but I put a safety margin

            ## Fill waveform statistic information
            le.stat.tel[telId] = LightEvent.WaveformStat(waveform)

            integration_method = None
            if evt.trigger.event_type == EventType.SKY_PEDESTAL:
                integration_method = None
            elif (evt.trigger.event_type == EventType.FLATFIELD or ffcandidate) and (
                npix > ffNPixThreshold or ffNoInternal
            ):
                integration_method = IntegrationMethod.PEAKSEARCH
                # integration_method = IntegrationMethod.NNSEARCH
            else:
                integration_method = IntegrationMethod.PEAKSEARCH

            if integration_method is not None:
                hdw_fail_wvf = default_mask | np.expand_dims(bad_hdw, axis=-1)

                # ped = np.mean( waveform[0:16], axis=-1, keepdims=True )
                ped = np.mean(waveform, axis=-1, keepdims=True, where=ped_mask)
                # embed()
                excess_wvf = waveform - ped
                evt_charge, evt_timing = SignalIntegration(
                    excess_wvf,
                    hdw_fail_wvf,
                    method=integration_method,
                    left_bound=5,
                    right_bound=9,
                    camera=camera,
                    peakpositions=None,
                )

                excess_wvf[hdw_fail_wvf] = 0.0
                fwhm = get_fwhms_fast(excess_wvf)

                le.charge.tel[telId] = evt_charge.astype(np.float16)
                le.charge.method = integration_method
                le.timing.tel[telId] = evt_timing.astype(np.int16)
                le.fwhm.tel[telId] = fwhm.astype(np.float16)

            # embed()

            if evt.trigger.event_type == EventType.SKY_PEDESTAL:
                mean_wvf = np.mean(waveform, axis=-1)
                std_wvf = np.std(waveform, axis=-1)
                noisy = std_wvf > noiseThresh
                mean_wvf[bad_hdw | noisy] = 0.0
                le.pedestal.tel[telId] = mean_wvf.copy()
            elif (
                evt.trigger.event_type == EventType.FLATFIELD
                and npix < 30
                and not ffNoInternal
            ):  # 30 is arbitrary, I don't know
                ## Likely a Random Pedestal --> Treate it like pedestal
                mean_wvf = np.mean(waveform, axis=-1)
                std_wvf = np.std(waveform, axis=-1)
                noisy = std_wvf > noiseThresh
                mean_wvf[bad_hdw | noisy] = 0.0
                le.pedestal.tel[telId] = mean_wvf.copy()

            outFile.append(le)
    return 0


if __name__ == "__main__":
    ExtractLightEvent(sys.argv[1:])
