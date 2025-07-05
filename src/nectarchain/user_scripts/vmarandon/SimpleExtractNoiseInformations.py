try:
    import argparse
    import sys
    from functools import partial

    import numpy as np
    from ctapipe.containers import ArrayEventContainer, EventType
    from ctapipe.core import Container, Field, Map, TelescopeComponent, Tool
    from ctapipe.core.traits import Bool, Integer, Long, Path
    from ctapipe.instrument import SubarrayDescription
    from FileHandler import GetNectarCamEvents
    from IPython import embed
    from NoiseInfos import NoiseInfos
    from Stats import CameraSampleStats, CameraStats, Stats
    from tqdm import tqdm

    # from DataUtils import GetNectarCamEvents, GetDefaultDataPath
    from Utils import (
        CustomFormatter,
        GetDefaultDataPath,
        GetEventTypeFromString,
        save_simple_data,
    )

except ImportError as e:
    print(e)
    raise SystemExit


def SimpleExtractNoiseInformations(arglist):
    p = argparse.ArgumentParser(
        description="Show stats on the hardware_failing_pixels flag",
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
        "--telid",
        dest="telId",
        type=int,
        default=0,
        help="Telescope id for which we show the data distributions",
    )
    p.add_argument(
        "--event-type",
        dest="eventType",
        type=str,
        default="SKY_PEDESTAL",
        help="Event type to be used for the pedestal extraction",
    )
    p.add_argument(
        "--max-event",
        dest="maxevent",
        type=int,
        default=-1,
        help="Max number of event to analyze",
    )

    args = p.parse_args(arglist)

    if args.run is None:
        p.print_help()
        return -1

    if args.dataPath is None:
        path = GetDefaultDataPath()
    else:
        path = args.dataPath

    # print(path)
    print(f"Extracting noise infos for run [{args.run}]")
    events = GetNectarCamEvents(run=args.run, path=path, applycalib=False)
    # print(type(events))
    # print(len(events))
    event_type = GetEventTypeFromString(args.eventType)

    FirstEventSeen = False

    noiseinfos = NoiseInfos()

    firstEventTime = None

    default_mask = None
    default_fft_mask = None

    nMaxEvent = min(args.maxevent, len(events)) if args.maxevent > 0 else len(events)

    try:
        for ie, evt in enumerate(tqdm(events, total=nMaxEvent)):
            hardware_failing = evt.mon.tel[
                args.telId
            ].pixel_status.hardware_failing_pixels
            if ie >= nMaxEvent:
                break
            if not FirstEventSeen:
                FirstEventSeen = True
                continue

            if firstEventTime is None:
                firstEventTime = evt.trigger.time

            if noiseinfos.firstEventTime is None:
                noiseinfos.firstEventTime = evt.trigger.time
            noiseinfos.lastEventTime = evt.trigger.time

            if evt.trigger.event_type not in noiseinfos.nEventTypes:
                noiseinfos.nEventTypes[evt.trigger.event_type] = 0
            noiseinfos.nEventTypes[evt.trigger.event_type] += 1

            # if evt.trigger.event_type == EventType.SKY_PEDESTAL or args.run == 4938:
            if evt.trigger.event_type == event_type:
                # 4938 is a SPE run for which I forgot to put the HV on
                hdw_fail = evt.mon.tel[args.telId].pixel_status.hardware_failing_pixels
                hdw_good = ~hdw_fail

                wvf = evt.r0.tel[args.telId].waveform
                std_wvf = wvf.std(axis=-1)
                noisy_pixels = std_wvf > 4.5
                clean_pixels = ~noisy_pixels

                noiseinfos.stats.add(noisy_pixels.astype(int), validmask=hdw_good)
                noiseinfos.nEvents += 1

                noiseinfos.std_stats.add(std_wvf, validmask=hdw_good)
                noiseinfos.std_noise_stats.add(
                    std_wvf, validmask=(hdw_good & noisy_pixels)
                )

                if default_mask is None:
                    default_mask = np.full_like(wvf, fill_value=True, dtype=bool)

                noiseinfos.waveform.add(
                    wvf, validmask=(default_mask & hdw_good[:, :, None])
                )

                ## FFT part.
                ## Separate the FFT for the noisy and for the cleaned pixels
                fft_wvf = np.fft.rfft(wvf, axis=-1)
                fft_norm2 = np.real(fft_wvf * np.conjugate(fft_wvf))

                if default_fft_mask is None:
                    default_fft_mask = np.full_like(
                        fft_norm2, fill_value=True, dtype=bool
                    )

                if noiseinfos.fft_all is None:
                    noiseinfos.fft_all = Stats(shape=fft_norm2.shape)
                    noiseinfos.fft_noise = Stats(shape=fft_norm2.shape)
                    noiseinfos.fft_clean = Stats(shape=fft_norm2.shape)

                good_hdw_fft_mask = default_fft_mask & hdw_good[:, :, None]
                good_hdw_fft_mask = good_hdw_fft_mask & (
                    (np.min(fft_norm2, axis=-1) > 1.0e-3)[:, :, None]
                )  ## Sometimes there is values close to 0 in the FFT, remove those
                noiseinfos.fft_all.add(fft_norm2, validmask=good_hdw_fft_mask)
                noiseinfos.fft_noise.add(
                    fft_norm2, validmask=(good_hdw_fft_mask & noisy_pixels[:, :, None])
                )
                noiseinfos.fft_clean.add(
                    fft_norm2, validmask=(good_hdw_fft_mask & clean_pixels[:, :, None])
                )

    except Exception as e:
        print(e)

    noiseinfos.livetime = (evt.trigger.time - firstEventTime).sec

    # save data
    save_simple_data(noiseinfos, f"run_{args.run}_noiseinfos.pickle")


if __name__ == "__main__":
    SimpleExtractNoiseInformations(sys.argv[1:])
