try:
    import os
    import re
    from glob import glob
    import numpy as np
    import datetime
    from astropy.time import Time
    import astropy
    import protozfits
    from collections.abc import Iterable
    from tqdm import tqdm
except ImportError as err:
    print(err)
    raise SystemExit


def to_datetime(t):
    if t is None:
        t_corr = None
    elif isinstance(t, datetime.datetime):
        if t.tzinfo is None:
            # Assume this is actually utc
            t_corr = t.replace(tzinfo=datetime.timezone.utc)
        else:
            t_corr = t
    elif isinstance(t, astropy.time.core.Time):
        t_corr = t.utc.to_datetime(timezone=datetime.timezone.utc)
    elif isinstance(t, Iterable):
        # print(f"t is iterable: {t}")
        t_corr = list(map(to_datetime, t))
    else:
        raise ValueError(
            f"tmin (type: {type(t)}) is not of type datetime --> Problem !"
        )

    if isinstance(t, np.ndarray):  # Convert to ndarray if this was given
        t_corr = np.array(t_corr)

    return t_corr


def GetDefaultDataPath(default_path="./"):
    return os.environ.get("NECTARCAMDATA", default_path)


def GetRunURL(run, path):
    pattern = f"NectarCAM.Run{run:04}."
    runpath = ""
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            if f.startswith(pattern):
                runpath = dirpath
                break
    return runpath + "/" + pattern + "*.fits.fz"


def GetFirstLastEventTime(run, path=None):
    if path is None:
        path = GetDefaultDataPath()

    try:
        files = glob(GetRunURL(run, path))
        files.sort()
        evt_times = list()
        if len(files) > 0:
            for file in tqdm(files):
                with protozfits.File(file, pure_protobuf=False) as f:
                    nEvents = len(f.Events)
                    # get first event time
                    ranges = [range(nEvents), reversed(range(nEvents))]
                    for r in ranges:
                        for i in r:
                            t_s = f.Events[i].event_time_s
                            if t_s != 0:
                                t_qns = f.Events[i].event_time_qns
                                evt_times.append(
                                    Time(t_s, t_qns * 1.0e-9 / 4.0, format="unix_tai")
                                )
                                break
            print(len(evt_times))
            evt_times.sort()
            return evt_times[0], evt_times[-1]
        else:
            print(f"Can't find files for run {run}")
    except Exception as err:
        print(err)


def FindFile(filename, path):
    for dirpath, _, filenames in os.walk(path):
        if filename in filenames:
            # print(dirpath,filename)
            return os.path.join(dirpath, filename)


def FindFiles(filename, path, recursive=True, remove_hidden_files=True):
    # As it is regular expression, you should not use * but .* , etc...
    filename = filename.replace(".*", "*").replace(
        "*", ".*"
    )  # dirty trick to have the wild card * working as one can use in a command line
    files = list()
    for dirpath, _, filenames in os.walk(path):
        # Go for a pedestrian way as list comprehension
        # is a bit unreadable in this case:
        for name in filenames:
            matchPattern = re.match(filename, name)
            hidden_file = name.startswith(".")
            if matchPattern and not (hidden_file and remove_hidden_files):
                files.append(os.path.abspath(os.path.join(dirpath, name)))
        if not recursive:
            break
    return files


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
    # print(str_time)
    return str_time


def GetDBNameFromTime(t):
    return "nectarcam_monitoring_db_" + GetDAQDateFromTime(t) + ".sqlite"
