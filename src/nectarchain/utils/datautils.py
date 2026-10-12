import datetime
import logging
import os
import re
from collections.abc import Iterable
from glob import glob
from pathlib import Path

import astropy
import numpy as np
import protozfits
from astropy.time import Time
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    filename=f"{os.environ.get('NECTARCHAIN_LOG', '/tmp')}/{os.getpid()}/"
    f"{Path(__file__).stem}_{os.getpid()}.log",
    handlers=[logging.getLogger("__main__").handlers],
)
log = logging.getLogger(__name__)


def to_datetime(t):
    """Convert a datetime-like object to a timezone-aware UTC datetime.

    Parameters
    ----------
    t : datetime.datetime, astropy.time.Time, iterable, or None
        Input time to convert.

    Returns
    -------
    datetime.datetime or list or np.ndarray
        Converted datetime(s) in UTC.
    """
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
        t_corr = list(map(to_datetime, t))
    else:
        raise ValueError(
            f"tmin (type: {type(t)}) is not of type datetime --> Problem !"
        )

    if isinstance(t, np.ndarray):  # Convert to ndarray if this was given
        t_corr = np.array(t_corr)

    return t_corr


def GetDefaultDataPath(default_path="./"):
    """Get the default data path from the NECTARCAMDATA environment variable.

    Parameters
    ----------
    default_path : str, optional
        Fallback path if the environment variable is not set.

    Returns
    -------
    str
        Data directory path.
    """
    return os.environ.get("NECTARCAMDATA", default_path)


def GetRunURL(run, path):
    """Build a glob pattern for run files in a given path.

    Parameters
    ----------
    run : int
        Run number.
    path : str
        Root directory to search.

    Returns
    -------
    str
        Glob pattern matching the run's FITS files.
    """
    pattern = f"NectarCAM.Run{run:04}."
    runpath = ""
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            if f.startswith(pattern):
                runpath = dirpath
                break
    return runpath + "/" + pattern + "*.fits.fz"


def GetFirstLastEventTime(run, path=None):
    """Get the first and last event timestamps for a given run.

    Parameters
    ----------
    run : int
        Run number.
    path : str, optional
        Data directory path. Defaults to the NECTARCAMDATA path.

    Returns
    -------
    tuple of astropy.time.Time or None
        First and last event times, or None if no files are found.
    """
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
            log.info(len(evt_times))
            evt_times.sort()
            return evt_times[0], evt_times[-1]
        else:
            log.error(f"Can't find files for run {run}")
    except Exception as err:
        log.error(err)


def FindFile(filename, path):
    """Find a file by name in a directory tree.

    Parameters
    ----------
    filename : str
        Name of the file to find.
    path : str
        Root directory to search.

    Returns
    -------
    str or None
        Full path to the file, or None if not found.
    """
    for dirpath, _, filenames in os.walk(path):
        if filename in filenames:
            return os.path.join(dirpath, filename)


def FindFiles(filename, path, recursive=True, remove_hidden_files=True):
    """Find files matching a regex pattern in a directory tree.

    Parameters
    ----------
    filename : str
        Regex pattern to match file names against.
    path : str
        Root directory to search.
    recursive : bool, optional
        Whether to search subdirectories recursively.
    remove_hidden_files : bool, optional
        Whether to exclude hidden files (starting with '.').

    Returns
    -------
    list
        Sorted list of absolute paths to matching files.
    """
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
    """Convert a time to the DAQ noon-anchored datetime.

    Parameters
    ----------
    t : astropy.time.Time or datetime.datetime
        Input time.

    Returns
    -------
    datetime.datetime
        DAQ datetime (noon of the same or previous day).
    """
    if isinstance(t, astropy.time.core.Time):
        log.info("GetDAQTimeFromTime> converting to datetime")
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
    """Get the DAQ date string from a datetime.

    A datetime is expected.

    Parameters
    ----------
    t : datetime.datetime
        Input datetime.

    Returns
    -------
    str
        DAQ date in YYYY-MM-DD format.
    """

    if t.hour >= 12:
        str_time = t.strftime("%Y-%m-%d")
    else:
        t_past = t - datetime.timedelta(seconds=86400)
        str_time = t_past.strftime("%Y-%m-%d")
    # print(str_time)
    return str_time


def GetDBNameFromTime(t):
    """Get the SQLite database file name for a given time.

    Parameters
    ----------
    t : datetime.datetime
        Input datetime.

    Returns
    -------
    str
        Database file name.
    """
    return "nectarcam_monitoring_db_" + GetDAQDateFromTime(t) + ".sqlite"
