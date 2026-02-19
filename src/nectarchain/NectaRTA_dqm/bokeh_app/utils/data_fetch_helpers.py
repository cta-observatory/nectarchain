# Author: Julian Hamo
# Contact mail: julian.hamo@ijclab.in2p3.fr

"""
This module stores data fetching helpers for the RTA of NectarCAM.
"""

# imports
import os
import h5py
from pathlib import Path

# Bokeh imports
from .utils_helpers import hdf5Proxy


__all__ = ["safe_close_file", "open_file_from_selection", "fetch_stream"]


def _get_latest_file(ressource_path, extension=".h5"):
    """Open latest .h5 file from the ressource directory.

    Parameters
    ----------
    ressource_path : string
        Ressource path to find files.
    extension : string, optional
        Extension of the format for files.
        Default is .h5.

    Returns
    -------
    out : h5py.File or None
        The opened HDF5 file if successful.
        Returns ``None`` if any error occured.

    Examples
    --------
    >>> ressource_path = "../../example_data"
    >>> print(_get_latest_file(ressource_path).filename.split("/")[-1])
    dl1_sb_id_1_obs_id_20549_tel_id_1_line_idx_0thread_idx0th_file_idx5file_idx.h5

    """

    # Find the latest file
    filepath = max(
        Path(ressource_path).glob("*"+extension),
        key=lambda f: f.stat().st_mtime
    )
    try:
        # Try to open .h5 file
        return h5py.File(filepath, "r")
    except Exception as e:
        # Return None if an error occured
        print(f"_get_latest_file: failed reading {filepath}: {e}")
        return None
        
def safe_close_file(fobj):
    """Safely close ``fobj`` file. If ``fobj`` can not be closed, no nothing.

    Parameters
    ----------
    fobj: h5py.File-like or None or Any
        file object to close

    Returns
    -------
    out : None

    """

    try:
        # Defensive format handling for clsoing methods
        if fobj is not None and hasattr(fobj, "close"):
            fobj.close()
    except Exception:
        pass


def open_file_from_selection(
        sel_value, ressource_path, real_time_tag,
        extension=".h5", time_parentkeys=None, time_childkeys=None
    ):
    """Return an open h5py.File-like object for selection.
    If ``sel_value == real_time_tag``, returns ``-get_latest_file(ressource_path)``.
    Else, expects ``sel_value`` to be a filename (without path and extension).

    Parameters
    ----------
    sel_value : string
        Either ```real_time_tag``` or name of the file to load.
    ressource_path : string
        Ressource path to find .h5 files.
    real_time_tag : string
        Tag representing the real-time mode.
        Stored in static.constants.json.
        Default is None.
    extension : string, optional
        Extension of the format for files.
        Default is .h5.
    time_parentkeys : list of strings, optional
        Parentkeys of data that can be time ordered.
        Default is ``None``, meaning nothing to be sorted.
    time_childkeys : list of strings, optional
        Childkeys of data that can be time ordered.
        Default is ``None``, meaning nothing to be sorted.

    Returns
    -------
    file : h5py.File or None
        The opened file if successful, otherwise ``None``.
    path : str or None
        The resolved path to the file, otherwise ``None``.

    """
    
    if sel_value is None:
        return None, None

    if sel_value == real_time_tag:
        file = _get_latest_file(ressource_path, extension=extension)
        # Might not be useful to sort by time
        # if the final goal is to listen to stream directly.
        # fileproxy = hdf5Proxy(file)
        # for time_parentkey, time_childkey in zip(time_parentkeys, time_childkeys):
        #     fileproxy[time_parentkey].sort_from_key(time_childkey)
        # path = getattr(fileproxy, "filename", None)
        path = getattr(file, "filename", None)
        return file, path

    try:
        filepath = (Path(ressource_path) / (sel_value + extension)).resolve()
    except Exception as e:
        print(f"open_file_for_selection: failed file opening: {e}")
        return None, None

    try:
        if os.path.exists(filepath):
            file = h5py.File(filepath, "r")
            fileproxy = hdf5Proxy(file)
            for time_parentkey, time_childkey in zip(time_parentkeys, time_childkeys):
                fileproxy[time_parentkey].sort_from_key(time_childkey)
            return fileproxy, filepath
        else:
            print(f"open_file_for_selection: failed file opening: {filepath} not found.")
            return None, None
    except Exception as e:
        print(f"open_file_for_selection: failed file opening: {e}")
        return None, None
    
def fetch_stream():
    pass