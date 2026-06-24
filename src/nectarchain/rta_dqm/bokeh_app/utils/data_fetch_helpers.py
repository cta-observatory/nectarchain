"""
This module stores data fetching helpers for the RTA of NectarCAM.
"""

import logging

# imports
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import h5py
from watchdog.events import FileSystemEventHandler

# Bokeh RTA imports
from .utils_helpers import hdf5Proxy

__all__ = ["safe_close_file", "open_file_from_selection"]

logger = logging.getLogger(__name__)


class LatestFilesHandler(FileSystemEventHandler):
    def __init__(self, latest_files, n_files):
        self.latest_files = latest_files
        self.n_files = n_files

    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)

        if path.suffix != ".h5":
            return

        self.latest_files.append(path)

        while len(self.latest_files) > self.n_files:
            self.latest_files.popleft()


@contextmanager
def agglomerate_DL1(list_filepaths):
    """Aggolemerate multiple DL1 files into one"""
    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    tmp_path = tmp.name
    # logger.info(f"Building agglomerated temporary file {tmp.name}")
    tmp.close()

    out = h5py.File(tmp_path, "w")

    f_out = None

    try:
        for i, fp in enumerate(list_filepaths):
            with h5py.File(fp, "r") as f:
                grp = out.create_group(f"file_{i}")
                for name in f:
                    f.copy(name, grp, name=name)

        out.flush()

        f_out = h5py.File(tmp_path, "r")
        yield f_out

    finally:
        if f_out is not None:
            f_out.close()
        out.close()
        os.remove(tmp_path)


def _get_latest_file(
    resource_path, extension=".h5", n_rec=10, n_writing_files=8, n_files=15
):
    """Open latest .h5 file from the resource directory.

    Parameters
    ----------
    resource_path : string
        resource path to find files.
    extension : string, optional
        Extension of the format for files.
        Default is .h5.
    n_rec : int, optional
        Maximum number of retries of search of the files before raising a warning.
        Default is 10.
    n_writing_files : int, optional
        Number of files to be written at the same time.
        Corresponds to number of lines times number of threads.
        Default is `2*4=8`.
    n_files : int, optional
        Maximum number of files to agglomerate.
        Default is 10.

    Returns
    -------
    out : h5py.File or None
        The opened HDF5 file if successful.
        Returns ``None`` if any error occured.

    Examples
    --------
    >>> resource_path = "../../example_data"
    >>> print(_get_latest_file(resource_path).filename.split("/")[-1])
    dl1_sb_id_1_obs_id_20549_tel_id_1_line_idx_0thread_idx0th_file_idx5file_idx.h5

    """

    # Find the latest file
    rec = 0
    while (
        len(list(Path(resource_path).glob("*" + extension))) <= n_writing_files
        and rec < n_rec
    ):
        time.sleep(10)
        rec += 1
        logger.info(
            f"Empty DL1 directory {resource_path} - attempt {rec+1}: sleeping 10s..."
        )
    if (
        rec >= n_rec
        and len(list(Path(resource_path).glob("*" + extension))) <= n_writing_files
    ):
        logger.warning(f"_get_latest_file: failed reading {resource_path}")
        return None
    filepaths = sorted(
        Path(resource_path).glob("*" + extension), key=lambda f: f.stat().st_mtime
    )
    try:
        # Try to open .h5 second to last file
        return agglomerate_DL1(
            filepaths[-(n_writing_files + n_files + 1) : -(n_writing_files + 1)]
        )
    except Exception as e:
        # Return None if an error occured
        logger.warning(
            f"""
_get_latest_file: failed reading files \
{filepaths[-(n_writing_files+n_files+1):-(n_writing_files+1)]}: {e}
            """
        )
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
    sel_value,
    resource_path,
    real_time_tag,
    extension=".h5",
    time_parentkey=None,
    time_childkey=None,
    group_parentkeys=None,
):
    """Return an open h5py.File-like object for selection.
    If ``sel_value == real_time_tag``, returns ``-get_latest_file(resource_path)``.
    Else, expects ``sel_value`` to be a filename (without path and extension).

    Parameters
    ----------
    sel_value : string
        Either ```real_time_tag``` or name of the file to load.
    resource_path : string
        Resource path to find .h5 files.
    real_time_tag : string
        Tag representing the real-time mode.
        Stored in static.constants.json.
        Default is None.
    extension : string, optional
        Extension of the format for files.
        Default is .h5.
    time_parentkey : string, optional
        Parentkey for time to sort data.
        Default is ``None``, meaning nothing is sorted.
    time_childkey : string, optional
        Childkey for time to sort data.
        Default is ``None``, meaning nothing is sorted.
    group_parentkeys : list of strings, optional
        Parentkeys of data to be time ordered.
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
        with _get_latest_file(resource_path, extension=extension) as file:
            fileproxy = hdf5Proxy(file)
        if time_parentkey is not None and time_childkey is not None:
            try:
                sort_indexes = fileproxy[time_parentkey].sort_from_key(time_childkey)
                for group_parentkey in group_parentkeys:
                    fileproxy[group_parentkey].mask(sort_indexes)
            except Exception as e:
                logger.warning(
                    f"open_file_for_selection: failed data time sorting: {e}"
                )
        path = getattr(fileproxy, "filename", None)
        return fileproxy, path

    try:
        filepath = (Path(resource_path) / (sel_value + extension)).resolve()
    except Exception as e:
        logger.warning(f"open_file_for_selection: failed file opening: {e}")
        return None, None

    try:
        if os.path.exists(filepath):
            file = h5py.File(filepath, "r")
            fileproxy = hdf5Proxy(file)
            if time_parentkey is not None and time_childkey is not None:
                try:
                    sort_indexes = fileproxy[time_parentkey].sort_from_key(
                        time_childkey
                    )
                    for group_parentkey in group_parentkeys:
                        fileproxy[group_parentkey].mask(sort_indexes)
                except Exception as e:
                    logger.warning(
                        f"open_file_for_selection: failed data time sorting: {e}"
                    )
            return fileproxy, filepath
        else:
            logger.warning(
                f"open_file_for_selection: failed file opening: {filepath} not found."
            )
            return None, None
    except Exception as e:
        logger.warning(f"open_file_for_selection: failed file opening: {e}")
        return None, None
