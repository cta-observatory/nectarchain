# Author: Julian Hamo
# Contact mail: julian.hamo@ijclab.in2p3.fr

"""
This module stores the Bokeh webpage header helpers for the RTA of NectarCAM.
"""

# imports
import logging
import os
import time

# Bokeh imports
from bokeh.models import Select, Div
from bokeh.layouts import column

# Bokeh RTA imports
from ..update_helpers import (
    start_periodic_updates,
    stop_periodic_updates,
    update_all_figures,
)
from ..data_fetch_helpers import (
    safe_close_file,
    open_file_from_selection,
    _get_latest_file,
)

__all__ = ["make_select_run", "make_status_col", "make_header_menu"]

logger = logging.getLogger(__name__)


def _list_runs(ressource_path, extension=".h5"):
    """List the name of the files in the directory of path ```ressource_path```
    with the format ```extension```.

    Parameters
    ----------
    ressource_path : string
        Path of the directory where to find the files to list.
        Can be relative or absolute (careful if it is relative,
        might be an issue for portability).
    extension : string, optional
        Extension of the format of files to be listed.
        Default is .h5.

    Returns
    -------
    filenames : array_like
        Array of the names of the files fitting the requirements,
        without the extension format.

    Examples
    --------
    >>> ressource_path = "../../example_data"
    >>> len(_list_runs(ressource_path))
    5

    """

    try:
        return [
            item[: -len(extension)]
            for item in os.listdir(ressource_path)
            if item.endswith(extension)
        ]
    except Exception:
        pass


def _set_status_text(msgs, status_col=None):
    """Helper to set status in header if status_col exists.

    Parameters
    ----------
    msg_file : list of string
        Text to update the file_div and time_div in status_col
    status_col : column, optional
        Bokeh column layout for the status of the webpage.
        Default is None, meaning nothing is happening.

    Returns
    -------
    out : None

    """

    try:
        # check if status_col has text attribute to be updated
        if hasattr(status_col, "children") and len(status_col.children) > 1:
            for index in range(2):
                child = status_col.children[index]
                if hasattr(child, "text"):
                    child.text = msgs[index]
    # error raised
    except Exception:
        pass
    return


def _on_header_select_change(
    attr,
    old,
    new,
    fobj=None,
    fpath=None,
    ressource_path=None,
    status_col=None,
    real_time_tag=None,
    default_update_ms=None,
    display_registry=None,
    widgets=None,
    time_parentkeys=None,
    time_childkeys=None,
):
    """Callback when header selector changes value.
    Select run to be displayed or real time mode.

    Parameters
    ----------
    attr, old, new :
        Arguments for the Bokeh .on_change method
    fobj : hdf5 file, optional
        File to open, default is None.
    fpath : string, optional
        Path of fobj, default is None.
    ressource_path : string, optional
        Path of directory to browse for files, default is None.
    status_col : Div, optional
        Bokeh column with the divider for the status of the webpage.
        Default is None.
    real_time_tag: string, optional
        Tag representing the real-time mode.
        Stored in static.constants.json.
        Default is None.
    default_update_ms: int, optional
        Refresh period in real-time mode.
        Stored in static.constants.json.
        Default is None.
    display_registry : list
        Storage of all the displays for later update.
    widgets : dict
        Storage of all the widgets for later use and update.
    time_parentkeys : list of strings, optional
        Parentkeys of data that can be time ordered.
        Default is ``None``, meaning nothing to be sorted.
    time_childkeys : list of strings, optional
        Childkeys of data that can be time ordered.
        Default is ``None``, meaning nothing to be sorted.

    Returns
    -------
    out : hdf5 file, string
        Opened file and its path.
        Every failure results in None, None.

    """

    sel = new

    # close previously opened non-latest file (if any)
    # (for now consider only format handled i.e .h5 files)
    try:
        if fobj is not None:
            safe_close_file(fobj)
    except Exception:
        pass

    # open right file depending on the Select value
    if sel == real_time_tag:
        logger.info(f"Real-time mode: {time.strftime('%H:%M:%S')}")
        try:
            # real time will have to be replaced by ...data_fetch_helpers.fetch_stream()
            fobj, fpath = open_file_from_selection(
                sel,
                ressource_path=ressource_path,
                real_time_tag=real_time_tag,
            )
            # Update and start periodic updates
            update_all_figures(fobj, display_registry=display_registry, widgets=widgets)
            widgets["PERIODIC_CB_ID"] = start_periodic_updates(
                file=fobj,
                display_registry=display_registry,
                widgets=widgets,
                status_col=status_col,
                interval_ms=default_update_ms,
            )
            # Display
            _set_status_text(
                [
                    f"Loaded file: {fobj.filename}",
                    f"Last update: {time.strftime('%H:%M:%S')}",
                ],
                status_col=status_col,
            )
            return fobj, fpath
        except Exception as e:
            _set_status_text(
                [f"Failed to start real-time mode: {e}"] * 2, status_col=status_col
            )
            logger.warning(f"Failed to start real-time mode: {e}")
            return None, None
    else:
        logger.info(f"Reading mode (filename: {sel}): {time.strftime('%H:%M:%S')}")
        # open the selected file and update once (no periodic updates)'
        try:
            widgets["PERIODIC_CB_ID"] = stop_periodic_updates(widgets)
            fobj, fpath = open_file_from_selection(
                sel,
                ressource_path=ressource_path,
                real_time_tag=real_time_tag,
                time_parentkeys=time_parentkeys,
                time_childkeys=time_childkeys,
            )
            # issue with opening the specific file
            if fobj is None:
                _set_status_text(
                    [f"Could not open selected file: {sel}"] * 2, status_col=status_col
                )
                return None, None
            update_all_figures(fobj, display_registry=display_registry, widgets=widgets)
            _set_status_text(
                [
                    f"Loaded file: {fobj.filename}",
                    f"Last update: {time.strftime('%H:%M:%S')}",
                ],
                status_col=status_col,
            )
            return fobj, fpath
        # errors raised
        except Exception as e:
            _set_status_text(
                [f"Could not open selected file: {sel} : {e}"] * 2,
                status_col=status_col,
            )
            logger.warning(f"Could not open selected file: {e}")
            return None, None


def make_select_run(list_file, real_time_tag):
    """Create the Select widget for RTA modes.

    Modes:
        - Real time: retrieve data from RTA stream to display.
        - Reading mode: retrieve data from selected file to display.

    Parameters
    ----------
    list_file : array_like
        List of files in the storage directory.
    real_time_tag : string
        Tag representing the real-time mode.
        Stored in static.constants.json.

    Returns
    -------
    out : Select
        Bokeh Select widget to display for selection.

    """

    return Select(
        title="Run selected:",
        value=real_time_tag,
        options=list(list_file) + [real_time_tag],
    )


def make_status_col(file):
    """Create the column layout of the metadata depending on the mode.

    Modes:
        - Real time: the listened stream.
        - Reading mode: the selected file.

    Parameters
    ----------
    file : hdf5 file
        File where data is stored.
        **Need to add stream too.**

    Returns
    -------
    out : column
        Bokeh column layout of the file/stream metadata.

    """

    file_div = Div(text=f"Loaded file: {file.filename}")
    time_div = Div(text=f"Last update: {time.strftime('%H:%M:%S')}")
    return column(file_div, time_div)


def make_header_menu(ressource_path, real_time_tag, file=None, extension=".h5"):
    """Create full header menu.

    Parameters
    ----------
    ressource_path : string
        Path of the directory where to find the files to list.
        Can be relative or absolute (careful if it is relative,
        might be an issue for portability).
    file : hdf5 file, optional
        File of data to display.
        Default is None, meaning last file of the storage directory is taken.
    extension : string, optional
        Extension of the format of files to be listed.
        Default is .h5.

    Returns
    -------
    out : tuple of (Select, column of Div)
        Full header menu

    """

    if file is None:
        file = _get_latest_file(ressource_path, extension=extension)
    list_file = _list_runs(ressource_path, extension=extension)
    run_choice_slidedown = make_select_run(list_file, real_time_tag)

    status_col = make_status_col(file)
    return run_choice_slidedown, status_col
