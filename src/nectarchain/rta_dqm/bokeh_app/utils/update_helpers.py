"""
This module stores Bokeh webpage update helpers for the RTA of NectarCAM.
"""

# imports
import logging
import time
from functools import partial

# Bokeh imports
from bokeh.plotting import curdoc

from .data_fetch_helpers import _get_latest_file
from .display.camera_mapping import update_camera_display
from .display.histogram import (
    update_annulus,
    update_display_hist,
    update_display_hist_for_1d,
)
from .display.summary import update_summary_card
from .display.timeline import update_timelines

# Bokeh RTA imports
from .utils_helpers import hdf5Proxy

__all__ = ["periodic_update_display", "start_periodic_updates", "stop_periodic_updates"]

logger = logging.getLogger(__name__)


def update_all_figures(file, display_registry, widgets):
    """Update all displays in display_registry according to their ._meta
    using data from file and widgets.
    Only figures are updated (widgets are left untouched).

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    display_registry : list
        Storage of all the displays for later update.
    widgets : list
        Storage of all interactive widgets for manual update.

    Returns
    -------
    out : None

    """

    for disp in display_registry:
        meta = getattr(disp, "_meta", None)
        if not meta:
            continue

        dtype = meta.get("type", "").lower()

        if dtype == "summary_card":
            run_widget = widgets.get("camera_run")
            run_index = (
                int(run_widget.value)
                if getattr(run_widget, "value", None) is not None
                else -1
            )
            parent = meta.get("parentkey")
            child = meta.get("childkey")
            update_summary_card(disp, file, parent, child, run_index=run_index)

        if dtype == "hist_1d":
            parent = meta.get("parentkey")
            child = meta.get("childkey")
            label = meta.get("label", meta.get("childkey", "value"))
            # read widget values (fall back to meta defaults)
            n_bins_widget = widgets.get("hist_bins")
            n_bins = (
                int(n_bins_widget.value)
                if getattr(n_bins_widget, "value", None) is not None
                else meta.get("n_bins", 50)
            )
            update_display_hist_for_1d(disp, parent, child, file, label, n_bins)

        elif dtype == "hist_avg":
            parent = meta.get("parentkey")
            child = meta.get("childkey")
            label = meta.get("label", meta.get("childkey", "value"))
            # read widget values (fall back to meta defaults)
            n_runs_widget = widgets.get("hist_runs")
            n_bins_widget = widgets.get("hist_bins")
            n_runs = (
                int(n_runs_widget.value)
                if getattr(n_runs_widget, "value", None) is not None
                else meta.get("n_runs", 1)
            )
            n_bins = (
                int(n_bins_widget.value)
                if getattr(n_bins_widget, "value", None) is not None
                else meta.get("n_bins", 50)
            )
            update_display_hist(disp, parent, child, file, label, n_runs, n_bins)

        elif dtype == "annulus":
            parent = meta.get("parentkey")
            child = meta.get("childkey")
            update_annulus(disp, parent, child, file)

        elif dtype == "camera":
            # choose run index from widget if present
            run_widget = widgets.get("camera_run")
            run_index = (
                int(run_widget.value)
                if getattr(run_widget, "value", None) is not None
                else -1
            )
            image_parent = meta.get("image_parentkey")
            param_parent = meta.get("parameter_parentkeys")
            child = meta.get("childkey")
            param = meta.get("parameterkeys")
            update_camera_display(
                disp,
                child,
                image_parent,
                param_parent,
                param,
                file,
                run_index=run_index,
            )

        elif dtype.startswith("timeline"):
            parent = meta.get("parentkey")
            child = meta.get("childkey")
            time_parent = meta.get("time_parentkey")
            time_child = meta.get("time_childkey")
            cached_timelines = meta.get("cached_timelines")
            update_timelines(
                disp, cached_timelines, parent, child, time_parent, time_child, file
            )


def update_timestamp(status_col):
    """Update the time of last update of the page.

    Parameters
    ----------
    status_col : column
        Bokeh column layout for the status of the webpage.

    Returns
    -------
    out : None

    """

    ts = time.strftime("%H:%M:%S")
    logger.info(f"Real time mode: updating figure - {ts}")
    status_col.children[1].text = f"Last update: {ts}"


def periodic_update_display(
    file_list,
    display_registry,
    widgets,
    status_col,
    time_parentkey=None,
    time_childkey=None,
):
    """Update all the figures of the webpage and the status divider.

    Parameters
    ----------
    file_list : list
        list of the files to agglomerate.
    display_registry : list
        Storage of all the displays for later update.
    widgets : list
        Storage of all interactive widgets for manual update.
    status_col : column
        Bokeh column layout for the status of the webpage.
    time_parentkey : string, optional
        Parentkey for time to sort data.
        Default is ``None``, meaning nothing is sorted.
    time_childkey : string, optional
        Childkey for time to sort data.
        Default is ``None``, meaning nothing is sorted.

    Returns
    -------
    out : None

    """
    with _get_latest_file(file_list) as fileHDF5:
        fileproxy = hdf5Proxy(fileHDF5)
    if time_parentkey is not None and time_childkey is not None:
        try:
            sort_indexes = fileproxy[time_parentkey].sort_from_key(time_childkey)
            for group_parentkey in fileproxy.parentkeys:
                if group_parentkey.startswith("dl1"):
                    fileproxy[group_parentkey].mask(sort_indexes)
        except Exception as e:
            logger.warning(f"periodic_update_display: failed data time sorting: {e}")
    update_all_figures(fileproxy, display_registry, widgets)
    update_timestamp(status_col)


def start_periodic_updates(
    file_list,
    resource_path,
    display_registry,
    widgets,
    status_col,
    interval_ms=1000,
    time_parentkey=None,
    time_childkey=None,
):
    """Start the periodic update of the webpage every ``interval_ms`` milliseconds.

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    display_registry : list
        Storage of all the displays for later update.
    widgets : list
        Storage of all interactive widgets for manual update.
    status_col : column
        Bokeh column layout for the status of the webpage.
    interval_ms : int, optional
        Interval between each refresh, in milliseconds.
        Default is 1000, i.e. every second.
    time_parentkey : string, optional
        Parentkey for time to sort data.
        Default is ``None``, meaning nothing is sorted.
    time_childkey : string, optional
        Childkey for time to sort data.
        Default is ``None``, meaning nothing is sorted.

    Returns
    -------
    periodic_cb_id : callback
        Callback function used for the update

    """

    periodic_cb_id = curdoc().add_periodic_callback(
        partial(
            periodic_update_display,
            file_list=file_list,
            display_registry=display_registry,
            widgets=widgets,
            status_col=status_col,
            time_parentkey=time_parentkey,
            time_childkey=time_childkey,
        ),
        interval_ms,
    )
    widgets["PERIODIC_CB_ID"] = periodic_cb_id
    return


def stop_periodic_updates(widgets):
    """Stop the periodic update of the webpage.

    Parameters
    ----------
    widgets : list
        Storage of all interactive widgets for manual update.

    Returns
    -------
    out : None

    """

    try:
        curdoc().remove_periodic_callback(widgets["PERIODIC_CB_ID"])
    except Exception:
        pass
    widgets["PERIODIC_CB_ID"] = None
    logger.info(f"Periodic updates stopped (id={None})")
    return
